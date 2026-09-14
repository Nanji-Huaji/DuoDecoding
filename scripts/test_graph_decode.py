#!/usr/bin/env python
"""CUDA Graph 集成的**等价性**验证（动 src/ 之前的前置测试）。

要回答的问题（比"快多少"更重要）：
  1. StaticCache + 固定 mask + 图回放，能否产出与现有 DynamicCache eager 路径
     **逐 token 完全相同**的结果？
  2. **回滚后继续**是否仍然一致？（真实解码里每轮都会回滚，这是最易错的地方）
  3. 每步的 logits 差异有多大（数值容差）？

做法：参考路径用**真实的 KVCacheModel**（DynamicCache），图路径用 StaticCache，
喂同样的 prompt，在 --temp 0（argmax，确定性）下比对逐 token 输出。

用法: .venv/bin/python scripts/test_graph_decode.py [模型路径]
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM, StaticCache

sys.path.insert(0, ".")
from src.graph_decode import GraphDecodeRunner  # noqa: E402
from src.model_gpu import KVCacheModel  # noqa: E402

MAX_LEN = 256


def compare(path: str, prompt_len: int = 8, n_steps: int = 24, rollback_at: int = 16):
    print("=" * 78)
    print(f"模型 {path}")
    print("=" * 78)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    # 与真实加载一致：model_loading.py 用 load_kwargs 打开它，acc_head 才有输入
    model.config.output_hidden_states = True
    torch.manual_seed(0)
    prompt = torch.randint(100, 30000, (1, prompt_len), device="cuda")

    # ── 参考路径：真实 KVCacheModel（DynamicCache），temp 0 ⇒ argmax 确定性
    ref = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=MAX_LEN)
    ref_tokens, ref_logits, ref_hiddens = [], [], []
    q = ref._forward_with_kvcache(prompt)
    ref_logits.append(ref.logits_history[0, -1, :].float().clone())  # 原始 logits
    ref_hiddens.append(ref.hidden_states[-1][:, -1, :].float().clone())  # 每步都记 ✓
    tok = q.argmax(dim=-1, keepdim=True)
    ref_tokens.append(tok)
    for _ in range(n_steps - 1):
        q = ref._decode_step(tok)
        ref_logits.append(ref.logits_history[0, -1, :].float().clone())
        ref_hiddens.append(ref.hidden_states[-1][:, -1, :].float().clone())
        tok = q.argmax(dim=-1, keepdim=True)
        ref_tokens.append(tok)

    # ── 图路径
    gd = GraphDecodeRunner(model, max_len=MAX_LEN)
    gq = gd.prefill(prompt)
    g_logits = [gq.float().clone()]  # 图路径返回的就是原始 logits
    g_tokens = [gq.argmax(dim=-1, keepdim=True)]
    hidden_diffs = []
    for i in range(n_steps - 1):
        gq, g_hidden = gd.step(g_tokens[-1])
        g_logits.append(gq.float().clone())
        g_tokens.append(gq.argmax(dim=-1, keepdim=True))
        # acc_head 依赖 hidden_states[-1] ⇒ 必须与参考路径逐元素对齐。
        # 下标注意：ref_hiddens[0] 是 prefill 的输出，图第 i 步对应 ref_hiddens[i+1]。
        ref_h = ref_hiddens[i + 1].unsqueeze(0)
        hidden_diffs.append(float((ref_h - g_hidden.float()).abs().max()))

    ref_seq = torch.cat(ref_tokens, dim=1)
    g_seq = torch.cat(g_tokens, dim=1)
    same_tokens = bool(torch.equal(ref_seq, g_seq))
    agree = float((ref_seq == g_seq).float().mean())
    maxdiff = max(float((a - b).abs().max()) for a, b in zip(ref_logits, g_logits))

    print(f"  生成 {n_steps} 步")
    print(f"  逐 token 完全一致        : {same_tokens}   一致率 {agree * 100:.1f}%")
    rel = maxdiff / max(float(a.abs().max()) for a in ref_logits)
    print(f"  各步 |Δlogits| 最大绝对值  : {maxdiff:.4f}（同量纲原始 logits）")
    print(f"     相对幅度              : {rel * 100:.3f}%  ⚠ bf16 下 1e-2 量级属正常")

    # ── 回滚后继续：两边都回滚到 rollback_at，再各走 6 步
    ref.rollback(rollback_at)
    gd.rollback(rollback_at)
    r_cont, g_cont = [], []
    tok = ref_tokens[rollback_at - 1]
    for _ in range(6):
        q = ref._decode_step(tok)
        tok = q.argmax(dim=-1, keepdim=True)
        r_cont.append(tok)
        gq, _ = gd.step(g_tokens[rollback_at - 1] if not g_cont else g_cont[-1])
        gt = gq.argmax(dim=-1, keepdim=True)
        g_cont.append(gt)
    r_seq2 = torch.cat(r_cont, dim=1)
    g_seq2 = torch.cat(g_cont, dim=1)
    same_after = bool(torch.equal(r_seq2, g_seq2))
    print(f"  hidden_states[-1] |Δ| 最大 : {max(hidden_diffs):.5f}   ← acc_head 的输入")
    print(f"  回滚到 {rollback_at} 后继续 6 步一致 : {same_after}")
    if not same_after:
        print(f"    参考: {r_seq2.tolist()}")
        print(f"    图内: {g_seq2.tolist()}")

    ok = same_tokens and same_after
    print(f"  ⇒ {'✓ 通过' if ok else '✗ 不一致，不可集成'}")
    print()
    del model, ref, gd
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    paths = sys.argv[1:] or ["llama/llama-68m"]
    print(f"GPU {torch.cuda.get_device_name(0)}\n")
    results = []
    for p in paths:
        try:
            results.append(compare(p))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{p} 失败: {type(exc).__name__}: {str(exc)[:200]}\n")
            results.append(False)
    print("=" * 78)
    print("全部通过 ✓ 可以集成" if all(results) else "存在不一致 ✗ 先解决再集成")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
