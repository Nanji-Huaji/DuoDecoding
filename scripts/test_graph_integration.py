#!/usr/bin/env python
"""CUDA Graph 集成测试：走**真实的 KVCacheModel API**，比对开关打开前后的行为。

与 test_graph_decode.py 的分工：
  · test_graph_decode.py 验证 GraphDecodeRunner 这个组件本身与 eager 等价；
  · 本脚本验证**集成后**的 KVCacheModel（_prefill / _generate_with_kvcache /
    rollback / current_length）在 use_cuda_graph 开关两侧完全一致。

模拟真实解码循环：每轮 generate(gamma) 拿到 gamma 个草稿 token，再回滚到
实际接受的位置继续 —— 回滚是投机解码里最容易出错的地方。

用法: .venv/bin/python scripts/test_graph_integration.py [模型路径]
"""
from __future__ import annotations

import sys
import time

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from src.model_gpu import KVCacheModel  # noqa: E402

MAX_LEN = 128


def make_cache(model, use_graph: bool) -> KVCacheModel:
    return KVCacheModel(
        model, temperature=0.0, top_k=0, top_p=0,
        max_length=MAX_LEN, use_cuda_graph=use_graph,
    )


def run_rounds(cache: KVCacheModel, prompt: torch.Tensor, gamma: int, rounds: int):
    """模拟投机解码：每轮生成 gamma 个 token，回滚到随机接受长度，再继续。"""
    out_tokens = []
    out_hidden = []
    cache._forward_with_kvcache(prompt)
    torch.manual_seed(1234)  # 回滚长度序列在两条路径上必须一致
    accepts = [torch.randint(1, gamma, (1,)).item() for _ in range(rounds)]
    torch.manual_seed(0)

    x = prompt
    for r in range(rounds):
        seq = cache.generate(x, gamma)
        new = seq[:, x.shape[1]:]
        out_tokens.append(new.clone())
        h = cache.hidden_states[-1][:, -1, :].float().clone()
        out_hidden.append(h)
        # 回滚到"接受"位置（模拟真实拒绝采样后的裁剪）
        keep = int(accepts[r])
        end_pos = cache.current_length - (gamma - keep)
        cache.rollback(end_pos)
        x = seq[:, : end_pos]
    return out_tokens, out_hidden, accepts


def compare(path: str, gamma: int = 16, rounds: int = 8) -> bool:
    print("=" * 78)
    print(f"模型 {path}   gamma={gamma}  rounds={rounds}")
    print("=" * 78)
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    model.config.output_hidden_states = True
    torch.manual_seed(0)
    prompt = torch.randint(100, 30000, (1, 8), device="cuda")

    # ── eager ──────────────────────────────────────────────────────────
    eager = make_cache(model, use_graph=False)
    t0 = time.perf_counter()
    e_tokens, e_hidden, accepts = run_rounds(eager, prompt, gamma, rounds)
    torch.cuda.synchronize()
    eager_s = time.perf_counter() - t0

    # ── graph ──────────────────────────────────────────────────────────
    graph = make_cache(model, use_graph=True)
    t0 = time.perf_counter()
    g_tokens, g_hidden, _ = run_rounds(graph, prompt, gamma, rounds)
    torch.cuda.synchronize()
    graph_s = time.perf_counter() - t0

    tok_same = all(bool(torch.equal(a, b)) for a, b in zip(e_tokens, g_tokens))
    n_tok = sum(t.numel() for t in e_tokens)
    agree = sum(int((a == b).sum()) for a, b in zip(e_tokens, g_tokens)) / max(n_tok, 1)
    hid_diff = max(float((a - b).abs().max()) for a, b in zip(e_hidden, g_hidden))

    print(f"  回滚长度序列（两条路径共用）: {accepts}")
    print(f"  逐 token 完全一致 : {tok_same}   一致率 {agree * 100:.1f}%（共 {n_tok} token）")
    print(f"  hidden |Δ| 最大   : {hid_diff:.5f}")
    print(f"  耗时 eager {eager_s * 1000:7.1f} ms   graph {graph_s * 1000:7.1f} ms"
          f"   ⇒ {eager_s / graph_s:.2f}×")
    ok = tok_same and hid_diff < 1.0
    print(f"  ⇒ {'✓ 通过' if ok else '✗ 不一致'}")
    print()
    del model, eager, graph
    torch.cuda.empty_cache()
    return ok


def main() -> int:
    paths = sys.argv[1:] or ["llama/llama-68m"]
    print(f"GPU {torch.cuda.get_device_name(0)}\n")
    results = []
    for p in paths:
        try:
            results.append(compare(p))
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            results.append(False)
    print("=" * 78)
    print("集成验证通过 ✓" if all(results) else "集成验证失败 ✗")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
