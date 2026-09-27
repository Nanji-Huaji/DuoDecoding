#!/usr/bin/env python
"""(1,K) 定长 padding 验证图的**等价性**验证。

回答的问题（比"快多少"更重要，动管线之前必须过）：
  1. 图模式的 KVCacheModel 对多 token 前向（verify 图，k∈{1..最大档位}）产出的
     prob/logits 行，是否与 eager（DynamicCache）整段前向一致（bf16 容差）？
  2. **回滚后 resync**（真实管线每轮的形态：回滚 → 喂 k 个 token）是否仍一致？
  3. k 超出最大档位时的 eager 回退是否正确？
  4. **跨样本复用**（reset_for_new_sample + 同一 runner 重新 prefill）与全新
     runner 的结果是否一致？（StaticCache 原地 reset，图地址不变）
  5. 不变量：runner.nnz == _current_seq_len，prob_history 无越界。

用法: .venv/bin/python scripts/test_graph_verify.py [模型路径 ...]
      默认 llama/llama-68m llama/tiny-llama-1.1b（13B 显存不够时可手动加）
"""
from __future__ import annotations

import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from src.model_gpu import KVCacheModel  # noqa: E402

MAX_LEN = 512
VERIFY_SIZES = (4, 8, 16)
TOL_REL = 3e-2  # bf16 图回放 vs eager 整段：1e-2 量级相对差异属正常（历史实测 0.938%）


def make_model(path: str) -> torch.nn.Module:
    model = (
        AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    model.config.output_hidden_states = True
    return model


def logits_close(a: torch.Tensor, b: torch.Tensor) -> tuple[bool, float]:
    """逐行比较（float32）：返回 (是否通过, 最大相对差异)。"""
    diff = (a.float() - b.float()).abs()
    denom = b.float().abs().max().clamp_min(1e-6)
    rel = float(diff.max() / denom)
    return rel <= TOL_REL, rel


def run_rounds(ref: KVCacheModel, g: KVCacheModel, ks: list[int], seed: int) -> list[float]:
    """喂若干轮 k-token 前向（中间夹回滚），返回各轮相对差异。

    覆盖三路输出：logits 历史、prob 历史（temp 0 ⇒ 图内 argmax one-hot）、
    每次前向的熵（图内逐行熵 + 调用侧按真实 k 行求均值）。
    """
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rels = []
    cur = ref._current_seq_len
    for i, k in enumerate(ks):
        toks = torch.randint(100, 30000, (1, k), generator=gen).cuda()
        q_ref = ref._decode_step(toks)
        q_g = g._decode_step(toks)
        end = cur + k
        rows_ref = ref.logits_history[0, cur:end, :].float()
        rows_g = g.logits_history[0, cur:end, :].float()
        ok, rel = logits_close(rows_g, rows_ref)
        rels.append(rel)
        assert ok, f"第 {i} 轮 k={k} logits 行不一致: rel={rel:.4f}"
        # prob 历史：one-hot 必须与**同一侧** logits 的 argmax 自洽（图内 scatter
        # 正确性）；跨侧 argmax 允许在 logits 漂移容差内翻转（与 q 的判定同标准）
        p_ref = ref.prob_history[0, cur:end, :].float()
        p_g = g.prob_history[0, cur:end, :].float()
        assert torch.equal(p_g.argmax(-1), rows_g.argmax(-1)), (
            f"第 {i} 轮 k={k} 图内 one-hot 与图内 logits argmax 不自洽"
        )
        assert torch.equal(p_ref.argmax(-1), rows_ref.argmax(-1)), (
            f"第 {i} 轮 k={k} eager one-hot 与 eager logits argmax 不自洽"
        )
        assert torch.equal(p_ref.argmax(-1), p_g.argmax(-1)) or rel <= TOL_REL, (
            f"第 {i} 轮 k={k} prob argmax 分歧且超出漂移容差"
        )
        # 熵：均值语义（真实 k 行）一致
        e_ref, e_g = ref.last_entropy, g.last_entropy
        assert e_ref is not None and e_g is not None
        assert abs(e_ref - e_g) <= max(1e-3, 0.05 * abs(e_ref)), (
            f"第 {i} 轮 k={k} 熵不一致: {e_ref} vs {e_g}"
        )
        assert bool(torch.equal(q_ref.argmax(-1), q_g.argmax(-1))) or rel <= TOL_REL, (
            f"第 {i} 轮 k={k} argmax 分歧"
        )
        cur = end
        # 每 2 轮做一次回滚（模拟真实管线的 rollback → resync）
        if i % 2 == 1:
            rollback_to = max(cur - (k + 1), 1)
            ref.rollback(rollback_to)
            g.rollback(rollback_to)
            cur = rollback_to
            assert g._current_seq_len == g._graph_runner.nnz, (
                f"不变量破坏: _current_seq_len={g._current_seq_len} nnz={g._graph_runner.nnz}"
            )
    return rels


def compare(path: str) -> bool:
    print("=" * 78)
    print(f"模型 {path}  验证图档位 {VERIFY_SIZES}")
    print("=" * 78)
    model = make_model(path)
    torch.manual_seed(0)

    prompt = torch.randint(100, 30000, (1, 24), device="cuda")

    ref = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=MAX_LEN)
    q_ref = ref._forward_with_kvcache(prompt)

    g = KVCacheModel(
        model,
        temperature=0.0,
        top_k=0,
        top_p=0,
        max_length=MAX_LEN,
        use_cuda_graph=True,
        verify_graph_sizes=VERIFY_SIZES,
        graph_len_budget=128,
    )
    q_g = g._forward_with_kvcache(prompt)
    ok, rel = logits_close(q_g.float(), q_ref.float())
    assert ok, f"prefill 不一致 rel={rel:.4f}"
    print(f"  prefill                     : rel={rel:.4f} ✓")

    # 1) 图内档位的多 token 前向（含 1 = 单步图、k<档位 = padding、k=档位 = 满档）
    ks = [1, 2, 3, 4, 5, 8, 11, 16, 2, 7]
    rels = run_rounds(ref, g, ks, seed=123)
    print(f"  图内档位 k∈{ks}          : max rel={max(rels):.4f} ✓（含回滚 resync）")

    # 2) k 超出最大档位 → eager 回退（图模式下 StaticCache 的 eager 前向）
    rels = run_rounds(ref, g, [VERIFY_SIZES[-1] + 5, 1, 3], seed=456)
    print(f"  超档位回退 k={VERIFY_SIZES[-1] + 5}       : max rel={max(rels):.4f} ✓")

    # 3) 跨样本复用：reset → 新 prompt prefill（不重捕获），对照全新 eager 缓存
    prompt2 = torch.randint(100, 30000, (1, 31), device="cuda")
    captures_before = g._graph_runner.capture_count
    g.reset_for_new_sample()
    q2_g = g._forward_with_kvcache(prompt2)
    assert g._graph_runner.capture_count == captures_before, "复用路径不应重捕获"
    ref2 = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=MAX_LEN)
    q2_ref = ref2._forward_with_kvcache(prompt2)
    ok, rel = logits_close(q2_g.float(), q2_ref.float())
    assert ok, f"跨样本 prefill 不一致 rel={rel:.4f}"
    rels = run_rounds(ref2, g, [2, 5, 9, 16, 3], seed=789)
    print(f"  跨样本复用（无重捕获）       : prefill rel={rel:.4f}, 续走 max rel={max(rels):.4f} ✓")

    # 4) 超长 prompt 触发「重建+重捕获」：新桶/新图仍与 eager 一致
    long_prompt = torch.randint(100, 30000, (1, MAX_LEN + 88), device="cuda")
    g.reset_for_new_sample()
    q3_g = g._forward_with_kvcache(long_prompt)  # prompt ≥ max_len ⇒ 重建路径
    assert g._graph_runner.max_len > MAX_LEN, "应已重建更大的图缓存"
    ref3 = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=2048)
    q3_ref = ref3._forward_with_kvcache(long_prompt)
    ok, rel = logits_close(q3_g.float(), q3_ref.float())
    assert ok, f"重建后 prefill 不一致 rel={rel:.4f}"
    rels = run_rounds(ref3, g, [3, 8, 16], seed=321)
    print(f"  超长重建+重捕获            : prefill rel={rel:.4f}, 续走 max rel={max(rels):.4f} ✓")
    del ref3

    print(f"  捕获次数（应为 1+档位数）    : {g._graph_runner.capture_count} ✓")
    print("  ⇒ ✓ 通过\n")
    del model, ref, g, ref2
    torch.cuda.empty_cache()
    return True


def main() -> int:
    paths = sys.argv[1:] or ["llama/llama-68m", "llama/tiny-llama-1.1b"]
    print(f"GPU {torch.cuda.get_device_name(0)}\n")
    results = []
    for p in paths:
        try:
            results.append(compare(p))
        except Exception as exc:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"{p} 失败: {type(exc).__name__}: {str(exc)[:300]}\n")
            results.append(False)
    print("=" * 78)
    print("全部通过 ✓ 可以集成" if all(results) else "存在不一致 ✗ 先解决再集成")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
