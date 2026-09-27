#!/usr/bin/env python
"""验证图长序列压力测试：模拟真实管线的几百次「随机 k 前向 + 随机回滚」。

test_graph_verify.py 覆盖形态正确性（~20 次操作）；本脚本把量级加到
300+ 次操作、序列长到 ~1200 token，k 与回滚深度都随机（贴合真实分布）。

对照基线的语义（诊断结论，见 docs/verify_graph_fix.md）：
  * **主断言：图路径 vs StaticCache-eager** —— 这是"图机器是否忠实回放"的
    正确对照，长序列上必须保持 bf16 图回放级（~1%）；
  * StaticCache vs DynamicCache 的差是**模型级数值路径性质**（注意力 kernel
    分支不同），随上下文累积到 ~4%，与图无关 —— 历史配对 A/B 已证明该差
    不改变准确率（0.25 = 0.25），只作信息打印不作断言。

不变量（每步检查）：
  * runner.nnz == _current_seq_len
  * 图路径不重捕获（capture_count 恒定）

用法: .venv/bin/python scripts/stress_graph_verify.py [模型路径] [--ops 300]
"""
from __future__ import annotations

import argparse
import random
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from src.model_gpu import KVCacheModel  # noqa: E402

TOL_GRAPH = 2.5e-2  # 图回放 vs StaticCache-eager：bf16 图级容差。
# 依据：真实失配（KV 错位/掩码错）是 O(1) 量级（rel ~ 100%）；bf16 图回放
# 的正常漂移在 300 次操作的极值统计下到 ~2.1%（68M 实测 2.13%、1.1B 1.84%），
# 取 2.5% 留出极值余量而不放松对真 bug 的分辨力。


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", nargs="?", default="llama/tiny-llama-1.1b")
    ap.add_argument("--ops", type=int, default=300)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    print(f"GPU {torch.cuda.get_device_name(0)}  模型 {args.model}  操作数 {args.ops}")
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True,
        )
        .cuda()
        .eval()
    )
    model.config.output_hidden_states = True
    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    prompt = torch.randint(100, 30000, (1, 48), device="cuda")
    sizes = (4, 8, 16, 24, 32)
    ref = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=2048)
    g = KVCacheModel(
        model, temperature=0.0, top_k=0, top_p=0, use_cuda_graph=True,
        verify_graph_sizes=sizes, graph_len_budget=128,
    )
    # StaticCache-eager 对照：同款图模式构造，摘掉图 ⇒ 全走 StaticCache eager
    se = KVCacheModel(
        model, temperature=0.0, top_k=0, top_p=0, use_cuda_graph=True,
        verify_graph_sizes=sizes, graph_len_budget=128,
    )
    ref._forward_with_kvcache(prompt.clone())
    g._forward_with_kvcache(prompt.clone())
    se._forward_with_kvcache(prompt.clone())
    se._graph_runner.graph = None
    captures0 = g._graph_runner.capture_count

    max_rel_graph = 0.0
    max_rel_static_vs_dyn = 0.0
    n_verify = n_step = n_fallback = 0
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    for op in range(args.ops):
        # 随机操作：1/3 概率草稿单步，2/3 概率验证前向（k ∈ [1, 21]）
        if rng.random() < 0.33:
            k = 1
        else:
            k = rng.randint(1, 21)
        toks = torch.randint(100, 30000, (1, k), generator=gen).cuda()
        cur = ref._current_seq_len
        ref._decode_step(toks)
        g._decode_step(toks)
        se._decode_step(toks)
        if k == 1:
            n_step += 1
        elif g._graph_runner._pick_verify_size(k) is not None:
            n_verify += 1
        else:
            n_fallback += 1

        end = cur + k
        row_se = se.logits_history[0, cur:end, :].float()
        row_dyn = ref.logits_history[0, cur:end, :].float()
        rel_graph = float(
            (g.logits_history[0, cur:end, :].float() - row_se).abs().max()
            / row_se.abs().max().clamp_min(1e-6)
        )
        max_rel_graph = max(max_rel_graph, rel_graph)
        max_rel_static_vs_dyn = max(
            max_rel_static_vs_dyn,
            float((row_se - row_dyn).abs().max() / row_dyn.abs().max().clamp_min(1e-6)),
        )
        assert rel_graph <= TOL_GRAPH, f"op{op} k={k}: 图vs静态缓存eager rel={rel_graph:.4f}"
        assert g._current_seq_len == g._graph_runner.nnz, (
            f"op{op}: nnz 不变量破坏 {g._current_seq_len} != {g._graph_runner.nnz}"
        )

        # 随机回滚：保留前 1..k 个新 token（模拟接受计数），或整轮回退
        if rng.random() < 0.8:
            rb = cur + rng.randint(0, k)
            ref.rollback(rb)
            g.rollback(rb)
            se.rollback(rb)
            assert g._current_seq_len == g._graph_runner.nnz
        # 序列长度护栏：贴近图缓存桶容量就回滚（桶 = 512；越桶会触发图模式的
        # eager 回退容量护栏 —— 那是另一个测试该管的事）
        cap = g._graph_runner.max_len - 64
        if ref._current_seq_len > cap:
            ref.rollback(60)
            g.rollback(60)
            se.rollback(60)

    assert g._graph_runner.capture_count == captures0, "压力过程中不应重捕获"
    print(f"  操作 {args.ops} 次（单步 {n_step} / 验证图 {n_verify} / 超档回退 {n_fallback}）")
    print(f"  图 vs StaticCache-eager   max rel = {max_rel_graph:.4f}（容差 {TOL_GRAPH}）← 主断言")
    print(f"  StaticCache vs Dynamic    max rel = {max_rel_static_vs_dyn:.4f}（模型级数值路径差，仅信息）")
    print(f"  nnz 不变量全程保持 ✓，无重捕获 ✓（capture_count={captures0}）")
    print("  ⇒ ✓ 压力测试通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())

