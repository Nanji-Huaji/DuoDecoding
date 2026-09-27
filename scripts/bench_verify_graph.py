#!/usr/bin/env python
"""(1,K) padding 验证图 vs eager 整段前向 的微基准。

回答的问题：真实管线里每轮的「resync+草稿 合成验证前向」（k≈接受+γ），
图回放比 eager 整段快多少？以及单步图、跨样本 prefill（不重捕获）各多少。

模拟真实轮形态（不是孤立计时）：
    verify k=17 → rollback(2) → resync k=2 → step ×2 → rollback
eager 用 DynamicCache（与现网一致），图用 StaticCache+padding 档位。

用法:
  .venv/bin/python scripts/bench_verify_graph.py                    # 68m + 1.1b
  .venv/bin/python scripts/bench_verify_graph.py llama/tiny-llama-1.1b
  .venv/bin/python scripts/bench_verify_graph.py --gamma1 16 llama/llama-2-13b  # 显存够时
注意：计时必须在 GPU 空闲窗口做（wait_gpu_idle.py 放行后），否则只有相对意义。
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from src.model_gpu import KVCacheModel  # noqa: E402


def bench_one(path: str, gamma1: int, n_rounds: int, prompt_len: int, quant: str = "none") -> None:
    print("=" * 78)
    print(f"模型 {path}  γ={gamma1}  轮数 {n_rounds}  量化 {quant}")
    from src.model_loading import build_quant_config

    quant_cfg = build_quant_config(path, quantization=quant, compute_dtype=torch.bfloat16)
    load_kwargs: dict = {}
    if quant_cfg is not None:
        load_kwargs["quantization_config"] = quant_cfg
    model = (
        AutoModelForCausalLM.from_pretrained(
            path, dtype=torch.bfloat16, attn_implementation="sdpa",
            local_files_only=True, **load_kwargs,
        )
        .cuda()
        .eval()
    )
    model.config.output_hidden_states = True
    sizes = sorted({8, 16, 24, 32, gamma1 + 8})
    torch.manual_seed(0)
    prompt = torch.randint(100, 30000, (1, prompt_len), device="cuda")

    def round_loop(cache: KVCacheModel, seed: int) -> float:
        gen = torch.Generator(device="cpu").manual_seed(seed)
        cache._forward_with_kvcache(prompt.clone())
        base = cache._current_seq_len
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_rounds):
            k = gamma1 + 1
            toks = torch.randint(100, 30000, (1, k), generator=gen).cuda()
            cache._decode_step(toks)          # 验证前向（resync+草稿合成）
            cache.rollback(base + 2)          # 接受 1 + bonus
            resync = torch.randint(100, 30000, (1, 2), generator=gen).cuda()
            cache._decode_step(resync)        # 下一轮 resync
            cache._decode_step(resync[:, :1]) # 草稿单步
            cache.rollback(base + 1)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n_rounds * 1000

    # 热身 + 计时：eager
    eager = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=2048)
    round_loop(eager, 0)
    eager_ms = [round_loop(eager, 0) for _ in range(3)]

    # 图（含一次性捕获成本，单独计时）
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g = KVCacheModel(
        model,
        temperature=0.0,
        top_k=0,
        top_p=0,
        use_cuda_graph=True,
        verify_graph_sizes=sizes,
        graph_len_budget=256,
    )
    round_loop(g, 0)  # 首次 prefill 内含捕获
    torch.cuda.synchronize()
    capture_s = time.perf_counter() - t0
    g_ms = [round_loop(g, 0) for _ in range(3)]

    # 跨样本复用的 prefill（不重捕获）
    g.reset_for_new_sample()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    g._forward_with_kvcache(prompt.clone())
    torch.cuda.synchronize()
    reuse_prefill_ms = (time.perf_counter() - t0) * 1000
    eager2 = KVCacheModel(model, temperature=0.0, top_k=0, top_p=0, max_length=2048)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    eager2._forward_with_kvcache(prompt.clone())
    torch.cuda.synchronize()
    eager_prefill_ms = (time.perf_counter() - t0) * 1000

    e_med, g_med = statistics.median(eager_ms), statistics.median(g_ms)
    print(f"  模拟轮（verify k={gamma1+1} + rollback + resync 2 + step）:")
    print(f"    eager : {e_med:7.2f} ms/轮")
    print(f"    graph : {g_med:7.2f} ms/轮   ⇒ {e_med / g_med:.2f}× 加速")
    print(f"  捕获成本（一次性，{1 + len(sizes)} 张图）: {capture_s:.2f} s")
    print(f"  跨样本 prefill: graph {reuse_prefill_ms:.1f} ms vs eager {eager_prefill_ms:.1f} ms"
          f"（含 graph 路径的 prob/logits 写回）")
    print()
    del model, eager, g, eager2
    torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="*", default=["llama/llama-68m", "llama/tiny-llama-1.1b"])
    ap.add_argument("--gamma1", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--prompt_len", type=int, default=96)
    ap.add_argument("--quant", type=str, default="none", help="none | 4bit | auto（镜像管线 --target_quantization）")
    args = ap.parse_args()
    print(f"GPU {torch.cuda.get_device_name(0)}\n")
    for p in args.models:
        bench_one(p, args.gamma1, args.rounds, args.prompt_len, quant=args.quant)
    return 0


if __name__ == "__main__":
    sys.exit(main())
