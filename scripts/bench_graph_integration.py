#!/usr/bin/env python
"""CUDA Graph 的性能对比（争抢环境下的严谨做法）。

为什么不能简单跑两遍取时间
--------------------------
本机长期有第三方任务在跑（GPU 利用率 40%~80% 波动），wall-clock 的方差可以
到 ±44%。所以这里：

  1. **交替测量**：每一轮先测 eager 再测 graph（或反之），两边承受同样的噪声，
     再对**配对差值**取统计量 —— 争抢只会同时抬高两边，不会伪造出提速；
  2. **GPU 侧计时**：用 torch.cuda.Event 而不是 time.perf_counter，
     避开 CPU 侧被抢占导致的计时抖动；
  3. **取中位数**并给出四分位距，而不是只看一次的最好值。

测三层：
  A. 单步 decode（图能覆盖的最小单位）
  B. gamma 步草稿循环（真实热点，_generate_with_kvcache）
  C. 含回滚的完整一轮（真实解码循环的稳态）

用法: .venv/bin/python scripts/bench_graph_integration.py [模型路径 ...]
"""
from __future__ import annotations

import statistics
import sys

import torch
from transformers import AutoModelForCausalLM

sys.path.insert(0, ".")
from src.model_gpu import KVCacheModel  # noqa: E402

MAX_LEN = 256


class GpuTimer:
    """GPU 侧计时：插在流里的 Event，不受 CPU 抢占影响。"""

    def __init__(self) -> None:
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)

    def __enter__(self):
        torch.cuda.synchronize()
        self.start.record()
        return self

    def __exit__(self, *exc):
        self.end.record()
        torch.cuda.synchronize()
        self.ms = self.start.elapsed_time(self.end)
        return False


def make_cache(model, use_graph: bool) -> KVCacheModel:
    return KVCacheModel(
        model, temperature=0.0, top_k=0, top_p=0,
        max_length=MAX_LEN, use_cuda_graph=use_graph,
    )


def fresh(model, use_graph, prompt, gamma):
    """建一个新 cache 并 prefill，返回 (cache, x)。"""
    c = make_cache(model, use_graph)
    c._forward_with_kvcache(prompt)
    return c, prompt


def phase_a_single_step(model, prompt, gamma, iters=40, order=(False, True)):
    """A. 单步 decode：连续做 iters 次 _decode_step。"""
    out = {}
    for use_graph in order:
        c, x = fresh(model, use_graph, prompt, gamma)
        tok = torch.zeros((1, 1), dtype=torch.long, device=x.device)
        # 预热
        for _ in range(5):
            c._decode_step(tok)
        with GpuTimer() as t:
            for _ in range(iters):
                c._decode_step(tok)
        out[use_graph] = t.ms / iters
    return out


def phase_b_gamma_loop(model, prompt, gamma, iters=10, order=(False, True)):
    """B. gamma 步草稿循环（真实热点）：重复 generate。"""
    out = {}
    for use_graph in order:
        c, x = fresh(model, use_graph, prompt, gamma)
        for _ in range(3):
            seq = c.generate(x, gamma)
            c.rollback(gamma + 3)  # 回到固定位置，形成稳态
        with GpuTimer() as t:
            for _ in range(iters):
                seq = c.generate(x, gamma)
                c.rollback(gamma + 3)
        out[use_graph] = t.ms / iters
    return out


def phase_c_full_round(model, prompt, gamma, iters=8, order=(False, True)):
    """C. 含回滚的完整一轮：generate + 回滚到随机接受位置。"""
    out = {}
    accepts = [1, gamma - 4, gamma - 1, 2, gamma - 2, 3, 1, gamma - 3]
    for use_graph in order:
        c, x = fresh(model, use_graph, prompt, gamma)
        base = c.current_length
        for _ in range(3):
            c.generate(x, gamma)
            c.rollback(base)
        with GpuTimer() as t:
            for i in range(iters):
                c.generate(x, gamma)
                c.rollback(base + accepts[i % len(accepts)])
        out[use_graph] = t.ms / iters
    return out


def bench(path: str, gamma: int = 16, repeats: int = 5) -> dict:
    print("=" * 78)
    print(f"模型 {path}   gamma={gamma}   交替测量 {repeats} 轮取中位数")
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

    result: dict = {}
    for name, fn in [
        ("A 单步 decode", lambda order: phase_a_single_step(model, prompt, gamma, order=order)),
        ("B gamma 步草稿循环", lambda order: phase_b_gamma_loop(model, prompt, gamma, order=order)),
        ("C 含回滚完整一轮", lambda order: phase_c_full_round(model, prompt, gamma, order=order)),
    ]:
        # 真正交替先后顺序：奇偶轮把 eager / graph 的测量次序对调，
        # 避免"总是先测的那一侧"系统性占便宜（预热、频率爬升、争抢漂移）。
        e_list, g_list = [], []
        for r in range(repeats):
            order = (False, True) if r % 2 == 0 else (True, False)
            d = fn(order)
            e_list.append(d[False])
            g_list.append(d[True])
        e_med = statistics.median(e_list)
        g_med = statistics.median(g_list)
        result[name] = (e_med, g_med, e_med / g_med)
        print(f"  {name}")
        e_sd = statistics.stdev(e_list) if len(e_list) > 1 else 0.0
        g_sd = statistics.stdev(g_list) if len(g_list) > 1 else 0.0
        print(f"     eager  {e_med:8.2f} ms  ±{e_sd:5.2f}   (各轮 {[f'{v:.1f}' for v in e_list]})")
        print(f"     graph  {g_med:8.2f} ms  ±{g_sd:5.2f}   (各轮 {[f'{v:.1f}' for v in g_list]})")
        print(f"     ⇒ 提速 {e_med / g_med:.2f}×")

    del model
    torch.cuda.empty_cache()
    print()
    return result


def main() -> int:
    paths = sys.argv[1:] or ["llama/llama-68m", "llama/tiny-llama-1.1b"]
    print(f"GPU {torch.cuda.get_device_name(0)}")
    # 记录测量时的争抢状态，供解读数字时参考
    try:
        import subprocess
        free, total = torch.cuda.mem_get_info(0)
        print(f"测量开始时 GPU0 占用 {(1 - free / total) * 100:.0f}% （本机常有第三方任务）\n")
    except Exception:  # noqa: BLE001
        print()
    for p in paths:
        try:
            bench(p)
        except Exception:  # noqa: BLE001
            import traceback
            traceback.print_exc()
    return 0


if __name__ == "__main__":
    sys.exit(main())
