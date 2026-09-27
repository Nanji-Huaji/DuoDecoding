#!/usr/bin/env python3
"""汇总 CUDA Graph 配对 A/B 的结果。

读 exp/ab_<mode>_r<N>/ 下的 metrics.json，按轮次并排打印。
关键区分：
  * **确定性指标**（准确率、生成 token 数、各级前向次数）配对后必须完全一致 ——
    如果不一致，说明开关改变了行为，是 bug 而不是性能差异；
  * **耗时类指标**会随机器争抢波动，配对设计就是为了把它们的影响均摊掉。
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys

HDR = ["轮", "模式", "准确率", "tok/s", "生成tok", "68M前向", "1.1B前向",
       "13B前向", "68M(s)", "1.1B(s)", "13B(s)", "通信(s)", "排队(s)"]
KEYS = ["accuracy", "throughput", "generated_tokens", "little_forward_times",
        "draft_forward_times", "target_forward_times", "little_computation_time",
        "draft_computation_time", "target_computation_time",
        "communication_time", "queuing_time"]
W = [4, 6, 9, 8, 9, 9, 10, 9, 8, 8, 8, 8, 8]
DETERMINISTIC = ["accuracy", "generated_tokens", "little_forward_times",
                 "draft_forward_times", "target_forward_times"]


def load(pattern: str) -> dict[int, dict[str, dict]]:
    rows: dict[int, dict[str, dict]] = {}
    for d in sorted(glob.glob(pattern)):
        m = re.match(r"(?:ab|vg\d*)_(eager|cuda)_r(\d+)", os.path.basename(d))
        if not m:
            continue
        met: dict = {}
        for f in glob.glob(f"{d}/*_metrics.json"):
            met = json.load(open(f))
        if met:
            rows.setdefault(int(m.group(2)), {})[m.group(1)] = met
    return rows


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "exp/ab_*_r*"
    rows = load(pattern)
    if not rows:
        print(f"  没有找到匹配 {pattern} 的结果")
        return 1
    print("  " + " ".join(h.rjust(w) for h, w in zip(HDR, W)))
    for rnd in sorted(rows):
        for mode in ("eager", "cuda"):
            met = rows[rnd].get(mode)
            if not met:
                continue
            vals = [str(rnd), mode]
            for k in KEYS:
                v = met.get(k)
                if isinstance(v, float):
                    vals.append(f"{v:.2f}")
                elif v is None:
                    vals.append("-")
                else:
                    vals.append(str(v))
            print("  " + " ".join(v.rjust(w) for v, w in zip(vals, W)))
        e, c = rows[rnd].get("eager"), rows[rnd].get("cuda")
        if e and c:
            bad = [k for k in DETERMINISTIC if e.get(k) != c.get(k)]
            if bad:
                print(f"      ✗ 第 {rnd} 轮确定性指标不一致: {bad} —— 开关改变了行为！")
            else:
                print(f"      ✓ 第 {rnd} 轮确定性指标完全一致")
            te, tc = e.get("throughput"), c.get("throughput")
            if isinstance(te, float) and isinstance(tc, float) and tc:
                faster = tc / te
                tag = "更快" if faster >= 1 else "更慢"
                print(f"        吞吐: eager {te:.2f} → cuda {tc:.2f} tok/s"
                      f"  ({faster:.2f}× {tag})")

    # 跨轮汇总吞吐
    pairs = [
        (rows[r]["eager"]["throughput"], rows[r]["cuda"]["throughput"])
        for r in sorted(rows)
        if "eager" in rows[r] and "cuda" in rows[r]
        and isinstance(rows[r]["eager"].get("throughput"), float)
        and isinstance(rows[r]["cuda"].get("throughput"), float)
    ]
    if pairs:
        import statistics
        e_med = statistics.median(p[0] for p in pairs)
        c_med = statistics.median(p[1] for p in pairs)
        print()
        print(f"  跨轮中位数: eager {e_med:.2f} tok/s   cuda {c_med:.2f} tok/s"
              f"   ⇒ 快 {c_med / e_med:.2f}×")
        print("  （墙钟吞吐受第三方负载影响大；确定性指标与质量才是主论据）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
