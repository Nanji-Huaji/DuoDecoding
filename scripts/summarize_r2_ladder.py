#!/usr/bin/env python
"""R2 阶梯汇总: 每 bw 点的静态 k 阶梯 vs DRA(自适应k), 输出 tok/fwd 与 Thr。
用法: .venv/bin/python scripts/summarize_r2_ladder.py [TAGPFX=r215] [OUT=stdout]"""
import glob, json, sys

pfx = sys.argv[1] if len(sys.argv) > 1 else "r215"
out = open(sys.argv[2], "w") if len(sys.argv) > 2 else sys.stdout


def load(tag):
    fs = glob.glob(f"exp/{tag}/*_metrics.json")
    if not fs:
        return None
    m = json.load(open(fs[0]))
    tf = m["generated_tokens"] / max(1, m["target_forward_times"])
    return tf, m["throughput"]


W = out.write
W(f"{'bw':>5} | {'k=64':>15} {'k=256':>15} {'k=1024':>15} | {'DRA':>13} {'vs最优':>8}\n")
W(f"{'':>5} | {'(tokfwd/Thr)':>15} {'(tokfwd/Thr)':>15} {'(tokfwd/Thr)':>15} | {'(tokfwd/Thr)':>13} {'':>8}\n")
W("-" * 82 + "\n")
for bw in ["0.5", "1", "2", "5"]:
    cells, best = [], -1.0
    for k in ["64", "256", "1024"]:
        r = load(f"{pfx}_bw{bw}_k{k}_thr04")
        if r:
            cells.append(f"{r[0]:6.2f}/{r[1]:5.2f}")
            best = max(best, r[0])
        else:
            cells.append(f"{'—':>15}")
    d = load(f"{pfx}_bw{bw}_dra")
    if d:
        dra = f"{d[0]:6.2f}/{d[1]:5.2f}"
        gap = f"{(d[0]-best)/best*100:+.1f}%" if best > 0 else "—"
    else:
        dra, gap = f"{'—':>13}", "—"
    W(f"{bw:>5} | {cells[0]:>15} {cells[1]:>15} {cells[2]:>15} | {dra:>13} {gap:>8}\n")
W("-" * 82 + "\n")
W("列: tok/fwd(每验证块) 与 Thr(tok/s); DRA=k自适应(cap1024)+thr自适应\n")
if out is not sys.stdout:
    out.close()
