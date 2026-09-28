#!/usr/bin/env python
"""低延迟(N=40) vs t5a(N=80, NTT50) 主对比汇总: Thr / tok-per-fwd / Cloud$ / acc。
用法: .venv/bin/python scripts/summarize_lowlat.py"""
import glob, json, sys

OUT = open("exp_logs/lowlat40_summary.txt", "w")


def load(prefix, tag):
    fs = glob.glob(f"exp/{prefix}_{tag}/*_metrics.json")
    if not fs:
        return None
    m = json.load(open(fs[0]))
    tf = m["target_forward_times"]
    return {
        "Thr": m["throughput"],
        "tf": m["generated_tokens"] / max(1, tf),
        "fwd": tf,
        "gen": m["generated_tokens"],
        "cloud": 4.05 * m["wall_time"] / 3600,
        "acc": m.get("accuracy"),
    }


ROWS = [
    ("DSD  (dist_spec)", "dsd"),
    ("DSSD (dist_split)", "dssd"),
    ("CUHLM(uncert)", "cuhlm"),
    ("CEE-SD γ5", "ceesd_g5"),
    ("CEE-SD γ16", "ceesd_g16"),
]
W = OUT.write
W("低延迟域主对比 (NTT15+ping trace, N=40) vs t5a (NTT50, N=80)\n")
W(f"{'方法':<16} | {'低延迟 Thr':>9} {'tokfwd':>6} {'Fwd':>5} {'acc':>6} | "
  f"{'t5a Thr':>7} {'tokfwd':>6} {'Fwd':>5} | {'Thr增幅':>7}\n")
W("-" * 88 + "\n")
for name, tag in ROWS:
    lo = load("lowlat40", tag)
    hi = load("t5a", tag)
    if lo is None:
        W(f"{name:<16} | {'(未跑)':>9}\n")
        continue
    acc = f"{lo['acc']:.4f}" if lo["acc"] is not None else "—"
    if hi:
        W(f"{name:<16} | {lo['Thr']:>9.2f} {lo['tf']:>6.2f} {lo['fwd']:>5d} {acc:>6} | "
          f"{hi['Thr']:>7.2f} {hi['tf']:>6.2f} {hi['fwd']:>5d} | "
          f"{(lo['Thr']-hi['Thr'])/hi['Thr']*100:+6.1f}%\n")
    else:
        W(f"{name:<16} | {lo['Thr']:>9.2f} {lo['tf']:>6.2f} {lo['fwd']:>5d} {acc:>6} | "
          f"{'—':>7} {'—':>6} {'—':>5} | {'—':>7}\n")
W("-" * 88 + "\n")
W("注: 两列同计费/同k300/同RL策略, 差值纯attributable于 NTT50(固定)→15(实测trace)\n")
OUT.close()
sys.stdout.write(open("exp_logs/lowlat40_summary.txt").read())
