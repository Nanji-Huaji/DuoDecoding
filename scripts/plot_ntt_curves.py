#!/usr/bin/env python
"""NTT 敏感性曲线：带记录的工件 × NTT 阶梯 → 离线重放 → 指标-NTT 曲线。

与带宽曲线（plot_bw_curves.py）共用工件与重放内核；本脚本扫 NTT：
两档带宽（46 = NTT 主导区 / 5 = 发射时长主导区）× 双口径（fluid+per_round
/ instant 旧口径），显示口径选择本身改变 NTT 弹性（旧口径每轮 2×NTT，
斜率翻倍）。默认 50ms 与论文 Table II 的 76.36ms 在轴上标注。

用法:
    .venv/bin/python scripts/plot_ntt_curves.py
    .venv/bin/python scripts/plot_ntt_curves.py --ntts 1,10,50,100,300
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.plot_bw_curves import (  # noqa: E402
    MODES,
    OLD_FROM_OWN_ARTIFACT,
    OLD_RT,
    find_artifact,
    rebill,
)

DEF_NTTS = [0.5, 1, 2, 5, 10, 20, 50, 76.36, 100, 200]
PAPER_NTTS = {"50": "default", "76.36": "Table II"}


def main() -> int:
    ap = argparse.ArgumentParser(description="NTT 敏感性曲线（离线重放）")
    ap.add_argument("--ntts", type=str, default=",".join(map(str, DEF_NTTS)))
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    ap.add_argument("--bandwidths", type=str, default="46,5")
    ap.add_argument("--out", type=str, default="experiment_results/ntt_curves")
    args = ap.parse_args()
    ntts = [float(x) for x in args.ntts.split(",")]
    bws = [float(x) for x in args.bandwidths.split(",")]

    data = {}  # data[label][arm][(bw, ntt)] = {thr, comm}
    for label in args.modes.split(","):
        mode_value, gamma = MODES[label]
        m = json.loads(find_artifact(mode_value, gamma).read_text())
        mo = m
        if label in OLD_FROM_OWN_ARTIFACT:
            mo = json.loads(
                find_artifact(mode_value, gamma, arm="old").read_text()
            )
        data[label] = {"new": {}, "old": {}}
        for bw in bws:
            for ntt in ntts:
                data[label]["new"][(bw, ntt)] = rebill(
                    m, bw, "fluid", "per_round", ntt)
                data[label]["old"][(bw, ntt)] = rebill(
                    mo, bw, "instant", OLD_RT[label], ntt)

    colors = {"dsd": "#1f77b4", "dssd": "#2ca02c",
              "tk_slt": "#d62728", "cuhlm": "#9467bd"}
    fig, axes = plt.subplots(2, len(bws), figsize=(6.2 * len(bws), 8.4),
                             sharex=True)
    if len(bws) == 1:
        axes = axes.reshape(2, 1)
    for col, bw in enumerate(bws):
        for label in data:
            c = colors.get(label)
            xs = sorted(ntts)
            new_y = [data[label]["new"][(bw, n)]["thr"] for n in xs]
            old_y = [data[label]["old"][(bw, n)]["thr"] for n in xs]
            axes[0][col].plot(xs, new_y, "-o", color=c, label=label)
            axes[0][col].plot(xs, old_y, "--s", color=c, alpha=0.5)
            new_c = [data[label]["new"][(bw, n)]["comm"] for n in xs]
            old_c = [data[label]["old"][(bw, n)]["comm"] for n in xs]
            axes[1][col].plot(xs, new_c, "-o", color=c)
            axes[1][col].plot(xs, old_c, "--s", color=c, alpha=0.5)
        for row in (0, 1):
            ax = axes[row][col]
            ax.set_xscale("log")
            ax.set_xticks(ntts)
            ax.set_xticklabels([str(n) for n in ntts], rotation=45)
            for v, _tag in PAPER_NTTS.items():
                if float(v) in ntts:
                    ax.axvline(float(v), color="gray", ls=":",
                               lw=0.8, alpha=0.7)
            ax.grid(alpha=0.3)
        axes[0][col].set_title(f"{int(bw)} Mbps", fontsize=11)
    axes[0][0].set_ylabel("throughput (tok/s)")
    axes[1][0].set_ylabel("communication time (s)")
    for ax in axes[1]:
        ax.set_xlabel("NTT edge-cloud (ms)")
    axes[0][0].legend(fontsize=9)
    fig.suptitle("NTT sensitivity: solid = fluid + per_round (paper) · "
                 "dashed = instant (legacy) · dotted vlines = 50ms default "
                 "& 76.36ms (Table II)", y=0.995, fontsize=10)
    fig.tight_layout()
    out_png = f"{args.out}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"图: {out_png}")

    # 数字表（默认 NTT 档附近 + 两端）
    lines = ["# NTT 敏感性曲线（离线重放）", "",
             "| 带宽 | NTT ms | " + " | ".join(
                 f"{m} 吞吐 | {m} 通信s" for m in data) + " |",
             "|---|---|" + "---|" * (2 * len(data))]
    for bw in bws:
        for ntt in ntts:
            row = [f"| {int(bw)} | {ntt:g} |"]
            for label in data:
                n, o = data[label]["new"][(bw, ntt)], data[label]["old"][(bw, ntt)]
                cell = f"{n['thr']:.2f} ({o['thr']:.2f})"
                row.append(f"{cell} | {n['comm']:.0f} ({o['comm']:.0f}) |")
            lines.append(" ".join(row))
    lines += ["", "括号内 = 旧口径（instant；tk_slt/cuhlm 另加 per_transfer 2×NTT，",
              "NTT 斜率因此翻倍）。50ms 为默认，76.36ms 对应论文 Table II。"]
    out_md = f"{args.out}.md"
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"表: {out_md}")

    # 内部一致性：NTT=50 与带宽曲线同点位应一致
    print("\n内部一致性（NTT=50ms 应与 bw_curves 同点位一致）:")
    for bw in bws:
        for label in data:
            v = data[label]["new"][(bw, 50.0)]["comm"]
            print(f"  {label:6s} @{int(bw)}Mbps: comm={v:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
