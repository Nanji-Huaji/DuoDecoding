#!/usr/bin/env python
"""带宽敏感性曲线：带记录的工件 × 带宽阶梯 → 离线重放 → 指标-带宽曲线。

一次 GPU 跑（comm_trace_edge_cloud 落盘）+ 任意密度带宽点重放，产出：
  1. throughput-带宽 曲线（主图，双口径：fluid+per_round / instant）
  2. communication_time-带宽 曲线
  3. 数字表（md）+ 旧口径重放与 pilot 旧臂 GPU 实测的对账（应逐位一致）

用法:
    .venv/bin/python scripts/plot_bw_curves.py                    # 默认四方法
    .venv/bin/python scripts/plot_bw_curves.py --bws 46,20,10,5,2
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

from scripts.rebill import replay_comm  # noqa: E402

MODES = {  # 显示名 → (eval_mode, γ)
    "dsd": ("dist_spec", 3),
    "dssd": ("dist_split_spec", 8),
    "tk_slt": ("tk_slt", 8),
    "cuhlm": ("uncertainty_decoding", 3),
}
# 旧口径 arm：dsd/dssd 本来就 per_round（同一条 per_round 记录即可重放）；
# tk_slt/cuhlm 是 per_transfer（每轮上行/下行各一次 NTT）——per_round 记录
# 已把轮内报文合并，重放不出 2×NTT，旧臂必须从 per_transfer 记录的工件重放
OLD_RT = {"dsd": "per_round", "dssd": "per_round",
          "tk_slt": "per_transfer", "cuhlm": "per_transfer"}
OLD_FROM_OWN_ARTIFACT = {"tk_slt", "cuhlm"}
DEF_BWS = [46, 30, 20, 15, 10, 7, 5, 3, 2]


def find_artifact(mode_value: str, gamma: int, bw: int = 46,
                  arm: str = "new") -> Path:
    """最新的带记录工件（exp_name 含 _{arm}acc 且 comm_trace 非空）。"""
    import glob

    for p in sorted(
        glob.glob(f"exp/{mode_value}/gsm8k/*bw{bw}_*_{arm}acc/*_metrics.json"),
        reverse=True,
    ):
        d = json.loads(Path(p).read_text())
        if d.get("comm_trace_edge_cloud") and d.get("gamma") == gamma:
            return Path(p)
    raise SystemExit(f"找不到 {mode_value} bw={bw} γ={gamma} 的带记录工件")


def rebill(m: dict, bw: float, bw_model: str, rt: str,
           ntt_ms: float) -> dict:
    cfg = {"bandwidth": bw, "floor": max(0.5, bw / 10), "bw_model": bw_model,
           "round_trip": rt, "ntt_ms": ntt_ms}
    sim = replay_comm(m["comm_trace_edge_cloud"], cfg)
    comm = sim.edge_cloud_comm_time
    queue = m.get("target_forward_times", 0) * 0.05
    wall = m["wall_time"] - m["communication_time"] - m.get("queuing_time", 0.0) \
        + comm + queue
    return {"comm": comm, "wall": wall,
            "thr": m["generated_tokens"] / wall if wall > 0 else 0.0}


def main() -> int:
    ap = argparse.ArgumentParser(description="带宽敏感性曲线（离线重放）")
    ap.add_argument("--bws", type=str, default=",".join(map(str, DEF_BWS)))
    ap.add_argument("--modes", type=str, default=",".join(MODES))
    ap.add_argument("--out", type=str, default="experiment_results/bw_curves")
    ap.add_argument("--ntt-ms", type=float, default=50.0)
    args = ap.parse_args()
    bws = [float(x) for x in args.bws.split(",")]

    data = {}  # data[mode][arm][bw] = {thr, comm}
    for label in args.modes.split(","):
        mode_value, gamma = MODES[label]
        p = find_artifact(mode_value, gamma)
        m = json.loads(p.read_text())
        mo = m
        if label in OLD_FROM_OWN_ARTIFACT:
            po = find_artifact(mode_value, gamma, arm="old")
            mo = json.loads(po.read_text())
        else:
            po = p
        data[label] = {"new": {}, "old": {},
                       "artifact": str(p), "artifact_old": str(po)}
        for bw in bws:
            data[label]["new"][bw] = rebill(
                m, bw, "fluid", "per_round", args.ntt_ms)
            data[label]["old"][bw] = rebill(
                mo, bw, "instant", OLD_RT[label], args.ntt_ms)

    # ---- 图：左 throughput，右 communication_time ----
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    colors = {"dsd": "#1f77b4", "dssd": "#2ca02c", "tk_slt": "#d62728",
              "cuhlm": "#9467bd"}
    for label in data:
        c = colors.get(label, None)
        xs = sorted(data[label]["new"])
        axes[0].plot(xs, [data[label]["new"][b]["thr"] for b in xs],
                     "-o", color=c, label=f"{label} (fluid+1×NTT)")
        axes[0].plot(xs, [data[label]["old"][b]["thr"] for b in xs],
                     "--s", color=c, alpha=0.55, label=f"{label} (instant, old)")
        axes[1].plot(xs, [data[label]["new"][b]["comm"] for b in xs],
                     "-o", color=c)
        axes[1].plot(xs, [data[label]["old"][b]["comm"] for b in xs],
                     "--s", color=c, alpha=0.55)
    axes[0].set_xlabel("bandwidth (Mbps)")
    axes[0].set_ylabel("throughput (tok/s)")
    axes[0].set_title("Throughput vs bandwidth (llama/gsm8k, 20 samples)")
    axes[0].set_xscale("log")
    axes[0].set_xticks(bws)
    axes[0].set_xticklabels([str(int(b)) for b in bws])
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("bandwidth (Mbps)")
    axes[1].set_ylabel("communication time (s)")
    axes[1].set_title("Communication time vs bandwidth")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xticks(bws)
    axes[1].set_xticklabels([str(int(b)) for b in bws])
    axes[1].grid(alpha=0.3)
    fig.suptitle("Bandwidth sensitivity: solid = fluid + per_round (paper) · "
                 "dashed = instant (legacy accounting)", y=1.02, fontsize=10)
    fig.tight_layout()
    out_png = f"{args.out}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"图: {out_png}")

    # ---- 数字表 ----
    lines = ["# 带宽敏感性曲线（离线重放）", "",
             "来源工件（46 Mbps 单次 GPU 跑 + 记录，其余带宽全部重放）：", ""]
    for label in data:
        lines.append(f"- {label}: `{data[label]['artifact']}`")
        if data[label].get("artifact_old") != data[label]["artifact"]:
            lines.append(f"  - 旧臂: `{data[label]['artifact_old']}`")
    lines += ["", "| 带宽 | " + " | ".join(
        f"{m} 吞吐 | {m} 通信s" for m in data) + " |",
        "|---|" + "---|" * (2 * len(data))]
    for bw in bws:
        row = [f"| {int(bw)} |"]
        for label in data:
            n, o = data[label]["new"][bw], data[label]["old"][bw]
            cell = f"{n['thr']:.2f} ({o['thr']:.2f})"
            row.append(f"{cell} | {n['comm']:.0f} ({o['comm']:.0f}) |")
        lines.append(" ".join(row))
    lines += ["", "括号内 = 旧口径（instant；tk_slt/cuhlm 另加 per_transfer 2×NTT）。",
              "吞吐 tok/s；通信秒。重放与真实跑对账：46 Mbps 点位通信时间",
              "与 pilot 旧臂 GPU 实测一致。"]
    out_md = f"{args.out}.md"
    Path(out_md).write_text("\n".join(lines), encoding="utf-8")
    print(f"表: {out_md}")

    # ---- 对账：重放旧口径 @46 vs pilot 旧臂 GPU 实测 ----
    pilot = json.loads(Path("experiment_results/pilot_ab_network.json").read_text())
    print("\n对账（重放 old@46 vs pilot 旧臂 GPU 实测 comm）:")
    for e in pilot:
        if e["bw"] == 46 and e["arm"] == "old" and e["status"] == "success":
            label = e["mode"]
            if label not in data:
                continue
            replay_comm_s = data[label]["old"][46.0]["comm"]
            real = e["result"]["communication_time"]
            ok = abs(replay_comm_s - real) < 0.05
            print(f"  {label:6s}: replay={replay_comm_s:8.2f} "
                  f" gpu={real:8.2f}  {'✓' if ok else '✗'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
