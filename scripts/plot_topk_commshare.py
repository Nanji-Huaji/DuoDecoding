#!/usr/bin/env python
"""top-k → 通信时间构成: (a) 实测 RTT/payload 占比 (b) 弱带宽/低NTT投影"""
import json, glob
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

KS = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
rows = []
for K in KS:
    m = json.load(open(glob.glob(f"exp/ksweep_{K:04d}/*_metrics.json")[0]))
    ct = m["communication_time"]
    conn = m["connect_times"]["edge_cloud"]
    ntt = conn * 0.05
    bw = m["edge_cloud_bandwidth_mbps_stats"]["mean"] * 1e6
    pay = m["edge_cloud_data_bytes"] / bw
    rows.append({
        "k": K, "ct": ct, "ntt": ntt, "pay": pay,
        "oth": ct - ntt - pay,
        "bpr": m["edge_cloud_data_bytes"] / m["target_forward_times"],
    })

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))

# (a) 实测: 堆叠占比 + 绝对通信时间(右轴)
ax = axes[0]
ks = [r["k"] for r in rows]
rtt = np.array([100 * r["ntt"] / r["ct"] for r in rows])
pay = np.array([100 * r["pay"] / r["ct"] for r in rows])
oth = np.array([100 * r["oth"] / r["ct"] for r in rows])
ax.stackplot(ks, [rtt, pay, oth], labels=["RTT (NTT×rounds)", "payload (bytes/BW)", "other (edge_end link)"],
             colors=["tab:blue", "tab:red", "tab:gray"], alpha=0.85)
ax.set_xscale("log", base=2)
ax.set_xlabel("uplink top-$k$")
ax.set_ylabel("share of communication time (%)")
ax.set_ylim(0, 100.5)
ax.set_title("(a) Measured comm-time split (WAN: NTT 50 ms, ~42.5 Mbps)")
ax2 = ax.twinx()
ax2.plot(ks, [r["ct"] for r in rows], "k^--", lw=1, ms=5, label="total comm time (s)")
ax2.set_ylabel("total comm time (s)")
ax2.set_ylim(50, 75)
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8)
ax.annotate("RTT count falls with $k$:\n1355→1121 rounds\n(acceptance ↑ → fewer rounds)",
            xy=(1, 68.4), xytext=(6, 72.5), fontsize=7.5,
            arrowprops=dict(arrowstyle="->", lw=0.8))

# (b) 投影: payload占比 = (B/轮/BW) / (NTT + B/轮/BW), 实测B/轮
ax = axes[1]
NTT_MS = 0.05
for bw_mbps, color, ls in [(46, "tab:green", "-"), (10, "tab:orange", "-"),
                           (5, "tab:red", "-"), (1, "tab:purple", "-")]:
    bw = bw_mbps * 1e6
    bpr = np.array([r["bpr"] for r in rows])
    share = 100 * (bpr / bw) / (NTT_MS + bpr / bw)
    ax.plot(ks, share, ls, color=color, marker=".", ms=4,
            label=f"BW={bw_mbps} Mbps, NTT=50 ms")
# 近边缘体制: 低NTT
bw = 5e6
bpr = np.array([r["bpr"] for r in rows])
share = 100 * (bpr / bw) / (0.005 + bpr / bw)
ax.plot(ks, share, "--", color="tab:brown", marker=".", ms=4,
        label="BW=5 Mbps, NTT=5 ms (near-edge)")
ax.set_xscale("log", base=2)
ax.set_xlabel("uplink top-$k$")
ax.set_ylabel("payload share of comm time (%)")
ax.set_title("(b) Projected payload share (per-round model, measured $B$/round)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("figures/topk_comm_share.png", dpi=160)
print("→ figures/topk_comm_share.png")
for r in rows:
    print(f"k={r['k']:>4}: RTT {100*r['ntt']/r['ct']:.2f}% payload {100*r['pay']/r['ct']:.3f}% "
          f"({r['bpr']:.0f} B/round, comm {r['ct']:.1f}s)")
