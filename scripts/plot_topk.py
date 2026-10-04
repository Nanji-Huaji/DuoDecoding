#!/usr/bin/env python
"""top-k 扫描两图: (1) k→接受率/吞吐/字节; (2) 草稿分布集中度(排序PDF+质量分布)"""
import json, glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ---- 图1: k vs 接受率 (端到端) ----
rows = []
for K in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
    fs = glob.glob(f"exp/ksweep_{K:04d}/*_metrics.json")
    if not fs:
        continue
    m = json.load(open(fs[0]))
    tf = m["target_forward_times"]
    rows.append({
        "k": K,
        "racce": 100 * m["draft_accepted_tokens"] / max(1, m["draft_generated_tokens"]),
        "tf": m["generated_tokens"] / tf,
        "bpr": m.get("edge_cloud_data_bytes", 0) / max(1, tf),
        "acc": m.get("accuracy"),
    })

# ---- trace: 覆盖率曲线 + 集中度 ----
tr = []
p = "exp/topk_trace_true.jsonl"
try:
    for line in open(p):
        line = line.strip()
        if line:
            tr.append(json.loads(line))
except FileNotFoundError:
    print(f"警告: {p} 不存在, 图2与覆盖率层将跳过", file=sys.stderr)

K_GRID = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
cov_t = []   # P(目标argmax token ∈ 草稿top-k)
cov_a = []   # P(草稿token ∈ 自身top-k) —— 平凡地=1? 不: tok_rank≤k
if tr:
    for k in K_GRID:
        ct = sum(1 for r in tr if r["t_argmax_rank"] <= k) / len(tr)
        ca = sum(1 for r in tr if r["tok_rank"] <= k) / len(tr)
        cov_t.append(ct)
        cov_a.append(ca)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
if rows:
    ks = [r["k"] for r in rows]
    ax.plot(ks, [r["racce"] for r in rows], "o-", color="tab:red", label="accept rate $R_{acce}$")
    ax2 = ax.twinx()
    ax2.plot(ks, [r["tf"] for r in rows], "s--", color="tab:blue", label="tokens/forward")
    ax2.set_ylabel("tokens / forward", color="tab:blue")
    ax2.set_xscale("log", base=2)
    if tr:
        ax.plot(K_GRID, [100 * c for c in cov_t], "^:", color="tab:gray",
                label="P(target token $\\in$ draft top-$k$)")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("uplink top-$k$")
    ax.set_ylabel("accept rate (%)", color="tab:red")
    ax.set_title("(a) Acceptance vs uplink top-$k$ (CEE-SD, temp 0.7)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    # 曲线分布: 覆盖率占上带, R_acce/tok-per-fwd 占下带, 中部空 → 图例置中左
    ax.legend(h1 + h2, l1 + l2, loc="center left", fontsize=8)
    ax.grid(alpha=0.3)

ax = axes[1]
if rows:
    ks = [r["k"] for r in rows]
    ax.plot(ks, [r["bpr"] for r in rows], "d-", color="tab:green")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("uplink top-$k$")
    ax.set_ylabel("uplink bytes / round")
    ax.set_title("(b) Uplink cost vs top-$k$")
    ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("exp/figs/topk_acceptance.png", dpi=160)
print("图1 → exp/figs/topk_acceptance.png")

# ---- 图2: 分布集中度 ----
if tr:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) 一个典型位置的排序概率 (取熵最接近中位数的位置)
    ents = [r["entropy"] for r in tr]
    med = float(np.median(ents))
    ex = min(tr, key=lambda r: abs(r["entropy"] - med))
    top20 = ex["top20"]
    ax = axes[0]
    ax.bar(range(1, len(top20) + 1), top20, color="tab:blue", alpha=0.85)
    ax.set_yscale("log")
    ax.set_xlabel("rank (sorted draft-token probability)")
    ax.set_ylabel("p(rank)")
    cum8 = sum(top20[:8])
    ax.set_title(
        f"(a) Sorted draft distribution, one step\n"
        f"(median-entropy step: H={ex['entropy']:.2f}; top-8 mass={cum8:.3f})"
    )
    ax.grid(alpha=0.3, axis="y", which="both")

    # (b) 各位置的前k累积质量分布 (集中度全景)
    ax = axes[1]
    for k, color in [(8, "tab:red"), (32, "tab:orange"), (256, "tab:green")]:
        idx = [i for i, kk in enumerate(ex["k_grid"]) if kk == k]
        if not idx:
            continue
        j = idx[0]
        vals = [min(1.0, r["cum_mass"][j]) for r in tr if len(r["cum_mass"]) > j]
        ax.hist(vals, bins=40, range=(0, 1), histtype="step", lw=1.8,
                color=color, label=f"top-{k}")
    ax.set_xlabel("cumulative probability mass of top-$k$ draft tokens")
    ax.set_ylabel("# positions")
    ax.set_title("(b) Concentration across steps (all drafted positions)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig("exp/figs/dist_concentration.png", dpi=160)
    print("图2 → exp/figs/dist_concentration.png")
    # 文字摘要
    for k in (8, 32, 256):
        idx = [i for i, kk in enumerate(tr[0]["k_grid"]) if kk == k][0]
        vals = [r["cum_mass"][idx] for r in tr if len(r["cum_mass"]) > idx]
        print(f"top-{k:>3} 质量: 中位 {np.median(vals):.4f}, "
              f"P10 {np.percentile(vals,10):.4f}, P90 {np.percentile(vals,90):.4f}")
    print(f"目标argmax秩: 中位 {np.median([r['t_argmax_rank'] for r in tr]):.0f}, "
          f"P90 {np.percentile([r['t_argmax_rank'] for r in tr],90):.0f}")
    print(f"覆盖率 P(目标∈top-k): " + ", ".join(
        f"k={k}:{100*c:.1f}%" for k, c in zip(K_GRID, cov_t)))
else:
    print("无 trace, 图2跳过")
