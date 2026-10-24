# sweeps/ — 扫描摘要数据（图的 provenance）

`topk_ksweep_summary.csv`：`scripts/k_sweep.sh` 扫描（k∈{1..1024} ×
adaptive_tridecoding）每个 k 的 metrics 摘要，从本地 `exp/ksweep_*/`
（被 gitignore）提取，供 `scripts/plot_topk.py` / `plot_topk_commshare.py`
产出的图（`figures/topk_acceptance.png`、`topk_comm_share.png`）核对来源。

**扫描体制（≠ protocol.md §2 冻结体制，引用结论时必须标注）**：
temp 0.7（冻结值 0.0）、N=40（冻结 80）、`--no-charge_residual_payload`、
`--transfer_top_k_cap 0`、γ1=γ2=16；NTT 50 ms / 带宽 563+46 Mbps（与冻结值同）。

重跑：`GPU=<id> bash scripts/k_sweep.sh`，再按本目录 CSV 的列定义从
`exp/ksweep_*/*_metrics.json` 重新提取。
