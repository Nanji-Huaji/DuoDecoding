"""Analyze why cee_dsd/cee_dssd underperform dsd/dssd."""

import json
import os
import glob
from collections import defaultdict
from typing import Dict, List, Tuple


def collect_metrics(exp_root: str = "exp") -> Dict[str, List[dict]]:
    """Collect all metrics JSONs grouped by eval_mode."""
    results = defaultdict(list)
    for path in glob.glob(f"{exp_root}/**/*_metrics.json", recursive=True):
        with open(path) as f:
            data = json.load(f)
        mode = data.get("eval_mode", "unknown")
        results[mode].append(data)
    return results


def summarize(mode: str, records: List[dict]) -> dict:
    """Compute average stats for a mode's experiments."""
    if not records:
        return {}
    n = len(records)

    def avg(key):
        vals = [r.get(key, 0) for r in records if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0

    def safe_div(a, b):
        return a / b if b != 0 else 0

    return {
        "count": n,
        "throughput": avg("throughput"),
        "wall_time_s": avg("wall_time"),
        "generated_tokens": avg("generated_tokens"),
        # 推测效率
        "draft_accept_rate": safe_div(
            sum(r.get("draft_accepted_tokens", 0) for r in records),
            sum(r.get("draft_generated_tokens", 1) for r in records),
        ),
        "little_accept_rate": safe_div(
            sum(r.get("little_accepted_tokens", 0) for r in records),
            sum(r.get("little_generated_tokens", 1) for r in records),
        ),
        # 计算开销
        "draft_forward_times": avg("draft_forward_times"),
        "target_forward_times": avg("target_forward_times"),
        "little_forward_times": avg("little_forward_times"),
        # 通信开销
        "communication_time_s": avg("communication_time"),
        "computation_time_s": avg("computation_time"),
        "queuing_time_s": avg("queuing_time"),
        "edge_end_comm_time_s": avg("edge_end_comm_time"),
        # 带宽 (MB)
        "edge_cloud_data_mb": avg("edge_cloud_data_bytes") / 1e6,
        "edge_end_data_mb": avg("edge_end_data_bytes") / 1e6,
        # 连接次数
        "connect_cloud": avg("connect_times.get('edge_cloud', 0)") if False
        else sum(
            r.get("connect_times", {}).get("edge_cloud", 0) for r in records
        )
        / n,
        "connect_end": sum(
            r.get("connect_times", {}).get("edge_end", 0) for r in records
        )
        / n,
        # draft 模型信息
        "draft_model": records[0].get("draft_model", "?"),
        "little_model": records[0].get("little_model", "?"),
        "target_model": records[0].get("target_model", "?"),
    }


def compute_connect_times_avg(records, key):
    vals = [r.get("connect_times", {}).get(key, 0) for r in records]
    return sum(vals) / len(vals) if vals else 0


def print_table(modes: List[str], summaries: Dict[str, dict]):
    """Print side-by-side comparison."""
    print("=" * 130)
    print(f"{'Metric':<32}", end="")
    for m in modes:
        print(f"{m:<24}", end="")
    print()
    print("-" * 130)

    rows = [
        ("throughput (tok/s)", "throughput"),
        ("wall_time (s)", "wall_time_s"),
        ("generated_tokens", "generated_tokens"),
        ("", ""),  # separator
        ("draft accept rate", "draft_accept_rate"),
        ("little accept rate", "little_accept_rate"),
        ("draft_forward_times", "draft_forward_times"),
        ("target_forward_times", "target_forward_times"),
        ("little_forward_times", "little_forward_times"),
        ("", ""),  # separator
        ("communication_time (s)", "communication_time_s"),
        ("computation_time (s)", "computation_time_s"),
        ("queuing_time (s)", "queuing_time_s"),
        ("edge_end_comm_time (s)", "edge_end_comm_time_s"),
        ("", ""),  # separator
        ("edge_cloud_data (MB)", "edge_cloud_data_mb"),
        ("edge_end_data (MB)", "edge_end_data_mb"),
        ("", ""),  # separator
        ("connect_times.cloud", "connect_cloud"),
        ("connect_times.end", "connect_end"),
    ]

    for label, key in rows:
        if not label:
            print("-" * 130)
            continue
        print(f"{label:<32}", end="")
        for m in modes:
            s = summaries.get(m, {})
            val = s.get(key, 0)
            if isinstance(val, float):
                print(f"{val:<24.2f}", end="")
            else:
                print(f"{str(val):<24}", end="")
        print()

    # Model info
    print("-" * 130)
    print(f"{'Model chain':<32}", end="")
    for m in modes:
        s = summaries.get(m, {})
        chain = f"{s.get('little_model','?') if s.get('little_model','?') != s.get('draft_model','?') else ''}"
        if chain:
            chain = f"{chain} → "
        chain += f"{s.get('draft_model','?')} → {s.get('target_model','?')}"
        print(f"{chain:<24}", end="")
    print()
    print("=" * 130)


def compare_cee_vs_non_cee(metrics: Dict[str, List[dict]]):
    """Compare cee_dsd vs dsd and cee_dssd vs dssd."""
    print("\n" + "=" * 130)
    print("🔍 cee_dsd vs dsd")
    print("=" * 130)
    modes1 = ["dist_spec", "cee_dsd"]
    summaries = {m: summarize(m, metrics.get(m, [])) for m in modes1}
    print_table(modes1, summaries)

    print("\n" + "=" * 130)
    print("🔍 cee_dssd vs dssd")
    print("=" * 130)
    modes2 = ["dist_split_spec", "cee_dssd"]
    summaries = {m: summarize(m, metrics.get(m, [])) for m in modes2}
    print_table(modes2, summaries)


def compare_by_gamma(metrics: Dict[str, List[dict]]):
    """Break down cee_dsd/cee_dssd by gamma1/gamma2 combination."""
    print("\n" + "=" * 130)
    print("🔍 cee_dsd 按 gamma 组合拆解")
    print("=" * 130)
    records = metrics.get("cee_dsd", [])
    if not records:
        print("  No data yet.")
        return
    groups = defaultdict(list)
    for r in records:
        g1 = r.get("gamma1", "?")
        g2 = r.get("gamma2", "?")
        groups[(g1, g2)].append(r)

    print(f"  {'gamma1/gamma2':<16} {'throughput':>12} {'accept_rate':>14} {'wall_time':>12} {'samples':>8}")
    print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12} {'-'*8}")
    for (g1, g2) in sorted(groups.keys()):
        recs = groups[(g1, g2)]
        s = summarize("", recs)
        print(
            f"  {f'{g1}/{g2}':<16} {s['throughput']:>12.2f} "
            f"{s['draft_accept_rate']:>14.2%} {s['wall_time_s']:>12.1f} "
            f"{len(recs):>8}"
        )

    print("\n" + "=" * 130)
    print("🔍 cee_dssd 按 gamma 组合拆解")
    print("=" * 130)
    records = metrics.get("cee_dssd", [])
    if not records:
        print("  No data yet.")
        return
    groups = defaultdict(list)
    for r in records:
        g1 = r.get("gamma1", "?")
        g2 = r.get("gamma2", "?")
        groups[(g1, g2)].append(r)

    print(f"  {'gamma1/gamma2':<16} {'throughput':>12} {'accept_rate':>14} {'wall_time':>12} {'samples':>8}")
    print(f"  {'-'*16} {'-'*12} {'-'*14} {'-'*12} {'-'*8}")
    for (g1, g2) in sorted(groups.keys()):
        recs = groups[(g1, g2)]
        s = summarize("", recs)
        print(
            f"  {f'{g1}/{g2}':<16} {s['throughput']:>12.2f} "
            f"{s['draft_accept_rate']:>14.2%} {s['wall_time_s']:>12.1f} "
            f"{len(recs):>8}"
        )


def main():
    metrics = collect_metrics("exp")
    print(f"Loaded {sum(len(v) for v in metrics.values())} metric files across "
          f"{len(metrics)} modes")
    print(f"Modes found: {sorted(metrics.keys())}")

    compare_cee_vs_non_cee(metrics)
    compare_by_gamma(metrics)

    # Root cause summary
    print("\n" + "=" * 130)
    print("📊 根因分析要点")
    print("=" * 130)
    print("""
1. little_accept_rate: little→draft 的接受率是关键
   - 如果 < 60%, 则 little 模型推测的 token 一半以上被 draft 拒绝
   - 被拒绝的 token 浪费了: little 前向计算 + edge_end 带宽 + 额外的连接开销

2. draft_forward_times 对比: cee 模式下 draft 模型总前向次数是否远超非 cee?
   - cee 每轮 draft 既要验证 little 的 token 又要自己推测 → 前向次数堆叠

3. communication_time: cee 的双跳通信时间是否抵得过非 cee 的单跳?
   - edge_end 带宽极高时, edge_end 通信时间微不足道 (≈0.04s)
   - 但 connect_times.end 大幅增加连接开销

4. computation_time: cee 多了 little model 的计算量
""")

    # Check experiment completion
    total_modes = ["cee_dsd", "cee_dssd", "dist_spec", "dist_split_spec"]
    for m in total_modes:
        count = len(metrics.get(m, []))
        print(f"  {m}: {count} experiments completed")


if __name__ == "__main__":
    main()
