"""Focused comparison of cee vs non-cee from current experiment scan only."""

import json
import os
import glob
from collections import defaultdict

# Filter to experiments from today's scan (20260505_181158)
TODAY_FILE = "experiment_results/experiment_summary_20260505_181158.json"


def load_summary(path: str):
    with open(path) as f:
        return json.load(f)


def get_metrics_for_exp_name(exp_name: str) -> dict:
    """Find metrics JSON for a given exp_name."""
    base = f"exp/{exp_name}"
    if os.path.isdir(base):
        for root, _, files in os.walk(base):
            for file in files:
                if file.endswith("_metrics.json"):
                    with open(os.path.join(root, file)) as f:
                        return json.load(f)
    return {}


def main():
    results = load_summary(TODAY_FILE)
    print(f"Loaded {len(results)} experiments from {TODAY_FILE}")

    # Group by eval_mode and model series
    groups = defaultdict(list)
    for r in results:
        mode = r.get("config", {}).get("eval_mode", "?")
        dm = r.get("config", {}).get("draft_model", "?")
        tm = r.get("config", {}).get("target_model", "?")
        lm = r.get("config", {}).get("little_model", "?")
        g1 = r.get("config", {}).get("gamma1", "?")
        g2 = r.get("config", {}).get("gamma2", "?")
        status = r.get("status", "?")
        groups[(mode, dm, tm, lm)].append((r, g1, g2, status))

    # Target modes
    targets = ["dist_spec", "cee_dsd", "dist_split_spec", "cee_dssd"]

    for (mode, dm, tm, lm), items in groups.items():
        if mode not in targets:
            continue
        # Filter to llama series or show all
        if "llama" not in dm and "qwen" not in dm.lower():
            continue

        print(f"\n{'='*100}")
        chain = f"{lm} → {dm} → {tm}" if mode.startswith("cee") else f"{dm} → {tm}"
        print(f"Mode: {mode}  |  Chain: {chain}")
        print(f"{'gamma1/g':<10} {'gamma2':<8} {'throughput':>12} {'draft_acc':>12} "
              f"{'little_acc':>12} {'wall(s)':>10} {'comm(s)':>10} {'comp(s)':>10} {'status':>10}")
        print(f"{'-'*98}")

        for r, g1, g2, status in sorted(items, key=lambda x: (x[1], x[2])):
            m = get_metrics_for_exp_name(r.get("exp_name", ""))
            if not m:
                continue
            tp = m.get("throughput", 0)
            da = m.get("draft_accepted_tokens", 0) / max(m.get("draft_generated_tokens", 1), 1)
            la = m.get("little_accepted_tokens", 0) / max(m.get("little_generated_tokens", 1), 1)
            wt = m.get("wall_time", 0)
            comm = m.get("communication_time", 0)
            comp = m.get("computation_time", 0)
            print(f"{str(g1):<10} {str(g2):<8} {tp:>12.2f} {da:>12.2%} "
                  f"{la:>12.2%} {wt:>10.1f} {comm:>10.1f} {comp:>10.1f} {status:>10}")


if __name__ == "__main__":
    main()
