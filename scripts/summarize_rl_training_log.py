"""Summarize an RL training/eval log produced by eval/eval_mixed.py.

Usage:
    python scripts/summarize_rl_training_log.py <logfile> [--tail N]

Extracts per-sample TPS ("Average Generation Speed"), best-TPS events and the
final RL epsilon, and prints a compact summary plus a per-sample TPS series
useful for before/after comparisons.
"""

import argparse
import re
import statistics
from pathlib import Path

SPEED_RE = re.compile(r"Average Generation Speed:\s*([\d.]+)\s*tokens/s")
BEST_RE = re.compile(
    r"\[(?P<agent>rl_adapter_\w+)\] New Best TPS:\s*([\d.]+)"
)
EPS_RE = re.compile(
    r"\[(?P<agent>rl_adapter_\w+)\] Step:\s*\d+.*?Epsilon:\s*([\d.]+)"
)


def summarize(path: Path, tail: int | None = None) -> dict:
    text = path.read_text(errors="replace")
    speeds = [float(m) for m in SPEED_RE.findall(text)]
    best_events = [
        (m.group("agent"), float(v))
        for m in re.finditer(
            r"\[(?P<agent>rl_adapter_\w+)\] New Best TPS:\s*([\d.]+)", text
        )
    ]
    epsilons = [
        (m.group("agent"), float(v))
        for m in re.finditer(
            r"\[(?P<agent>rl_adapter_\w+)\] Step:\s*\d+.*?Epsilon:\s*([\d.]+)", text
        )
    ]
    # Final epsilon per agent = value of the last matching line
    final_eps = {}
    for agent, value in epsilons:
        final_eps[agent] = value

    series = speeds[-tail:] if tail else speeds
    best_tps = {}
    for agent, value in best_events:
        best_tps[agent] = max(best_tps.get(agent, -1.0), value)

    return {
        "n_samples": len(speeds),
        "mean_tps_all": statistics.fmean(speeds) if speeds else 0.0,
        "mean_tps_tail": statistics.fmean(series) if series else 0.0,
        "max_tps": max(speeds) if speeds else 0.0,
        "best_tps": best_tps,
        "final_epsilon": final_eps,
        "speeds": speeds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logfile")
    parser.add_argument("--tail", type=int, default=50)
    args = parser.parse_args()

    summary = summarize(Path(args.logfile), args.tail)
    print(f"Log: {args.logfile}")
    print(f"Samples: {summary['n_samples']}")
    print(f"Mean TPS (all): {summary['mean_tps_all']:.2f}")
    print(f"Mean TPS (last {args.tail}): {summary['mean_tps_tail']:.2f}")
    print(f"Max TPS: {summary['max_tps']:.2f}")
    print(f"Best TPS by agent: {summary['best_tps']}")
    print(f"Final epsilon: {summary['final_epsilon']}")


if __name__ == "__main__":
    main()
