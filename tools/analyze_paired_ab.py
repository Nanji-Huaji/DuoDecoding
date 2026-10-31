#!/usr/bin/env python3
"""汇总 paired A/B 结果：median、paired ratio、硬件无关的结构指标。

输入目录约定（scripts/run_paired_ablation.sh 生成）：

    <root>/rep1/A/metrics.json, run.jsonl
    <root>/rep1/B/metrics.json, run.jsonl
    ...

用法：
    python tools/analyze_paired_ab.py <root> [--json-out summary.json]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _load_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _median(values):
    return statistics.median(values) if values else None


def collect(root: Path) -> dict:
    reps: dict[int, dict[str, dict]] = {}
    for rep_dir in sorted(root.glob("rep*")):
        rep = int(rep_dir.name.replace("rep", ""))
        slots = {}
        for slot in ("A", "B"):
            d = rep_dir / slot
            m = d / "metrics.json"
            if not m.is_file():
                continue
            slots[slot] = {
                "name": (d / "mode_name.txt").read_text().strip()
                if (d / "mode_name.txt").is_file()
                else slot,
                "metrics": _load_metrics(m),
                "jsonl": _load_jsonl(d / "run.jsonl")
                if (d / "run.jsonl").is_file()
                else [],
            }
        if slots:
            reps[rep] = slots
    return {"root": str(root), "reps": reps}


def summarize(data: dict) -> dict:
    root = data["root"]
    reps = data["reps"]
    names = {}
    per_mode = {"A": [], "B": []}
    for rep, slots in reps.items():
        for slot, item in slots.items():
            names[slot] = item["name"]
            per_mode[slot].append(item["metrics"])

    def agg(ms, key):
        vals = [m[key] for m in ms if m.get(key) is not None]
        return _median(vals)

    summary = {"root": root, "reps": len(reps), "names": names, "modes": {}}
    for slot in ("A", "B"):
        ms = per_mode[slot]
        summary["modes"][slot] = {
            k: agg(ms, k)
            for k in (
                "throughput",
                "communication_time",
                "generated_tokens",
                "target_forward_times",
                "draft_forward_times",
                "little_forward_times",
                "edge_cloud_data_bytes",
                "accuracy",
            )
        }

    pairs = []
    for rep, slots in sorted(reps.items()):
        if "A" not in slots or "B" not in slots:
            continue
        a, b = slots["A"]["metrics"], slots["B"]["metrics"]
        ja, jb = slots["A"]["jsonl"], slots["B"]["jsonl"]
        tokens = sum(x.get("gen_ids") == y.get("gen_ids") for x, y in zip(ja, jb))
        pair = {"rep": rep, "token_match": tokens, "n": min(len(ja), len(jb))}
        for key in (
            "throughput",
            "communication_time",
            "target_forward_times",
            "draft_forward_times",
            "little_forward_times",
            "edge_cloud_data_bytes",
        ):
            if a.get(key) and b.get(key):
                pair[f"ratio_{key}"] = b[key] / a[key]
        pairs.append(pair)

    summary["pairs"] = pairs
    summary["paired_ratio_median"] = {
        k: _median([p[k] for p in pairs if k in p])
        for k in pairs[0]
        if k.startswith("ratio_")
    } if pairs else {}
    return summary


def render_markdown(s: dict) -> str:
    names = s["names"]
    a_name, b_name = names.get("A", "A"), names.get("B", "B")
    lines = [
        f"# Paired A/B 汇总（{s['reps']} 轮）",
        "",
        f"- A = `{a_name}`，B = `{b_name}`",
        "",
        "| 指标 | A | B | B/A（median） |",
        "|---|---:|---:|---:|",
    ]
    keys = (
        "throughput",
        "communication_time",
        "generated_tokens",
        "target_forward_times",
        "draft_forward_times",
        "little_forward_times",
        "edge_cloud_data_bytes",
        "accuracy",
    )
    for k in keys:
        a = s["modes"]["A"].get(k)
        b = s["modes"]["B"].get(k)
        ratio = s["paired_ratio_median"].get(f"ratio_{k}")
        lines.append(
            f"| {k} | {a} | {b} | {ratio:.4f} |" if ratio is not None else f"| {k} | {a} | {b} | — |"
        )
    lines += ["", "## 硬件无关结构指标（per token）", "", "| 指标 | A | B | B/A |", "|---|---:|---:|---:|"]
    for key in ("target_forward_times", "draft_forward_times", "little_forward_times", "edge_cloud_data_bytes"):
        av = s["modes"]["A"].get(key)
        bv = s["modes"]["B"].get(key)
        at = s["modes"]["A"].get("generated_tokens")
        bt = s["modes"]["B"].get("generated_tokens")
        if not all(isinstance(x, (int, float)) and x for x in (av, bv, at, bt)):
            continue
        a_per, b_per = av / at, bv / bt
        lines.append(f"| {key}/token | {a_per:.4f} | {b_per:.4f} | {b_per/a_per:.4f} |")
    lines += ["", "## 逐轮 token 一致率", "", "| rep | 一致 | 样本数 |", "|---|---:|---:|"]
    for p in s["pairs"]:
        lines.append(f"| {p['rep']} | {p['token_match']} | {p['n']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--markdown-out", default=None)
    args = ap.parse_args(argv)
    data = collect(Path(args.root))
    s = summarize(data)
    md = render_markdown(s)
    print(md)
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(s, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out:
        Path(args.markdown_out).write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
