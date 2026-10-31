#!/usr/bin/env python3
"""跨机器复现的分布一致性检查（R2 的补充口径）。

同机回归用 test/golden/compare.py 的 LOCKED/token 逐位规则；换 GPU/驱动后
token 序列不再逐位一致，但仍应满足：

- 生成长度分布接近（two-sample KS D ≤ 0.2）
- 生成 token 的 unigram 分布接近（cosine ≥ 0.99，TV ≤ 0.05）
- accuracy 差异 ≤ 0.05（若有 correct 字段）
- token 一致率只作诊断，不判失败

用法：
    python tools/compare_distributions.py baseline.jsonl new.jsonl
    python tools/compare_distributions.py baseline.jsonl new.jsonl --json
"""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def ks_lengths(a: list[int], b: list[int]) -> float:
    """two-sample KS 统计量（按取值合并并列，避免 128 截断造成的假跳变）。"""
    ca, cb = collections.Counter(a), collections.Counter(b)
    n, m = len(a), len(b)
    ai = bi = 0
    d = 0.0
    for value in sorted(set(ca) | set(cb)):
        ai += ca[value]
        bi += cb[value]
        d = max(d, abs(ai / n - bi / m))
    return d


def unigram(records: list[dict]) -> collections.Counter:
    c: collections.Counter = collections.Counter()
    for r in records:
        c.update(r.get("gen_ids") or [])
    return c


def cosine_tv(ca: collections.Counter, cb: collections.Counter) -> tuple[float, float]:
    vocab = set(ca) | set(cb)
    va = [ca.get(k, 0) for k in vocab]
    vb = [cb.get(k, 0) for k in vocab]
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    cos = dot / (na * nb) if na and nb else 0.0
    tv = 0.5 * sum(abs(x / sum(va) - y / sum(vb)) for x, y in zip(va, vb))
    return cos, tv


def compare(
    baseline: list[dict],
    new: list[dict],
    *,
    max_ks: float = 0.2,
    min_cosine: float = 0.99,
    max_tv: float = 0.05,
    max_acc_diff: float = 0.05,
) -> dict:
    la = [len(r.get("gen_ids") or []) for r in baseline]
    lb = [len(r.get("gen_ids") or []) for r in new]
    d = ks_lengths(la, lb)
    ca, cb = unigram(baseline), unigram(new)
    cos, tv = cosine_tv(ca, cb)

    acc_a = acc_b = None
    if baseline and all("correct" in r for r in baseline):
        acc_a = sum(bool(r["correct"]) for r in baseline) / len(baseline)
    if new and all("correct" in r for r in new):
        acc_b = sum(bool(r["correct"]) for r in new) / len(new)
    acc_diff = abs(acc_a - acc_b) if acc_a is not None and acc_b is not None else None

    token_match = sum(
        a.get("gen_ids") == b.get("gen_ids") for a, b in zip(baseline, new)
    )

    checks = {
        "length_ks": d <= max_ks,
        "unigram_cosine": cos >= min_cosine,
        "unigram_tv": tv <= max_tv,
        "accuracy": acc_diff is None or acc_diff <= max_acc_diff,
    }
    return {
        "samples": [len(baseline), len(new)],
        "tokens": [sum(la), sum(lb)],
        "length_mean": [sum(la) / len(la), sum(lb) / len(lb)],
        "length_ks": d,
        "unigram_cosine": cos,
        "unigram_tv": tv,
        "accuracy": [acc_a, acc_b],
        "accuracy_diff": acc_diff,
        "token_match": token_match,
        "thresholds": {
            "max_ks": max_ks,
            "min_cosine": min_cosine,
            "max_tv": max_tv,
            "max_acc_diff": max_acc_diff,
        },
        "checks": checks,
        "pass": all(checks.values()),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--max-ks", type=float, default=0.2)
    ap.add_argument("--min-cosine", type=float, default=0.99)
    ap.add_argument("--max-tv", type=float, default=0.05)
    ap.add_argument("--max-acc-diff", type=float, default=0.05)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    result = compare(
        load_jsonl(args.baseline),
        load_jsonl(args.new),
        max_ks=args.max_ks,
        min_cosine=args.min_cosine,
        max_tv=args.max_tv,
        max_acc_diff=args.max_acc_diff,
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(f"samples: {result['samples']}")
        print(f"tokens:  {result['tokens']}")
        print(
            f"length:  mean {result['length_mean'][0]:.2f} / "
            f"{result['length_mean'][1]:.2f}, KS D={result['length_ks']:.3f} "
            f"(<= {args.max_ks})"
        )
        print(
            f"unigram: cosine={result['unigram_cosine']:.5f} (>= {args.min_cosine}), "
            f"TV={result['unigram_tv']:.4f} (<= {args.max_tv})"
        )
        print(
            f"accuracy: {result['accuracy']} diff={result['accuracy_diff']} "
            f"(<= {args.max_acc_diff})"
        )
        print(f"token match: {result['token_match']}/{result['samples'][0]} (diagnostic)")
        print("RESULT", "PASS" if result["pass"] else "FAIL")
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
