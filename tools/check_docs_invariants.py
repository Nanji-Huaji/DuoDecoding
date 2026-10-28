#!/usr/bin/env python3
"""检查 Markdown 文档改写是否动了不该动的内容。

用于 humanizer-zh 之类的风格清洗：叙述可以变，代码块、行内代码、表格、
标题序列、URL 必须逐字不变。

用法（仓库根目录）：
    python tools/check_docs_invariants.py --base HEAD
    python tools/check_docs_invariants.py --base HEAD README.md docs/checkpoints.md
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIST_FILE = ROOT / ".refactor" / "docs_humanize" / "files.txt"

FENCE_RE = re.compile(r"```[^\n]*\n(.*?)```", re.S)
INLINE_RE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
URL_RE = re.compile(r"https?://[^\s)\]>]+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.M)
TABLE_RE = re.compile(r"^\|.*\|$", re.M)


def invariants(text: str) -> dict[str, Counter]:
    without_fences = FENCE_RE.sub("", text)
    return {
        "code_blocks": Counter(m.strip() for m in FENCE_RE.findall(text)),
        "inline_code": Counter(INLINE_RE.findall(without_fences)),
        "tables": Counter(m.strip() for m in TABLE_RE.findall(without_fences)),
        "headings": Counter(f"{m[0].strip()} {m[1].strip()}" for m in HEADING_RE.findall(text)),
        "urls": Counter(URL_RE.findall(text)),
    }


def compare(base: str, new: str) -> list[str]:
    a, b = invariants(base), invariants(new)
    problems: list[str] = []
    for key in a:
        missing = a[key] - b[key]
        added = b[key] - a[key]
        for item, count in missing.items():
            problems.append(f"{key}: 丢失 {count}x {item[:80]!r}")
        for item, count in added.items():
            problems.append(f"{key}: 新增 {count}x {item[:80]!r}")
    return problems


def _base_text(path: str, base: str) -> str:
    return subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=ROOT).decode(
        "utf-8"
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="HEAD")
    ap.add_argument("paths", nargs="*")
    args = ap.parse_args(argv)
    paths = args.paths
    if not paths and LIST_FILE.is_file():
        paths = [
            line.strip()
            for line in LIST_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    if not paths:
        print("没有待检查文件；传入 paths 或先生成 .refactor/docs_humanize/files.txt")
        return 2
    bad = 0
    for path in paths:
        current = (ROOT / path)
        if not current.is_file():
            print(f"SKIP {path}: 当前文件不存在")
            continue
        try:
            base_text = _base_text(path, args.base)
        except subprocess.CalledProcessError:
            print(f"SKIP {path}: base {args.base} 中不存在")
            continue
        problems = compare(base_text, current.read_text(encoding="utf-8"))
        if problems:
            bad += 1
            print(f"FAIL {path}")
            for p in problems[:20]:
                print("  -", p)
        else:
            print(f"OK   {path}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
