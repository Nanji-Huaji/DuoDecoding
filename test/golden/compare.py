"""golden 对照工具——重构期间行为漂移的执法者。

用法（在 worktree 根目录；脚本直调，避开与 stdlib `test` 包的命名冲突）：
  python test/golden/compare.py <golden_tag> <new_metrics.json>
      # 重构后的新 run 对照 golden 固件
  python test/golden/compare.py --stability <run1.json> <run2.json>
      # 校准双跑：同配置两遍，判定字段稳定性

字段分类规则（Phase 0.1 校准双跑裁决前的初版）：
  TIMING   实测计时（wall_time / throughput / *_computation_time）——
           GPU 调度噪音，只报告漂移百分比，不判失败
  LOCKED   其余全部字段（token 计数 / 接受数 / 前向数 / 通信模拟 /
           配置回显 / 历史记录）——精确相等，漂移即 FAIL
  NEW      新 run 独有的字段（如 0.4 之后新增的 git_sha）——信息性
           列出，不判失败

退出码：0 = 通过；1 = 有 LOCKED 字段漂移或缺失。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GOLDEN_ROOT = Path(__file__).parent / "gammar"

#: 实测计时字段——物理上不可复现，golden 不锁（Phase 0.1 可修订此表）
TIMING_FIELDS = frozenset(
    {
        "wall_time",
        "throughput",
        "computation_time",
        "little_computation_time",
        "draft_computation_time",
        "target_computation_time",
    }
)


def classify(field: str) -> str:
    return "timing" if field in TIMING_FIELDS else "locked"


def load_metrics(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def golden_path(tag: str) -> Path:
    p = GOLDEN_ROOT / tag / "metrics.json"
    if not p.exists():
        available = sorted(
            d.name for d in GOLDEN_ROOT.iterdir() if d.is_dir()
        )
        raise SystemExit(
            f"golden tag {tag!r} 不存在（可用: {available}）"
        )
    return p


def _fmt(v) -> str:
    s = str(v)
    return s if len(s) <= 48 else s[:45] + "..."


def compare_metrics(baseline: dict, new: dict) -> tuple[bool, list[str]]:
    """逐字段对照。返回 (是否通过, 报告行)。baseline 可为 golden 或 run1。"""
    lines: list[str] = []
    ok = True
    fails = 0

    for field in sorted(set(baseline) | set(new)):
        if field not in baseline:
            lines.append(f"  ℹ NEW     {field} = {_fmt(new[field])}")
            continue
        if field not in new:
            if classify(field) == "timing":
                lines.append(f"  ℹ 缺失    {field}（timing，忽略）")
                continue
            lines.append(f"  ✗ 缺失    {field}")
            ok = False
            fails += 1
            continue

        vb, vn = baseline[field], new[field]
        if vb == vn:
            continue

        kind = classify(field)
        if kind == "timing":
            try:
                rel = abs(vn - vb) / max(abs(vb), 1e-9) * 100
                lines.append(
                    f"  ⚠ timing  {field}: {_fmt(vb)} → {_fmt(vn)} ({rel:+.2f}%)"
                )
            except TypeError:
                lines.append(f"  ⚠ timing  {field}: {_fmt(vb)} → {_fmt(vn)}")
        else:
            lines.append(f"  ✗ FAIL    {field}: {_fmt(vb)} → {_fmt(vn)}")
            ok = False
            fails += 1

    verdict = "PASS" if ok else f"FAIL（{fails} 个锁定字段漂移）"
    lines.append(f"  判定: {verdict}")
    return ok, lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stability", action="store_true",
                        help="对照两份 run（校准模式），而非对照 golden")
    parser.add_argument("a", help="golden tag 或 run1 的 metrics.json")
    parser.add_argument("b", nargs="?", help="run2 的 metrics.json")
    args = parser.parse_args(argv)

    if args.stability:
        if not args.b:
            parser.error("--stability 需要两份 metrics.json")
        baseline, new = load_metrics(args.a), load_metrics(args.b)
        header = f"[STABILITY] {args.a}  vs  {args.b}"
    else:
        if args.b is None:
            parser.error("用法: python -m test.golden.compare <tag> <new.json>")
        baseline = load_metrics(golden_path(args.a))
        new = load_metrics(args.b)
        header = f"[GOLDEN] tag={args.a}  new={args.b}"

    ok, lines = compare_metrics(baseline, new)
    print(header)
    print("\n".join(lines))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
