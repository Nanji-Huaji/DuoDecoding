"""把 exp 里逐样本的 `gen_ids` 收成 token 级 golden。

`metrics.json` 只锁聚合口径（token 总数、接受数、通信字节），两份不同的
token 序列完全可能撞出同一组指标。`gen_ids` 才是「每步选了什么」的直接
观测，而原始 run 的 jsonl 已经在主树 exp/ 里——收割是零成本，不需要 GPU。

用法（worktree 根目录）：

    # 首次收割 / 补新 tag
    python tools/harvest_tokens.py --exp /path/to/main/exp

    # 只核对，不写盘
    python tools/harvest_tokens.py --exp /path/to/main/exp --check

产出：test/golden/gammar/<tag>/tokens.json

之后每次重跑 golden 配置，用

    python test/golden/compare.py --tokens <tag> <new_dir>/*.jsonl

逐样本对齐 token 序列。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
GOLDEN_ROOT = ROOT / "test" / "golden" / "gammar"


def _sha16(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def render(payload: dict) -> str:
    """一行一个样本：重跑后 diff 只落在漂移的样本上。"""
    meta = json.dumps(payload["_meta"], ensure_ascii=False)
    rows = [json.dumps(sample, ensure_ascii=False) for sample in payload["samples"]]
    body = ",\n  ".join(rows)
    return '{\n "_meta": %s,\n "samples": [\n  %s\n ]\n}\n' % (meta, body)


def _load_run(run_dir: Path) -> tuple[Path, list[dict]]:
    files = sorted(run_dir.glob("*.jsonl"))
    if len(files) != 1:
        raise SystemExit(f"{run_dir}: 期望 1 个 jsonl，实际 {len(files)}")
    path = files[0]
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return path, records


def harvest_tag(tag: str, meta: dict, exp_root: Path) -> dict:
    run_name = Path(meta["original_run_dir"]).name
    run_dir = exp_root / "gammar" / run_name
    if not run_dir.is_dir():
        raise SystemExit(f"{tag}: 原始 run 不存在 {run_dir}")

    path, records = _load_run(run_dir)
    try:
        source = str(path.relative_to(exp_root.parent))
    except ValueError:
        source = str(path)
    # eval 侧 sample_idx 是"每个 task 内第几条"，num_samples_per_task=1 时
    # 恒为 0，不能当身份——身份用 run 内的记录顺序（index）。
    samples = [
        {
            "index": position,
            "sample_idx": record.get("sample_idx", 0),
            "gen_ids": list(record["gen_ids"]),
        }
        for position, record in enumerate(records)
    ]
    return {
        "_meta": {
            "tag": tag,
            "source": source,
            "source_sha256_16": _sha16(path),
            "samples": len(samples),
            "total_tokens": sum(len(sample["gen_ids"]) for sample in samples),
            "original_run_dir": meta["original_run_dir"],
        },
        "samples": samples,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", required=True, help="主树 exp 目录")
    parser.add_argument("--out", default=str(GOLDEN_ROOT), help="golden 根目录")
    parser.add_argument("--tag", action="append", default=[], help="只处理指定 tag")
    parser.add_argument("--check", action="store_true", help="只对比不写盘")
    args = parser.parse_args(argv)

    exp_root = Path(args.exp).resolve()
    out_root = Path(args.out)
    manifest = json.loads((out_root / "manifest.json").read_text(encoding="utf-8"))

    tags = sorted(manifest["tags"])
    if args.tag:
        tags = [tag for tag in tags if tag in set(args.tag)]
        missing = set(args.tag) - set(tags)
        if missing:
            raise SystemExit(f"未知 tag: {sorted(missing)}")

    drifted = 0
    for tag in tags:
        payload = harvest_tag(tag, manifest["tags"][tag], exp_root)
        target = out_root / tag / "tokens.json"
        text = render(payload)
        if args.check:
            current = target.read_text(encoding="utf-8") if target.exists() else ""
            status = "OK" if current == text else "DRIFT"
            drifted += status == "DRIFT"
            print(f"  {status:5s} {tag}  {payload['_meta']['samples']} 样本 / "
                  f"{payload['_meta']['total_tokens']} tokens")
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"  ✓ {tag}  {payload['_meta']['samples']} 样本 / "
              f"{payload['_meta']['total_tokens']} tokens  → {target.relative_to(ROOT)}")

    if args.check and drifted:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
