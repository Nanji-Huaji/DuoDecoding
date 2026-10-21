#!/usr/bin/env python3
"""把 staging 里的训练产物显式晋升为生产 checkpoint，并更新 manifest。

训练时用 ``--rl_output_dir checkpoints/staging/<run-id>``，产物写到
``<staging>/<role>/<pair>/{latest,best}.pth``；本脚本负责：

1. 校验 payload（DDQN 三件套 + 动作格）；
2. 旧生产文件备份到 ``checkpoints/archive/<timestamp>/``；
3. 复制到 ``checkpoints/rl/<role>/<pair>/``；
4. 重算 sha256/size，更新 ``checkpoints/manifest.json``（带 provenance）。

用法：
    python tools/promote_checkpoint.py --staging checkpoints/staging/run1 --dry-run
    python tools/promote_checkpoint.py --staging checkpoints/staging/run1 \\
        --roles main,little --evidence "golden q15 PASS"
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
MANIFEST = CKPT / "manifest.json"
ROLE_DIR = {"main": "rl_main", "little": "rl_little"}
REQUIRED_KEYS = {"policy_net", "target_net", "optimizer"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_payload(path: Path) -> dict:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - 统一补上文件上下文
        raise ValueError(f"{path}: 无法读取 checkpoint: {exc}") from exc
    if not isinstance(payload, dict) or not REQUIRED_KEYS.issubset(payload):
        missing = REQUIRED_KEYS - set(payload if isinstance(payload, dict) else ())
        raise ValueError(f"{path}: 不是有效的 RL checkpoint（缺 {sorted(missing)}）")
    return payload


def _backup(existing: Path, backup_dir: Path) -> None:
    if not existing.exists():
        return
    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.move(str(existing), str(backup_dir / existing.name))


def promote(
    *,
    staging: Path,
    roles: list[str],
    pair_filter: str | None = None,
    dry_run: bool = True,
    evidence: str | None = None,
    as_best: bool = False,
    root: Path = ROOT,
) -> list[dict]:
    root = Path(root)
    ckpt = root / "checkpoints"
    manifest_path = ckpt / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else {"version": 1, "entries": {}}
    )
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    promoted: list[dict] = []

    for role in roles:
        role_root = staging / role
        if not role_root.is_dir():
            continue
        for pair_dir in sorted(p for p in role_root.iterdir() if p.is_dir()):
            pair = pair_dir.name
            if pair_filter and pair != pair_filter:
                continue
            files = sorted(pair_dir.iterdir())
            if not any(p.name.startswith("latest") for p in files):
                print(f"skip {role}/{pair}: 没有 latest.pth")
                continue
            for src in files:
                if src.suffix == ".tmp":
                    continue
                if src.suffix == ".pth":
                    validate_payload(src)
                dst = ckpt / "rl" / role / pair / src.name
                rel = str(dst.relative_to(root))
                record = {"role": ROLE_DIR[role], "pair": pair, "path": rel}
                print(
                    f"{'would promote' if dry_run else 'promote'} {src} -> {rel}"
                )
                if dry_run:
                    promoted.append(record)
                    continue
                backup_dir = ckpt / "archive" / stamp / role / pair
                _backup(dst, backup_dir)
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                digest = sha256_file(dst)
                manifest["entries"][rel] = {
                    "remote_path": rel,
                    "sha256": digest,
                    "size": dst.stat().st_size,
                    "role": ROLE_DIR[role],
                    "pair": pair,
                    "status": "production",
                    "legacy_paths": [
                        f"checkpoints/rl_agents/{role}/{pair}/{src.name}"
                    ],
                    "provenance": {
                        "promoted_at": stamp,
                        "staging": str(src.relative_to(root)),
                        "evidence": evidence or "",
                    },
                }
                record["sha256"] = digest
                record["size"] = dst.stat().st_size
                promoted.append(record)

                # 生产 best 由晋升决定，而不是训练吞吐
                if as_best and src.name == "latest.pth":
                    best_dst = dst.with_name("best.pth")
                    _backup(best_dst, backup_dir)
                    shutil.copy2(dst, best_dst)
                    best_rel = str(best_dst.relative_to(root))
                    manifest["entries"][best_rel] = {
                        "remote_path": best_rel,
                        "sha256": sha256_file(best_dst),
                        "size": best_dst.stat().st_size,
                        "role": ROLE_DIR[role],
                        "pair": pair,
                        "status": "production",
                        "designation": "best",
                        "legacy_paths": [
                            f"checkpoints/rl_agents/{role}/{pair}/best.pth"
                        ],
                        "provenance": {
                            "promoted_at": stamp,
                            "staging": str(src.relative_to(root)),
                            "evidence": evidence or "",
                            "designated_best": True,
                        },
                    }
                    print(f"designate best: {best_rel}")

    if not dry_run and promoted:
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"manifest updated: {manifest_path}")
    return promoted


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--staging", default=str(CKPT / "staging"))
    ap.add_argument("--roles", default="main,little")
    ap.add_argument("--pair", default=None)
    ap.add_argument("--evidence", default=None)
    ap.add_argument("--as-best", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    roles = [r.strip() for r in args.roles.split(",") if r.strip()]
    promoted = promote(
        staging=Path(args.staging),
        roles=roles,
        pair_filter=args.pair,
        dry_run=args.dry_run,
        evidence=args.evidence,
        as_best=args.as_best,
    )
    print(f"{len(promoted)} files {'planned' if args.dry_run else 'promoted'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
