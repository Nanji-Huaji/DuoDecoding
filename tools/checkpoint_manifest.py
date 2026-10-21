#!/usr/bin/env python3
"""生成 / 校验 checkpoint 清单（checkpoints/manifest.json）。

清单是唯一权威索引，覆盖两处物理根：
- checkpoints/（RL agents、历史实验、系列目录）
- src/SpecDec_pp/checkpoints/acc_head/（SpecDec_pp 子模块里的验收头）

条目的 ``remote_path`` 是仓库相对路径（也用作 Hugging Face 仓库内路径），
下载后会落在同一位置，因此代码路径不需要改。

用法：
    python tools/checkpoint_manifest.py --write
    python tools/checkpoint_manifest.py --check
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
ACC = ROOT / "src" / "SpecDec_pp" / "checkpoints" / "acc_head"
MANIFEST = CKPT / "manifest.json"
ACC_REGISTRY = ROOT / "src" / "SpecDec_pp" / "checkpoints" / "acc_head_registry.json"
SERIES = {"qwen15", "qwen", "llama", "vicuna", "best", "tinyllama"}
SKIP_NAMES = {"manifest.json", "README.md", "readme.md", "training_status.json"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(rel: Path) -> dict:
    parts = rel.parts
    if parts[0] == "rl" and len(parts) >= 4 and parts[1] in {"main", "little"}:
        role = "rl_main" if parts[1] == "main" else "rl_little"
        return {
            "role": role,
            "pair": parts[2],
            "status": "production",
            "legacy_paths": [
                f"checkpoints/rl_agents/{Path(*parts[1:])}"
            ],
        }
    if parts[0] in SERIES:
        return {"role": "series_legacy", "pair": None, "status": "legacy"}
    if parts[0].startswith(("rl_", "qwen15_backup_")):
        return {"role": "experiment_legacy", "pair": None, "status": "legacy"}
    return {"role": "other", "pair": None, "status": "legacy"}


def _acc_sources() -> list[dict]:
    if not ACC_REGISTRY.is_file():
        return []
    return json.loads(ACC_REGISTRY.read_text(encoding="utf-8"))


def entries() -> dict:
    out = {}
    acc_sources = _acc_sources()
    for path in sorted(CKPT.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.name in SKIP_NAMES:
            continue
        rel = path.relative_to(CKPT)
        out[f"checkpoints/{rel}"] = {
            "remote_path": f"checkpoints/{rel}",
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            **classify(rel),
        }
    for path in sorted(ACC.rglob("*")):
        if path.is_symlink() or not path.is_file() or path.name in SKIP_NAMES:
            continue
        if any(part in {".cache", ".git"} for part in path.parts):
            continue
        rel = path.relative_to(ROOT / "src" / "SpecDec_pp" / "checkpoints")
        parts = rel.parts  # acc_head/<pair>/...
        remote = str(Path("src") / "SpecDec_pp" / "checkpoints" / rel)
        local_abs = str(path)
        hf_repo = None
        hf_path = None
        for entry in acc_sources:
            local_root = str(ROOT / entry["local_path"])
            if local_abs.startswith(local_root + "/"):
                suffix = local_abs[len(local_root) + 1 :]
                hf_repo = entry["hf_repo"]
                hf_path = f"{entry['hf_subpath']}/{suffix}"
                break
        meta = {
            "remote_path": remote,
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
            "role": "acc_head",
            "pair": parts[1] if len(parts) > 1 else None,
            "status": "production",
        }
        if hf_repo:
            meta["hf_repo"] = hf_repo
            meta["hf_path"] = hf_path
        out[remote] = meta
    return out


def build() -> dict:
    return {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hub": {"repo_id": None, "repo_type": "model", "revision": "main"},
        "entries": entries(),
    }


def check(manifest: dict) -> int:
    bad = 0
    for remote, meta in manifest["entries"].items():
        path = ROOT / remote
        if not path.is_file():
            print(f"MISSING {remote}")
            bad += 1
        elif path.stat().st_size != meta["size"] or sha256_file(path) != meta["sha256"]:
            print(f"CHANGED {remote}")
            bad += 1
    listed = set(entries())
    for extra in sorted(listed - set(manifest["entries"])):
        print(f"EXTRA   {extra}")
    print(f"\n{len(manifest['entries'])} entries, {bad} bad, {len(listed - set(manifest['entries']))} extra")
    return 1 if bad or (listed - set(manifest["entries"])) else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)
    if args.check:
        return check(json.loads(MANIFEST.read_text(encoding="utf-8")))
    manifest = build()
    MANIFEST.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {MANIFEST} ({len(manifest['entries'])} entries)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
