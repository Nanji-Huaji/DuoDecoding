#!/usr/bin/env python3
"""按 checkpoints/manifest.json 下载并校验 checkpoint，并恢复旧路径符号链接。

用法：
    python tools/fetch_checkpoints.py --dry-run
    python tools/fetch_checkpoints.py --only production
    python tools/fetch_checkpoints.py --verify-only

下载源优先取 manifest["hub"]["repo_id"]（Hugging Face），可用 --repo-id 覆盖。
校验失败会立即退出，不会留下"看似可用"的文件。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
MANIFEST = CKPT / "manifest.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def link_legacy(entry: dict, root: Path) -> list[Path]:
    created = []
    for legacy in entry.get("legacy_paths") or []:
        link = ROOT / legacy
        target = os.path.relpath(ROOT / entry["remote_path"], link.parent)
        if link.is_symlink():
            if str(link.readlink()) == target:
                continue
            link.unlink()
        elif link.exists():
            print(f"skip link（已存在且非链接）: {legacy}")
            continue
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        created.append(link)
    return created


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--root", default=str(CKPT))
    ap.add_argument("--repo-id", default=None)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--only", choices=["production", "all"], default="production")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    hub = manifest.get("hub") or {}
    repo_id = args.repo_id or hub.get("repo_id")
    revision = args.revision or hub.get("revision", "main")
    root = Path(args.root)

    entries = [
        e for e in manifest["entries"].values()
        if args.only == "all" or e.get("status") == "production"
    ]
    missing = [e for e in entries if not (ROOT / e["remote_path"]).is_file()]
    print(f"{len(entries)} entries, {len(missing)} missing (only={args.only})")

    requires_net = [e for e in missing if not args.verify_only]
    if requires_net and not repo_id:
        print("manifest 未配置 hub.repo_id，且存在缺失文件；请用 --repo-id 指定")
        return 2
    if requires_net:
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            print("需要 huggingface_hub: pip install huggingface_hub")
            return 2

    for entry in entries:
        remote = entry["remote_path"]
        path = ROOT / remote
        if not path.is_file() and not args.verify_only:
            if args.dry-run:
                print(f"would download {entry['path']}")
                continue
            print(f"download {entry['path']}")
            hf_hub_download(
                repo_id=repo_id,
                filename=entry["path"],
                revision=revision,
                local_dir=str(root),
            )
        if not path.is_file():
            print(f"MISSING {remote}")
            return 1
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            print(f"SHA MISMATCH {remote}: {digest} != {entry['sha256']}")
            return 1
        for link in link_legacy(entry, root):
            print(f"link {link.relative_to(ROOT)}")

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
