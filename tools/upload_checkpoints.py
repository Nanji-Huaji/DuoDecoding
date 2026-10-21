#!/usr/bin/env python3
"""把 checkpoints/ 按 manifest 上传到 Hugging Face Hub。

用法：
    python tools/upload_checkpoints.py --repo-id <user>/DuoDecoding-checkpoints --create
    python tools/upload_checkpoints.py --repo-id ... --dry-run

需要 `hf auth login` 或 HF_TOKEN。上传前会重新生成 manifest 并写入 hub.repo_id。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import checkpoint_manifest as cm


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo-id", required=True)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--create", action="store_true")
    ap.add_argument("--only", choices=["production", "all"], default="production")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    manifest = cm.build()
    manifest["hub"] = {
        "repo_id": args.repo_id,
        "repo_type": "model",
        "revision": args.revision,
    }
    entries = [
        e for e in manifest["entries"].values()
        if args.only == "all" or e.get("status") == "production"
    ]
    print(f"{len(entries)} files -> {args.repo_id}@{args.revision} (only={args.only})")
    if args.dry_run:
        for e in entries:
            print(f"  {e['remote_path']}")
        return 0

    from huggingface_hub import HfApi

    api = HfApi()
    if args.create:
        api.create_repo(args.repo_id, repo_type="model", exist_ok=True)
    for i, entry in enumerate(entries, 1):
        print(f"[{i}/{len(entries)}] {entry['remote_path']}")
        api.upload_file(
            path_or_fileobj=cm.ROOT / entry["remote_path"],
            path_in_repo=entry["remote_path"],
            repo_id=args.repo_id,
            revision=args.revision,
        )
    manifest_path = cm.CKPT / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    api.upload_file(
        path_or_fileobj=manifest_path,
        path_in_repo="manifest.json",
        repo_id=args.repo_id,
        revision=args.revision,
    )
    print("done")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    raise SystemExit(main())
