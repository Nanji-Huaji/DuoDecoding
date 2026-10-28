#!/usr/bin/env python3
"""把 configs/model_repos.json 的 hf_repo 固定到当前 commit（hf_revision）。

需要能访问 huggingface.co；本机可走 Clash：

    export HTTP_PROXY=http://127.0.0.1:7890
    export HTTPS_PROXY=http://127.0.0.1:7890
    unset ALL_PROXY   # 本机 ALL_PROXY 指向 tailscaled，不是外网代理

用法：
    python tools/pin_model_revisions.py --dry-run
    python tools/pin_model_revisions.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "model_repos.json"


def pin(config: Path = CONFIG, dry_run: bool = False) -> int:
    from huggingface_hub import HfApi

    cfg = json.loads(config.read_text(encoding="utf-8"))
    api = HfApi()
    revisions: dict[str, str] = {}
    failures = 0
    for alias, entry in sorted(cfg["models"].items()):
        repo = entry.get("hf_repo")
        if not repo:
            continue
        if repo not in revisions:
            try:
                revisions[repo] = api.model_info(repo).sha
                print(f"OK   {repo} -> {revisions[repo][:12]}")
            except Exception as exc:  # noqa: BLE001
                print(f"FAIL {repo}: {type(exc).__name__}: {str(exc)[:100]}")
                revisions[repo] = ""
                failures += 1
        sha = revisions[repo]
        if not sha:
            continue
        if entry.get("hf_revision") != sha:
            entry["hf_revision"] = sha
            if dry_run:
                print(f"  would pin {alias}")
    if not dry_run:
        config.write_text(
            json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"wrote {config}")
    return 1 if failures else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    return pin(Path(args.config), args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
