#!/usr/bin/env python3
"""checkpoint 一键入口：pull / push / verify / promote / status。

    python tools/checkpoints.py pull        # 下载全部 production（RL + acc heads）并校验
    python tools/checkpoints.py push --repo-id <user>/DuoDecoding-checkpoints --commit
                                            # 刷新 manifest + 上传 RL + 提交清单
    python tools/checkpoints.py verify      # 本地 manifest + sha 校验
    python tools/checkpoints.py promote --staging checkpoints/staging/<run> --as-best
    python tools/checkpoints.py status      # 清单概览 / 缺失生产文件

push 默认只上传无公共源的 RL 条目（acc heads 已在 ArcticHuaji/specdecpp-acc-heads）；
--commit 会把更新后的 checkpoints/manifest.json 提交到 git。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
sys.path.insert(0, str(TOOLS))

import checkpoint_manifest as cm  # noqa: E402


def cmd_status(argv: list[str]) -> int:
    manifest = json.loads(cm.MANIFEST.read_text(encoding="utf-8"))
    entries = manifest["entries"].values()
    prod = [e for e in entries if e.get("status") == "production"]
    legacy = [e for e in entries if e.get("status") == "legacy"]
    missing = [e for e in prod if not (ROOT / e["remote_path"]).is_file()]
    hub = manifest.get("hub", {}).get("repo_id")
    print(f"entries: {len(manifest['entries'])}（production {len(prod)} / legacy {len(legacy)}）")
    print(f"hub: {hub or '(未配置，push 时用 --repo-id)'}")
    print(f"production 大小: {sum(e['size'] for e in prod) / 1e6:.1f} MB")
    if missing:
        print(f"缺失 production 文件 {len(missing)} 个：")
        for e in missing[:10]:
            print(f"  {e['remote_path']}")
        return 1
    print("production 文件齐全")
    return 0


def cmd_pull(argv: list[str]) -> int:
    import fetch_checkpoints

    return fetch_checkpoints.main(argv)


def cmd_verify(argv: list[str]) -> int:
    import fetch_checkpoints

    rc_manifest = cm.main(["--check"])
    rc_files = fetch_checkpoints.main(["--verify-only", "--only", "production"])
    return 1 if (rc_manifest or rc_files) else 0


def cmd_promote(argv: list[str]) -> int:
    import promote_checkpoint

    return promote_checkpoint.main(argv)


def cmd_push(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="checkpoints.py push")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--create", action="store_true")
    parser.add_argument("--only", choices=["production", "all"], default="production")
    parser.add_argument("--groups", default="rl")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--commit", action="store_true")
    parser.add_argument(
        "--message",
        default="chore(checkpoints): 更新 checkpoint manifest",
    )
    args = parser.parse_args(argv)

    import upload_checkpoints

    rc = upload_checkpoints.main(
        [
            "--repo-id", args.repo_id,
            "--revision", args.revision,
            "--only", args.only,
            "--groups", args.groups,
            *(["--create"] if args.create else []),
            *(["--dry-run"] if args.dry_run else []),
        ]
    )
    if rc or args.dry_run:
        return rc

    manifest = ROOT / "checkpoints" / "manifest.json"
    subprocess.run(["git", "add", str(manifest)], cwd=ROOT, check=False)
    if args.commit:
        diff = subprocess.run(
            ["git", "diff", "--cached", "--quiet", "--", str(manifest)],
            cwd=ROOT,
        )
        if diff.returncode == 0:
            print("manifest 无变化，跳过 commit")
        else:
            subprocess.run(
                ["git", "commit", "-m", args.message, "--", str(manifest)],
                cwd=ROOT,
                check=False,
            )
    return 0


COMMANDS = {
    "pull": cmd_pull,
    "push": cmd_push,
    "verify": cmd_verify,
    "promote": cmd_promote,
    "status": cmd_status,
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    handler = COMMANDS.get(cmd)
    if handler is None:
        print(f"未知子命令 {cmd!r}；可用: {sorted(COMMANDS)}")
        return 2
    return handler(rest)


if __name__ == "__main__":
    raise SystemExit(main())
