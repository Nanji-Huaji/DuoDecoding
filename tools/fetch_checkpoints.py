#!/usr/bin/env python3
"""统一下载全部实验 checkpoint（RL agents + SpecDec++ acc heads）。

每个条目自带来源：
- RL agents：manifest["hub"]["repo_id"]（本仓库的 HF 镜像，需先配置/上传）
- acc heads：``ArcticHuaji/specdecpp-acc-heads``（已有公共仓库，按
  ``acc_head_registry.json`` 的 hf_subpath 逐文件取）

下载后逐个校验 sha256，并按 manifest 的 legacy_paths 建兼容符号链接。

用法：
    python tools/fetch_checkpoints.py --dry-run
    python tools/fetch_checkpoints.py --groups acc_head
    python tools/fetch_checkpoints.py --verify-only
    python tools/fetch_checkpoints.py --rl-repo <user>/DuoDecoding-checkpoints
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CKPT = ROOT / "checkpoints"
MANIFEST = CKPT / "manifest.json"
ACC_REGISTRY = ROOT / "src" / "SpecDec_pp" / "checkpoints" / "acc_head_registry.json"
ASSETS = ROOT / "scripts" / "setup" / "assets.json"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def group_of(entry: dict) -> str:
    role = entry.get("role", "")
    if role.startswith("rl_"):
        return "rl"
    if role == "acc_head":
        return "acc_head"
    return "legacy"


def link_legacy(entry: dict) -> list[Path]:
    created = []
    for legacy in entry.get("legacy_paths") or []:
        link = ROOT / legacy
        target = os.path.relpath(ROOT / entry["remote_path"], link.parent)
        if link.is_symlink():
            if str(link.readlink()) == target:
                continue
            link.unlink()
        elif link.exists():
            continue  # 已由目录级符号链接覆盖
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(target)
        created.append(link)
    return created


def _acc_complete(local: Path) -> bool:
    return (local / "model.safetensors").exists() or (local / "pytorch_model.bin").exists()


def sync_acc_heads(args, dry_run: bool) -> int:
    """按 acc_head_registry.json 补全缺失的验收头（支持镜像/代理）。"""
    if not ACC_REGISTRY.is_file():
        print(f"skip acc heads：找不到 {ACC_REGISTRY}")
        return 0
    registry = json.loads(ACC_REGISTRY.read_text(encoding="utf-8"))
    if args.acc_head_pairs:
        wanted = {p.strip() for p in args.acc_head_pairs if p.strip()}
    elif args.all_acc_heads:
        wanted = None
    else:
        required = json.loads(ASSETS.read_text(encoding="utf-8"))["acc_head_required"]
        wanted = {p.split("/")[0] + "|" + p.split("/")[1] for p in required}

    todo = []
    for entry in registry:
        key = f"{entry['source']}--to--{entry['target']}"
        run = entry["hf_subpath"].split("/")[-1]
        if wanted is not None and f"{key}|{run}" not in wanted and entry["hf_subpath"] not in wanted:
            continue
        local = ROOT / entry["local_path"]
        if _acc_complete(local):
            continue
        todo.append(entry)

    if not todo:
        return 0
    print(f"acc heads 待补 {len(todo)} 个")
    for entry in todo:
        label = f"{entry['source']}--to--{entry['target']}/{entry['hf_subpath'].split('/')[-1]}"
        if dry_run:
            print(f"would download acc_head {label} <- {entry['hf_repo']}")
            continue
        print(f"download acc_head {label} <- {entry['hf_repo']}")
        from huggingface_hub import snapshot_download

        local = ROOT / entry["local_path"]
        local.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            snapshot_download(
                repo_id=entry["hf_repo"],
                local_dir=tmp,
                allow_patterns=[f"{entry['hf_subpath']}/*"],
                token=os.environ.get("HF_TOKEN"),
                max_workers=args.workers,
            )
            src = Path(tmp) / entry["hf_subpath"]
            if not src.exists():
                print(f"FAIL {label}: 远端缺少 {entry['hf_subpath']}")
                return 1
            for item in src.iterdir():
                dst = local / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dst)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(MANIFEST))
    ap.add_argument("--rl-repo", default=None, help="覆盖 manifest.hub.repo_id")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--only", choices=["production", "all"], default="production")
    ap.add_argument("--groups", default="rl,acc_head", help="rl,acc_head,legacy 组合")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--acc-head-pairs", default="", help="逗号分隔的 pair 或 pair/run")
    ap.add_argument("--all-acc-heads", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    hub = manifest.get("hub") or {}
    rl_repo = args.rl_repo or hub.get("repo_id")
    revision = args.revision or hub.get("revision", "main")
    groups = {g.strip() for g in args.groups.split(",") if g.strip()}

    if "acc_head" in groups:
        rc = sync_acc_heads(args, args.dry_run)
        if rc:
            return rc

    entries = [
        e for e in manifest["entries"].values()
        if (args.only == "all" or e.get("status") == "production")
        and group_of(e) in groups
    ]
    print(f"{len(entries)} entries (groups={sorted(groups)}, only={args.only})")

    missing, need_download = [], []
    for entry in entries:
        dest = ROOT / entry["remote_path"]
        if dest.is_file():
            continue
        missing.append(entry)
        repo = entry.get("hf_repo") or (rl_repo if group_of(entry) == "rl" else None)
        if repo:
            need_download.append((entry, repo))
        elif not args.verify_only:
            print(f"NO SOURCE {entry['remote_path']}（rl_repo 未配置）")

    if args.verify_only:
        for entry in missing:
            print(f"MISSING {entry['remote_path']}")
        return 1 if missing else 0

    if need_download and not args.dry_run:
        from huggingface_hub import hf_hub_download

    for entry, repo in need_download:
        dest = ROOT / entry["remote_path"]
        hf_path = entry.get("hf_path") or entry["remote_path"]
        if args.dry_run:
            print(f"would download {dest.relative_to(ROOT)} <- {repo}:{hf_path}")
            continue
        print(f"download {dest.relative_to(ROOT)} <- {repo}:{hf_path}")
        with tempfile.TemporaryDirectory() as tmp:
            hf_hub_download(
                repo_id=repo,
                filename=hf_path,
                revision=revision,
                local_dir=tmp,
            )
            src = Path(tmp) / hf_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)

    bad = 0
    for entry in entries:
        dest = ROOT / entry["remote_path"]
        if not dest.is_file():
            print(f"MISSING {entry['remote_path']}")
            bad += 1
            continue
        digest = sha256_file(dest)
        if digest != entry["sha256"]:
            print(f"SHA MISMATCH {entry['remote_path']}: {digest} != {entry['sha256']}")
            bad += 1
            continue
        for link in link_legacy(entry):
            print(f"link {link.relative_to(ROOT)}")
    print("OK" if bad == 0 else f"{bad} bad")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
