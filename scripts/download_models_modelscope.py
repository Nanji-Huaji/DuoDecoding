#!/usr/bin/env python3
"""用 ModelScope 下载大型模型（国内网络优先）。

模型清单与本地目录在 ``configs/model_repos_modelscope.json``：
- ``modelscope_id``：ModelScope 模型 id（与 HF id 常常不同）
- ``local_dir``：下载目标；已有 ``config.json`` + 权重则跳过
- ``verified``：是否已人工确认 id；false 会打印警告

用法：
    python scripts/download_models_modelscope.py --list
    python scripts/download_models_modelscope.py --only qwen15 --dry-run
    python scripts/download_models_modelscope.py --only qwen3 qwen15
    python scripts/download_models_modelscope.py --only llama --model-id llama-68m=AI-ModelScope/llama-68m
    python scripts/download_models_modelscope.py --verify-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "model_repos.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin")


def load_config(path: Path = CONFIG) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_complete(local: Path) -> bool:
    if not (local / "config.json").is_file():
        return False
    return any(p.suffix in WEIGHT_SUFFIXES for p in local.iterdir() if p.is_file())


def select(cfg: dict, only: list[str]) -> list[str]:
    if not only:
        return sorted(cfg["models"])
    aliases: list[str] = []
    for token in only:
        if token in cfg["models"]:
            aliases.append(token)
        elif token in cfg.get("groups", {}):
            aliases.extend(cfg["groups"][token])
        else:
            raise SystemExit(
                f"未知别名/组 {token!r}；可用组: {sorted(cfg.get('groups', {}))}"
            )
    return aliases


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", default=[], help="别名或组（qwen15/qwen3/llama/vicuna）")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--model-id", action="append", default=[], help="别名=ModelScope id 覆盖")
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args(argv)

    cfg = load_config(Path(args.config))
    overrides = dict(item.split("=", 1) for item in args.model_id if "=" in item)
    aliases = select(cfg, args.only)

    if args.list:
        print(f"{'alias':22s} {'modelscope_id':45s} {'local_dir':40s} 状态")
        for alias in aliases:
            m = cfg["models"][alias]
            local = ROOT / m["local_dir"]
            state = "complete" if is_complete(local) else "missing"
            print(
                f"{alias:22s} {str(m.get('modelscope_id')):45s} "
                f"{m['local_dir']:40s} {state}"
            )
        return 0

    missing = 0
    for alias in aliases:
        m = cfg["models"][alias]
        local = ROOT / m["local_dir"]
        if is_complete(local) and not args.force:
            print(f"[skip] {alias}: {m['local_dir']} 已存在")
            continue
        if args.verify_only:
            print(f"[missing] {alias}: {m['local_dir']}")
            missing += 1
            continue
        model_id = overrides.get(alias) or m.get("modelscope_id")
        if not model_id:
            print(f"[skip] {alias}: 未配置 modelscope_id（用 --model-id {alias}=<id> 指定）")
            missing += 1
            continue
        if not m.get("verified", True):
            print(f"[warn] {alias}: ModelScope id 未验证 {model_id}，如 404 请用 --model-id 覆盖")
        revision = args.revision or m.get("revision")
        print(f"[download] {alias} <- {model_id}" + (f"@{revision}" if revision else ""))
        if args.dry_run:
            continue
        try:
            from modelscope import snapshot_download
        except ImportError:
            print("需要 modelscope：pip install modelscope")
            return 2
        local.mkdir(parents=True, exist_ok=True)
        kwargs = {"local_dir": str(local)}
        if revision:
            kwargs["revision"] = revision
        try:
            snapshot_download(model_id, **kwargs)
        except TypeError:
            # 旧版 modelscope 不支持 local_dir：先落 cache，再告知路径
            path = snapshot_download(model_id, revision=revision)
            print(f"该 modelscope 版本不支持 local_dir；已下载到 {path}，请手动迁移")
            continue
        if not is_complete(local):
            print(f"[error] {alias}: 下载后仍缺 config.json/权重（{local}）")
            missing += 1
        else:
            print(f"[ok] {alias} -> {m['local_dir']}")

    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
