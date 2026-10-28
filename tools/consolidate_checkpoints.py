#!/usr/bin/env python3
"""把分散的 checkpoint 合并到单一根目录，并用符号链接保持旧路径可用。

当前布局（2026-10-21）：
- ``checkpoints/``：RL agents（``rl_agents``）、历史实验目录（``rl_*``）、
  系列目录（``qwen15``/``qwen``/``llama``/``vicuna``/``best``）
- ``src/SpecDec_pp/checkpoints/``：acc heads 与旧系列目录

目标布局：
- ``checkpoints/rl/{main,little}/<pair>/...``（原 ``rl_agents``）
- ``checkpoints/acc_head/<pair>/<run>/...``（原 ``src/SpecDec_pp/checkpoints/acc_head``）
- 其余旧目录保持在 ``checkpoints/`` 下，由 manifest 标 status=legacy
- 旧路径全部用相对符号链接指回新位置

用法：
    python tools/consolidate_checkpoints.py            # dry-run，只打印计划
    python tools/consolidate_checkpoints.py --apply    # 执行移动 + 建链接
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _rel(target: Path) -> str:
    return str(target.relative_to(ROOT))


def _move(src: Path, dst: Path, apply: bool) -> None:
    print(f"move  {_rel(src)} -> {_rel(dst)}")
    if not apply:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        raise SystemExit(f"目标已存在，拒绝覆盖: {_rel(dst)}")
    shutil.move(str(src), str(dst))


def _symlink(link: Path, target: str, apply: bool) -> None:
    print(f"link  {_rel(link)} -> {target}")
    if not apply:
        return
    if link.is_symlink():
        if str(link.readlink()) == target:
            return
        link.unlink()
    elif link.exists():
        # 合并后源目录会空出来；允许删除空目录再建链接
        if link.is_dir() and not any(link.iterdir()):
            link.rmdir()
        else:
            raise SystemExit(f"路径已存在且不是符号链接: {_rel(link)}")
    link.symlink_to(target)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true", help="执行（默认 dry-run）")
    args = ap.parse_args(argv)
    apply = args.apply

    ckpt = ROOT / "checkpoints"

    # rl_agents -> rl，旧路径保留链接
    if (ckpt / "rl_agents").is_dir() and not (ckpt / "rl_agents").is_symlink():
        _move(ckpt / "rl_agents", ckpt / "rl", apply)
        _symlink(ckpt / "rl_agents", "rl", apply)
    else:
        print("skip  checkpoints/rl_agents（已是链接或不存在）")

    # 注意：src/SpecDec_pp/checkpoints/acc_head 属于 SpecDec_pp 子模块，
    # 不移动到主仓（会弄脏子模块）。它由 checkpoints/manifest.json 统一索引，
    # remote_path 保持 src/SpecDec_pp/checkpoints/... 不变。

    print("\n完成" + ("" if apply else "（dry-run；加 --apply 执行）"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
