"""Golden 固件的外部 checkpoint 资产检查。

14 项历史 golden 资产（baselines 7 + gammar 7）依赖未入库的 RL checkpoint。
本模块把判定收敛为一条规则，供两套固件共用：

- 已存在的文件永远按 manifest 的 ``sha256_16`` 断言，漂移立刻失败；
- 缺失默认 ``pytest.skip``（= 未验收，不是通过），并给出原因；
- ``--require-golden-assets`` 严格模式下缺失改为失败。

先检查全部已存在文件，再处理缺失，避免某个 role 缺失掩盖另一个 role 的漂移。
参考路径统一相对仓库根解析（不依赖调用者 cwd）。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _resolve(path: str, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def assert_golden_checkpoints(
    specs: Mapping[str, Mapping[str, str]],
    *,
    require: bool,
    repo_root: Path = REPO_ROOT,
) -> None:
    """校验一个 golden tag 绑定的 checkpoint；漂移失败，缺失 skip 或失败。"""
    missing: list[str] = []
    drifted: list[str] = []
    for role, spec in specs.items():
        path = _resolve(spec["path"], repo_root)
        if not path.is_file():
            missing.append(f"{role}: {path}")
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest[:16] != spec["sha256_16"]:
            drifted.append(
                f"{role}: {path} sha256_16={digest[:16]} != {spec['sha256_16']}"
            )

    if drifted or (missing and require):
        lines = ["golden checkpoint 校验失败："]
        if drifted:
            lines += ["指纹漂移（golden 需重新收割）：", *drifted]
        if missing:
            heading = (
                "--require-golden-assets 要求全部资产存在，但缺少："
                if require else "另有缺失资产（本次失败原因是上面的指纹漂移）："
            )
            lines += [heading, *missing]
        pytest.fail("\n".join(lines))

    if missing:
        pytest.skip(
            "缺少 golden checkpoint（未验收，非通过；"
            "用 --require-golden-assets 强制失败）：\n" + "\n".join(missing)
        )
