"""模式归档守卫。

2026-10-19 起 `ceesd_without_arp` 被标为 legacy：它走的是旧验证/回滚路径，
不是 CEE-SD 的有效消融。这里锁住三件事：
- legacy 模式 CLI 启动必须告警（可解析，用于复现历史 run）；
- 核心方法不得误报 legacy；
- docs/modes_inventory.md 的 L0 清单必须覆盖注册表里的每个名字（防再次过期）。
"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

import src.utils as utils


def _parse(monkeypatch, mode: str):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prog",
            "--eval_mode", mode,
            "--draft_model", "llama-68m",
            "--target_model", "llama-2-13b",
        ],
    )
    with patch("src.utils.model_zoo"):
        return utils.parse_arguments()


@pytest.mark.parametrize("mode", sorted(utils.LEGACY_EVAL_MODES))
def test_archived_mode_prints_legacy_warning(monkeypatch, capsys, mode):
    _parse(monkeypatch, mode)
    out = capsys.readouterr().out
    assert "[legacy]" in out and mode in out


@pytest.mark.parametrize("mode", ["adaptive_tridecoding", "dist_spec"])
def test_supported_mode_has_no_legacy_warning(monkeypatch, capsys, mode):
    _parse(monkeypatch, mode)
    assert "[legacy]" not in capsys.readouterr().out


def test_inventory_covers_every_registered_name():
    import src.baselines  # noqa: F401  触发注册
    import src.engine  # noqa: F401
    from src.register import Register

    doc = (
        Path(__file__).resolve().parents[1] / "docs" / "modes_inventory.md"
    ).read_text(encoding="utf-8")
    missing = [
        name for name in Register._DECODING_REGISTRY if f"`{name}`" not in doc
    ]
    assert not missing, f"modes_inventory.md 漏了注册名: {sorted(missing)}"
