"""ModelScope 下载脚本的离线行为（不触网）。"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import download_models_modelscope as dm  # noqa: E402


def test_list_offline(capsys):
    assert dm.main(["--list"]) == 0
    out = capsys.readouterr().out
    assert "qwen15-7b" in out
    assert "modelscope_id" in out


def test_dry_run_group_offline(capsys):
    assert dm.main(["--only", "qwen15", "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "qwen15-0.5b" in out


def test_verify_only_reports_missing_for_unknown_config(tmp_path):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        '{"groups": {}, "models": {"m": {"modelscope_id": "x/y", '
        '"local_dir": "does/not/exist", "verified": true}}}',
        encoding="utf-8",
    )
    assert dm.main(["--verify-only", "--config", str(cfg)]) == 1


def test_unknown_alias_rejected():
    with pytest.raises(SystemExit):
        dm.main(["--only", "no-such-group", "--list"])
