"""一键 checkpoint CLI 的派发测试（临时 manifest，不访问网络/生产资产）。

status/pull 都必须在测试自带的 root + manifest 上工作：干净 clone 没有生产
checkpoint 时这些测试仍能验证真实的成功/缺失返回码与输入透传。
"""

import hashlib
import json
from pathlib import Path

from tools import checkpoints

REMOTE = "checkpoints/rl/main/example/latest.pth"


def _entry(remote_path: str, payload: bytes) -> dict:
    return {
        "remote_path": remote_path,
        "status": "production",
        "role": "rl_main",
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _manifest(root: Path, entries: dict[str, dict]) -> Path:
    path = root / "checkpoints" / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"hub": {"repo_id": "example/hub"}, "entries": entries}),
        encoding="utf-8",
    )
    return path


def _asset(root: Path, remote_path: str, payload: bytes) -> None:
    path = root / remote_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _isolate_status(tmp_path, monkeypatch, entries: dict[str, dict]) -> None:
    manifest = _manifest(tmp_path, entries)
    monkeypatch.setattr(checkpoints.cm, "MANIFEST", manifest)
    monkeypatch.setattr(checkpoints, "ROOT", tmp_path)


def test_status_reports_complete_production_from_tmp_manifest(
    tmp_path, monkeypatch, capsys
):
    payload = b"weights-v1"
    _asset(tmp_path, REMOTE, payload)
    _isolate_status(tmp_path, monkeypatch, {REMOTE: _entry(REMOTE, payload)})

    assert checkpoints.main(["status"]) == 0

    out = capsys.readouterr().out
    assert "entries: 1" in out
    assert "production 1" in out
    assert "production 文件齐全" in out


def test_status_missing_production_returns_failure(tmp_path, monkeypatch, capsys):
    _isolate_status(tmp_path, monkeypatch, {REMOTE: _entry(REMOTE, b"weights-v1")})

    assert checkpoints.main(["status"]) == 1

    out = capsys.readouterr().out
    assert "缺失 production 文件 1 个" in out
    assert REMOTE in out


def _isolate_pull(tmp_path, monkeypatch, entries: dict[str, dict]) -> None:
    import fetch_checkpoints

    manifest = _manifest(tmp_path, entries)
    monkeypatch.setattr(fetch_checkpoints, "ROOT", tmp_path)
    monkeypatch.setattr(fetch_checkpoints, "MANIFEST", manifest)
    # 生产 acc-head 注册表可能存在于本机；测试只服务 rl 走查，明确隔离。
    monkeypatch.setattr(fetch_checkpoints, "ACC_REGISTRY", tmp_path / "absent.json")


def test_pull_dry_run_verifies_supplied_manifest_without_downloading(
    tmp_path, monkeypatch, capsys
):
    payload = b"weights-v1"
    _asset(tmp_path, REMOTE, payload)
    _isolate_pull(tmp_path, monkeypatch, {REMOTE: _entry(REMOTE, payload)})

    assert checkpoints.main(["pull", "--dry-run"]) == 0

    out = capsys.readouterr().out
    assert "1 entries" in out
    assert "would download" not in out  # 文件已就位，不该走到下载分支
    assert "OK" in out


def test_pull_dry_run_reports_missing_entry_from_supplied_manifest(
    tmp_path, monkeypatch, capsys
):
    _isolate_pull(tmp_path, monkeypatch, {REMOTE: _entry(REMOTE, b"weights-v1")})

    assert checkpoints.main(["pull", "--dry-run"]) == 1

    out = capsys.readouterr().out
    assert "1 entries" in out
    assert f"would download {REMOTE}" in out
    assert f"MISSING {REMOTE}" in out


def test_unknown_command_rejected(capsys):
    assert checkpoints.main(["nope"]) == 2
    assert "未知子命令" in capsys.readouterr().out


def test_push_dispatches_to_uploader(monkeypatch):
    calls = []

    def fake_main(argv):
        calls.append(argv)
        return 0

    import upload_checkpoints

    monkeypatch.setattr(upload_checkpoints, "main", fake_main)
    rc = checkpoints.main(
        ["push", "--repo-id", "example/repo", "--dry-run"]
    )
    assert rc == 0
    assert "--repo-id" in calls[0]
    assert "example/repo" in calls[0]
