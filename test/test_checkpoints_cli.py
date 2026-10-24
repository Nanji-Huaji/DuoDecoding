"""一键 checkpoint CLI 的派发测试（不访问网络）。"""

from tools import checkpoints


def test_status_reports_production(capsys):
    assert checkpoints.main(["status"]) == 0
    out = capsys.readouterr().out
    assert "entries:" in out and "production" in out


def test_pull_dry_run_works_offline():
    assert checkpoints.main(["pull", "--dry-run"]) == 0


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
