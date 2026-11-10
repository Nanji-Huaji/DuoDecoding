"""CPU gate orchestration contracts; never recursively run pytest here."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools import run_cpu_gate as gate


def test_commands_always_run_complete_test_tree_and_record_junit(tmp_path):
    for mode in ("default", "importlib"):
        report = tmp_path / f"{mode}.xml"
        command = gate.pytest_command("python-for-test", mode, report)
        assert command[:6] == ["python-for-test", "-B", "-m", "pytest", "-q", "-p"]
        assert "no:cacheprovider" in command
        assert command[command.index("-o") + 1] == "addopts="
        assert command[-2:] == ["test", f"--junitxml={report}"]
        assert ("--import-mode=importlib" in command) == (mode == "importlib")
    with pytest.raises(ValueError, match="Unsupported"):
        gate.pytest_command("python", "silent-subset", tmp_path / "bad.xml")


def test_source_snapshot_includes_new_and_deleted_files_not_gitlink_dirs(
    tmp_path, monkeypatch,
):
    (tmp_path / "tools").mkdir()
    source = tmp_path / "tools" / "new.py"
    source.write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "src" / "vendor").mkdir(parents=True)
    root_entry = tmp_path / "exp.py"
    root_entry.write_text("value = 1\n", encoding="utf-8")

    def fake_git(root, *args):
        assert root == tmp_path
        if args[0] == "ls-files":
            assert "--others" in args and "--exclude-standard" in args
            assert "*.py" in args  # Includes real test imports such as root exp.py.
            return "tools/new.py\0src/deleted.py\0src/vendor\0exp.py\0"
        if args[0] == "rev-parse":
            return "sha\n"
        if args[0] == "status":
            return "?? tools/new.py\n D src/deleted.py\n"
        if args[0] == "submodule":
            return " vendor-revision src/vendor\n"
        raise AssertionError(args)

    monkeypatch.setattr(gate, "_git", fake_git)
    before = gate.source_snapshot(tmp_path)
    assert set(before["files"]) == {"tools/new.py", "src/deleted.py", "exp.py"}
    root_entry.write_text("value = 2\n", encoding="utf-8")
    after_root_change = gate.source_snapshot(tmp_path)
    assert after_root_change != before
    assert before["files"]["src/deleted.py"] == "deleted"
    assert before["git_sha"] == "sha"
    source.write_text("x = 2\n", encoding="utf-8")
    assert gate.source_snapshot(tmp_path) != before


def _setup(monkeypatch, *, snapshots=None, returncodes=(0, 0)):
    snapshots = iter(snapshots or [{"sha": "same"}, {"sha": "same"}])
    monkeypatch.setattr(gate, "source_snapshot", lambda root: next(snapshots))
    monkeypatch.setattr(gate, "environment_metadata", lambda: {"test": True})
    calls = []
    codes = iter(returncodes)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        kwargs["stdout"].write("test output\n")
        return SimpleNamespace(returncode=next(codes))

    monkeypatch.setattr(gate.subprocess, "run", run)
    return calls


def _manifest(output: Path):
    return json.loads((output / "manifest.json").read_text(encoding="utf-8"))


def test_gate_runs_both_modes_offline_with_gpu_hidden(tmp_path, monkeypatch):
    calls = _setup(monkeypatch)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    monkeypatch.setenv("PYTEST_ADDOPTS", "-k no_tests")
    monkeypatch.setenv("PYTEST_PLUGINS", "ambient_plugin")
    monkeypatch.setenv("PRESERVED_TEST_VALUE", "yes")
    output = tmp_path / "evidence"
    assert gate.run_gate(tmp_path, output) == 0
    assert len(calls) == 2
    for command, kwargs in calls:
        assert kwargs["cwd"] == tmp_path.resolve()
        assert kwargs["check"] is False
        assert command[0] == gate.sys.executable
        assert kwargs["env"]["PRESERVED_TEST_VALUE"] == "yes"
        assert "PYTEST_ADDOPTS" not in kwargs["env"]
        assert "PYTEST_PLUGINS" not in kwargs["env"]
        for key, value in gate.CPU_ENV.items():
            assert kwargs["env"][key] == value
    manifest = _manifest(output)
    assert manifest["status"] == "passed"
    assert manifest["source_unchanged"] is True
    assert [r["mode"] for r in manifest["results"]] == ["default", "importlib"]
    assert (output / "default.log").read_text() == "test output\n"
    assert (output / "importlib.log").read_text() == "test output\n"


@pytest.mark.parametrize("codes", [(1, 0), (0, 2), (5, 5)])
def test_any_pytest_failure_or_no_collection_fails_gate(tmp_path, monkeypatch, codes):
    calls = _setup(monkeypatch, returncodes=codes)
    output = tmp_path / "evidence"
    assert gate.run_gate(tmp_path, output) == 1
    assert len(calls) == 2
    assert _manifest(output)["status"] == "failed"


def test_source_change_during_tests_invalidates_green_gate(tmp_path, monkeypatch):
    _setup(monkeypatch, snapshots=[{"hash": "old"}, {"hash": "new"}])
    output = tmp_path / "evidence"
    assert gate.run_gate(tmp_path, output) == 1
    manifest = _manifest(output)
    assert manifest["status"] == "failed"
    assert manifest["source_unchanged"] is False
    assert all(item["returncode"] == 0 for item in manifest["results"])


def test_existing_evidence_directory_is_never_overwritten(tmp_path, monkeypatch):
    calls = _setup(monkeypatch)
    output = tmp_path / "evidence"
    output.mkdir()
    marker = output / "old.txt"
    marker.write_text("keep me")
    with pytest.raises(FileExistsError):
        gate.run_gate(tmp_path, output)
    assert marker.read_text() == "keep me"
    assert not calls


def test_real_pytest_cannot_hide_failure_via_config_or_explicit_plugin(
    tmp_path, monkeypatch,
):
    # A real subprocess, unlike orchestration mocks: collect-only would return
    # zero without executing the failing test, and explicit plugins bypass
    # PYTEST_DISABLE_PLUGIN_AUTOLOAD unless separately removed.
    (tmp_path / "test").mkdir()
    (tmp_path / "test" / "test_failure.py").write_text(
        "def test_must_run():\n    assert False\n", encoding="utf-8",
    )
    (tmp_path / "pytest.ini").write_text(
        "[pytest]\naddopts = --collect-only\n", encoding="utf-8",
    )
    monkeypatch.setenv("PYTEST_PLUGINS", "plugin_that_must_never_be_loaded")
    monkeypatch.setenv("PYTEST_ADDOPTS", "--collect-only")
    monkeypatch.setattr(gate, "source_snapshot", lambda root: {"snapshot": "fixed"})
    output = tmp_path / "evidence"
    assert gate.run_gate(tmp_path, output) == 1
    manifest = _manifest(output)
    assert [r["returncode"] for r in manifest["results"]] == [1, 1]
    for mode in ("default", "importlib"):
        assert "1 failed" in (output / f"{mode}.log").read_text(encoding="utf-8")
        assert (output / f"{mode}.xml").is_file()


def test_execution_error_records_incomplete_manifest_and_raises(tmp_path, monkeypatch):
    _setup(monkeypatch)

    def fail(*args, **kwargs):
        raise OSError("interpreter unavailable")

    monkeypatch.setattr(gate.subprocess, "run", fail)
    output = tmp_path / "evidence"
    with pytest.raises(OSError, match="interpreter unavailable"):
        gate.run_gate(tmp_path, output)
    manifest = _manifest(output)
    assert manifest["status"] == "error"
    assert manifest["results"] == []
    assert "interpreter unavailable" in manifest["error"]
