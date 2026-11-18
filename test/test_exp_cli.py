"""Exercise the experiment launcher without allocating GPUs or loading models."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

import exp

ROOT = Path(__file__).resolve().parents[1]


def test_import_ignores_host_arguments_and_does_not_build_a_matrix(tmp_path):
    code = """
import sys
sys.argv = ['another-script', '--extra-modes=not-an-exp-mode']
from exp import create_config
import exp
assert 'config_to_run' not in vars(exp)
assert not {'torch', 'transformers', 'pynvml'}.intersection(sys.modules)
assert create_config('large')['eval_mode'] == 'large'
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        check=True,
        timeout=30,
        env={**os.environ, "PYTHONPATH": str(ROOT), "EXP_GAMMA_SCHEDULE": "invalid"},
    )
    assert not list(tmp_path.iterdir())


def test_dry_run_accepts_extra_modes_without_touching_disk(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        exp,
        "run_experiments_parallel",
        lambda *a, **kw: pytest.fail("launched workers"),
    )
    assert (
        exp.main(
            ["--extra-modes=tridecoding", "--only-modes", "tridecoding", "--dry-run"]
        )
        == 0
    )
    assert "本次待跑 27 个" in capsys.readouterr().out
    assert exp.main(["--dry-run"]) == 0
    assert "本次待跑 135 个" in capsys.readouterr().out
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "options", [["--extra-modes", "dsd"], ["--only-modes", "missing"]]
)
def test_cli_rejects_invalid_selection(options, capsys):
    assert exp.main([*options, "--dry-run"]) != 0
    assert options[0] in capsys.readouterr().err


def test_resume_and_merge_keep_history_when_reusing_summary_path(tmp_path, monkeypatch):
    configs = exp.filter_configs_by_mode(exp.build_matrix(), "dsd")
    prior = [
        {"config": c, "exp_name": c["exp_name"], "status": "success"} for c in configs
    ]
    prior[-1]["status"] = "failed"
    source, output = tmp_path / "previous.json", tmp_path / "merged.json"
    exp.save_summary(source, prior)

    def run(pending, *, max_workers, log_dir, summary_file):
        assert max_workers == 2 and log_dir == "exp_logs"
        assert [exp.run_identity(c) for c in pending] == [exp.run_identity(configs[-1])]
        result = [{"config": pending[0], "exp_name": "rerun", "status": "success"}]
        exp.save_summary(summary_file, result)
        return result

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(exp, "run_experiments_parallel", run)
    assert (
        exp.main(
            [
                "--only-modes",
                "dist_spec",
                "--max-workers",
                "2",
                "--resume-from",
                str(source),
                "--summary-file",
                str(source),
                "--merge-from",
                str(source),
                "--merged-output",
                str(output),
            ]
        )
        == 0
    )
    assert [e["exp_name"] for e in exp.load_summary(source)] == ["rerun"]
    merged = exp.load_summary(output)
    assert len(merged) == len(configs)
    assert not exp.filter_configs_for_resume(configs, output)


def test_atomic_summary_write_preserves_last_snapshot_on_failure(tmp_path):
    output = tmp_path / "nested" / "results.json"
    exp.save_summary(output, [{"status": "success"}])
    with pytest.raises(TypeError):
        exp.save_summary(output, [{"bad": object()}])
    assert exp.load_summary(output) == [{"status": "success"}]
    assert list(output.parent.iterdir()) == [output]
