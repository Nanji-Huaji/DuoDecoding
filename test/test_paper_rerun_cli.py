"""Paper launcher contracts; no real GPU/driver calls, datasets, or model loads."""

import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.experiments import paper_rerun as cli

ROOT = Path(__file__).resolve().parents[1]
# nvidia-smi reports MiB, not the VRAM printed on the box.
CARDS = {
    2: ("NVIDIA RTX A6000", 48683),
    4: ("NVIDIA RTX 6000 Ada", 48683),
    7: ("NVIDIA GeForce RTX 3090", 24576),
}
UUIDS = {i: f"GPU-00000000-0000-0000-0000-{i:012x}" for i in CARDS}
EXEC = ["--stage", "cee-plus46", "--execute", "--gpus", "2,4", "--run-id", "pilot.1"]
OFFLINE = dict(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_DATASETS_OFFLINE="1")


def test_default_preview_is_stable_and_lightweight(tmp_path):
    code = """
import subprocess, sys, socket
from contextlib import redirect_stdout
from io import StringIO

def forbidden(*a, **kw):
    raise AssertionError('preview attempted a process, input preflight or network call')
subprocess.Popen = socket.socket.connect = forbidden
from src.experiments import paper_rerun as cli
cli._freeze_inputs = cli._visible_cuda_uuids = forbidden
out = []
for flags in ([], ['--dry-run']):
    with redirect_stdout(StringIO()) as buf:
        assert cli.main(['--stage', 'cee-plus46', *flags]) == 0
    out.append(buf.getvalue())
assert out[0] == out[1] and 'cells=9/9' in out[0]
assert out[0].count('unassigned') == 9
for flags, expected in [
    (['--gpus', '7'], ['7'] * 9),
    (['--gpus', '7,3', '0'], ['7,3,0'] * 9),
    (['--gpus', '7,3', '--gpus', '0'], ['7,3', '0'] * 4 + ['7,3']),
]:
    with redirect_stdout(StringIO()) as buf:
        assert cli.main(['--stage', 'cee-plus46', '--dry-run', *flags]) == 0
    assert [line.split()[2] for line in buf.getvalue().splitlines()[1:]] == expected
assert not {'torch', 'transformers', 'datasets', 'pynvml'}.intersection(sys.modules)
assert 'src.experiments.paper_inputs' not in sys.modules
"""
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=tmp_path,
        check=True,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
        timeout=30,
    )
    assert not list(tmp_path.iterdir())


def _samples():
    return {
        "mt_bench_noeval": {
            "rows": [
                dict(question_id=n, category="math", turns=["a", "b"])
                for n in range(80)
            ]
        },
        "gsm8k": {
            "rows": [
                dict(question_sha256=hashlib.sha256(str(n).encode()).hexdigest())
                for n in range(80)
            ]
        },
        "humaneval": {"rows": [dict(task_id=f"task{n}") for n in range(80)]},
    }


@pytest.fixture
def offline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    for var in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", *OFFLINE):
        monkeypatch.delenv(var, raising=False)

    def forbidden(*a, **kw):
        pytest.fail("unexpected GPU probe / subprocess")

    monkeypatch.setattr(cli.subprocess, "run", forbidden)
    monkeypatch.setattr(cli.subprocess, "Popen", forbidden)
    monkeypatch.setattr(cli, "_visible_cuda_uuids", forbidden)
    freeze = Mock(
        return_value=dict(
            source={"head": "test", "files": {"long": True}},
            samples=_samples(),
            models=["local weights"],
            worker_env=OFFLINE,
        )
    )
    monkeypatch.setattr(cli, "_freeze_inputs", freeze)
    return freeze


def test_preflight_and_selection_are_readonly(offline, tmp_path, capsys):
    ids = [cli._cell_id(c) for c in cli.matrix.build_rerun_matrix("quality")]
    assert cli.main(["--stage", "quality", "--preflight", "--cells", *ids[:2]]) == 0
    assert [cli._cell_id(c) for c in offline.call_args.args[0]] == ids[:2]
    out = capsys.readouterr().out
    assert "cells=2/24" in out and "local weights" in out and '"rows"' not in out
    assert cli.main(["--stage", "quality", "--exclude-cells", ",".join(ids[:2])]) == 0
    assert "cells=22/24" in capsys.readouterr().out
    for selection in (
        ["--cells", ids[0], ids[0]],
        ["--cells", "missing"],
        ["--exclude-cells", *ids],
    ):
        assert cli.main(["--stage", "quality", *selection]) != 0
    assert not list(tmp_path.iterdir())
    assert offline.call_count == 1


def test_cell_identity_ignores_only_placement_and_name():
    c = cli.matrix.build_rerun_matrix("main46")[0]
    renamed = dict(c, CUDA_VISIBLE_DEVICES="17", exp_name="older")
    assert cli._cell_id(renamed) == cli._cell_id(c)
    # Same displayed axes but a different semantic threshold must get a different path.
    other = dict(c, draft_target_threshold=0.345)
    assert cli._cell_id(c).split(".")[-1] != cli._cell_id(other).split(".")[-1]
    with pytest.raises(ValueError, match="collision"):
        cli._select([c, renamed], [cli._cell_id(c)], [])


@pytest.mark.parametrize(
    "args",
    [
        ["--stage", "core", "--execute"],
        EXEC[:-2],
        [*EXEC[:3], *EXEC[-2:]],
        [*EXEC[:4], "2,2", *EXEC[-2:]],
        [*EXEC[:4], "7,2,07", *EXEC[-2:]],
        [*EXEC[:4], "4,2", "--gpus", "7,4", *EXEC[-2:]],
        [*EXEC[:4], "2,,7", *EXEC[-2:]],
        [*EXEC[:4], "GPU-x,4", *EXEC[-2:]],
        [*EXEC[:-1], "../escape"],
        ["--stage", "quality", *EXEC[2:]],
    ],
)
def test_execution_requires_all_authorizations(offline, tmp_path, args):
    assert cli.main(args) != 0
    offline.assert_not_called()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "mask",
    ["0,1", "2,3", "", "-1", "MIG-x", "2,2,4", f"{UUIDS[2]},{UUIDS[2]},{UUIDS[4]}"],
)
def test_inherited_mask_cannot_be_widened(offline, tmp_path, monkeypatch, mask):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    assert cli.main(EXEC) != 0
    offline.assert_not_called()
    assert not list(tmp_path.iterdir())


def _gpu(index, query):
    name, total = CARDS[index]
    values = {
        "index": index,
        "name": name,
        "uuid": UUIDS[index],
        "memory.total": total,
        "memory.used": 8,
        "utilization.gpu": 0,
        "driver_version": "580.0",
    }
    return ", ".join(
        str(values[k]) for k in query.removeprefix("--query-gpu=").split(",")
    )


@pytest.mark.parametrize(
    "problem",
    [
        "missing-name",
        "uuid",
        "duplicate",
        "mem",
        "util",
        "pid",
        "unknown",
        "driver-mask",
    ],
)
def test_hardware_checks_fail_closed(offline, tmp_path, monkeypatch, problem):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,4")
    monkeypatch.setattr(
        cli,
        "_visible_cuda_uuids",
        lambda: {UUIDS[2]} if problem == "driver-mask" else set(UUIDS.values()),
    )

    def smi(i, q):
        if q == "--query-compute-apps=pid":
            return "123" if problem == "pid" else ""
        row = _gpu(i, q)
        if problem == "unknown":
            raise OSError("nvidia-smi unavailable")
        return row.replace(
            {
                "missing-name": CARDS[i][0],
                "uuid": UUIDS[i],
                "duplicate": UUIDS[i],
                "mem": ", 8,",
                "util": ", 0,",
            }.get(problem, "NOT IN ROW"),
            {
                "missing-name": "",
                "uuid": "GPU-nope",
                "duplicate": UUIDS[2],
                "mem": ", 1024,",
                "util": ", 8,",
            }.get(problem, ""),
        )

    monkeypatch.setattr(cli, "_smi", smi)
    assert cli.main(EXEC) != 0
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("problem", ["duplicate", "busy", "outside allocation"])
def test_every_gpu_in_every_group_is_verified(offline, tmp_path, monkeypatch, problem):
    # A bad third device must not reach any worker, even in a second group.
    permitted = [2, 4] if problem == "outside allocation" else [2, 4, 7]
    monkeypatch.setenv("NVIDIA_VISIBLE_DEVICES", ",".join(UUIDS[i] for i in permitted))

    def smi(argv, **kwargs):
        i, query = int(argv[2]), argv[3]
        if query == "--query-compute-apps=pid":
            return SimpleNamespace(
                stdout="4242" if i == 7 and problem == "busy" else ""
            )
        row = _gpu(i, query)
        if i == 7 and problem == "duplicate":
            row = row.replace(UUIDS[7], UUIDS[2])
        return SimpleNamespace(stdout=row)

    monkeypatch.setattr(cli.subprocess, "run", smi)
    assert cli.main([*EXEC[:4], "2,4", "--gpus", "7", *EXEC[-2:]]) == 2
    assert not list(tmp_path.iterdir())


def test_input_error_never_launches_or_reserves_output(offline, tmp_path):
    offline.side_effect = RuntimeError("missing deployed head")
    assert cli.main(EXEC) != 0
    assert cli.main(["--stage", "quality", "--preflight"]) != 0
    assert not list(tmp_path.iterdir())


def _fake_rows(task):
    if task == "mt_bench_noeval":
        return [
            dict(question_id=n, category="math", choices=[{"turns": ["x", "y"]}])
            for n in range(80)
        ]
    if task == "gsm8k":
        return [dict(question=str(n)) for n in range(80)]
    return [dict(task_id=f"task{n}") for n in range(80)]


@pytest.mark.parametrize(
    "groups, expected, fault",
    [
        (["7"], [[7]] * 4, None),  # even a 24 GiB card may attempt a cell
        (["7,2,4"], [[7, 2, 4]] * 4, None),  # one ordered mask seen by the loader
        (["7,4", "2"], [[7, 4], [2], [7, 4], [2]], None),
        (["7", "2", "4"], [[7], [2], [4], [7]], None),
        (["7", "2", "4"], [[7]], None),  # unused groups do not start workers
        *(
            (["2", "4"], [[2], [4]] * 2, fault)
            for fault in (
                None,
                "worker failure",
                "missing",
                "wrong order",
                "missing turn",
            )
        ),
    ],
)
def test_execution_uses_only_the_assigned_cards(
    offline, tmp_path, monkeypatch, groups, expected, fault
):
    # Only replace the external processes. Exercise real worker leases, masks,
    # log names, per-cell clocks, durable summaries and single-use reservation.
    def smi(argv, *, capture_output, check, timeout, text):
        assert argv[:2] == ["nvidia-smi", "-i"] and check and timeout > 0
        if "--query-compute-apps=pid" in argv:
            return SimpleNamespace(stdout="")
        return SimpleNamespace(stdout=_gpu(int(argv[2]), argv[3]))

    monkeypatch.setattr(cli.subprocess, "run", smi)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", ",".join(UUIDS.values()))
    expected_masks = [",".join(UUIDS[i] for i in group) for group in expected]
    ready = threading.Barrier(len(set(expected_masks)), timeout=5)
    launches = []

    class Pipe:
        def __init__(self, argv, *, env, **kwargs):
            assert argv == [sys.executable, "-m", "eval.worker"]
            self.gpu = env["CUDA_VISIBLE_DEVICES"]
            assert self.gpu in expected_masks and all(
                env[k] == v for k, v in OFFLINE.items()
            )
            self.first = True
            self.stdin = self.stdout = self
            launches.append(self.gpu)

        def write(self, line):
            if self.first:
                self.first = False
                ready.wait()  # independent groups can actually work at the same time
            job = json.loads(line)
            c, path = job["config"], Path(job["log_file"])
            assert self.gpu == c["CUDA_VISIBLE_DEVICES"]
            path.write_text("fake worker log")
            task = cli._task(c)
            rows = _fake_rows(task)
            if fault == "missing":
                rows.pop()
            elif fault == "wrong order":
                rows[0] = rows[1]
            elif fault == "missing turn" and task == "mt_bench_noeval":
                rows[0]["choices"][0]["turns"].pop()
            sample_file = (
                Path(c["exp_name"])
                / f"{c['eval_mode']}_{task.removesuffix('_noeval')}.jsonl"
            )
            sample_file.write_text("\n".join(map(json.dumps, rows)) + "\n")
            status = "failed" if fault == "worker failure" else "success"
            self.response = json.dumps(
                dict(
                    exp_name=c["exp_name"], status=status, result={}, log_file=str(path)
                )
            )

        def readline(self):
            return self.response

        def poll(self):
            return None

        def close(self):
            pass

        def flush(self):
            pass

        def wait(self, timeout):
            return 0

    monkeypatch.setattr(cli.subprocess, "Popen", Pipe)
    ticks = threading.local()

    def clock():
        ticks.value = getattr(ticks, "value", 0.0) + 1.25
        return ticks.value

    monkeypatch.setattr(cli, "time", SimpleNamespace(perf_counter=clock))
    source = cli.matrix.build_rerun_matrix("cee-plus46")
    picked = [source[i] for i in (0, 3, 6, 7)[: len(expected)]]
    ids = [cli._cell_id(c) for c in picked]
    run_args = [
        *EXEC[:3],
        *(arg for group in groups for arg in ("--gpus", group)),
        *EXEC[-2:],
    ]
    assert cli.main([*run_args, "--cells", ",".join(ids)]) == int(fault is not None)
    assert sorted(launches) == sorted(set(expected_masks))
    assert not any(k in os.environ for k in OFFLINE)
    assert [cli._cell_id(c) for c in offline.call_args.args[0]] == ids
    dest = tmp_path / "experiments" / "paper_reruns" / "pilot.1"
    manifest = json.loads((dest / "manifest.json").read_text())
    results = json.loads((dest / "summary.json").read_text())
    assert manifest["selection"]["include"] == ids and manifest["stage_size"] == 9
    assert manifest["inputs"] == offline.return_value
    assert manifest["gpu_groups"] == [[int(i) for i in g.split(",")] for g in groups]
    assert [j["gpu_indices"] for j in manifest["cells"]] == expected
    assert [
        j["config"]["CUDA_VISIBLE_DEVICES"] for j in manifest["cells"]
    ] == expected_masks
    for job, group in zip(manifest["cells"], expected, strict=True):
        assert job.get("gpu_index") == (group[0] if len(group) == 1 else None)
    assert [(c["name"], c["memory_total_mib"]) for c in manifest["gpus"]] == [
        CARDS[int(i)] for g in groups for i in g.split(",")
    ]
    assert {r["cell_id"] for r in results} == set(ids)
    for entry in results:
        config = entry["config"]
        assert cli._cell_id(config) == entry["cell_id"]
        assert entry["elapsed_seconds"] == 1.25
        assert Path(config["exp_name"]).parent == dest / "cells"
        assert Path(entry["log_file"]).parent == dest / "logs"
        if entry["status"] == "success":
            assert entry["sample_rows"] == 80
    assert bool([r for r in results if r["status"] != "success"]) == bool(fault)
    # No reuse, even for another stage/subset with the same run-id.
    assert cli.main(["--stage", "static46", *EXEC[2:]]) != 0
    assert json.loads((dest / "summary.json").read_text()) == results
    assert len(launches) == len(set(expected_masks)) and offline.call_count == 1
