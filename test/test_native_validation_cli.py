"""Public native CLI/result contract only; no CUDA/query/model execution."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "tools" / "run_native_validation.py"


UUIDS = [
    "GPU-00000000-0000-0000-0000-000000000001",
    "GPU-00000000-0000-0000-0000-000000000002",
]


class FakeExternal:
    """Injected OS/source adapter; no GPU or process syscall is reachable."""

    def snapshot(self):
        return {
            "git_sha": "a" * 40,
            "source_status": "",
            "submodules": "",
            "files": {
                "src/cee_proposal.py": "b" * 64,
                "tools/run_native_validation.py": "c" * 64,
            },
        }

    def environment(self):
        from tools.run_cpu_gate import CPU_ENV

        return {
            "python": "test",
            "executable": sys.executable,
            "platform": "test",
            "packages": {"torch": "test", "transformers": "test"},
            "overrides": CPU_ENV,
        }

    def fsync(self, fd):
        os.fsync(fd)

    def tools(self):
        return {
            "python": {"path": sys.executable, "sha256": "d" * 64},
            "nvidia_smi": None,
        }

    def query(self, uuids):
        raise AssertionError("read-only validation must not query GPUs")


def inputs(tmp_path, external):
    from tools.run_cpu_gate import pytest_command

    cfg = {
        "schema": "duodecoding.native-config.v1",
        "profile": "tiny",
        "gpu_uuids": UUIDS,
        "max_seconds": 120,
        "continue_on_numerical_mismatch": True,
        "local": None,
    }
    config = tmp_path / "config.json"
    config.write_text(json.dumps(cfg))
    results = []
    for mode in ("default", "importlib"):
        junit = tmp_path / f"{mode}.xml"
        junit.write_text(
            '<testsuites><testsuite tests="1" failures="0" errors="0" skipped="0">'
            '<testcase name="public"/></testsuite></testsuites>'
        )
        (tmp_path / f"{mode}.log").write_text("1 passed\n")
        results.append(
            {
                "mode": mode,
                "command": pytest_command(sys.executable, mode, junit),
                "returncode": 0,
                "seconds": 1.0,
                "log": f"{mode}.log",
                "junit": f"{mode}.xml",
            }
        )
    manifest = {
        "schema": "duodecoding.cpu-gate.v1",
        "status": "passed",
        "source_before": external.snapshot(),
        "source_after": external.snapshot(),
        "source_unchanged": True,
        "environment": external.environment(),
        "golden_assets": {"required": False, "flag": "--require-golden-assets"},
        "results": results,
    }
    cpu = tmp_path / "manifest.json"
    cpu.write_text(json.dumps(manifest))
    return config, cpu


def test_read_only_plan_has_exact_full_matrix_and_bound_cpu_evidence(tmp_path, capsys):
    from tools.run_native_validation import main

    external = FakeExternal()
    cfg, cpu = inputs(tmp_path, external)
    assert (
        main(
            ["--validate-only", "--config", str(cfg), "--cpu-manifest", str(cpu)],
            adapters=external,
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["execution"] == "not_started" and not report["acceptance_eligible"]
    planned = report["plan"]
    assert len(planned) == len({row["id"] for row in planned}) == 336
    assert {row["dtype"] for row in planned} == {"float32", "bfloat16"}
    assert {row["topology"] for row in planned} == {"single", "ld-t", "l-dt"}
    assert {row["scenario"] for row in planned} == {
        "budget0",
        "budget1",
        "budget2",
        "budget3",
        "budget12",
        "new-sample",
        "first-token-eos",
    }
    assert report["binding"]["source"] == external.snapshot()
    assert report["binding"]["cpu"]["status"] == "passed"
    assert report["configuration"]["gpu_uuids"] == UUIDS
    assert all("prefix" in row["required_checks"] for row in planned)
    assert report["binding"]["tools"] == external.tools()


@pytest.mark.parametrize("setting", ["omitted", False, True, 0, 1, None, "true", []])
def test_xorg_policy_is_explicit_strict_boolean_and_defaults_off(
    tmp_path, capsys, setting
):
    from tools.run_native_validation import main

    external = FakeExternal()
    cfg, cpu = inputs(tmp_path, external)
    if setting != "omitted":
        value = json.loads(cfg.read_text())
        value["ignore_xorg"] = setting
        cfg.write_text(json.dumps(value))
    args = ["--validate-only", "--config", str(cfg), "--cpu-manifest", str(cpu)]
    if setting == "omitted" or type(setting) is bool:
        assert main(args, adapters=external) == 0
        report = json.loads(capsys.readouterr().out)
        assert report["configuration"]["ignore_xorg"] is (setting is True)
        assert len(report["plan"]) == 336 and not report["acceptance_eligible"]
    else:
        with pytest.raises(SystemExit) as error:
            main(args, adapters=external)
        assert error.value.code == 2 and "ignore_xorg" in capsys.readouterr().err


@pytest.fixture
def xorg_external():
    from contextlib import nullcontext
    from types import SimpleNamespace
    from xml.etree import ElementTree as ET

    from tools.native_gpu_guard import NativeSystem

    class XorgSystem(FakeExternal):
        query = NativeSystem.query

        def __init__(self):
            self.now = 0
            self.memory_mib = 10
            self.utilization = 0
            self.launched = self.cleaned = False
            self.child = None
            self.processes = [
                [
                    {
                        "pid": 2834,
                        "type": "G",
                        "process_name": "/usr/lib/xorg/Xorg",
                        "used_memory": "4 MiB",
                    }
                ]
                for _ in UUIDS
            ]

        def lease(self, ids):
            return nullcontext()

        def clock(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

        def nvidia_smi(self, ids, options):
            assert ids == UUIDS
            if options == ["-q", "-x"]:
                root = ET.Element("nvidia_smi_log")
                for identity, records in zip(ids, self.processes, strict=True):
                    gpu = ET.SubElement(root, "gpu")
                    ET.SubElement(gpu, "uuid").text = identity
                    processes = ET.SubElement(gpu, "processes")
                    for record in records:
                        process = ET.SubElement(processes, "process_info")
                        for key, value in record.items():
                            if value is not None:
                                ET.SubElement(process, key).text = str(value)
                return ET.tostring(root, encoding="unicode")
            assert options == [
                "--query-gpu=index,uuid,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ]
            return "\n".join(
                f"{index + 2},{identity},{self.memory_mib},"
                f"{self.utilization},test-driver"
                for index, identity in enumerate(ids)
            )

        def launch(self, output, env):
            self.launched = True
            self.child = SimpleNamespace(pid=424242, poll=lambda: 0)
            return self.child

        def owns(self, pid, child):
            return pid == child.pid

        def cleanup(self, child):
            self.cleaned = True
            return {"session": child.pid, "remaining": []}

    return XorgSystem()


@pytest.mark.parametrize("setting", ["omitted", False, True])
def test_xorg_exemption_is_opt_in_and_keeps_admission_monitor_evidence(
    tmp_path, xorg_external, setting
):
    from tools.run_native_validation import main

    cfg, cpu = inputs(tmp_path, xorg_external)
    if setting != "omitted":
        config = json.loads(cfg.read_text())
        config["ignore_xorg"] = setting
        cfg.write_text(json.dumps(config))
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=xorg_external,
    )
    report = json.loads((output / "result.json").read_text())
    assert report["acceptance_eligible"] is False
    enabled = setting is True
    assert xorg_external.launched is enabled
    observations = [event for event in report["guard_events"] if "gpus" in event]
    assert len(observations) == (4 if enabled else 1)
    for event in observations:
        assert event["ignore_xorg"] is enabled
        assert event["effective_pids"] == {
            identity: [] if enabled else [2834] for identity in UUIDS
        }
        assert len(event["ignored_xorg"]) == (2 if enabled else 0)
        for identity in UUIDS:
            row = event["gpus"][identity]
            assert row["pids"] == [2834] and row["memory_mib"] == 10
            process = row["processes"][0]
            assert {
                key: process[key] for key in xorg_external.processes[0][0]
            } == xorg_external.processes[0][0]
            assert (
                "<process_name>/usr/lib/xorg/Xorg</process_name>" in process["raw_xml"]
            )
    if enabled:
        assert status == 1 and xorg_external.cleaned  # No simulated worker proof.
        assert report["worker_returncode"] == 0 and report["execution"] == "incomplete"
        assert [event["event"] for event in observations] == [
            "idle-observation-1",
            "idle-observation-2",
            "immediate-prelaunch-check",
            "monitor",
        ]
        assert observations[1]["time"] - observations[0]["time"] >= 30
        assert observations[-1]["foreign"] == []
        assert report["binding_before"] == report["binding_after"]
    else:
        assert status == 3 and report["execution"] == "not_started"
        assert report["outcome_category"] == "resource-refused"


@pytest.mark.parametrize(
    "inconsistency", ["missing-duplicate", "unlisted-compute", "omitted-details"]
)
def test_xorg_requires_complete_consistent_details_for_every_pid_observation(
    tmp_path, xorg_external, inconsistency
):
    from tools.run_native_validation import main

    native_query = xorg_external.query

    def incomplete_query(ids):
        rows = native_query(ids)
        row = rows[UUIDS[1]]
        if inconsistency == "missing-duplicate":
            row["pids"].append(2834)
        elif inconsistency == "unlisted-compute":
            row["pids"] = []
            row["processes"][0]["type"] = "C"
        else:
            row.pop("processes")
        return rows

    xorg_external.query = incomplete_query
    cfg, cpu = inputs(tmp_path, xorg_external)
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=True)))
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=xorg_external,
    )
    report = json.loads((output / "result.json").read_text())
    assert status == 3 and not xorg_external.launched
    event = report["guard_events"][0]
    assert event["ignored_xorg"] == []
    assert event["effective_pids"][UUIDS[0]] == [2834]
    assert event["gpus"][UUIDS[0]]["pids"] == [2834]


@pytest.mark.parametrize(
    "field,value",
    [
        ("pid", "9999"),
        ("type", "C"),
        ("process_name", "/tmp/Xorg"),
        ("used_memory", "N/A"),
    ],
)
def test_xorg_rejects_ambiguous_native_process_fields(
    tmp_path, xorg_external, field, value
):
    from tools.run_native_validation import main

    nvidia_output = xorg_external.nvidia_smi

    def ambiguous_output(ids, options):
        text = nvidia_output(ids, options)
        if options == ["-q", "-x"]:
            text = text.replace(
                f"</{field}>", f"</{field}><{field}>{value}</{field}>", 1
            )
        return text

    xorg_external.nvidia_smi = ambiguous_output
    cfg, cpu = inputs(tmp_path, xorg_external)
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=True)))
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=xorg_external,
        )
        == 3
    )
    report = json.loads((output / "result.json").read_text())
    assert not xorg_external.launched and not report["acceptance_eligible"]
    assert report["outcome_category"] == "resource-refused"


@pytest.mark.parametrize("observation", [1, 2, 3, 4])
@pytest.mark.parametrize(
    "field,value",
    [
        ("type", "C"),
        ("type", "C+G"),
        ("type", None),
        ("process_name", "Xorg"),
        ("process_name", "/tmp/Xorg"),
        ("process_name", "/usr/lib/xorg/../xorg/Xorg"),
        ("process_name", None),
        ("used_memory", "N/A"),
        ("used_memory", None),
        ("pid", 0),
        ("pid", -1),
        ("pid", "2834.5"),
        ("same_pid_extra", "C"),
        ("same_pid_extra", None),
    ],
)
def test_xorg_never_exempts_unsafe_evidence_at_any_guard_stage(
    tmp_path, xorg_external, observation, field, value
):
    from tools.run_native_validation import main

    query = xorg_external.query
    calls = 0

    def changing_query(ids):
        nonlocal calls
        calls += 1
        if calls == observation:
            if field == "same_pid_extra":
                xorg_external.processes[1].append(
                    dict(xorg_external.processes[1][0], type=value)
                )
            else:
                for processes in xorg_external.processes:
                    processes[0][field] = value
        return query(ids)

    xorg_external.query = changing_query
    cfg, cpu = inputs(tmp_path, xorg_external)
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=True)))
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=xorg_external,
    )
    report = json.loads((output / "result.json").read_text())
    assert status == (1 if observation == 4 else 3)
    assert xorg_external.launched is (observation == 4)
    assert xorg_external.cleaned is (observation == 4)
    assert report["acceptance_eligible"] is False
    if field == "pid" and value == "2834.5":
        assert "invalid literal" in report["error"]
    else:
        event = [event for event in report["guard_events"] if "gpus" in event][-1]
        assert event["ignored_xorg"] == []
        assert event["effective_pids"] == {
            identity: row["pids"] for identity, row in event["gpus"].items()
        }
        if observation == 4:
            assert event["foreign"]
        assert all(row["processes"] for row in event["gpus"].values())


@pytest.mark.parametrize("observation", [1, 2, 3])
@pytest.mark.parametrize("memory,utilization", [(256, 0), (10, 1), (-1, 0)])
def test_xorg_does_not_relax_idle_resource_thresholds(
    tmp_path, xorg_external, observation, memory, utilization
):
    from tools.run_native_validation import main

    query = xorg_external.query
    calls = 0

    def busy_query(ids):
        nonlocal calls
        calls += 1
        if calls == observation:
            xorg_external.memory_mib, xorg_external.utilization = memory, utilization
        return query(ids)

    xorg_external.query = busy_query
    cfg, cpu = inputs(tmp_path, xorg_external)
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=True)))
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=xorg_external,
        )
        == 3
    )
    report = json.loads((output / "result.json").read_text())
    assert (
        not xorg_external.launched and report["outcome_category"] == "resource-refused"
    )
    event = report["guard_events"][-1]
    assert len(event["ignored_xorg"]) == 2
    assert all(
        row["memory_mib"] == memory and row["utilization"] == utilization
        for row in event["gpus"].values()
    )


def test_xorg_monitor_allows_only_owned_compute_alongside_exempt_graphics(
    tmp_path, xorg_external
):
    from tools.run_native_validation import main

    query = xorg_external.query

    def own_compute_query(ids):
        if xorg_external.launched:
            xorg_external.memory_mib, xorg_external.utilization = 4096, 90
            for processes in xorg_external.processes:
                processes.append(
                    {
                        "pid": xorg_external.child.pid,
                        "type": "C",
                        "process_name": sys.executable,
                        "used_memory": "4000 MiB",
                    }
                )
        return query(ids)

    xorg_external.query = own_compute_query
    cfg, cpu = inputs(tmp_path, xorg_external)
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=True)))
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=xorg_external,
        )
        == 1
    )
    report = json.loads((output / "result.json").read_text())
    event = [event for event in report["guard_events"] if event["event"] == "monitor"][
        0
    ]
    assert event["foreign"] == [] and report["worker_returncode"] == 0
    assert event["effective_pids"] == dict.fromkeys(UUIDS, [xorg_external.child.pid])
    assert len(event["ignored_xorg"]) == 2
    assert all(row["process"]["pid"] == 2834 for row in event["ignored_xorg"])
    assert xorg_external.cleaned and not report["acceptance_eligible"]


@pytest.mark.parametrize(
    "fault", ["short-idle", "change-before-launch", "change-after-launch"]
)
def test_xorg_policy_keeps_idle_duration_and_binding_guards(
    tmp_path, xorg_external, fault
):
    from tools.run_native_validation import main

    cfg, cpu = inputs(tmp_path, xorg_external)
    cfg.write_text(json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=True)))
    query = xorg_external.query
    calls = 0

    def changed_configuration_query(ids):
        nonlocal calls
        calls += 1
        rows = query(ids)
        if (fault == "change-before-launch" and calls == 2) or (
            fault == "change-after-launch" and calls == 4
        ):
            cfg.write_text(
                json.dumps(dict(json.loads(cfg.read_text()), ignore_xorg=False))
            )
        return rows

    xorg_external.query = changed_configuration_query
    if fault == "short-idle":

        def insufficient_sleep(seconds):
            xorg_external.now += seconds - 1

        xorg_external.sleep = insufficient_sleep
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=xorg_external,
    )
    report = json.loads((output / "result.json").read_text())
    assert status == (3 if fault == "short-idle" else 2)
    assert xorg_external.launched is (fault == "change-after-launch")
    assert xorg_external.cleaned is (fault == "change-after-launch")
    assert report["configuration"]["ignore_xorg"] is True
    assert not report["acceptance_eligible"]
    if fault == "short-idle":
        assert "less than 30 seconds" in report["error"]
    else:
        assert report["binding_status"] == "failed"
        assert report["outcome_category"] == "invalid-contract"


def test_local_profile_requires_hashed_explicit_assets_and_fixed_bf16_matrix(
    tmp_path, capsys
):
    from tools.run_native_validation import main

    external = FakeExternal()
    cfg, cpu = inputs(tmp_path, external)
    import hashlib

    models = {}
    for role in ("little", "draft", "target"):
        root = tmp_path / role
        root.mkdir()
        (root / "config.json").write_bytes(b"{}")
        models[role] = {
            "path": str(root),
            "files": [
                {
                    "name": "config.json",
                    "size": 2,
                    "sha256": hashlib.sha256(b"{}").hexdigest(),
                }
            ],
        }
    assets = tmp_path / "assets.json"
    assets.write_text(
        json.dumps({"schema": "duodecoding.native-assets.v1", "models": models})
    )
    value = json.loads(cfg.read_text())
    value.update(
        profile="local", local={"topology": "ld-t", "asset_manifest": str(assets)}
    )
    cfg.write_text(json.dumps(value))
    assert (
        main(
            ["--validate-only", "--config", str(cfg), "--cpu-manifest", str(cpu)],
            adapters=external,
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert len(report["plan"]) == 56
    assert {row["dtype"] for row in report["plan"]} == {"bfloat16"}
    assert {row["topology"] for row in report["plan"]} == {"ld-t"}
    assert (
        report["binding"]["assets"]["models"]["target"]["files"][0]["sha256"]
        == hashlib.sha256(b"{}").hexdigest()
    )
    (tmp_path / "target" / "config.json").write_bytes(b"[]")
    with pytest.raises(SystemExit) as failure:
        main(
            ["--validate-only", "--config", str(cfg), "--cpu-manifest", str(cpu)],
            adapters=external,
        )
    assert failure.value.code == 2
    assert "asset" in capsys.readouterr().err.lower()


def test_busy_authorized_gpu_never_launches_and_leaves_atomic_not_started_result(
    tmp_path,
):
    from contextlib import nullcontext

    from tools.run_native_validation import main

    class Busy(FakeExternal):
        def lease(self, ids):
            return nullcontext()

        def query(self, ids):
            assert ids == UUIDS
            return {
                u: {"index": i, "memory_mib": 0, "utilization": 0, "pids": [123]}
                for i, u in enumerate(ids)
            }

        def clock(self):
            return 0.0

        def launch(self, *args):
            raise AssertionError("busy GPU must not launch")

    external = Busy()
    cfg, cpu = inputs(tmp_path, external)
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=external,
    )
    assert status == 3
    result = json.loads((output / "result.json").read_text())
    assert result["outcome_category"] == "resource-refused"
    assert result["execution"] == "not_started"
    assert result["functional"] == result["numerical"] == "not_run"
    assert result["acceptance_eligible"] is False
    assert "idle" in result["error"].lower()
    assert not list(output.glob("*.tmp"))


@pytest.mark.parametrize("logging_failure", [False, True])
def test_worker_exit_without_evidence_is_incomplete_and_cleans_surviving_children(
    tmp_path, monkeypatch, logging_failure
):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tools.run_native_validation import main

    class Exited(FakeExternal):
        now = 0.0
        cleaned = False
        launched = False
        queries = 0

        def lease(self, ids):
            return nullcontext()

        def clock(self):
            if self.launched and logging_failure:
                raise OSError("simulated logging clock failure")
            return self.now

        def sleep(self, seconds):
            self.now += seconds

        def query(self, ids):
            self.queries += 1
            return {
                u: {"index": i, "memory_mib": 0, "utilization": 0, "pids": []}
                for i, u in enumerate(ids)
            }

        def launch(self, output, env):
            self.launched = True
            assert self.now >= 30 and self.queries == 3
            assert env["CUDA_VISIBLE_DEVICES"] == ",".join(UUIDS)
            assert env["HF_HUB_OFFLINE"] == env["TRANSFORMERS_OFFLINE"] == "1"
            assert (
                "PYTHONPATH" not in env
                and "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in env
            )
            return SimpleNamespace(pid=4242, poll=lambda: 0)

        def owns(self, pid, child):
            return True

        def cleanup(self, child):
            assert child.pid == 4242
            self.cleaned = True  # Also called when the leader has already exited.
            return {"session": child.pid, "remaining": []}

    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    external = Exited()
    cfg, cpu = inputs(tmp_path, external)
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=external,
    )
    result = json.loads((output / "result.json").read_text())
    assert status != 0 and external.launched and external.cleaned
    assert result["execution"] == "incomplete"
    assert result["functional"] == result["numerical"] == "not_run"
    assert result["acceptance_eligible"] is False
    assert result["cleanup"]["remaining"] == []
    assert len(result["unexecuted_checks"]) == 336
    assert all(row["checks"] and row["reason"] for row in result["unexecuted_checks"])


@pytest.mark.parametrize("numerical_failure", [False, True])
@pytest.mark.parametrize(
    "proof_fault",
    [
        None,
        "empty-cases",
        "missing-check",
        "nan",
        "fallback",
        "late-reset",
        "recapture",
        "missing-forward",
        "runner-replaced",
        "missing-model",
        "wrong-placement",
        "eager-role",
        "nonfinite-pass",
    ],
)
def test_completed_matrix_retains_numerical_failure_and_injected_runs_never_accept(
    tmp_path, numerical_failure, proof_fault
):
    import hashlib
    from contextlib import nullcontext
    from types import SimpleNamespace

    from tools.run_native_validation import main

    class Completed(FakeExternal):
        now = 0.0

        def clock(self):
            return self.now

        def sleep(self, amount):
            self.now += amount

        def lease(self, ids):
            return nullcontext()

        def query(self, ids):
            return {
                u: {"index": i, "memory_mib": 0, "utilization": 0, "pids": []}
                for i, u in enumerate(ids)
            }

        def owns(self, pid, child):
            return True

        def cleanup(self, child):
            return {"session": child.pid, "remaining": []}

        def launch(self, output, env):
            request = json.loads((output / "request.json").read_text())
            cases = []
            for index, case in enumerate(request["plan"]):
                checks = dict.fromkeys(
                    ("prefix", "budget", "token_range", "cpu_rng", "cuda_rng"), "passed"
                )
                if case["budget"]:
                    checks.update(
                        dict.fromkeys(
                            (
                                "metrics",
                                "cache_state",
                                "cache_probabilities",
                                "forward_rows",
                            ),
                            "passed",
                        )
                    )
                if case["temperature"] == 0 and case["scenario"] != "first-token-eos":
                    checks["greedy_target"] = "passed"
                events = []
                if case["backend"] == "graph" and case["scenario"] in (
                    "budget12",
                    "new-sample",
                ):
                    runner = index // 7 + 1
                    base = {
                        "case_id": case["id"],
                        "runner": runner,
                        "cache": runner + 1000,
                        "role": "target",
                        "capture_count": 4,
                        "max_len": 64,
                        "device": "cuda:0",
                    }
                    if case["scenario"] == "new-sample":
                        events.append(dict(base, operation="reset"))
                    events.extend(
                        [dict(base, operation="step"), dict(base, operation="verify")]
                    )
                if numerical_failure and case["scenario"] == "budget1":
                    checks["cache_probabilities"] = "failed"
                cases.append(
                    {
                        "id": case["id"],
                        "checks": checks,
                        "graph_events": events,
                        "tokens": [
                            [1, 2, 3]
                            + [0]
                            * (
                                1
                                if case["scenario"] == "first-token-eos"
                                else case["budget"]
                            )
                        ],
                        "graph_runners": {
                            "little": {
                                "runner": index // 7 + 10001,
                                "cache": index // 7 + 11001,
                                "capture_count": 4,
                                "max_len": 64,
                            },
                            "draft": {
                                "runner": index // 7 + 20001,
                                "cache": index // 7 + 21001,
                                "capture_count": 4,
                                "max_len": 64,
                            },
                            "target": {
                                "runner": index // 7 + 1,
                                "cache": index // 7 + 1001,
                                "capture_count": 4,
                                "max_len": 64,
                            },
                        }
                        if events
                        else {},
                        "numerical_observations": [
                            {
                                "kind": kind,
                                "nonfinite": False,
                                "role": "target",
                                "rows": 1,
                                "atol": 0.008
                                if case["dtype"] == "bfloat16" and case["temperature"]
                                else 0.00004,
                                "rtol": 0.02 if case["dtype"] == "bfloat16" else 0.0005,
                                "max_abs": 0.1
                                if checks.get("cache_probabilities") == "failed"
                                else 0,
                                "passed": kind == "forward"
                                or checks.get("cache_probabilities") != "failed",
                            }
                            for kind in ("cache", "forward")
                        ]
                        if case["budget"]
                        else [],
                    }
                )
            encoded = json.dumps(
                request, sort_keys=True, allow_nan=False, separators=(",", ":")
            ).encode()
            proof = {
                "schema": "duodecoding.native-worker.v1",
                "request_sha256": hashlib.sha256(encoded).hexdigest(),
                "source_before": self.snapshot(),
                "source_after": self.snapshot(),
                "environment": self.environment(),
                "tools": self.tools(),
                "worker_environment": env,
                "gpu_uuids": UUIDS,
                "execution": "completed",
                "cases": cases,
                "smoke": {
                    "checks": {
                        "bf16_gemm": "passed",
                        "graph_replay": "passed",
                        "private_rng": "passed",
                        "cross_device_copy": "passed",
                    },
                    "devices": [{"uuid": u, "graph_replays": 3} for u in UUIDS],
                },
            }
            proof["cuda_runtime"] = "test-runtime"
            proof["peak_allocated_bytes"] = [1, 1]
            proof["models"] = []
            for dtype in ("float32", "bfloat16"):
                for topology, devices in (
                    ("single", (0, 0, 0)),
                    ("ld-t", (0, 0, 1)),
                    ("l-dt", (0, 1, 1)),
                ):
                    proof["models"].append(
                        {
                            "dtype": dtype,
                            "topology": topology,
                            "roles": {
                                role: {
                                    "class": "test.FakeModel",
                                    "dtype": "torch." + dtype,
                                    "device": f"cuda:{device}",
                                    "parameters": 1,
                                    "quantized": False,
                                    "attention": "sdpa",
                                    "asset_path": None,
                                }
                                for role, device in zip(
                                    ("little", "draft", "target"), devices, strict=True
                                )
                            },
                        }
                    )
            if proof_fault == "nonfinite-pass":
                proof["cases"][1]["numerical_observations"][-1]["nonfinite"] = True
            elif proof_fault == "missing-model":
                proof["models"].pop()
            elif proof_fault == "wrong-placement":
                proof["models"][0]["roles"]["target"]["device"] = "cuda:1"
            elif proof_fault == "empty-cases":
                proof["cases"] = []
            elif proof_fault == "missing-check":
                del proof["cases"][1]["checks"]["prefix"]
            elif proof_fault == "nan":
                proof["cases"][1]["numerical_observations"][0]["max_abs"] = float("nan")
            elif proof_fault == "missing-forward":
                proof["cases"][1]["numerical_observations"] = proof["cases"][1][
                    "numerical_observations"
                ][:1]
            else:
                for row in proof["cases"]:
                    if proof_fault == "eager-role" and row["graph_runners"]:
                        row["graph_runners"]["draft"] = None
                    if row["id"].endswith("new-sample"):
                        if proof_fault == "runner-replaced" and row["graph_runners"]:
                            row["graph_runners"]["target"]["runner"] += 1000
                        if proof_fault == "late-reset":
                            row["graph_events"].reverse()
                        for event in row["graph_events"]:
                            if proof_fault == "fallback" and event["operation"] in (
                                "step",
                                "verify",
                            ):
                                event["operation"] = "fallback"
                            elif proof_fault == "recapture":
                                event["capture_count"] += 1
            (output / "worker-result.json").write_text(json.dumps(proof))
            return SimpleNamespace(pid=4242, poll=lambda: 2 if numerical_failure else 0)

    external = Completed()
    cfg, cpu = inputs(tmp_path, external)
    output = tmp_path / "run"
    status = main(
        [
            "--config",
            str(cfg),
            "--cpu-manifest",
            str(cpu),
            "--output-dir",
            str(output),
            "--authorize-gpu-execution",
        ],
        adapters=external,
    )
    report = json.loads((output / "result.json").read_text())
    if proof_fault:
        assert status != 0 and report["acceptance_eligible"] is False
        assert report["execution"] == "incomplete"
        assert report.get("error")
        return
    assert report["execution"] == "completed", report.get("error")
    assert report["functional"] == "passed"
    assert len(report["token_pairs"]["eager_graph"]) == 168
    assert len(report["token_pairs"]["dsd_dssd"]) == 168
    assert all(row["passed"] for rows in report["token_pairs"].values() for row in rows)
    assert report["numerical"] == ("failed" if numerical_failure else "passed")
    assert report["acceptance_eligible"] is False
    assert report["verdict"] == (
        "failed-numerical-equivalence" if numerical_failure else "injected-not-accepted"
    )
    assert status != 0 if numerical_failure else status == 0


def test_real_worker_rejects_unbound_environment_before_cuda_import(tmp_path):
    import time
    from contextlib import nullcontext

    from tools.native_gpu_guard import NativeSystem
    from tools.run_native_validation import main

    class Hidden(FakeExternal, NativeSystem):
        offset = 0

        def lease(self, ids):
            return nullcontext()

        def clock(self):
            return time.monotonic() + self.offset

        def sleep(self, seconds):
            if seconds == 30:
                self.offset += 30
            else:
                time.sleep(0.01)

        def query(self, ids):
            return {
                u: {"index": i, "memory_mib": 0, "utilization": 0, "pids": []}
                for i, u in enumerate(ids)
            }

        def launch(self, output, env):
            return NativeSystem.launch(self, output, dict(env, CUDA_VISIBLE_DEVICES=""))

    external = Hidden()
    cfg, cpu = inputs(tmp_path, external)
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=external,
        )
        != 0
    )
    worker = json.loads((output / "worker-result.json").read_text())
    assert worker["gpu_initialized"] is False
    assert "environment" in worker["error"].lower()
    assert worker["execution"] == "incomplete"


@pytest.mark.parametrize(
    "invalid",
    ["configuration", "cpu-status", "missing-mode", "missing-junit", "stale-source"],
)
def test_invalid_contract_persists_not_started_result_without_gpu_queries(
    tmp_path, invalid
):
    from tools.run_native_validation import main

    external = FakeExternal()
    cfg, cpu = inputs(tmp_path, external)
    if invalid == "configuration":
        value = json.loads(cfg.read_text())
        value["gpu_uuids"] = [0, 1]
        cfg.write_text(json.dumps(value))
    else:
        value = json.loads(cpu.read_text())
        if invalid == "cpu-status":
            value["status"] = "failed"
        elif invalid == "missing-mode":
            value["results"].pop()
        elif invalid == "missing-junit":
            (tmp_path / "default.xml").unlink()
        else:
            value["source_before"]["git_sha"] = "d" * 40
        cpu.write_text(json.dumps(value))
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=external,
        )
        == 2
    )
    report = json.loads((output / "result.json").read_text())
    assert report["execution"] == "not_started"
    assert report["binding_status"] == "failed"
    assert report["outcome_category"] == "invalid-contract"
    assert report["acceptance_eligible"] is False


@pytest.mark.parametrize(
    "ending",
    [
        "leader-exited",
        "timeout",
        "foreign",
        "SIGTERM",
        "SIGINT",
        "logging-failure",
        "launch-SIGTERM",
        "fsync-failure",
        "vanished-foreign",
    ],
)
def test_owned_cpu_session_is_cleaned_without_signaling_foreign_processes(
    tmp_path, ending
):
    import os
    import select
    import signal
    import time
    from contextlib import nullcontext

    from tools.native_gpu_guard import NativeSystem
    from tools.run_native_validation import main

    old_handlers = {
        s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP)
    }
    old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, set())
    stranger = subprocess.Popen(
        [sys.executable, "-I", "-B", "-c", "import time; time.sleep(120)"],
        start_new_session=True,
        env={"CUDA_VISIBLE_DEVICES": ""},
    )

    class CpuSession(FakeExternal, NativeSystem):
        offset = 0
        queries = 0
        child = None
        descendant = None
        fsync_failed = False

        def fsync(self, fd):
            if (
                self.child is not None
                and ending == "fsync-failure"
                and not self.fsync_failed
            ):
                self.fsync_failed = True
                raise OSError("simulated fsync failure")
            os.fsync(fd)

        def lease(self, ids):
            return nullcontext()

        def clock(self):
            if self.child is not None and ending == "logging-failure":
                raise OSError("logging failure after successful launch")
            return time.monotonic() + self.offset

        def sleep(self, seconds):
            self.offset += seconds

        def query(self, ids):
            self.queries += 1
            if self.queries == 4 and ending in ("SIGTERM", "SIGINT", "SIGHUP"):
                os.kill(os.getpid(), getattr(signal, ending))
            if self.queries == 4 and ending == "vanished-foreign":
                stranger.kill()
                stranger.wait(timeout=5)
            pids = (
                []
                if self.child is None
                else [
                    stranger.pid
                    if ending in ("foreign", "vanished-foreign")
                    else self.descendant
                ]
            )
            return {
                u: {"index": i, "memory_mib": 0, "utilization": 0, "pids": pids}
                for i, u in enumerate(ids)
            }

        def launch(self, output, env):
            grandchild_code = (
                "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                "print('ready',flush=True); time.sleep(120)"
            )
            code = (
                "import subprocess,sys,time; "
                f"c=subprocess.Popen([sys.executable,'-I','-B','-c',{grandchild_code!r}],"
                "stdout=subprocess.PIPE,text=True); "
                "c.stdout.readline(); print(c.pid,flush=True); "
                + ("sys.exit(0)" if ending == "leader-exited" else "time.sleep(120)")
            )
            self.child = subprocess.Popen(
                [sys.executable, "-I", "-B", "-c", code],
                start_new_session=True,
                stdout=subprocess.PIPE,
                text=True,
                env={"CUDA_VISIBLE_DEVICES": ""},
            )
            assert select.select([self.child.stdout], [], [], 5)[0], (
                "CPU session failed to start"
            )
            self.descendant = int(self.child.stdout.readline())
            self.child.stdout.close()
            if ending in ("SIGHUP", "launch-SIGHUP"):
                # The outer pytest process must also own cleanup on the red path:
                # an unhandled HUP terminates this isolated supervisor immediately.
                ownership = [
                    {
                        "pid": pid,
                        "start": (Path("/proc") / str(pid) / "stat")
                        .read_text()
                        .rsplit(")", 1)[1]
                        .split()[19],
                    }
                    for pid in (self.child.pid, self.descendant, stranger.pid)
                ]
                (tmp_path / "hup-owned-processes.json").write_text(
                    json.dumps(ownership)
                )
            if ending == "leader-exited":
                assert self.child.wait(timeout=5) == 0
            elif ending in ("launch-SIGTERM", "launch-SIGHUP"):
                os.kill(os.getpid(), getattr(signal, ending.removeprefix("launch-")))
            return self.child

        def cleanup(self, child):
            if ending in ("SIGHUP", "launch-SIGHUP"):
                os.kill(os.getpid(), signal.SIGHUP)
            return NativeSystem.cleanup(self, child)

    external = CpuSession()
    try:
        cfg, cpu = inputs(tmp_path, external)
        value = json.loads(cfg.read_text())
        value["max_seconds"] = 1
        cfg.write_text(json.dumps(value))
        output = tmp_path / "run"
        assert (
            main(
                [
                    "--config",
                    str(cfg),
                    "--cpu-manifest",
                    str(cpu),
                    "--output-dir",
                    str(output),
                    "--authorize-gpu-execution",
                ],
                adapters=external,
            )
            != 0
        )
        report = json.loads((output / "result.json").read_text())
        assert report["execution"] == "incomplete"
        assert report["acceptance_eligible"] is False
        assert report["cleanup"]["remaining"] == []
        if ending == "fsync-failure":
            assert external.fsync_failed and "fsync" in report["error"]
            assert not (output / "result.json.tmp").exists()
        if ending == "vanished-foreign":
            assert "foreign" in report["error"] and stranger.poll() is not None
        else:
            assert stranger.poll() is None
        assert external.child.poll() is not None
        proc = Path("/proc") / str(external.descendant) / "stat"
        assert not proc.exists() or proc.read_text().rsplit(")", 1)[1].split()[0] == "Z"
        assert {s: signal.getsignal(s) for s in old_handlers} == old_handlers
        assert signal.pthread_sigmask(signal.SIG_BLOCK, set()) == old_mask
        if ending in ("SIGHUP", "launch-SIGHUP"):
            assert report["outcome_category"] == "interrupted"
            assert "SIGHUP" in report["error"]
    finally:
        stranger.kill()
        stranger.wait(timeout=5)
        if external.child is not None:
            NativeSystem.cleanup(external, external.child)


@pytest.mark.parametrize("ending", ["SIGHUP", "launch-SIGHUP"])
def test_ssh_hangup_cleans_worker_in_isolated_public_cli_process(tmp_path, ending):
    import signal
    import time

    # Never deliver an unhandled HUP to pytest. The isolated probe calls the
    # public coordinator; its outer owner can kill every test-owned CPU process
    # even when the red baseline exits before the coordinator's finally block.
    code = (
        "import runpy,signal,sys; from pathlib import Path; "
        "signal.signal(signal.SIGHUP,signal.SIG_DFL); "
        "signal.pthread_sigmask(signal.SIG_UNBLOCK,{signal.SIGHUP}); "
        "signal.pthread_sigmask(signal.SIG_BLOCK,{signal.SIGUSR1}); "
        "sys.path.insert(0,sys.argv[1]); "
        "tests=runpy.run_path(sys.argv[2]); "
        "tests['test_owned_cpu_session_is_cleaned_without_signaling_foreign_processes']"
        "(Path(sys.argv[3]),sys.argv[4])"
    )
    log_path = tmp_path / "hup-supervisor.log"
    with log_path.open("w") as log:
        probe = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-B",
                "-c",
                code,
                str(ROOT),
                __file__,
                str(tmp_path),
                ending,
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
        )
    try:
        probe.wait(timeout=25)
        assert probe.returncode == 0, log_path.read_text()
    finally:
        if probe.poll() is None:
            probe.kill()
        probe.wait(timeout=5)
        ownership = tmp_path / "hup-owned-processes.json"
        records = json.loads(ownership.read_text()) if ownership.exists() else []
        for record in records:
            proc = Path("/proc") / str(record["pid"]) / "stat"
            try:
                current = proc.read_text().rsplit(")", 1)[1].split()
                if current[19] == record["start"] and current[0] != "Z":
                    os.kill(record["pid"], signal.SIGKILL)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass
        deadline = time.monotonic() + 5
        while True:
            running = []
            for record in records:
                try:
                    current = (Path("/proc") / str(record["pid"]) / "stat").read_text()
                    fields = current.rsplit(")", 1)[1].split()
                    if fields[19] == record["start"] and fields[0] != "Z":
                        running.append(record["pid"])
                except FileNotFoundError:
                    pass
            if not running or time.monotonic() >= deadline:
                break
            time.sleep(0.01)
        assert not running, f"test-owned CPU processes survived cleanup: {running}"


@pytest.mark.parametrize("marker", ["-", "+", "U"])
def test_uninitialized_or_unbound_submodule_refuses_even_matching_cpu_manifest(
    tmp_path, marker
):
    from tools.run_native_validation import main

    class UnsafeSubmodule(FakeExternal):
        def snapshot(self):
            return dict(
                super().snapshot(),
                submodules=marker + "a" * 40 + " third_party/model\n",
            )

    external = UnsafeSubmodule()
    cfg, cpu = inputs(tmp_path, external)
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=external,
        )
        == 2
    )
    report = json.loads((output / "result.json").read_text())
    assert "submodule" in report["error"] and report["execution"] == "not_started"


def test_abbreviated_authorization_is_not_explicit_authorization(tmp_path):
    output = tmp_path / "run"
    result = cli(
        "--authorize-gpu",
        "--config",
        str(tmp_path / "missing"),
        "--cpu-manifest",
        str(tmp_path / "missing-cpu"),
        "--output-dir",
        str(output),
    )
    assert result.returncode == 2 and "unrecognized arguments" in result.stderr
    assert not output.exists()


def cli(*args, optimized=False):
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            *(["-O"] if optimized else []),
            str(CLI),
            *map(str, args),
        ],
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
        capture_output=True,
        text=True,
        timeout=20,
    )


@pytest.mark.parametrize(
    "status,count", [("passed", 336), ("failed-numerical-equivalence", 56)]
)
def test_legacy_report_preserves_verdict_and_never_upgrades_missing_provenance(
    tmp_path, status, count
):
    original = {
        "schema": "duodecoding.native-cee-validation.v1",
        "status": status,
        "matrix_execution_completed": True,
        "cases": [{"label": str(i)} for i in range(count)],
        "numerical_mismatches": [{"label": str(i)} for i in range(18)]
        if count == 56
        else [],
    }
    path = tmp_path / "result.json"
    raw = json.dumps(original).encode()
    path.write_bytes(raw)
    result = cli("--legacy-report", path)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["original"] == original
    assert report["acceptance_eligible"] is False
    assert report["provenance"] == "unknown"
    assert path.read_bytes() == raw
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize(
    "process_xml",
    ["N/A", "<unsupported/>", "<process_info><pid>12345</pid></process_info>"],
)
def test_native_query_parser_refuses_unavailable_or_busy_process_table(
    tmp_path, process_xml
):
    import contextlib

    from tools.native_gpu_guard import NativeSystem
    from tools.run_native_validation import main

    class NativeQuery(FakeExternal, NativeSystem):
        now = 0.0
        query = NativeSystem.query
        launched = False

        def clock(self):
            return self.now

        def sleep(self, seconds):
            self.now += seconds

        def lease(self, ids):
            return contextlib.nullcontext()

        def nvidia_smi(self, ids, options):
            assert ids == UUIDS
            if options == ["-q", "-x"]:
                return (
                    "<nvidia_smi_log>"
                    + "".join(
                        f"<gpu><uuid>{identity}</uuid><processes>{process_xml}</processes></gpu>"
                        for identity in ids
                    )
                    + "</nvidia_smi_log>"
                )
            assert options == [
                "--query-gpu=index,uuid,memory.used,utilization.gpu,driver_version",
                "--format=csv,noheader,nounits",
            ]
            return "\n".join(
                f"{index},{identity},0,0,test-driver"
                for index, identity in enumerate(ids)
            )

        def launch(self, output, env):
            self.launched = True
            raise AssertionError("unavailable/busy process table must not reach launch")

    external = NativeQuery()
    cfg, cpu = inputs(tmp_path, external)
    output = tmp_path / "run"
    assert (
        main(
            [
                "--config",
                str(cfg),
                "--cpu-manifest",
                str(cpu),
                "--output-dir",
                str(output),
                "--authorize-gpu-execution",
            ],
            adapters=external,
        )
        == 3
    )
    report = json.loads((output / "result.json").read_text())
    assert not external.launched and report["execution"] == "not_started"
    assert report["acceptance_eligible"] is False


@pytest.mark.parametrize("optimized", [False, True])
def test_run_requires_authorization_before_even_reading_config(tmp_path, optimized):
    output = tmp_path / "new"
    result = cli(
        "--config",
        tmp_path / "absent.json",
        "--cpu-manifest",
        tmp_path / "absent-manifest.json",
        "--output-dir",
        output,
        optimized=optimized,
    )
    assert result.returncode == 2
    assert "authorization" in result.stderr.lower()
    assert not output.exists()
