"""Finite, fail-closed native validation records. Independent of Stage-A/v1."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from tools.run_cpu_gate import CPU_ENV, pytest_command


class ValidationError(ValueError):
    pass


def require(ok, message):
    if not ok:
        raise ValidationError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, allow_nan=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def read_json(path):
    def pairs(rows):
        result = {}
        for key, value in rows:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(Path(path).read_text(), object_pairs_hook=pairs)
    digest(value)  # Reject every NaN/Infinity, including nested values.
    return value


def atomic_json(path, value, *, fsync=os.fsync):
    path = Path(path)
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    temporary = path.with_name(path.name + ".tmp")
    created = False
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            created = True
            stream.write(text)
            stream.flush()
            fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        if created:
            temporary.unlink(missing_ok=True)
        raise


def keys(value, expected, label):
    require(
        isinstance(value, dict) and set(value) == set(expected),
        f"invalid {label} fields",
    )


def configuration(value):
    if isinstance(value, dict):
        value = {"ignore_xorg": False, **value}
    keys(
        value,
        {
            "schema",
            "profile",
            "gpu_uuids",
            "max_seconds",
            "continue_on_numerical_mismatch",
            "ignore_xorg",
            "local",
        },
        "configuration",
    )
    require(
        value["schema"] == "duodecoding.native-config.v1",
        "unknown configuration schema",
    )
    require(value["profile"] in ("tiny", "local"), "unknown native profile")
    ids = value["gpu_uuids"]
    require(
        isinstance(ids, list) and len(ids) == 2 and len(set(ids)) == 2,
        "exactly two ordered authorized GPU UUIDs required; no smaller matrix",
    )
    for item in ids:
        require(
            isinstance(item, str) and item.startswith("GPU-"),
            "canonical GPU UUID required",
        )
        try:
            canonical = "GPU-" + str(uuid.UUID(item[4:]))
        except ValueError as exc:
            raise ValidationError("canonical GPU UUID required") from exc
        require(item == canonical, "canonical GPU UUID required")
    require(
        type(value["max_seconds"]) is int and 1 <= value["max_seconds"] <= 14400,
        "max_seconds must be 1..14400",
    )
    require(
        type(value["continue_on_numerical_mismatch"]) is bool,
        "continuation must be bool",
    )
    require(type(value["ignore_xorg"]) is bool, "ignore_xorg must be bool")
    if value["profile"] == "tiny":
        require(value["local"] is None, "tiny profile forbids local settings")
    else:
        keys(value["local"], {"topology", "asset_manifest"}, "local profile")
        require(
            value["local"]["topology"] in ("ld-t", "l-dt"),
            "explicit two-device local topology required",
        )
        require(
            isinstance(value["local"]["asset_manifest"], str)
            and Path(value["local"]["asset_manifest"]).is_absolute(),
            "absolute asset manifest required",
        )
    return value


def plan(config):
    cases = []
    dtypes = ("float32", "bfloat16") if config["profile"] == "tiny" else ("bfloat16",)
    topologies = (
        ("single", "ld-t", "l-dt")
        if config["profile"] == "tiny"
        else (config["local"]["topology"],)
    )
    for dtype, topology, mode, backend, temp in itertools.product(
        dtypes,
        topologies,
        ("cee_dsd", "cee_dssd"),
        ("eager", "graph"),
        (0.0, 0.7),
    ):
        for scenario, budget, seed in (
            ("budget0", 0, 713),
            ("budget1", 1, 714),
            ("budget2", 2, 715),
            ("budget3", 3, 716),
            ("budget12", 12, 725),
            ("new-sample", 12, 918),
            ("first-token-eos", 12, 725),
        ):
            identity = f"{dtype}/{topology}/{mode}/{backend}/{temp}/{scenario}"
            cases.append(
                {
                    "id": identity,
                    "dtype": dtype,
                    "topology": topology,
                    "mode": mode,
                    "backend": backend,
                    "temperature": temp,
                    "scenario": scenario,
                    "budget": budget,
                    "seed": seed,
                }
            )
    for case in cases:
        case["required_checks"] = sorted(required_checks(case))
        atol, rtol = tolerances(case)
        case["tolerances"] = {"atol": atol, "rtol": rtol}
    return cases


def bind_cpu(path, source, environment):
    path = Path(path).resolve()
    manifest = read_json(path)
    require(
        manifest.get("schema") == "duodecoding.cpu-gate.v1"
        and manifest.get("status") == "passed",
        "successful CPU gate required",
    )
    require(
        manifest.get("source_unchanged") is True
        and manifest.get("source_before") == manifest.get("source_after") == source,
        "CPU manifest must bind EXACT current source snapshot",
    )
    require(
        manifest.get("environment") == environment
        and environment.get("overrides") == CPU_ENV,
        "CPU manifest environment mismatch",
    )
    results = manifest.get("results")
    require(
        isinstance(results, list) and len(results) == 2,
        "two successful CPU modes required",
    )
    golden = manifest.get("golden_assets", {})
    require(
        type(golden.get("required")) is bool
        and golden.get("flag") == "--require-golden-assets",
        "CPU asset policy missing",
    )
    artifacts = {}
    for mode, row in zip(("default", "importlib"), results, strict=True):
        junit = path.parent / f"{mode}.xml"
        log = path.parent / f"{mode}.log"
        expected = pytest_command(
            environment["executable"],
            mode,
            junit,
            require_golden_assets=golden["required"],
        )
        require(
            row.get("mode") == mode
            and type(row.get("returncode")) is int
            and row["returncode"] == 0
            and row.get("command") == expected
            and row.get("junit") == junit.name
            and row.get("log") == log.name,
            f"invalid CPU {mode} command/result",
        )
        require(
            type(row.get("seconds")) in (int, float)
            and math.isfinite(row["seconds"])
            and row["seconds"] >= 0,
            "invalid CPU duration",
        )
        xml = ET.parse(junit).getroot()
        tests = list(xml.iter("testcase"))
        require(
            bool(tests)
            and not list(xml.iter("failure"))
            and not list(xml.iter("error"))
            and any(test.find("skipped") is None for test in tests),
            f"CPU {mode} JUnit lacks executed passing tests",
        )
        require(
            all(
                int(suite.get("failures", "0")) == 0
                and int(suite.get("errors", "0")) == 0
                for suite in xml.iter("testsuite")
            ),
            "CPU JUnit reports failures",
        )
        artifacts[junit.name] = file_hash(junit)
        artifacts[log.name] = file_hash(log)
    return {
        "status": "passed",
        "manifest_sha256": file_hash(path),
        "artifacts": artifacts,
        "golden_assets": golden,
    }


def assets(config):
    if config["profile"] == "tiny":
        return None
    path = Path(config["local"]["asset_manifest"])
    manifest = read_json(path)
    keys(manifest, {"schema", "models"}, "assets manifest")
    require(
        manifest["schema"] == "duodecoding.native-assets.v1", "invalid assets schema"
    )
    keys(manifest["models"], {"little", "draft", "target"}, "asset model roles")
    observed = {}
    for role, entry in manifest["models"].items():
        keys(entry, {"path", "files"}, f"asset {role}")
        require(
            isinstance(entry["path"], str) and Path(entry["path"]).is_absolute(),
            "absolute asset directory required",
        )
        root = Path(entry["path"])
        require(root.is_dir(), "asset directory missing")
        require(
            isinstance(entry["files"], list) and entry["files"], "asset files required"
        )
        names, files = set(), []
        for row in entry["files"]:
            keys(row, {"name", "size", "sha256"}, "asset file")
            name = row["name"]
            require(
                isinstance(name, str)
                and name
                and not Path(name).is_absolute()
                and ".." not in Path(name).parts
                and name not in names,
                "invalid asset filename",
            )
            names.add(name)
            file = root / name
            require(
                type(row["size"]) is int
                and row["size"] >= 0
                and file.is_file()
                and file.stat().st_size == row["size"]
                and file_hash(file) == row["sha256"],
                f"asset hash/size mismatch: {file}",
            )
            files.append({**row, "resolved": str(file.resolve())})
        actual = set()
        for file in root.rglob("*"):
            require(
                not (file.is_symlink() and file.is_dir()),
                "asset directory symlinks forbidden",
            )
            if file.is_file():
                actual.add(file.relative_to(root).as_posix())
        require(
            actual == names and "config.json" in names, "asset inventory incomplete"
        )
        observed[role] = {"path": str(root), "files": files}
    return {"manifest_sha256": file_hash(path), "models": observed}


def required_checks(case):
    names = {"prefix", "budget", "token_range", "cpu_rng", "cuda_rng"}
    if case["budget"]:
        names |= {"metrics", "cache_state", "cache_probabilities", "forward_rows"}
    if case["temperature"] == 0 and case["scenario"] != "first-token-eos":
        names.add("greedy_target")
    return names


def tolerances(case):
    bf16 = case["dtype"] == "bfloat16"
    return (
        0.008 if bf16 and case["temperature"] else 0.00004,
        0.02 if bf16 else 0.0005,
    )


def graph_evidence(cases, planned):
    prior = {}
    witnesses = []
    replays = fallbacks = 0
    replay_operations = set()
    by_id = {case["id"]: case for case in cases}
    for case, spec in zip(cases, planned, strict=False):
        resets, first_replays, reused = {}, set(), set()
        for event in case["graph_events"]:
            keys(
                event,
                {
                    "case_id",
                    "operation",
                    "runner",
                    "cache",
                    "role",
                    "capture_count",
                    "max_len",
                    "device",
                },
                "graph event",
            )
            require(
                event["case_id"] == case["id"] and spec["backend"] == "graph",
                "graph event attribution mismatch",
            )
            require(
                event["operation"] in ("step", "verify", "reset", "fallback"),
                "invalid graph operation",
            )
            require(
                all(
                    type(event[x]) is int and event[x] > 0
                    for x in ("runner", "cache", "max_len")
                )
                and type(event["capture_count"]) is int
                and event["capture_count"] >= 0,
                "invalid graph weak identity/capture",
            )
            require(
                event["role"] in ("little", "draft", "target")
                and event["device"] in ("cuda:0", "cuda:1"),
                "invalid graph placement",
            )
            identity = event["runner"]
            signature = tuple(
                event[x]
                for x in ("cache", "capture_count", "max_len", "role", "device")
            )
            if event["operation"] == "reset":
                resets[identity] = signature
            elif event["operation"] == "fallback":
                fallbacks += 1
            else:
                require(event["capture_count"] > 0, "graph replay without capture")
                replays += 1
                replay_operations.add(event["operation"])
                if identity not in first_replays:
                    if (
                        resets.get(identity) == signature
                        and prior.get(identity) == signature
                    ):
                        reused.add(identity)
                    first_replays.add(identity)
                prior[identity] = signature
        if spec["backend"] == "graph" and spec["scenario"] == "new-sample":
            require(
                bool(reused), f"no reset-before-replay reuse evidence: {case['id']}"
            )
            primary = by_id[case["id"].rsplit("/", 1)[0] + "/budget12"]
            runners = case.get("graph_runners")
            keys(runners, {"little", "draft", "target"}, "per-role graph runners")
            for role, runner in runners.items():
                keys(
                    runner,
                    {"runner", "cache", "capture_count", "max_len"},
                    f"{role} graph inventory",
                )
                require(
                    all(type(value) is int and value > 0 for value in runner.values()),
                    f"{role} graph runner missing/capture invalid",
                )
            require(
                runners == primary.get("graph_runners"),
                "graph runner/cache/capture changed across samples",
            )
            require(
                reused
                <= {row["runner"] for row in runners.values() if row is not None},
                "graph replay witness disagrees with live runner inventory",
            )
            witnesses.append({"case_id": case["id"], "runners": sorted(reused)})
    require(
        replay_operations == {"step", "verify"},
        "successful step AND verify replay evidence required",
    )
    return {
        "replays": replays,
        "fallbacks": fallbacks,
        "reuse_witnesses": witnesses,
        "scope": (
            "Observed successful original runner calls only; "
            "not all graph branches/roles"
        ),
    }


def token_pairs(cases, planned):
    by_id = {row["id"]: row for row in cases}
    result = {"eager_graph": [], "dsd_dssd": []}
    for case, spec in zip(cases, planned, strict=False):
        tokens = case.get("tokens")
        require(
            isinstance(tokens, list)
            and len(tokens) == 1
            and isinstance(tokens[0], list)
            and tokens[0]
            and all(type(token) is int and token >= 0 for token in tokens[0]),
            "executed token evidence missing",
        )
        for label, field, left, right in (
            ("eager_graph", "backend", "eager", "graph"),
            ("dsd_dssd", "mode", "cee_dsd", "cee_dssd"),
        ):
            if spec[field] == left:
                peer = spec["id"].replace(f"/{left}/", f"/{right}/")
                require(peer in by_id, "token comparison pair missing")
                result[label].append(
                    {
                        "left": case["id"],
                        "right": peer,
                        "passed": tokens == by_id[peer].get("tokens"),
                    }
                )
    require(
        all(len(rows) == len(planned) // 2 for rows in result.values()),
        "token pair matrix incomplete",
    )
    return result


def model_evidence(worker, request):
    expected = list(
        dict.fromkeys((case["dtype"], case["topology"]) for case in request["plan"])
    )
    models = worker.get("models")
    require(
        isinstance(models, list)
        and [(row.get("dtype"), row.get("topology")) for row in models] == expected,
        "actual model group inventory incomplete",
    )
    for group in models:
        keys(group.get("roles"), {"little", "draft", "target"}, "actual model roles")
        devices = {"single": (0, 0, 0), "ld-t": (0, 0, 1), "l-dt": (0, 1, 1)}[
            group["topology"]
        ]
        for role, device in zip(("little", "draft", "target"), devices, strict=True):
            row = group["roles"][role]
            expected_path = (
                None
                if request["binding"]["assets"] is None
                else request["binding"]["assets"]["models"][role]["path"]
            )
            require(
                isinstance(row.get("class"), str)
                and row["class"]
                and row.get("dtype") == "torch." + group["dtype"]
                and row.get("device") == f"cuda:{device}"
                and type(row.get("parameters")) is int
                and row["parameters"] > 0
                and row.get("quantized") is False
                and row.get("attention") == "sdpa"
                and row.get("asset_path") == expected_path,
                "actual model class/dtype/placement/assets mismatch",
            )
    require(
        isinstance(worker.get("cuda_runtime"), str) and worker["cuda_runtime"],
        "actual CUDA runtime missing",
    )
    peak = worker.get("peak_allocated_bytes")
    require(
        isinstance(peak, list)
        and len(peak) == len(request["configuration"]["gpu_uuids"])
        and all(type(value) is int and value > 0 for value in peak),
        "actual peak allocation evidence missing",
    )


def worker_outcome(worker, request):
    require(
        worker.get("schema") == "duodecoding.native-worker.v1", "invalid worker schema"
    )
    require(
        worker.get("request_sha256") == digest(request),
        "worker request binding mismatch",
    )
    bound = request["binding"]
    require(
        worker.get("source_before") == worker.get("source_after") == bound["source"]
        and worker.get("environment") == bound["environment"]
        and worker.get("tools") == bound["tools"]
        and worker.get("worker_environment") == request["worker_environment"]
        and worker.get("gpu_uuids") == request["configuration"]["gpu_uuids"],
        "worker source/environment/UUID binding mismatch",
    )
    smoke = worker.get("smoke", {})
    keys(
        smoke.get("checks"),
        {"bf16_gemm", "graph_replay", "private_rng", "cross_device_copy"},
        "smoke checks",
    )
    smoke_passed = all(value == "passed" for value in smoke["checks"].values())
    devices = smoke.get("devices", [])
    require(
        [row.get("uuid") for row in devices] == worker["gpu_uuids"]
        and all(
            type(row.get("graph_replays")) is int and row["graph_replays"] == 3
            for row in devices
        ),
        "same-run smoke device/replay evidence missing",
    )
    planned, cases = request["plan"], worker.get("cases")
    require(
        isinstance(cases, list) and len(cases) <= len(planned), "invalid executed cases"
    )
    functional_failed = not smoke_passed
    numerical_failed = False
    all_checks_executed = True
    for case, spec in zip(cases, planned, strict=False):
        require(case.get("id") == spec["id"], "missing/duplicated/out-of-order case")
        checks = case.get("checks")
        keys(checks, required_checks(spec), "case checks")
        require(
            all(x in ("passed", "failed", "not_run") for x in checks.values()),
            "invalid check outcome",
        )
        all_checks_executed &= "not_run" not in checks.values()
        functional_failed |= any(
            v == "failed"
            for k, v in checks.items()
            if k not in ("cache_probabilities", "forward_rows")
        )
        observations = case.get("numerical_observations")
        require(isinstance(observations, list), "numerical observations required")
        for field, kind in (
            ("cache_probabilities", "cache"),
            ("forward_rows", "forward"),
        ):
            if spec["budget"] and checks[field] != "not_run":
                matching = [row for row in observations if row.get("kind") == kind]
                require(
                    bool(matching) and any(row.get("rows", 0) > 0 for row in matching),
                    f"empty {kind} numerical check",
                )
                require(
                    (checks[field] == "passed")
                    == all(row.get("passed") is True for row in matching),
                    f"{kind} comparison status disagrees with observations",
                )
        for row in observations:
            require(
                row.get("kind") in ("cache", "forward")
                and row.get("role") in ("little", "draft", "target")
                and type(row.get("rows")) is int
                and row["rows"] >= 0
                and type(row.get("passed")) is bool
                and type(row.get("nonfinite")) is bool
                and not (row["nonfinite"] and row["passed"])
                and type(row.get("max_abs")) in (int, float)
                and row["max_abs"] >= 0,
                "invalid numerical observation",
            )
            require(
                (row.get("atol"), row.get("rtol")) == tolerances(spec),
                "comparison thresholds changed",
            )
            numerical_failed |= not row["passed"]
        numerical_failed |= any(
            checks.get(k) == "failed" for k in ("cache_probabilities", "forward_rows")
        )
        require(isinstance(case.get("graph_events"), list), "graph events required")
    completed = len(cases) == len(planned) and all_checks_executed
    require(
        worker.get("execution") in ("completed", "incomplete"),
        "invalid execution status",
    )
    require(
        worker["execution"] != "completed" or completed,
        "completed matrix missing planned checks/cases",
    )
    completed &= worker["execution"] == "completed"
    if completed:
        model_evidence(worker, request)
    coverage = graph_evidence(cases, planned) if completed else None
    paired = token_pairs(cases, planned) if completed else None
    if paired:
        functional_failed |= any(
            not row["passed"] for rows in paired.values() for row in rows
        )
    return {
        "execution": "completed" if completed else "incomplete",
        "functional": "failed"
        if functional_failed
        else "passed"
        if completed
        else "not_run",
        "numerical": "failed"
        if numerical_failed
        else "passed"
        if completed
        else "not_run",
        "cases": cases,
        "smoke": smoke,
        "graph": coverage,
        "token_pairs": paired,
        "native_metadata": {
            key: worker.get(key)
            for key in ("models", "cuda_runtime", "peak_allocated_bytes")
        },
    }


def binding(config_path, cpu_path, external):
    config = configuration(read_json(config_path))
    source, environment = external.snapshot(), external.environment()
    require(
        all(line.startswith(" ") for line in source["submodules"].splitlines()),
        "uninitialized/mismatched/conflicted submodule not bound",
    )
    return config, {
        "source": source,
        "source_sha256": digest(source),
        "environment": environment,
        "tools": external.tools(),
        "configuration_sha256": file_hash(config_path),
        "cpu": bind_cpu(cpu_path, source, environment),
        "assets": assets(config),
    }
