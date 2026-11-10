"""Pure-stdlib stage-A evidence contract. Schema 1 is intentionally separate.

Validation is of recorded observations, not a proof of algorithm correctness,
CUDA equivalence, unbiasedness, quality, or a frozen execution interval.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from pathlib import Path

SCHEMA_VERSION = 2
MODES = {
    "sd",
    "dsd",
    "dist_spec",
    "dssd",
    "dist_split_spec",
    "adaptive_decoding",
    "adaptive_tridecoding",
    "cee_sd",
    "tridecoding",
    "cee_sd_no_arp",
    "cee_sd_opportunistic",
}
LIMITATIONS = (
    "Observed smoke only; no distribution/quality/performance acceptance. Different "
    "RNG flows need not produce identical tokens; lossy probability bits are not "
    "an unbiasedness claim. Source equality checks endpoints only. Single GPU only."
)
# A named, deliberately bounded profile: unknown knobs are rejected, not ignored.
DEFAULT_BEHAVIOR = {
    "temp": 0.7,
    "top_k": 0,
    "top_p": 0.0,
    "gamma": 3,
    "gamma1": 3,
    "gamma2": 2,
    "transfer_top_k": 32,
    "transfer_top_k_cap": 0,
    "prob_payload_bits": 16,
    "force_full_vocab_transfer": False,
    "comm_round_trip_mode": "per_round",
    "charge_residual_payload": True,
    "edge_cloud_bandwidth": 20.0,
    "edge_end_bandwidth": 100.0,
    "cloud_end_bandwidth": 100.0,
    "min_bandwidth_mbps": 5.0,
    "ntt_ms_edge_cloud": 10.0,
    "ntt_ms_edge_end": 1.0,
    "batch_delay": 0.0,
    "comm_bw_model": "instant",
    "arp_stop_mode": "per_token",
    "small_draft_threshold": 0.8,
    "draft_target_threshold": 0.8,
    "rl_action_space": "topk_thr",
    "rl_gamma_candidates": "2,4,8,16",
    "rl_factored_q": False,
    "rl_include_gamma_in_state": False,
    "state_bw_scaling": "linear",
    "state_latency_scaling": "linear",
    "graph_verify_sizes": "4,8,16,24,32,40,48,64",
    "model_dtype": "bf16",
}


class StageAError(ValueError):
    """Fail closed, including RL loaders which catch RuntimeError and fall back."""


def need(condition, message):
    if not condition:
        raise StageAError(message)


def integer(value, low=0, high=None):
    return type(value) is int and value >= low and (high is None or value <= high)


def number(value, low=0):
    return type(value) in (int, float) and math.isfinite(value) and value >= low


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def sha(value):
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(c in "0123456789abcdef" for c in value)
    )


def gpu_uuid(value):
    need(isinstance(value, str) and value.startswith("GPU-"), "full GPU UUID required")
    try:
        canonical = "GPU-" + str(uuid.UUID(value[4:]))
    except (ValueError, AttributeError) as exc:
        raise StageAError("full GPU UUID required") from exc
    need(value == canonical, "canonical full GPU UUID required (not ordinal/MIG)")
    return canonical


def keys(value, required, label):
    need(
        isinstance(value, dict) and set(value) == set(required),
        f"{label}: keys must be {sorted(required)}",
    )


def strict_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        need(math.isfinite(value), "non-finite JSON number")
    elif isinstance(value, list):
        for item in value:
            strict_json(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            need(isinstance(key, str), "JSON key must be string")
            strict_json(item)
    else:
        raise StageAError(f"not JSON: {type(value).__name__}")


def load_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            need(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        result = json.loads(
            Path(path).read_text(encoding="utf-8"), object_pairs_hook=unique
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise StageAError(f"cannot read JSON {path}: {exc}") from exc
    strict_json(result)
    return result


def validate_config(raw):
    keys(
        raw,
        {
            "mode",
            "backend",
            "gpu_uuids",
            "samples",
            "max_tokens",
            "seed",
            "cache_capacity",
            "max_seconds",
            "behavior",
        },
        "config",
    )
    strict_json(raw)
    need(
        isinstance(raw["mode"], str) and raw["mode"] in MODES,
        "unsupported stage-A mode",
    )
    need(raw["backend"] in ("eager", "graph"), "backend must be eager/graph")
    devices = raw["gpu_uuids"]
    need(
        isinstance(devices, list) and len(devices) == 1,
        "stage-A currently supports exactly one GPU UUID; "
        "cross-device is NOT supported",
    )
    gpu_uuid(devices[0])
    need(integer(raw["samples"], 1, 12), "samples must be 1..12")
    need(integer(raw["max_tokens"], 1, 256), "max_tokens must be 1..256")
    need(integer(raw["seed"], 0, 2**32 - 13), "seed out of range")
    need(integer(raw["cache_capacity"], 64, 4096), "cache_capacity must be 64..4096")
    need(
        number(raw["max_seconds"], 1) and raw["max_seconds"] <= 3600,
        "max_seconds must be 1..3600",
    )
    if raw["backend"] == "graph":
        need(raw["samples"] >= 2, "graph reuse requires at least two samples")
    knobs = raw["behavior"]
    need(
        isinstance(knobs, dict) and not set(knobs) - set(DEFAULT_BEHAVIOR),
        "unknown behavior keys (safety/resource overrides prohibited)",
    )
    behavior = dict(DEFAULT_BEHAVIOR, **knobs)
    for name, default in DEFAULT_BEHAVIOR.items():
        value = behavior[name]
        if type(default) is bool:
            need(type(value) is bool, f"{name}: expected bool")
        elif type(default) is int:
            need(integer(value, 0, 65536), f"{name}: expected bounded nonnegative int")
        elif type(default) is float:
            need(number(value), f"{name}: expected finite nonnegative number")
        else:
            need(isinstance(value, str), f"{name}: expected string")
    for name in ("gamma", "gamma1", "gamma2"):
        need(behavior[name] <= 32, f"{name}: maximum 32")
    for name in ("top_p", "small_draft_threshold", "draft_target_threshold"):
        need(behavior[name] <= 1, f"{name}: maximum 1")
    for name in (
        "edge_cloud_bandwidth",
        "edge_end_bandwidth",
        "cloud_end_bandwidth",
        "min_bandwidth_mbps",
    ):
        need(behavior[name] > 0, f"{name}: must be positive")
    enums = {
        "prob_payload_bits": (4, 8, 16, 32),
        "comm_round_trip_mode": ("per_round", "per_transfer"),
        "comm_bw_model": ("instant", "fluid"),
        "arp_stop_mode": ("cumulative", "per_token"),
        "rl_action_space": ("topk_thr", "topk_thr_gamma"),
        "state_bw_scaling": ("linear", "log"),
        "state_latency_scaling": ("linear", "centi", "log"),
        "model_dtype": ("bf16", "fp16", "fp32"),
    }
    for name, choices in enums.items():
        need(behavior[name] in choices, f"{name}: unsupported value")
    for name in ("graph_verify_sizes", "rl_gamma_candidates"):
        text = behavior[name].split(",")
        need(
            all(x.isdigit() and 1 <= int(x) <= 128 for x in text),
            f"{name}: invalid sizes",
        )
    return dict(raw, behavior=behavior)


def requirements(mode):
    """Load the lightweight authoritative table without importing src/__init__."""
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "src" / "mode_features.py"
    spec = importlib.util.spec_from_file_location("_stage_a_mode_features", path)
    module = importlib.util.module_from_spec(spec)
    # dataclasses on supported Python versions need a module entry.
    import sys

    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        result = vars(module.get_mode_spec(mode)).copy()
    finally:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
    return result


def roles_for(features):
    models = (
        ["draft", "target"]
        if features["models"] == "dual"
        else ["little", "draft", "target"]
    )
    heads = {
        "none": [],
        "draft_target": ["draft_target"],
        "both": ["small_draft", "draft_target"],
    }[features["acc_head"]]
    rl = [name for name in ("main", "little") if features[f"uses_{name}_rl"]]
    return {"models": models, "heads": heads, "rl": rl}


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def validate_assets(assets, features, *, verify_files=True):
    keys(
        assets,
        {"schema_version", "models", "heads", "rl", "tokenizer_vocab_sha256"},
        "assets",
    )
    need(
        type(assets["schema_version"]) is int and assets["schema_version"] == 2,
        "assets require schema_version 2",
    )
    need(sha(assets["tokenizer_vocab_sha256"]), "invalid tokenizer digest")
    for group, roles in roles_for(features).items():
        keys(
            assets[group], roles, f"assets.{group} (required roles; no silent fallback)"
        )
        for role, entry in assets[group].items():
            keys(
                entry,
                {"path", "files", *(["entrypoint"] if group == "rl" else [])},
                f"assets.{group}.{role}",
            )
            need(
                isinstance(entry["path"], str) and Path(entry["path"]).is_absolute(),
                "asset path must be absolute local directory",
            )
            root = Path(entry["path"])
            need(
                isinstance(entry["files"], list) and bool(entry["files"]),
                "asset files required",
            )
            declared = set()
            for item in entry["files"]:
                keys(item, {"name", "size", "sha256"}, "asset file")
                name = item["name"]
                need(
                    isinstance(name, str)
                    and name
                    and not Path(name).is_absolute()
                    and ".." not in Path(name).parts
                    and Path(name).as_posix() == name
                    and name != "."
                    and name not in declared,
                    "invalid/duplicate asset filename",
                )
                declared.add(name)
                need(
                    integer(item["size"]) and sha(item["sha256"]),
                    "invalid asset size/sha256",
                )
                if verify_files:
                    target = root / name
                    need(
                        target.is_file() and target.stat().st_size == item["size"],
                        f"asset missing/size mismatch: {target}",
                    )
                    need(
                        file_sha(target) == item["sha256"],
                        f"asset sha256 mismatch: {target}",
                    )
            if group == "rl":
                need(entry["entrypoint"] in declared, "RL entrypoint must be declared")
            if verify_files:
                need(root.is_dir(), f"asset directory missing: {root}")
                actual = set()
                for path in root.rglob("*"):
                    need(
                        not (path.is_symlink() and path.is_dir()),
                        "symlink asset directories unsupported",
                    )
                    if path.is_file() or path.is_symlink():
                        actual.add(path.relative_to(root).as_posix())
                need(
                    actual == declared,
                    f"undeclared/missing asset files in {root}: {actual ^ declared}",
                )
    return assets


def validate_metrics(metrics):
    """Validate each producer record before comparing, including equal timings.

    Required keys freeze get_empty_metrics' common shape; per-role wall times
    are optional extensions, not present in every algorithm's producer.
    """
    counts = set(
        (
            "generated_tokens little_forward_times draft_forward_times "
            "target_forward_times "
            "little_generated_tokens draft_generated_tokens little_accepted_tokens "
            "draft_accepted_tokens loop_times"
        ).split()
    )
    timings = set(
        (
            "wall_time throughput computation_time little_computation_time "
            "draft_computation_time target_computation_time "
            "arp_overhead_time dra_overhead_time"
        ).split()
    )
    optional_timings = {"little_wall_time", "draft_wall_time", "target_wall_time"}
    scalars = set(
        (
            "each_loop_draft_tokens comm_energy avg_top_k avg_draft_len "
            "edge_cloud_data_bytes edge_end_data_bytes cloud_end_data_bytes "
            "communication_time edge_end_comm_time queuing_time"
        ).split()
    )
    histories = set(
        (
            "edge_cloud_bandwidth_history edge_cloud_ntt_history "
            "edge_cloud_topk_history "
            "edge_cloud_draft_len_history little_entropy_history draft_entropy_history "
            "little_accept_rate_history draft_accept_rate_history "
            "little_accepted_vocab_rank_history draft_accepted_vocab_rank_history "
            "little_accepted_in_transfer_topk_history "
            "draft_accepted_in_transfer_topk_history "
            "little_accepted_transfer_topk_rank_history "
            "draft_accepted_transfer_topk_rank_history "
            "comm_trace_edge_cloud"
        ).split()
    )
    required = counts | timings | scalars | histories | {"accuracy", "connect_times"}
    need(isinstance(metrics, dict), "metrics must be an object")
    missing = required - metrics.keys()
    need(not missing, f"missing required metrics: {sorted(missing)}")
    for field in counts:
        need(
            integer(metrics[field]), f"invalid metric {field}: nonnegative int required"
        )
    for field in timings | scalars | (optional_timings & metrics.keys()):
        need(
            number(metrics[field]),
            f"invalid metric {field}: nonnegative finite number required",
        )
    for field in histories:
        need(isinstance(metrics[field], list), f"invalid metric {field}: list required")
    connections = metrics["connect_times"]
    need(
        isinstance(connections, dict)
        and set(connections) <= {"edge_cloud", "edge_end", "cloud_end"}
        and all(integer(x) for x in connections.values()),
        "invalid connect_times",
    )
    # accuracy has the producer's Optional[Any] type; strict_json already checked it.


def validate_snapshot(snapshot):
    keys(
        snapshot, {"git_sha", "source_status", "submodules", "files"}, "source snapshot"
    )
    need(
        isinstance(snapshot["files"], dict) and bool(snapshot["files"]),
        "empty source snapshot",
    )
    for path, value in snapshot["files"].items():
        need(
            not Path(path).is_absolute() and ".." not in Path(path).parts,
            "source path must be relative",
        )
        need(value == "deleted" or sha(value), "invalid source digest")
    need(
        any(p.startswith("src/") for p in snapshot["files"]),
        "source snapshot lacks src",
    )
    need(
        any(p.startswith("tools/") for p in snapshot["files"]),
        "source snapshot lacks tools",
    )
    for name in ("git_sha", "source_status", "submodules"):
        need(isinstance(snapshot[name], str), f"invalid snapshot {name}")


def graph_coverage(samples, requested):
    """Derive coverage from successful wrapped calls, not flags/object existence."""
    seen = {}
    reused = []
    replay_count = 0
    for sample in samples:
        need(isinstance(sample["graph_events"], list), "graph events must be a list")
        for event in sample["graph_events"]:
            keys(
                event,
                {
                    "object",
                    "kind",
                    "role",
                    "capture_count",
                    "cache",
                    "max_len",
                    "verify_sizes",
                },
                "graph event",
            )
            need(
                integer(event["object"], 1) and integer(event["cache"], 1),
                "invalid weak identity ordinal",
            )
            need(
                event["kind"] in ("step", "verify", "fallback", "reset", "prefill"),
                "invalid graph event",
            )
            need(event["role"] in ("little", "draft", "target"), "unknown graph role")
            need(
                integer(event["capture_count"]) and integer(event["max_len"], 1),
                "invalid graph capture/capacity",
            )
            need(
                isinstance(event["verify_sizes"], list)
                and all(integer(x, 2) for x in event["verify_sizes"]),
                "invalid verify sizes",
            )
            state = seen.setdefault(
                event["object"],
                {
                    "samples": set(),
                    "resets": {},
                    "reset_before_first_replay": set(),
                    "counts": set(),
                    "caches": set(),
                    "role": event["role"],
                },
            )
            need(state["role"] == event["role"], "graph identity changed role")
            sample_id = sample["sample_id"]
            signature = (
                event["cache"],
                event["capture_count"],
                event["max_len"],
                tuple(event["verify_sizes"]),
            )
            if event["kind"] == "reset":
                state["resets"][sample_id] = signature
            if event["kind"] in ("step", "verify"):
                need(event["capture_count"] > 0, "replay without capture")
                # Capture the causal relation at this sample's FIRST replay.
                # A later reset cannot retroactively justify earlier execution.
                if (
                    sample_id not in state["samples"]
                    and state["resets"].get(sample_id) == signature
                ):
                    state["reset_before_first_replay"].add(sample_id)
                state["samples"].add(sample_id)
                state["counts"].add(event["capture_count"])
                state["caches"].add(event["cache"])
                replay_count += 1
    for identity, state in seen.items():
        ordered = sorted(state["samples"])
        if (
            len(ordered) >= 2
            and set(ordered[1:]) <= state["reset_before_first_replay"]
            and len(state["counts"]) == len(state["caches"]) == 1
        ):
            reused.append(identity)
    if requested:
        need(
            replay_count > 0 and bool(reused),
            "graph requested but no real replay/reset/cross-sample reuse evidence",
        )
    else:
        need(not seen, "eager request unexpectedly executed graph runner")
    return {"replay_count": replay_count, "reused_runners": sorted(reused)}


def validate_payload(payload, *, expect_samples=None, source_digest=None):
    keys(
        payload,
        {
            "schema_version",
            "profile",
            "status",
            "configuration",
            "code",
            "assets",
            "effective",
            "runtime",
            "samples",
            "graph",
            "limitations",
        },
        "stage-A artifact",
    )
    strict_json(payload)
    need(
        type(payload["schema_version"]) is int and payload["schema_version"] == 2,
        "expected schema_version 2",
    )
    need(
        payload["profile"] == "stage-a" and payload["status"] == "observed",
        "not a completed stage-A observation",
    )
    need(payload["limitations"] == LIMITATIONS, "missing evidence limitations")
    config = validate_config(payload["configuration"])
    need(
        config == payload["configuration"],
        "configuration must contain resolved behavior",
    )
    code = payload["code"]
    keys(code, {"before", "after", "sha256", "endpoint_only"}, "code")
    validate_snapshot(code["before"])
    validate_snapshot(code["after"])
    need(code["before"] == code["after"], "source endpoints changed")
    need(
        code["endpoint_only"] is True and code["sha256"] == digest(code["before"]),
        "invalid source endpoint identity",
    )
    if source_digest is not None:
        need(
            sha(source_digest) and code["sha256"] == source_digest,
            "source digest differs from expected identity",
        )
    effective = payload["effective"]
    keys(
        effective,
        {
            "arguments",
            "call_kwargs",
            "binding",
            "consumption",
            "mode_features",
            "models",
            "heads",
            "rl",
            "checkpoint_provenance",
            "eos_ids",
        },
        "effective",
    )
    features = effective["mode_features"]
    need(
        features == requirements(config["mode"]),
        "mode_features differ from current contract",
    )
    consumption = effective["consumption"]
    keys(
        consumption,
        {"source", "accounting", "depth", "prob_payload_bits"},
        "consumption",
    )
    need(
        consumption["source"] == "runtime_introspection_not_branch_coverage",
        "runtime consumption evidence missing",
    )
    keys(
        consumption["accounting"],
        {
            "comm_round_trip_mode",
            "charge_residual_payload",
            "transfer_top_k_cap",
            "force_full_vocab_transfer",
        },
        "accounting consumption",
    )
    keys(consumption["depth"], {"gamma", "gamma1", "gamma2"}, "depth consumption")
    need(
        all(
            v in ("consumed", "not_consumed")
            for group in ("accounting", "depth")
            for v in consumption[group].values()
        ),
        "invalid consumption label",
    )
    bits = consumption["prob_payload_bits"]
    keys(
        bits,
        {"implementation_reads", "supports_lossy", "semantics"},
        "probability consumption",
    )
    need(
        type(bits["implementation_reads"]) is bool
        and bits["supports_lossy"] is features["supports_prob_bits"],
        "probability support mismatch",
    )
    expected_semantics = (
        "not_consumed"
        if not bits["implementation_reads"]
        else "lossy_configured"
        if config["behavior"]["prob_payload_bits"] < 16
        else "pass_through"
    )
    need(
        bits["semantics"] == expected_semantics,
        "probability bits are not actual wire-width labels",
    )
    need(
        config["behavior"]["prob_payload_bits"] >= 16
        or (bits["implementation_reads"] and bits["supports_lossy"]),
        "lossy bits not supported/consumed",
    )
    validate_assets(payload["assets"], features, verify_files=False)
    arguments = effective["arguments"]
    need(isinstance(arguments, dict), "effective arguments required")
    need(
        arguments.get("disable_rl_update") is True
        and arguments.get("disable_eos_stop") is False,
        "unsafe effective flags",
    )
    need(
        arguments.get("eval_mode") == config["mode"]
        and arguments.get("use_cuda_graph") == (config["backend"] == "graph"),
        "effective mode/backend mismatch",
    )
    for key, value in config["behavior"].items():
        need(arguments.get(key) == value, f"effective behavior mismatch: {key}")
    for key in ("max_tokens", "seed"):
        need(arguments.get(key) == config[key], f"effective budget mismatch: {key}")
    need(
        arguments.get("probe_cache_max_length") == config["cache_capacity"],
        "capacity mismatch",
    )
    need(
        arguments.get("use_rl_adapter")
        is bool(features["uses_main_rl"] or features["uses_little_rl"]),
        "required RL silently disabled",
    )
    need(
        arguments.get("rl_init_strategy") == "resume"
        and arguments.get("rl_output_dir") is None,
        "unsafe RL initialization/output",
    )
    need(
        arguments.get("use_stochastic_comm") is False
        and arguments.get("use_early_stopping") is False,
        "unexpected stopping/stochastic communication",
    )
    for role, entry in payload["assets"]["models"].items():
        need(
            arguments.get(f"{role}_model") == entry["path"]
            and arguments.get(f"{role}_quantization") == "none",
            "model asset/quantization mismatch",
        )
    expected_kwargs = {
        "transfer_top_k": config["behavior"]["transfer_top_k"],
        "use_precise_comm_sim": False,
        "use_stochastic_comm": False,
        "ntt_ms_edge_cloud": config["behavior"]["ntt_ms_edge_cloud"],
        "ntt_ms_edge_end": config["behavior"]["ntt_ms_edge_end"],
        "use_early_stopping": False,
    }
    need(effective["call_kwargs"] == expected_kwargs, "actual method kwargs mismatch")
    provenance = effective["checkpoint_provenance"]
    need(isinstance(provenance, dict), "checkpoint provenance missing")
    for role, entry in payload["assets"]["heads"].items():
        pkey = (
            "acc_head" if features["acc_head"] == "draft_target" else f"acc_head_{role}"
        )
        argkey = (
            "acc_head_path"
            if features["acc_head"] == "draft_target"
            else f"{role}_acc_head_path"
        )
        need(
            arguments.get(argkey) == entry["path"], "head argument provenance mismatch"
        )
        recorded = provenance.get(pkey, {})
        expected = {
            f["name"]: f["sha256"]
            for f in entry["files"]
            if ".cache" not in Path(f["name"]).parts
        }
        need(
            recorded.get("source") in ("manifest", "untracked")
            and {k: v.get("sha256") for k, v in recorded.get("files", {}).items()}
            == expected,
            "head checkpoint provenance mismatch",
        )
    for role, entry in payload["assets"]["rl"].items():
        path = str(Path(entry["path"]) / entry["entrypoint"])
        expected_sha = next(
            f["sha256"] for f in entry["files"] if f["name"] == entry["entrypoint"]
        )
        need(
            arguments.get(f"{role}_rl_path")
            == arguments.get(f"{role}_rl_best_path")
            == path,
            "RL fallback path mismatch",
        )
        need(
            provenance.get(f"rl_{role}", {}).get("sha256") == expected_sha,
            "RL checkpoint provenance mismatch",
        )
    eos = effective["eos_ids"]
    need(
        isinstance(eos, list) and eos and all(integer(x) for x in eos),
        "EOS ids required",
    )
    for group, roles in roles_for(features).items():
        keys(effective[group], roles, f"effective {group}")
        for role in roles:
            row = effective[group][role]
            required = {
                "models": {"path", "devices", "gpu_uuid", "dtype", "hf_device_map"},
                "heads": {"path", "stop_mode", "threshold", "device"},
                "rl": {
                    "path",
                    "loaded_path",
                    "sha256",
                    "device",
                    "action_space",
                    "gamma_candidates",
                    "topk_candidates",
                    "threshold_candidates",
                    "update_count",
                },
            }
            keys(row, required[group], f"effective.{group}.{role}")
            entry = payload["assets"][group][role]
            need(
                row["path"] == entry["path"], f"{group}.{role} provenance path mismatch"
            )
            if group == "models":
                need(
                    row["devices"] == ["cuda:0"]
                    and row["gpu_uuid"] == config["gpu_uuids"][0],
                    "actual model placement mismatch",
                )
                expected_dtype = {
                    "bf16": "torch.bfloat16",
                    "fp16": "torch.float16",
                    "fp32": "torch.float32",
                }[arguments["model_dtype"]]
                need(row["dtype"] == expected_dtype, "effective model dtype mismatch")
                need(
                    isinstance(row["hf_device_map"], dict)
                    and all(
                        (type(device) is int and device == 0)
                        or (isinstance(device, str) and device in ("cuda", "cuda:0"))
                        for device in row["hf_device_map"].values()
                    ),
                    "hf_device_map contradicts single-GPU placement",
                )
            elif group == "heads":
                need(row["device"] == "cuda:0", "head device mismatch")
                need(
                    row["stop_mode"] == arguments["arp_stop_mode"],
                    "head stop_mode mismatch",
                )
                need(
                    number(row["threshold"])
                    and row["threshold"] <= 1
                    and row["threshold"] == arguments[f"{role}_threshold"],
                    "head initial threshold mismatch",
                )
            else:
                expected = str(Path(entry["path"]) / entry["entrypoint"])
                need(
                    row["loaded_path"] == expected,
                    "RL checkpoint was not loaded from manifest entrypoint",
                )
                expected_sha = next(
                    f["sha256"]
                    for f in entry["files"]
                    if f["name"] == entry["entrypoint"]
                )
                need(row["sha256"] == expected_sha, "effective RL sha256 mismatch")
                need(row["device"] in ("cuda", "cuda:0"), "RL device mismatch")
                need(
                    row["action_space"] == arguments["rl_action_space"],
                    "RL action_space mismatch",
                )
                for field in (
                    "gamma_candidates",
                    "topk_candidates",
                    "threshold_candidates",
                ):
                    grid = row[field]
                    need(
                        isinstance(grid, list) and bool(grid),
                        f"RL {field} must be nonempty",
                    )
                    valid = (
                        all(number(x) and x <= 1 for x in grid)
                        if field == "threshold_candidates"
                        else all(integer(x, 1) for x in grid)
                    )
                    need(valid, f"invalid RL {field} values")
                    need(len(set(grid)) == len(grid), f"duplicate RL {field}")
                need(
                    row["gamma_candidates"]
                    == [int(x) for x in arguments["rl_gamma_candidates"].split(",")],
                    "RL gamma grid mismatch",
                )
                # Checkpoint-restored top-k/threshold grids may differ from CLI
                # defaults; their actual values are mandatory, not reconstructed.
                need(integer(row["update_count"]), "invalid RL update_count")
    binding = effective["binding"]
    need(
        isinstance(binding, dict) and config["mode"] in binding.get("aliases", []),
        "registry binding missing",
    )
    for name in ("module", "qualname", "file"):
        need(
            isinstance(binding.get(name), str) and binding[name],
            "binding source missing",
        )
    need(
        binding["file"] in code["before"]["files"], "binding is not in source snapshot"
    )
    runtime = payload["runtime"]
    keys(
        runtime,
        {
            "python",
            "invocation",
            "packages",
            "cuda",
            "attention_backend",
            "gpu_uuid",
            "gpu_name",
            "idle_before",
            "exclusive_after",
            "offline",
        },
        "runtime",
    )
    for key in ("python", "cuda", "attention_backend", "gpu_name"):
        need(isinstance(runtime[key], str) and runtime[key], f"runtime {key} missing")
    keys(runtime["invocation"], {"argv", "executable"}, "invocation")
    need(
        isinstance(runtime["invocation"]["argv"], list)
        and all(isinstance(x, str) for x in runtime["invocation"]["argv"])
        and isinstance(runtime["invocation"]["executable"], str),
        "invalid invocation",
    )
    need(
        runtime["offline"]
        == {
            key: "1"
            for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")
        },
        "offline environment missing",
    )
    need(
        isinstance(runtime, dict) and runtime.get("gpu_uuid") == config["gpu_uuids"][0],
        "runtime UUID mismatch",
    )
    need(
        runtime.get("idle_before") is True and runtime.get("exclusive_after") is True,
        "GPU exclusivity evidence missing",
    )
    need(
        isinstance(runtime.get("packages"), dict)
        and all(runtime["packages"].get(p) for p in ("torch", "transformers")),
        "package versions missing",
    )
    samples = payload["samples"]
    need(
        isinstance(samples, list) and len(samples) == config["samples"],
        "sample count mismatch/empty run",
    )
    if expect_samples is not None:
        need(
            integer(expect_samples, 1) and len(samples) == expect_samples,
            "expected sample count mismatch",
        )
    total_target = 0
    total_bytes = 0
    usage = {
        f"{group}.{role}": 0
        for group in ("models", "heads", "rl")
        for role in roles_for(features)[group]
    }
    prompts = set()
    for index, sample in enumerate(samples):
        keys(
            sample,
            {
                "sample_id",
                "prompt",
                "prompt_sha256",
                "seed",
                "prefix_ids",
                "gen_ids",
                "metrics",
                "graph_events",
                "resource_calls",
            },
            "sample",
        )
        need(
            type(sample["sample_id"]) is int
            and sample["sample_id"] == index
            and sample["seed"] == config["seed"] + index,
            "sample position/seed mismatch",
        )
        prompt = sample["prompt"]
        need(
            isinstance(prompt, str) and prompt and prompt not in prompts,
            "empty/duplicate prompt",
        )
        prompts.add(prompt)
        need(
            sample["prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest(),
            "prompt digest mismatch",
        )
        for key in ("prefix_ids", "gen_ids"):
            need(
                isinstance(sample[key], list)
                and sample[key]
                and all(integer(t) for t in sample[key]),
                f"invalid {key}",
            )
        gen = sample["gen_ids"]
        need(len(gen) <= config["max_tokens"], "generation exceeds budget")
        need(
            len(gen) == config["max_tokens"] or set(gen).intersection(eos),
            "short generation without EOS",
        )
        metrics = sample["metrics"]
        validate_metrics(metrics)
        need(
            isinstance(metrics, dict)
            and integer(metrics.get("generated_tokens"), 1)
            and metrics["generated_tokens"] == len(gen),
            "generated_tokens mismatch",
        )
        for role in ("little", "draft", "target"):
            need(
                integer(metrics.get(f"{role}_forward_times")),
                "missing/invalid forward count",
            )
        total_target += metrics["target_forward_times"]
        for link in ("edge_cloud", "edge_end", "cloud_end"):
            need(
                number(metrics.get(f"{link}_data_bytes")), "invalid communication bytes"
            )
        trace = metrics.get("comm_trace_edge_cloud")
        need(isinstance(trace, list), "communication trace missing")
        for row in trace:
            need(
                isinstance(row, list)
                and len(row) == 2
                and number(row[0])
                and integer(row[1]),
                "invalid communication trace",
            )
        amount = metrics["edge_cloud_data_bytes"]
        need(
            math.isclose(sum(x[0] for x in trace), amount, rel_tol=0, abs_tol=1e-6),
            "communication trace sum mismatch",
        )
        total_bytes += amount
        calls = sample["resource_calls"]
        keys(calls, usage, "resource calls")
        for key in usage:
            need(integer(calls[key]), "invalid resource call count")
            usage[key] += calls[key]
    need(total_target > 0, "no real target forward recorded")
    if config["mode"] != "sd":
        need(total_bytes > 0, "no communication recorded for distributed protocol")
    need(usage["models.target"] > 0, "no successful target model forward hook recorded")
    need(
        all(value > 0 for key, value in usage.items() if not key.startswith("models.")),
        "required ARP/RL resource never executed",
    )
    need(
        payload["graph"] == graph_coverage(samples, config["backend"] == "graph"),
        "graph summary mismatch",
    )
    return payload


def compare_stage_a(
    base,
    candidate,
    *,
    expect_samples,
    base_source,
    candidate_source,
    comparison="observations",
):
    report = {
        "schema_version": 2,
        "verdict": "INVALID",
        "case": "stage-a",
        "samples": expect_samples,
        "errors": [],
        "mismatches": [],
        "ignored_timing": [],
        "observations": [],
        "limitations": LIMITATIONS,
        "comparison": comparison,
    }
    try:
        need(comparison in ("observations", "exact-replay"), "unsupported comparison")
        need(
            sha(base_source) and sha(candidate_source),
            "v2 requires explicit 64-hex source digests, not HEAD revisions",
        )
        validate_payload(base, expect_samples=expect_samples, source_digest=base_source)
        validate_payload(
            candidate, expect_samples=expect_samples, source_digest=candidate_source
        )
        for key in ("configuration", "assets"):
            need(base[key] == candidate[key], f"incomparable {key}")
        for key in (
            "eos_ids",
            "models",
            "heads",
            "rl",
            "mode_features",
            "consumption",
            "arguments",
            "call_kwargs",
        ):
            need(
                base["effective"][key] == candidate["effective"][key],
                f"incomparable effective.{key}",
            )
        need(
            base["runtime"]["packages"] == candidate["runtime"]["packages"],
            "incomparable packages",
        )
        for key in (
            "python",
            "cuda",
            "attention_backend",
            "gpu_uuid",
            "gpu_name",
            "offline",
        ):
            need(
                base["runtime"][key] == candidate["runtime"][key],
                f"incomparable runtime.{key}",
            )
        if comparison == "exact-replay":
            need(
                base_source == candidate_source,
                "exact-replay requires identical source endpoints",
            )
            # Invocation includes unique output names, not an execution behavior.
            need(
                {k: v for k, v in base["runtime"].items() if k != "invocation"}
                == {k: v for k, v in candidate["runtime"].items() if k != "invocation"},
                "exact-replay requires identical runtime",
            )
        # Imported only here; schema-1's strict equality/timing policy stays unchanged.
        try:
            from .compare_refactor_smoke import TIMING_FIELDS, _same, _token_mismatch
        except ImportError:
            from compare_refactor_smoke import TIMING_FIELDS, _same, _token_mismatch
        for left, right in zip(base["samples"], candidate["samples"], strict=True):
            for key in ("sample_id", "prompt", "prompt_sha256", "seed", "prefix_ids"):
                need(_same(left[key], right[key]), f"incomparable sample {key}")
            deltas = []
            if not _same(left["gen_ids"], right["gen_ids"]):
                deltas.append(_token_mismatch(left, right))
            need(
                set(left["metrics"]) == set(right["metrics"]), "metric key sets differ"
            )
            for key, value in left["metrics"].items():
                other = right["metrics"][key]
                if not _same(value, other):
                    item = {
                        "kind": "metric",
                        "sample_id": left["sample_id"],
                        "field": key,
                        "base": value,
                        "candidate": other,
                    }
                    if key in TIMING_FIELDS:
                        need(number(value) and number(other), "invalid timing metric")
                        report["ignored_timing"].append(item)
                    else:
                        deltas.append(item)
            report["observations"].extend(deltas)
            if comparison == "exact-replay":
                report["mismatches"].extend(deltas)
    except (StageAError, KeyError, TypeError) as exc:
        report["errors"].append(str(exc))
        return report
    report["verdict"] = (
        ("FAIL" if report["mismatches"] else "PASS")
        if comparison == "exact-replay"
        else "OBSERVED"
    )
    return report
