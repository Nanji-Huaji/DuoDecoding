#!/usr/bin/env python3
"""严格对照两份 refactor smoke JSON（main vs candidate）。

只认 schema_version=1：固定 prompt / 48 token 序列 / per-sample metrics。
按位置精确配对，不使用 zip/min 截断；除显式 timing 字段外，metrics 的键集合
和 deep type+value 必须完全一致。退出码 PASS 0 / FAIL 1 / INVALID 2。

用法：
    .venv/bin/python tools/compare_refactor_smoke.py \
        --base base.json --candidate candidate.json --expect-samples 12 \
        --base-revision <40sha> --candidate-revision <40sha> [--json-out report.json]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
TOP_KEYS = frozenset(
    "schema_version case revision code configuration runtime assets samples".split()
)
CODE_KEYS = frozenset("definition_module definition_file source_sha256".split())
RUNTIME_KEYS = frozenset(
    "python torch cuda transformers gpu_name gpu_uuid attention_backend".split()
)
SAMPLE_KEYS = frozenset(
    "sample_id prompt prompt_sha256 seed prefix_ids gen_ids metrics".split()
)
# schema v1: all non-timing fields emitted by src.metrics.get_empty_metrics().
# Keep this literal so the comparator remains standalone across archived revisions.
REQUIRED_METRICS = frozenset(
    "generated_tokens little_forward_times draft_forward_times target_forward_times "
    "little_generated_tokens draft_generated_tokens little_accepted_tokens "
    "draft_accepted_tokens loop_times each_loop_draft_tokens comm_energy accuracy "
    "avg_top_k avg_draft_len "
    "edge_cloud_data_bytes edge_end_data_bytes cloud_end_data_bytes "
    "communication_time edge_end_comm_time queuing_time connect_times "
    "edge_cloud_bandwidth_history edge_cloud_ntt_history edge_cloud_topk_history "
    "edge_cloud_draft_len_history little_entropy_history draft_entropy_history "
    "little_accept_rate_history draft_accept_rate_history "
    "little_accepted_vocab_rank_history draft_accepted_vocab_rank_history "
    "little_accepted_in_transfer_topk_history draft_accepted_in_transfer_topk_history "
    "little_accepted_transfer_topk_rank_history "
    "draft_accepted_transfer_topk_rank_history "
    "comm_trace_edge_cloud".split()
)
TIMING_FIELDS = frozenset(
    "wall_time throughput computation_time little_computation_time "
    "draft_computation_time target_computation_time arp_overhead_time "
    "dra_overhead_time little_wall_time draft_wall_time target_wall_time".split()
)
LINK_KEYS = frozenset({"edge_cloud", "edge_end", "cloud_end"})
CONFIG_KEYS = frozenset(
    "eval_mode temp transfer_top_k prob_payload_bits comm_round_trip_mode "
    "charge_residual_payload gamma gamma1 gamma2 max_tokens samples seed "
    "use_cuda_graph use_rl_adapter disable_rl_update disable_eos_stop "
    "eos_token_ids model_dtype little_quantization draft_quantization "
    "target_quantization top_k top_p transfer_top_k_cap force_full_vocab_transfer "
    "edge_cloud_bandwidth edge_end_bandwidth cloud_end_bandwidth "
    "min_bandwidth_mbps comm_bw_model ntt_ms_edge_cloud ntt_ms_edge_end "
    "batch_delay use_stochastic_comm use_early_stopping".split()
)
CASE_PROFILES = {
    "legacy_static": {
        "temp": 0.0,
        "transfer_top_k": 32,
        "prob_payload_bits": 16,
        "comm_round_trip_mode": "per_transfer",
        "charge_residual_payload": False,
    },
    "honest_int8": {
        "temp": 0.7,
        "transfer_top_k": 8,
        "prob_payload_bits": 8,
        "comm_round_trip_mode": "per_round",
        "charge_residual_payload": True,
    },
}
SHA40 = re.compile(r"[0-9a-f]{40}\Z")
SHA64 = re.compile(r"[0-9a-f]{64}\Z")


class Invalid(Exception):
    """schema/配对/身份不合法；对应退出码 2。"""


def _need(ok: bool, path: str, message: str) -> None:
    if not ok:
        raise Invalid(f"{path}: {message}")


def _int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _num(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _sha(value: Any, length: int, path: str) -> None:
    pattern = SHA40 if length == 40 else SHA64
    _need(
        isinstance(value, str) and bool(pattern.fullmatch(value)),
        path,
        f"expected lowercase {length}-hex SHA, got {value!r}",
    )


def _text(value: Any, path: str) -> None:
    _need(
        isinstance(value, str) and bool(value.strip()),
        path,
        "expected non-empty string",
    )


def _keys(value: Any, keys: frozenset[str], path: str) -> None:
    _need(isinstance(value, dict), path, "expected object")
    _need(set(value) == keys, path, f"keys must be {sorted(keys)}")


def _json(value: Any, path: str) -> None:
    """JSON 形状 + 有限数；bool 不算数字。"""
    if value is None or isinstance(value, (str, bool)) or _int(value):
        return
    if isinstance(value, float):
        _need(math.isfinite(value), path, f"non-finite number {value!r}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _json(item, f"{path}[{index}]")
    elif isinstance(value, dict):
        for key, item in value.items():
            _need(isinstance(key, str), path, f"non-string key {key!r}")
            _json(item, f"{path}.{key}")
    else:
        raise Invalid(f"{path}: unsupported JSON value {type(value).__name__}")


def _same(left: Any, right: Any) -> bool:
    """deep type+value equality；1/True/1.0 互不相等。"""
    if type(left) is not type(right):
        return False
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _same(left[i], right[i]) for i in range(len(left))
        )
    if isinstance(left, dict):
        return set(left) == set(right) and all(_same(left[k], right[k]) for k in left)
    return left == right


def _first_diff(left: Any, right: Any, path: str = "") -> str:
    if type(left) is not type(right):
        return (
            f"{path or '<root>'}: type {type(left).__name__} != {type(right).__name__}"
        )
    if isinstance(left, dict):
        if set(left) != set(right):
            return f"{path or '<root>'}: keys {sorted(left)} != {sorted(right)}"
        for key in sorted(left):
            found = _first_diff(left[key], right[key], f"{path}.{key}" if path else key)
            if found:
                return found
        return ""
    if isinstance(left, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)} != {len(right)}"
        for index in range(len(left)):
            found = _first_diff(left[index], right[index], f"{path}[{index}]")
            if found:
                return found
        return ""
    return "" if left == right else f"{path}: {left!r} != {right!r}"


def _check_code(value: Any, label: str) -> None:
    path = f"{label}.code"
    _keys(value, CODE_KEYS, path)
    _text(value["definition_module"], f"{path}.definition_module")
    _text(value["definition_file"], f"{path}.definition_file")
    file_path = Path(value["definition_file"])
    _need(
        not file_path.is_absolute() and ".." not in file_path.parts,
        f"{path}.definition_file",
        "expected repo-relative path",
    )
    _sha(value["source_sha256"], 64, f"{path}.source_sha256")


def _check_runtime(value: Any, label: str) -> None:
    path = f"{label}.runtime"
    _keys(value, RUNTIME_KEYS, path)
    for key in RUNTIME_KEYS:
        _text(value[key], f"{path}.{key}")


def _check_assets(value: Any, label: str) -> None:
    path = f"{label}.assets"
    _need(isinstance(value, dict), path, "expected object")
    required = {
        "schema_version",
        "models",
        "tokenizer_vocab_sha256",
        "submodule_commit",
    }
    _need(required <= set(value), path, "missing required keys")
    _need(
        _int(value["schema_version"]) and value["schema_version"] == SCHEMA_VERSION,
        f"{path}.schema_version",
        f"expected {SCHEMA_VERSION}",
    )
    models = value["models"]
    _keys(models, frozenset({"small", "target"}), f"{path}.models")
    for role in ("small", "target"):
        model_path = f"{path}.models.{role}"
        _keys(models[role], frozenset({"path", "files"}), model_path)
        _text(models[role]["path"], f"{model_path}.path")
        files = models[role]["files"]
        _need(
            isinstance(files, list) and bool(files),
            f"{model_path}.files",
            "expected non-empty list",
        )
        for index, item in enumerate(files):
            item_path = f"{model_path}.files[{index}]"
            _keys(item, frozenset({"name", "size", "sha256"}), item_path)
            _text(item["name"], f"{item_path}.name")
            _need(
                _int(item["size"]) and item["size"] >= 0,
                f"{item_path}.size",
                "expected non-negative int",
            )
            _sha(item["sha256"], 64, f"{item_path}.sha256")
    _sha(value["tokenizer_vocab_sha256"], 64, f"{path}.tokenizer_vocab_sha256")
    _text(value["submodule_commit"], f"{path}.submodule_commit")


def _check_connect_times(value: Any, path: str) -> None:
    _keys(value, LINK_KEYS, path)
    for key in LINK_KEYS:
        _need(
            _int(value[key]) and value[key] >= 0,
            f"{path}.{key}",
            "expected non-negative int",
        )


def _check_trace(value: Any, edge_cloud_bytes: float, path: str) -> None:
    _need(isinstance(value, list), path, "expected list")
    for index, entry in enumerate(value):
        item_path = f"{path}[{index}]"
        _need(
            isinstance(entry, list) and len(entry) == 2,
            item_path,
            "expected [bytes, round]",
        )
        _need(
            _num(entry[0]) and entry[0] >= 0,
            f"{item_path}[0]",
            "expected finite non-negative bytes",
        )
        _need(
            _int(entry[1]) and entry[1] >= 0,
            f"{item_path}[1]",
            "expected non-negative round int",
        )
    if edge_cloud_bytes > 0:
        _need(bool(value), path, "must be non-empty when edge_cloud_data_bytes > 0")
    traced = sum(float(entry[0]) for entry in value)
    _need(
        math.isclose(traced, edge_cloud_bytes, rel_tol=0.0, abs_tol=1e-6),
        path,
        f"trace bytes {traced} != edge_cloud_data_bytes {edge_cloud_bytes}",
    )


def _check_config(config: Any, case: str, expect: int) -> None:
    _need(isinstance(config, dict), "configuration", "expected object")
    missing = CONFIG_KEYS - set(config)
    _need(not missing, "configuration", f"missing required keys {sorted(missing)}")

    def cfg(key: str, ok: bool, message: str) -> None:
        _need(ok, f"configuration.{key}", message)

    cfg(
        "eval_mode",
        isinstance(config["eval_mode"], str) and config["eval_mode"] == "tridecoding",
        "expected tridecoding",
    )
    cfg(
        "temp",
        _num(config["temp"]) and config["temp"] >= 0,
        "expected finite non-negative number",
    )
    for key in (
        "transfer_top_k",
        "prob_payload_bits",
        "gamma",
        "gamma1",
        "gamma2",
        "max_tokens",
    ):
        cfg(key, _int(config[key]) and config[key] >= 1, "expected positive int")
    cfg(
        "comm_round_trip_mode",
        isinstance(config["comm_round_trip_mode"], str)
        and config["comm_round_trip_mode"] in {"per_transfer", "per_round"},
        "expected per_transfer/per_round",
    )
    cfg(
        "charge_residual_payload",
        isinstance(config["charge_residual_payload"], bool),
        "expected bool",
    )
    cfg(
        "samples",
        _int(config["samples"]) and config["samples"] == expect,
        f"expected {expect}",
    )
    cfg(
        "seed",
        _int(config["seed"]) and config["seed"] >= 0,
        "expected non-negative int",
    )
    for key in (
        "use_cuda_graph",
        "use_rl_adapter",
        "use_stochastic_comm",
        "use_early_stopping",
    ):
        cfg(key, config[key] is False, "expected False")
    cfg("disable_rl_update", config["disable_rl_update"] is True, "expected True")
    cfg("disable_eos_stop", config["disable_eos_stop"] is False, "expected False")
    eos_ids = config["eos_token_ids"]
    cfg(
        "eos_token_ids",
        isinstance(eos_ids, list) and bool(eos_ids),
        "expected non-empty list",
    )
    cfg(
        "eos_token_ids",
        all(_int(token) and token >= 0 for token in eos_ids),
        "expected non-negative ints",
    )
    cfg("eos_token_ids", len(set(eos_ids)) == len(eos_ids), "duplicate token id")
    cfg(
        "model_dtype",
        isinstance(config["model_dtype"], str)
        and config["model_dtype"] in {"bf16", "fp16", "fp32"},
        "expected bf16/fp16/fp32",
    )
    for key in ("little_quantization", "draft_quantization", "target_quantization"):
        cfg(
            key,
            isinstance(config[key], str) and config[key] in {"auto", "4bit", "none"},
            "expected auto/4bit/none",
        )
    cfg(
        "top_k",
        _int(config["top_k"]) and config["top_k"] >= 0,
        "expected non-negative int",
    )
    cfg(
        "top_p",
        _num(config["top_p"]) and 0.0 <= config["top_p"] <= 1.0,
        "expected number in [0,1]",
    )
    cfg(
        "transfer_top_k_cap",
        _int(config["transfer_top_k_cap"]) and config["transfer_top_k_cap"] >= 0,
        "expected non-negative int",
    )
    cfg(
        "force_full_vocab_transfer",
        isinstance(config["force_full_vocab_transfer"], bool),
        "expected bool",
    )
    for key in (
        "edge_cloud_bandwidth",
        "edge_end_bandwidth",
        "cloud_end_bandwidth",
        "min_bandwidth_mbps",
    ):
        cfg(key, _num(config[key]) and config[key] > 0, "expected positive number")
    cfg(
        "comm_bw_model",
        isinstance(config["comm_bw_model"], str)
        and config["comm_bw_model"] in {"instant", "fluid"},
        "expected instant/fluid",
    )
    for key in ("ntt_ms_edge_cloud", "ntt_ms_edge_end", "batch_delay"):
        cfg(
            key,
            _num(config[key]) and config[key] >= 0,
            "expected finite non-negative number",
        )
    profile = CASE_PROFILES.get(case)
    _need(
        profile is not None,
        "case",
        f"unknown case {case!r}; expected one of {sorted(CASE_PROFILES)}",
    )
    for key, expected in profile.items():
        cfg(
            key, _same(config[key], expected), f"must be {expected!r} for case {case!r}"
        )


def _check_sample(sample: Any, index: int, label: str) -> None:
    path = f"{label}.samples[{index}]"
    _keys(sample, SAMPLE_KEYS, path)
    _need(
        _int(sample["sample_id"]) and sample["sample_id"] >= 0,
        f"{path}.sample_id",
        "expected non-negative int",
    )
    _text(sample["prompt"], f"{path}.prompt")
    _sha(sample["prompt_sha256"], 64, f"{path}.prompt_sha256")
    expected = hashlib.sha256(sample["prompt"].encode("utf-8")).hexdigest()
    _need(
        sample["prompt_sha256"] == expected,
        f"{path}.prompt_sha256",
        "does not match prompt",
    )
    _need(_int(sample["seed"]), f"{path}.seed", "expected int")
    for field in ("prefix_ids", "gen_ids"):
        values = sample[field]
        _need(
            isinstance(values, list) and bool(values),
            f"{path}.{field}",
            "expected non-empty list",
        )
        for pos, token in enumerate(values):
            _need(
                _int(token) and token >= 0,
                f"{path}.{field}[{pos}]",
                "expected non-negative int (not bool)",
            )
    metrics = sample["metrics"]
    _need(
        isinstance(metrics, dict) and bool(metrics),
        f"{path}.metrics",
        "expected non-empty object",
    )
    missing = REQUIRED_METRICS - set(metrics)
    _need(not missing, f"{path}.metrics", f"missing required keys {sorted(missing)}")
    for key in ("little_forward_times", "draft_forward_times", "target_forward_times"):
        _need(
            _int(metrics[key]) and metrics[key] >= 0,
            f"{path}.metrics.{key}",
            "expected non-negative int (not bool)",
        )
    for key in (
        "edge_cloud_data_bytes",
        "edge_end_data_bytes",
        "cloud_end_data_bytes",
        "communication_time",
        "edge_end_comm_time",
        "queuing_time",
    ):
        _need(
            _num(metrics[key]) and metrics[key] >= 0,
            f"{path}.metrics.{key}",
            "expected finite non-negative number",
        )
    _check_connect_times(metrics["connect_times"], f"{path}.metrics.connect_times")
    _check_trace(
        metrics["comm_trace_edge_cloud"],
        metrics["edge_cloud_data_bytes"],
        f"{path}.metrics.comm_trace_edge_cloud",
    )
    _need(
        _int(metrics["generated_tokens"]),
        f"{path}.metrics.generated_tokens",
        "expected int (not bool)",
    )
    _need(
        metrics["generated_tokens"] == len(sample["gen_ids"]),
        f"{path}.metrics.generated_tokens",
        f"{metrics['generated_tokens']} != len(gen_ids) {len(sample['gen_ids'])}",
    )


def _check_payload(payload: Any, label: str, revision: str, expect: int) -> None:
    _keys(payload, TOP_KEYS, label)
    _json(payload, label)
    _need(
        _int(payload["schema_version"]) and payload["schema_version"] == SCHEMA_VERSION,
        f"{label}.schema_version",
        f"expected {SCHEMA_VERSION}",
    )
    _text(payload["case"], f"{label}.case")
    _sha(payload["revision"], 40, f"{label}.revision")
    _need(
        payload["revision"] == revision,
        f"{label}.revision",
        f"!= expected {revision}",
    )
    _check_code(payload["code"], label)
    configuration = payload["configuration"]
    _check_config(configuration, payload["case"], expect)
    _check_runtime(payload["runtime"], label)
    _check_assets(payload["assets"], label)
    samples = payload["samples"]
    _need(isinstance(samples, list), f"{label}.samples", "expected list")
    _need(
        len(samples) == expect,
        f"{label}.samples",
        f"count {len(samples)} != --expect-samples {expect}",
    )
    seen_ids: set[int] = set()
    seen_shas: set[str] = set()
    for index, sample in enumerate(samples):
        _check_sample(sample, index, label)
        _need(
            sample["sample_id"] == index,
            f"{label}.samples[{index}].sample_id",
            f"expected {index}",
        )
        _need(
            sample["seed"] == configuration["seed"] + index,
            f"{label}.samples[{index}].seed",
            f"expected cfg.seed + {index} = {configuration['seed'] + index}",
        )
        gen_len = len(sample["gen_ids"])
        _need(
            gen_len <= configuration["max_tokens"],
            f"{label}.samples[{index}].gen_ids",
            f"length {gen_len} > max_tokens {configuration['max_tokens']}",
        )
        if gen_len < configuration["max_tokens"]:
            _need(
                bool(set(sample["gen_ids"]) & set(configuration["eos_token_ids"])),
                f"{label}.samples[{index}].gen_ids",
                "short generation must contain a configured EOS token",
            )
        _need(
            sample["sample_id"] not in seen_ids,
            f"{label}.samples[{index}].sample_id",
            "duplicate",
        )
        _need(
            sample["prompt_sha256"] not in seen_shas,
            f"{label}.samples[{index}].prompt_sha256",
            "duplicate",
        )
        seen_ids.add(sample["sample_id"])
        seen_shas.add(sample["prompt_sha256"])
    total_bytes = sum(
        float(sample["metrics"]["edge_cloud_data_bytes"]) for sample in samples
    )
    total_connect = sum(
        sample["metrics"]["connect_times"]["edge_cloud"] for sample in samples
    )
    total_target = sum(sample["metrics"]["target_forward_times"] for sample in samples)
    any_trace = any(sample["metrics"]["comm_trace_edge_cloud"] for sample in samples)
    _need(
        total_bytes > 0,
        f"{label}.samples",
        "aggregate edge_cloud_data_bytes must be > 0",
    )
    _need(
        total_connect > 0,
        f"{label}.samples",
        "aggregate edge_cloud connect_times must be > 0",
    )
    _need(
        total_target > 0,
        f"{label}.samples",
        "aggregate target_forward_times must be > 0",
    )
    _need(
        any_trace,
        f"{label}.samples",
        "at least one sample must have a non-empty edge_cloud trace",
    )


def _token_mismatch(base: dict, candidate: dict) -> dict:
    left, right = base["gen_ids"], candidate["gen_ids"]
    limit = len(left) if len(left) < len(right) else len(right)
    pos = next((i for i in range(limit) if left[i] != right[i]), limit)
    return {
        "kind": "token",
        "sample_id": base["sample_id"],
        "pos": pos,
        "base": left[pos] if pos < len(left) else None,
        "candidate": right[pos] if pos < len(right) else None,
        "base_len": len(left),
        "candidate_len": len(right),
    }


def _short(value: Any, limit: int = 160) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _base_report(base_revision: str, candidate_revision: str) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "verdict": "INVALID",
        "case": None,
        "samples": None,
        "base_revision": base_revision,
        "candidate_revision": candidate_revision,
        "errors": [],
        "mismatches": [],
        "ignored_timing": [],
    }


def compare(
    base: Any,
    candidate: Any,
    *,
    expect_samples: int,
    base_revision: str,
    candidate_revision: str,
) -> dict:
    report = _base_report(base_revision, candidate_revision)
    try:
        _need(
            _int(expect_samples) and expect_samples > 0,
            "--expect-samples",
            "expected positive int",
        )
        _sha(base_revision, 40, "--base-revision")
        _sha(candidate_revision, 40, "--candidate-revision")
        _check_payload(base, "base", base_revision, expect_samples)
        _check_payload(candidate, "candidate", candidate_revision, expect_samples)
        _need(base["case"] == candidate["case"], "case", "case mismatch")
        for name in ("configuration", "runtime", "assets"):
            _need(
                _same(base[name], candidate[name]),
                name,
                f"differs at {_first_diff(base[name], candidate[name]) or '<root>'}",
            )
        report["case"], report["samples"] = base["case"], expect_samples
        for index in range(expect_samples):
            left, right = base["samples"][index], candidate["samples"][index]
            for field in ("sample_id", "prompt", "prompt_sha256", "seed", "prefix_ids"):
                _need(
                    left[field] == right[field],
                    f"sample position {index}",
                    f"{field} mismatch",
                )
            if left["gen_ids"] != right["gen_ids"]:
                report["mismatches"].append(_token_mismatch(left, right))
            left_metrics, right_metrics = left["metrics"], right["metrics"]
            missing = sorted(set(left_metrics) - set(right_metrics))
            extra = sorted(set(right_metrics) - set(left_metrics))
            _need(
                not missing and not extra,
                f"sample {left['sample_id']}",
                f"metric keys differ; missing={missing}, extra={extra}",
            )
            for key in sorted(left_metrics):
                if key in TIMING_FIELDS:
                    _need(
                        _num(left_metrics[key]),
                        f"sample {left['sample_id']}.{key}",
                        "timing is not finite number",
                    )
                    _need(
                        _num(right_metrics[key]),
                        f"sample {left['sample_id']}.{key}",
                        "timing is not finite number",
                    )
                    if not _same(left_metrics[key], right_metrics[key]):
                        report["ignored_timing"].append(
                            {
                                "sample_id": left["sample_id"],
                                "field": key,
                                "base": _short(left_metrics[key]),
                                "candidate": _short(right_metrics[key]),
                            }
                        )
                elif not _same(left_metrics[key], right_metrics[key]):
                    report["mismatches"].append(
                        {
                            "kind": "metric",
                            "sample_id": left["sample_id"],
                            "field": key,
                            "base": _short(left_metrics[key]),
                            "candidate": _short(right_metrics[key]),
                            "detail": _first_diff(
                                left_metrics[key], right_metrics[key]
                            ),
                        }
                    )
    except Invalid as exc:
        report["errors"].append(str(exc))
        return report
    report["verdict"] = "FAIL" if report["mismatches"] else "PASS"
    return report


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise Invalid(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _load(path: str, label: str) -> Any:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise Invalid(f"{label}: not valid UTF-8: {path}: {exc}") from exc
    except OSError as exc:
        raise Invalid(f"{label}: cannot read {path}: {exc}") from exc
    try:
        return json.loads(text, object_pairs_hook=_no_duplicate_keys)
    except Invalid as exc:
        raise Invalid(f"{label}: {exc}") from exc
    except json.JSONDecodeError as exc:
        raise Invalid(f"{label}: invalid JSON {path}: {exc}") from exc


def _print_report(report: dict) -> None:
    print(f"{report['verdict']} case={report['case']!r} samples={report['samples']}")
    for error in report["errors"]:
        print(f"  schema/identity: {error}")
    for item in report["mismatches"][:10]:
        if item["kind"] == "token":
            print(
                f"  token sample={item['sample_id']} pos={item['pos']}: "
                f"{item['base']!r} -> {item['candidate']!r} "
                f"(len {item['base_len']} -> {item['candidate_len']})"
            )
        else:
            print(
                f"  metric sample={item['sample_id']} {item['field']}: "
                f"{item['base']} -> {item['candidate']}"
            )
    if len(report["mismatches"]) > 10:
        print(f"  ... {len(report['mismatches']) - 10} more mismatches")
    if report["ignored_timing"]:
        fields = sorted({item["field"] for item in report["ignored_timing"]})
        print(
            f"  ignored timing ({len(report['ignored_timing'])} differences): "
            f"{', '.join(fields)}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--expect-samples", type=int, required=True)
    parser.add_argument("--base-revision", required=True)
    parser.add_argument("--candidate-revision", required=True)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    try:
        report = compare(
            _load(args.base, "base"),
            _load(args.candidate, "candidate"),
            expect_samples=args.expect_samples,
            base_revision=args.base_revision,
            candidate_revision=args.candidate_revision,
        )
    except Invalid as exc:
        report = _base_report(args.base_revision, args.candidate_revision)
        report["errors"].append(str(exc))
    if args.json_out:
        try:
            path = Path(args.json_out)
            path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except OSError as exc:
            print(f"INVALID: cannot write --json-out: {exc}", file=sys.stderr)
            return 2
    _print_report(report)
    return {"PASS": 0, "FAIL": 1, "INVALID": 2}[report["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
