"""strict refactor smoke comparator 的离线测试（CPU，无需 GPU）。"""

from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import pytest

from src.metrics import get_empty_metrics

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_refactor_smoke as smoke  # noqa: E402

REV_A = "a" * 40
REV_B = "b" * 40
EOS_ID = 2
MAX_TOKENS = 4
SEED = 1234
# Independent producer contract: never derive fixture/required keys from the comparator.
NON_TIMING_METRICS = set(get_empty_metrics()) - {
    "wall_time",
    "throughput",
    "computation_time",
    "little_computation_time",
    "draft_computation_time",
    "target_computation_time",
    "arp_overhead_time",
    "dra_overhead_time",
}
PROFILES = {
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


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _metrics(
    tokens: list[int],
    *,
    edge_bytes: float = 0.0,
    trace: list | None = None,
    connect_edge: int = 0,
    target: int = 1,
) -> dict:
    if trace is None:
        trace = [[edge_bytes, 1]] if edge_bytes > 0 else []
    return {
        "generated_tokens": len(tokens),
        "little_generated_tokens": 2,
        "draft_generated_tokens": 2,
        "little_accepted_tokens": 1,
        "draft_accepted_tokens": 1,
        "loop_times": 1,
        "each_loop_draft_tokens": 2.0,
        "comm_energy": 0.1,
        "accuracy": None,
        "avg_top_k": 32.0,
        "avg_draft_len": 2.0,
        "edge_cloud_bandwidth_history": [],
        "edge_cloud_ntt_history": [],
        "edge_cloud_topk_history": [],
        "edge_cloud_draft_len_history": [],
        "little_entropy_history": [],
        "draft_entropy_history": [],
        "little_accept_rate_history": [],
        "draft_accept_rate_history": [],
        "little_accepted_vocab_rank_history": [],
        "draft_accepted_vocab_rank_history": [],
        "little_accepted_in_transfer_topk_history": [],
        "draft_accepted_in_transfer_topk_history": [],
        "little_accepted_transfer_topk_rank_history": [],
        "draft_accepted_transfer_topk_rank_history": [],
        "little_forward_times": 1,
        "draft_forward_times": 2,
        "target_forward_times": target,
        "edge_cloud_data_bytes": edge_bytes,
        "edge_end_data_bytes": 5.0,
        "cloud_end_data_bytes": 0.0,
        "communication_time": 1.0,
        "edge_end_comm_time": 0.1,
        "queuing_time": 0.2,
        "connect_times": {"edge_cloud": connect_edge, "edge_end": 1, "cloud_end": 0},
        "comm_trace_edge_cloud": trace,
        "wall_time": 2.0,
        "throughput": 12.0,
        "computation_time": 1.0,
        "little_computation_time": 0.1,
        "draft_computation_time": 0.2,
        "target_computation_time": 0.3,
        "arp_overhead_time": 0.0,
        "dra_overhead_time": 0.1,
        "little_wall_time": 0.1,
        "draft_wall_time": 0.2,
        "target_wall_time": 0.3,
    }


def _config(
    case: str = "legacy_static",
    *,
    samples: int = 2,
    max_tokens: int = MAX_TOKENS,
    seed: int = SEED,
    eos_ids: tuple[int, ...] = (EOS_ID,),
) -> dict:
    profile = PROFILES[case]
    return {
        "eval_mode": "tridecoding",
        "temp": profile["temp"],
        "transfer_top_k": profile["transfer_top_k"],
        "prob_payload_bits": profile["prob_payload_bits"],
        "comm_round_trip_mode": profile["comm_round_trip_mode"],
        "charge_residual_payload": profile["charge_residual_payload"],
        "gamma": 3,
        "gamma1": 3,
        "gamma2": 2,
        "max_tokens": max_tokens,
        "samples": samples,
        "seed": seed,
        "use_cuda_graph": False,
        "use_rl_adapter": False,
        "disable_rl_update": True,
        "disable_eos_stop": False,
        "eos_token_ids": list(eos_ids),
        "model_dtype": "bf16",
        "little_quantization": "none",
        "draft_quantization": "none",
        "target_quantization": "none",
        "top_k": 0,
        "top_p": 0.0,
        "transfer_top_k_cap": 0,
        "force_full_vocab_transfer": False,
        "edge_cloud_bandwidth": 20.0,
        "edge_end_bandwidth": 100.0,
        "cloud_end_bandwidth": 100.0,
        "min_bandwidth_mbps": 5.0,
        "comm_bw_model": "instant",
        "ntt_ms_edge_cloud": 10.0,
        "ntt_ms_edge_end": 1.0,
        "batch_delay": 0.0,
        "use_stochastic_comm": False,
        "use_early_stopping": False,
    }


def _payload(
    revision: str,
    *,
    case: str = "legacy_static",
    samples: int = 2,
    code_tag: str = "base",
) -> dict:
    rows = []
    for index in range(samples):
        prompt = f"prompt {index}"
        if index % 2 == 0:
            tokens = [10 + index, EOS_ID, 12 + index, 13 + index]
            metrics = _metrics(
                tokens,
                edge_bytes=12.0 + index,
                trace=[[12.0 + index, 1]],
                connect_edge=1,
                target=1,
            )
        else:
            tokens = [20 + index, EOS_ID, 22 + index]
            metrics = _metrics(
                tokens, edge_bytes=0.0, trace=[], connect_edge=0, target=0
            )
        rows.append(
            {
                "sample_id": index,
                "prompt": prompt,
                "prompt_sha256": _sha(prompt),
                "seed": SEED + index,
                "prefix_ids": [1, 2],
                "gen_ids": tokens,
                "metrics": metrics,
            }
        )
    return {
        "schema_version": 1,
        "case": case,
        "revision": revision,
        "code": {
            "definition_module": f"src.baselines.{code_tag}",
            "definition_file": f"src/baselines/{code_tag}.py",
            "source_sha256": _sha(f"source-{code_tag}"),
        },
        "configuration": _config(case, samples=samples),
        "runtime": {
            "python": "3.10.18",
            "torch": "2.6.0",
            "cuda": "12.4",
            "transformers": "4.57.6",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-1234",
            "attention_backend": "sdpa",
        },
        "assets": {
            "schema_version": 1,
            "models": {
                "small": {
                    "path": "Qwen/Qwen3-0.6B",
                    "files": [
                        {
                            "name": "model.safetensors",
                            "size": 100,
                            "sha256": _sha("small"),
                        }
                    ],
                },
                "target": {
                    "path": "Qwen/Qwen3-1.7B",
                    "files": [
                        {
                            "name": "model.safetensors",
                            "size": 200,
                            "sha256": _sha("target"),
                        }
                    ],
                },
            },
            "tokenizer_vocab_sha256": _sha("vocab"),
            "submodule_commit": "c" * 40,
        },
        "samples": rows,
    }


def _run(
    base: dict,
    cand: dict,
    *,
    expect: int = 2,
    base_rev: str = REV_A,
    cand_rev: str = REV_B,
) -> dict:
    return smoke.compare(
        base,
        cand,
        expect_samples=expect,
        base_revision=base_rev,
        candidate_revision=cand_rev,
    )


def _invalid(report: dict, needle: str) -> None:
    assert report["verdict"] == "INVALID", report
    assert any(needle in error for error in report["errors"]), report


def test_positive_allows_different_code_source_and_revision() -> None:
    report = _run(_payload(REV_A, code_tag="main"), _payload(REV_B, code_tag="core"))
    assert report["verdict"] == "PASS", report
    assert report["mismatches"] == []
    assert report["ignored_timing"] == []


def test_honest_profile_passes() -> None:
    report = _run(
        _payload(REV_A, case="honest_int8"), _payload(REV_B, case="honest_int8")
    )
    assert report["verdict"] == "PASS", report


def test_timing_differences_are_ignored_but_listed() -> None:
    base = _payload(REV_A)
    cand = copy.deepcopy(base)
    cand["revision"] = REV_B
    cand["samples"][0]["metrics"]["wall_time"] = 999.0
    cand["samples"][1]["metrics"]["target_computation_time"] = 888.0
    report = _run(base, cand)
    assert report["verdict"] == "PASS", report
    assert report["mismatches"] == []
    assert {item["field"] for item in report["ignored_timing"]} == {
        "wall_time",
        "target_computation_time",
    }


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), "slow", None])
def test_timing_must_be_finite_number(bad_value) -> None:
    base = _payload(REV_A)
    cand = copy.deepcopy(base)
    cand["revision"] = REV_B
    cand["samples"][0]["metrics"]["wall_time"] = bad_value
    _invalid(_run(base, cand), "wall_time")


def test_revision_must_match_cli_and_be_full_sha() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    _invalid(_run(base, cand, base_rev=REV_B), "revision")
    _invalid(_run(base, cand, base_rev="abc"), "40-hex")
    bad = _payload(REV_A)
    bad["revision"] = "not-a-sha"
    _invalid(_run(bad, cand), "revision")


def test_configuration_empty_or_degenerate_is_invalid() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    base["configuration"] = {}
    cand["configuration"] = {}
    _invalid(_run(base, cand), "missing required keys")
    base, cand = _payload(REV_A), _payload(REV_B)
    del cand["configuration"]["temp"]
    _invalid(_run(base, cand), "temp")


def test_configuration_profile_must_match_case() -> None:
    cand = _payload(REV_B, case="honest_int8")
    cand["configuration"]["temp"] = 0.0
    _invalid(_run(_payload(REV_A), cand), "must be 0.7")
    cand = _payload(REV_B, case="honest_int8")
    cand["configuration"]["transfer_top_k"] = 32
    _invalid(_run(_payload(REV_A), cand), "transfer_top_k")
    cand = _payload(REV_B)
    cand["case"] = "unknown_case"
    _invalid(_run(_payload(REV_A), cand), "unknown case")


def test_configuration_types_and_safety_flags() -> None:
    cand = _payload(REV_B)
    cand["configuration"]["use_cuda_graph"] = True
    _invalid(_run(_payload(REV_A), cand), "use_cuda_graph")
    cand = _payload(REV_B)
    cand["configuration"]["disable_eos_stop"] = True
    _invalid(_run(_payload(REV_A), cand), "disable_eos_stop")
    cand = _payload(REV_B)
    cand["configuration"]["eos_token_ids"] = []
    _invalid(_run(_payload(REV_A), cand), "eos_token_ids")
    cand = _payload(REV_B)
    cand["configuration"]["max_tokens"] = "4"
    _invalid(_run(_payload(REV_A), cand), "max_tokens")


def test_config_samples_must_equal_expect_n() -> None:
    cand = _payload(REV_B)
    cand["configuration"]["samples"] = 3
    _invalid(_run(_payload(REV_A), cand), "configuration.samples")


def test_empty_and_wrong_sample_counts_are_invalid() -> None:
    base = _payload(REV_A, samples=2)
    empty = copy.deepcopy(base)
    empty["revision"] = REV_B
    empty["samples"] = []
    _invalid(_run(base, empty, expect=2), "count 0")
    one = copy.deepcopy(base)
    one["revision"] = REV_B
    one["samples"] = one["samples"][:1]
    _invalid(_run(base, one, expect=2), "count 1")
    _invalid(_run(base, _payload(REV_B), expect=0), "positive")


def test_sample_id_seed_and_pairing_are_cross_checked() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    cand["samples"][0], cand["samples"][1] = cand["samples"][1], cand["samples"][0]
    _invalid(_run(base, cand), "sample_id")
    cand = _payload(REV_B)
    cand["samples"][1]["seed"] = 7
    _invalid(_run(base, cand), "seed")
    cand = _payload(REV_B)
    cand["samples"][1]["seed"] = SEED + 2
    _invalid(_run(base, cand), "seed")


def test_gen_length_and_eos_rules() -> None:
    # 短于 max_tokens 且无 EOS -> invalid；EOS 在 chunk 中间合法。
    cand = _payload(REV_B)
    cand["samples"][1]["gen_ids"] = [20, 21, 22]
    cand["samples"][1]["metrics"]["generated_tokens"] = 3
    _invalid(_run(_payload(REV_A), cand), "EOS")
    cand = _payload(REV_B)
    cand["samples"][1]["gen_ids"] = [21, EOS_ID, 23]
    cand["samples"][1]["metrics"]["generated_tokens"] = 3
    assert _run(_payload(REV_A), cand)["verdict"] == "PASS"
    # 等于 max_tokens 且无 EOS 合法；超过 max_tokens 非法。
    base = _payload(REV_A)
    cand = _payload(REV_B)
    for arm in (base, cand):
        arm["samples"][0]["gen_ids"] = [10, 11, 12, 13]
        arm["samples"][0]["metrics"]["generated_tokens"] = 4
    assert _run(base, cand)["verdict"] == "PASS"
    cand = _payload(REV_B)
    cand["samples"][1]["gen_ids"].extend([99, 100])
    cand["samples"][1]["metrics"]["generated_tokens"] = 5
    _invalid(_run(_payload(REV_A), cand), "max_tokens")


def test_reordered_samples_are_invalid_not_zipped() -> None:
    cand = _payload(REV_B)
    cand["samples"][0], cand["samples"][1] = cand["samples"][1], cand["samples"][0]
    _invalid(_run(_payload(REV_A), cand), "sample_id")


def test_missing_required_metric_in_one_or_both_is_invalid() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    del cand["samples"][0]["metrics"]["target_forward_times"]
    _invalid(_run(base, cand), "target_forward_times")
    del base["samples"][0]["metrics"]["target_forward_times"]
    _invalid(_run(base, cand), "target_forward_times")


def test_metric_fixture_covers_producer_non_timing_schema() -> None:
    assert NON_TIMING_METRICS <= set(_metrics([10, EOS_ID]))


@pytest.mark.parametrize("missing", sorted(NON_TIMING_METRICS))
def test_each_emitted_non_timing_metric_is_required_in_both_arms(missing: str) -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    for arm in (base, cand):
        for sample in arm["samples"]:
            del sample["metrics"][missing]
    _invalid(_run(base, cand), missing)


def test_new_metric_key_on_one_side_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["new_counter"] = 1
    _invalid(_run(_payload(REV_A), cand), "metric keys differ")


def test_unknown_metric_is_strict_not_default_ignored() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    base["samples"][0]["metrics"]["new_counter"] = 1
    cand["samples"][0]["metrics"]["new_counter"] = 2
    report = _run(base, cand)
    assert report["verdict"] == "FAIL", report
    assert any(
        item["kind"] == "metric" and item["field"] == "new_counter"
        for item in report["mismatches"]
    )


def test_token_drift_reports_first_position() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["gen_ids"][0] = 999
    report = _run(_payload(REV_A), cand)
    assert report["verdict"] == "FAIL", report
    token = next(item for item in report["mismatches"] if item["kind"] == "token")
    assert (token["sample_id"], token["pos"], token["base"], token["candidate"]) == (
        0,
        0,
        10,
        999,
    )


def test_token_length_drift_reports_length() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["gen_ids"].pop()
    cand["samples"][0]["metrics"]["generated_tokens"] = 3
    report = _run(_payload(REV_A), cand)
    assert report["verdict"] == "FAIL", report
    token = next(item for item in report["mismatches"] if item["kind"] == "token")
    assert token["pos"] == 3 and token["base_len"] == 4 and token["candidate_len"] == 3


def test_derived_generated_tokens_mismatch_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["generated_tokens"] = 5
    _invalid(_run(_payload(REV_A), cand), "generated_tokens")


def test_bool_is_not_int_in_samples_and_derived_count() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["sample_id"] = True
    _invalid(_run(_payload(REV_A), cand), "sample_id")
    cand = _payload(REV_B)
    cand["samples"][0]["gen_ids"][0] = True
    _invalid(_run(_payload(REV_A), cand), "gen_ids")
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["generated_tokens"] = True
    _invalid(_run(_payload(REV_A), cand), "generated_tokens")


def test_int_float_and_bool_metric_types_are_exact() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    base["samples"][0]["metrics"]["custom"] = 1
    cand["samples"][0]["metrics"]["custom"] = 1.0
    report = _run(base, cand)
    assert report["verdict"] == "FAIL", report
    assert any(item["field"] == "custom" for item in report["mismatches"])
    base, cand = _payload(REV_A), _payload(REV_B)
    base["configuration"]["temp"] = 0.0
    cand["configuration"]["temp"] = 0
    _invalid(_run(base, cand), "configuration.temp")


def test_nan_metric_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["communication_time"] = float("nan")
    _invalid(_run(_payload(REV_A), cand), "non-finite")


def test_runtime_or_assets_mismatch_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["runtime"]["torch"] = "2.7.0"
    _invalid(_run(_payload(REV_A), cand), "runtime")
    cand = _payload(REV_B)
    cand["assets"]["models"]["small"]["files"][0]["sha256"] = _sha("other")
    _invalid(_run(_payload(REV_A), cand), "assets")


def test_bad_code_sha_or_absolute_path_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["code"]["source_sha256"] = "xyz"
    _invalid(_run(_payload(REV_A), cand), "source_sha256")
    cand = _payload(REV_B)
    cand["code"]["definition_file"] = "/abs/src.py"
    _invalid(_run(_payload(REV_A), cand), "repo-relative")


def test_duplicate_sample_id_or_prompt_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["samples"][1]["sample_id"] = cand["samples"][0]["sample_id"]
    _invalid(_run(_payload(REV_A), cand), "sample_id")
    cand = _payload(REV_B)
    cand["samples"][1]["prompt"] = cand["samples"][0]["prompt"]
    cand["samples"][1]["prompt_sha256"] = cand["samples"][0]["prompt_sha256"]
    _invalid(_run(_payload(REV_A), cand), "duplicate")


def test_seed_prefix_and_prompt_sha_must_match() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    cand["samples"][0]["seed"] = 7
    _invalid(_run(base, cand), "seed")
    cand = _payload(REV_B)
    cand["samples"][0]["prefix_ids"] = [9]
    _invalid(_run(base, cand), "prefix_ids")
    cand = _payload(REV_B)
    cand["samples"][0]["prompt"] = "different"
    _invalid(_run(base, cand), "prompt_sha256")


def test_network_all_empty_is_invalid() -> None:
    base = _payload(REV_A)
    cand = _payload(REV_B)
    for arm in (base, cand):
        for sample in arm["samples"]:
            metrics = sample["metrics"]
            metrics["edge_cloud_data_bytes"] = 0.0
            metrics["comm_trace_edge_cloud"] = []
            metrics["connect_times"]["edge_cloud"] = 0
            metrics["target_forward_times"] = 0
    _invalid(_run(base, cand), "aggregate")


def test_single_first_stage_eos_sample_is_valid_if_other_samples_have_network() -> None:
    report = _run(_payload(REV_A), _payload(REV_B))
    assert report["verdict"] == "PASS", report
    assert _payload(REV_B)["samples"][1]["metrics"]["comm_trace_edge_cloud"] == []


def test_per_sample_bytes_positive_requires_trace() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["comm_trace_edge_cloud"] = []
    _invalid(_run(_payload(REV_A), cand), "must be non-empty")


@pytest.mark.parametrize(
    ("sample_index", "edge_bytes", "trace"),
    [
        (0, 12.0, [[0.0, 1]]),
        (0, 12.0, [[11.0, 1]]),
        (0, 12.0, [[13.0, 1]]),
        (1, 0.0, [[1.0, 1]]),
        (0, 1.0, [[1.000002, 1]]),
        (0, 1_000_000_000.0, [[1_000_000_001.0, 1]]),
    ],
)
def test_inconsistent_trace_is_invalid_even_when_identical_in_both_arms(
    sample_index: int, edge_bytes: float, trace: list
) -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    for arm in (base, cand):
        metrics = arm["samples"][sample_index]["metrics"]
        metrics["edge_cloud_data_bytes"] = edge_bytes
        metrics["comm_trace_edge_cloud"] = copy.deepcopy(trace)
    _invalid(_run(base, cand), "trace bytes")


@pytest.mark.parametrize(
    ("edge_bytes", "trace"),
    [
        (0.3, [[0.1, 1], [0.2, 1]]),
        (1.0, [[1.0000005, 1]]),
    ],
)
def test_trace_sum_allows_float_rounding_within_absolute_tolerance(
    edge_bytes: float, trace: list
) -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    for arm in (base, cand):
        metrics = arm["samples"][0]["metrics"]
        metrics["edge_cloud_data_bytes"] = edge_bytes
        metrics["comm_trace_edge_cloud"] = copy.deepcopy(trace)
    report = _run(base, cand)
    assert report["verdict"] == "PASS", report


def test_consistent_byte_accounting_drift_is_fail_not_invalid() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    metrics = cand["samples"][0]["metrics"]
    metrics["edge_cloud_data_bytes"] = 13.0
    metrics["comm_trace_edge_cloud"] = [[13.0, 1]]
    report = _run(base, cand)
    assert report["verdict"] == "FAIL", report
    assert {item["field"] for item in report["mismatches"]} == {
        "edge_cloud_data_bytes",
        "comm_trace_edge_cloud",
    }


def test_connect_times_and_trace_shapes_are_strict() -> None:
    cand = _payload(REV_B)
    del cand["samples"][0]["metrics"]["connect_times"]["cloud_end"]
    _invalid(_run(_payload(REV_A), cand), "connect_times")
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["comm_trace_edge_cloud"] = [[1]]
    _invalid(_run(_payload(REV_A), cand), "comm_trace_edge_cloud")
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["comm_trace_edge_cloud"] = [[1, 1.5]]
    _invalid(_run(_payload(REV_A), cand), "round")


def test_required_metric_structure_types() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["connect_times"] = []
    _invalid(_run(_payload(REV_A), cand), "connect_times")
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["comm_trace_edge_cloud"] = {}
    _invalid(_run(_payload(REV_A), cand), "comm_trace_edge_cloud")
    cand = _payload(REV_B)
    cand["samples"][0]["metrics"]["little_forward_times"] = 2.0
    _invalid(_run(_payload(REV_A), cand), "little_forward_times")


def test_schema_version_bool_is_not_one() -> None:
    base = _payload(REV_A)
    base["schema_version"] = True
    _invalid(_run(base, _payload(REV_B)), "schema_version")
    cand = _payload(REV_B)
    cand["assets"]["schema_version"] = True
    _invalid(_run(_payload(REV_A), cand), "schema_version")


def test_timing_key_must_exist_on_both_sides() -> None:
    cand = _payload(REV_B)
    del cand["samples"][0]["metrics"]["wall_time"]
    _invalid(_run(_payload(REV_A), cand), "metric keys differ")


def test_unknown_metric_same_value_passes() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    base["samples"][0]["metrics"]["custom"] = {"x": [1, 2]}
    cand["samples"][0]["metrics"]["custom"] = {"x": [1, 2]}
    assert _run(base, cand)["verdict"] == "PASS"


def test_empty_generated_sequence_is_invalid() -> None:
    cand = _payload(REV_B)
    cand["samples"][0]["gen_ids"] = []
    cand["samples"][0]["metrics"]["generated_tokens"] = 0
    _invalid(_run(_payload(REV_A), cand), "gen_ids")


def test_metric_bool_vs_int_is_fail_not_pass() -> None:
    base, cand = _payload(REV_A), _payload(REV_B)
    base["samples"][0]["metrics"]["custom"] = True
    cand["samples"][0]["metrics"]["custom"] = 1
    report = _run(base, cand)
    assert report["verdict"] == "FAIL", report
    assert any(item["field"] == "custom" for item in report["mismatches"])


def test_cli_exit_codes_and_json_out(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"
    base = _payload(REV_A)
    cand = _payload(REV_B)
    base_path.write_text(json.dumps(base), encoding="utf-8")
    cand_path.write_text(json.dumps(cand), encoding="utf-8")
    argv = [
        "--base",
        str(base_path),
        "--candidate",
        str(cand_path),
        "--expect-samples",
        "2",
        "--base-revision",
        REV_A,
        "--candidate-revision",
        REV_B,
    ]
    assert smoke.main(argv) == 0
    drift = copy.deepcopy(cand)
    drift["samples"][0]["gen_ids"][0] = 999
    cand_path.write_text(json.dumps(drift), encoding="utf-8")
    assert smoke.main(argv) == 1
    # Internally valid accounting drift is a comparison failure, not bad input.
    drift = copy.deepcopy(cand)
    drift["samples"][0]["metrics"]["edge_cloud_data_bytes"] = 13.0
    drift["samples"][0]["metrics"]["comm_trace_edge_cloud"] = [[13.0, 1]]
    cand_path.write_text(json.dumps(drift), encoding="utf-8")
    assert smoke.main(argv) == 1
    # The same well-shaped ledger with an unchanged aggregate is invalid evidence.
    drift["samples"][0]["metrics"]["comm_trace_edge_cloud"] = [[0.0, 1]]
    cand_path.write_text(json.dumps(drift), encoding="utf-8")
    assert smoke.main(argv) == 2
    bad_argv = list(argv)
    bad_argv[bad_argv.index("--base-revision") + 1] = "bad"
    out = tmp_path / "report.json"
    assert smoke.main(bad_argv + ["--json-out", str(out)]) == 2
    assert json.loads(out.read_text(encoding="utf-8"))["verdict"] == "INVALID"


def test_non_utf8_is_invalid_and_jsonout_is_written(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"
    base_path.write_bytes(b"\xff\xfe\x00bad")
    cand_path.write_text(json.dumps(_payload(REV_B)), encoding="utf-8")
    out = tmp_path / "report.json"
    argv = [
        "--base",
        str(base_path),
        "--candidate",
        str(cand_path),
        "--expect-samples",
        "2",
        "--base-revision",
        REV_A,
        "--candidate-revision",
        REV_B,
        "--json-out",
        str(out),
    ]
    assert smoke.main(argv) == 2
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "INVALID"
    assert any("UTF-8" in error for error in report["errors"])


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    cand_path = tmp_path / "cand.json"
    base_text = json.dumps(_payload(REV_A))
    base_path.write_text(
        base_text.replace(
            '"schema_version": 1', '"schema_version": 1, "schema_version": 1', 1
        ),
        encoding="utf-8",
    )
    cand_path.write_text(json.dumps(_payload(REV_B)), encoding="utf-8")
    out = tmp_path / "report.json"
    argv = [
        "--base",
        str(base_path),
        "--candidate",
        str(cand_path),
        "--expect-samples",
        "2",
        "--base-revision",
        REV_A,
        "--candidate-revision",
        REV_B,
        "--json-out",
        str(out),
    ]
    assert smoke.main(argv) == 2
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"] == "INVALID"
    assert any("duplicate JSON key" in error for error in report["errors"])


def test_config_enum_wrong_type_is_invalid_not_crash() -> None:
    cand = _payload(REV_B)
    cand["configuration"]["model_dtype"] = []
    _invalid(_run(_payload(REV_A), cand), "model_dtype")
    cand = _payload(REV_B)
    cand["configuration"]["comm_round_trip_mode"] = {}
    _invalid(_run(_payload(REV_A), cand), "comm_round_trip_mode")
