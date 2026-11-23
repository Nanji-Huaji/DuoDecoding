"""Paper rerun design invariants; no weights, torch, network or GPU required."""

from itertools import product

import pytest

from eval.worker import build_argv
from src.experiments.config import EvalDataset, EvalMode
from src.experiments.matrix import (
    build_rerun_matrix,
    llama_series,
    paper_table5_datasets,
    paper_table5_modes,
    paper_table5_series,
    qwen_1_5_series,
)


def semantics(c):
    return tuple(
        sorted(
            (k, v)
            for k, v in c.items()
            if k
            not in {
                "exp_name",
                "CUDA_VISIBLE_DEVICES",
            }
        )
    )


def keys(stage):
    configs = build_rerun_matrix(stage)
    distinct = {semantics(c) for c in configs}
    assert len(configs) == len(distinct)
    return distinct


def test_stage_membership_and_no_repeated_controls():
    main, plus, static, low = map(keys, ("main46", "cee-plus46", "static46", "lowbw"))
    assert list(map(len, (main, plus, static, low))) == [45, 9, 9, 90]
    core = keys("core")
    perf = keys("performance")
    assert len(core) == 63 and core == main | plus | static
    assert len(perf) == 153 and perf == core | low
    assert len(main | low) == 135
    assert keys("quality").isdisjoint(perf)
    main_axes = {
        (
            c["target_model"],
            c["eval_dataset"],
            c["eval_mode"],
            c["edge_cloud_bandwidth"],
        )
        for s in ("main46", "lowbw")
        for c in build_rerun_matrix(s)
    }
    assert main_axes == set(
        product(
            [s[-1] for s in paper_table5_series],
            paper_table5_datasets,
            paper_table5_modes,
            (46, 10, 5),
        )
    )
    with pytest.raises(ValueError, match="Unknown rerun stage"):
        build_rerun_matrix("typo")


def test_operating_points_ignore_old_shell_overrides(monkeypatch):
    monkeypatch.setenv("EXP_GAMMA_SCHEDULE", "typo")
    monkeypatch.setenv("CEESD_GAMMA1_LLAMA", "999")
    monkeypatch.setenv("CEESD_GAMMA2", "not-an-int")
    for c in build_rerun_matrix("performance"):
        mode = EvalMode(c["eval_mode"])
        cap = 5 if c["target_model"] == qwen_1_5_series[-1] else 16
        bw = c["edge_cloud_bandwidth"]
        assert c["sample_seed"] == c["seed"] == 1234
        assert (c["eval_data_num"], c["num_shots"], c["max_tokens"]) == (80, 3, 128)
        assert c["random_sample"] and c["use_cuda_graph"] and c["use_stochastic_comm"]
        assert c["comm_bw_model"] == "fluid" and not c["use_early_stopping"]
        assert c["min_bandwidth_mbps"] == max(0.5, bw / 10)
        argv = build_argv(c)
        assert argv[:6] == [
            "--protocol",
            "paper_table5",
            "--comm_accounting",
            "honest",
            "--temp",
            "0.0",
        ]
        assert argv.count("--protocol") == 1
        assert ("--humaneval_generate_only" in argv) == (
            c["eval_dataset"] == EvalDataset.humaneval
        )
        if mode in (EvalMode.tridecoding, EvalMode.ceesd):
            assert c["gamma1"] == c["gamma2"] == cap
        if mode == EvalMode.ceesd:
            assert c["use_rl_adapter"] and c["disable_rl_update"]
            for role in ("main", "little"):
                path = c[f"{role}_rl_path"]
                assert (
                    path.endswith("/latest.pth") and path == c[f"{role}_rl_best_path"]
                )
        else:
            assert not c["use_rl_adapter"]
        if mode in (EvalMode.dsd, EvalMode.dssd, EvalMode.tk_slt, EvalMode.cuhlm):
            gamma, k = {
                EvalMode.dsd: (3 if bw == 46 else 1, 0),
                EvalMode.dssd: (8, 0),
                EvalMode.tk_slt: (8, 320),
                EvalMode.cuhlm: (3, 300),
            }[mode]
            assert (c["gamma"], c["transfer_top_k"]) == (gamma, k)
        if mode == EvalMode.cuhlm:
            assert c["uncertainty_threshold"] == 0.08
        if mode == EvalMode.tridecoding:
            assert c["transfer_top_k"] == 1024


def test_extension_windows_and_executed_cuhlm_threshold():
    native = {
        c["eval_mode"]: c
        for c in build_rerun_matrix("main46")
        if c["target_model"] == llama_series[-1]
    }
    for c in build_rerun_matrix("cee-plus46"):
        assert (
            c["draft_model"] == llama_series[1] and c["target_model"] == llama_series[2]
        )
        assert not c["use_rl_adapter"]
        if c["eval_mode"] == EvalMode.cee_cuhlm:
            assert (c["gamma1"], c["gamma2"]) == (1, 1)
            assert c["draft_target_threshold"] == c["uncertainty_threshold"] == 0.08
            assert c["transfer_top_k"] == native[EvalMode.cuhlm]["transfer_top_k"]
            argv = build_argv(c)
            assert argv[argv.index("--draft_target_threshold") + 1] == "0.08"
        else:
            is_dsd = c["eval_mode"] == EvalMode.cee_dsd
            assert (c["gamma1"], c["gamma2"]) == ((1, 1) if is_dsd else (4, 3))
            assert c["transfer_top_k"] == 0
            assert (
                c["gamma1"] + 1 + c["gamma2"]
                == native[EvalMode.dsd if is_dsd else EvalMode.dssd]["gamma"]
            )


def test_fixed_five_alternative_is_separate_and_matched():
    static5, full5, main = keys("static5"), keys("full5"), keys("main46")
    assert len(static5) == 9 and len(full5) == 6
    assert len(static5 & keys("static46")) == 3
    assert full5.isdisjoint(main) and full5.isdisjoint(keys("core"))

    def pair(c):
        return c["target_model"], c["eval_dataset"], c["gamma1"], c["gamma2"]

    controls = [
        c
        for c in build_rerun_matrix("main46")
        if c["target_model"] == qwen_1_5_series[-1] and c["eval_mode"] == EvalMode.ceesd
    ]
    assert {pair(c) for c in build_rerun_matrix("static5")} == {
        pair(c) for c in controls + build_rerun_matrix("full5")
    }


def test_quality_has_paired_target_prompts_stop_and_separate_denominator():
    quality = build_rerun_matrix("quality")
    assert len(quality) == 24
    for series, task in product(
        paper_table5_series, (EvalDataset.gsm8k, EvalDataset.cnndm)
    ):
        group = [
            c
            for c in quality
            if c["target_model"] == series[-1] and c["eval_dataset"] == task
        ]
        assert {c["eval_mode"] for c in group} == {
            EvalMode.autoregression,
            EvalMode.cuhlm,
            EvalMode.cee_cuhlm,
            EvalMode.ceesd,
        }
        for c in group:
            assert c["eval_data_num"] == 500 and c["seed"] == c["sample_seed"] == 1234
            assert c["max_tokens"] == (1024 if task == EvalDataset.gsm8k else 160)
            argv = build_argv(c)
            assert (
                argv[:2] == ["--protocol", "honest"] and "--use_early_stopping" in argv
            )
            assert "--disable_eos_stop" not in argv
            if c["eval_mode"] == "large":
                assert (
                    c["draft_model"] == series[0]
                )  # template selector, not loaded weights
            if c["eval_mode"] == EvalMode.cee_cuhlm:
                assert c["draft_target_threshold"] == c["uncertainty_threshold"] == 0.08
