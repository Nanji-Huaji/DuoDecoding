"""exp.py 的断点续跑与结果合并回归测试。

为什么需要这组测试
------------------
`exp_name` 里带运行时间戳（`..._20261005_125853_876008`），同一个实验重跑一次就有
一个新的 `exp_name`。如果合并/续跑拿 `exp_name` 当键，重跑结果会被当成"新实验"插进
汇总，旧的 failed 条目原地留下 —— 表格里就会出现同一实验两行、其中一行是全 0。
本测试锁定：键必须是与时间戳和 GPU 分配无关的语义字段，且新结果覆盖旧条目。
"""

import json

import pytest

import exp


def make_entry(
    mode="adaptive_tridecoding",
    dataset="eval/eval_gsm8k.py",
    target="Qwen/Qwen3-14B",
    status="success",
    generated_tokens=10240,
    exp_name="run",
    gamma1=5,
):
    """构造一条与 experiment_summary_*.json 同构的汇总条目。"""
    config = {
        "eval_dataset": dataset,
        "eval_mode": mode,
        "little_model": "qwen/Qwen3-0.6B",
        "draft_model": "qwen/Qwen3-1.7B",
        "target_model": target,
        "edge_cloud_bandwidth": 46.0,
        "edge_end_bandwidth": 563,
        "cloud_end_bandwidth": 46.0,
        "batch_delay": 0.05,
        "gamma": 3,
        "gamma1": gamma1,
        "gamma2": gamma1,
        "max_tokens": 128,
        "num_shots": 3,
        "eval_data_num": 80,
        "num_samples_per_task": 1,
        "sample_seed": 1234,
        "random_sample": True,
        # 下面两个字段每次运行都会变，必须不进语义键
        "CUDA_VISIBLE_DEVICES": "0",
        "exp_name": f"{mode}/{dataset}/{exp_name}",
    }
    return {
        "exp_name": config["exp_name"],
        "result": {"eval_mode": mode, "generated_tokens": generated_tokens},
        "log_file": "",
        "status": status,
        "config": config,
    }


def dump_summary(tmp_path, entries, name="summary.json"):
    path = tmp_path / name
    path.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
    return path


class TestRunIdentity:
    def test_ignores_runtime_only_fields(self):
        """GPU 分配与 exp_name 时间戳不同，仍属同一个实验。"""
        first = make_entry(exp_name="run_a")
        second = make_entry(exp_name="run_b")
        second["config"]["CUDA_VISIBLE_DEVICES"] = "1"
        assert exp.run_identity(first["config"]) == exp.run_identity(second["config"])

    @pytest.mark.parametrize(
        "field,value",
        [
            ("eval_mode", "dist_spec"),
            ("eval_dataset", "eval/eval_humaneval.py"),
            ("target_model", "Qwen/Qwen1.5-7B-Chat"),
            ("gamma1", 3),
            ("edge_cloud_bandwidth", 5.0),
        ],
    )
    def test_distinguishes_semantic_fields(self, field, value):
        """真正决定"跑的是什么"的字段必须区分开。"""
        base = make_entry()
        other = make_entry()
        other["config"][field] = value
        if field == "gamma1":  # gamma1/gamma2 成对出现，保持一致
            other["config"]["gamma2"] = value
        assert exp.run_identity(base["config"]) != exp.run_identity(other["config"])

    def test_enum_and_str_agree(self):
        """config 里存 Enum 还是字符串，应算出同一个键。"""
        as_enum = make_entry()["config"]
        as_enum["eval_mode"] = exp.EvalMode.ceesd
        as_enum["eval_dataset"] = exp.EvalDataset.gsm8k
        assert exp.run_identity(as_enum) == exp.run_identity(make_entry()["config"])

    def test_int_and_float_bandwidth_agree(self):
        """46 与 46.0 是同一个带宽，不该算成两个实验。"""
        first = make_entry()["config"]
        first["edge_cloud_bandwidth"] = 46
        second = make_entry()["config"]
        second["edge_cloud_bandwidth"] = 46.0
        assert exp.run_identity(first) == exp.run_identity(second)


class TestMergeSummaries:
    def test_new_result_overrides_failed_entry(self):
        """重跑成功后，旧的 failed 条目应被替换而不是并存。"""
        base = [make_entry(status="failed", generated_tokens=0, exp_name="old")]
        base[0]["result"] = {"error": "实验失败，错误代码: 1"}
        rerun = [make_entry(status="success", exp_name="new")]

        merged = exp.merge_summaries(base, rerun)

        assert len(merged) == 1
        assert merged[0]["status"] == "success"
        assert merged[0]["exp_name"].endswith("new")

    def test_unrelated_entries_are_kept(self):
        base = [
            make_entry(dataset="eval/eval_gsm8k.py", exp_name="gsm"),
            make_entry(dataset="eval/eval_humaneval.py", exp_name="he"),
        ]
        rerun = [
            make_entry(dataset="eval/eval_gsm8k.py", exp_name="gsm_v2"),
        ]

        merged = exp.merge_summaries(base, rerun)

        assert len(merged) == 2
        by_dataset = {e["config"]["eval_dataset"]: e for e in merged}
        assert by_dataset["eval/eval_gsm8k.py"]["exp_name"].endswith("gsm_v2")
        assert by_dataset["eval/eval_humaneval.py"]["exp_name"].endswith("he")

    def test_is_idempotent(self):
        base = [make_entry(exp_name="a"), make_entry(dataset="eval/eval_humaneval.py")]
        assert exp.merge_summaries(base, base) == exp.merge_summaries(base)

    def test_keeps_entries_without_config(self):
        """早期异常条目没有 config，无从定位语义，按 exp_name 兜底保留。"""
        orphan = {
            "exp_name": "broken",
            "result": "执行异常: x",
            "log_file": "",
            "status": "exception",
        }
        merged = exp.merge_summaries([orphan], [make_entry()])
        assert len(merged) == 2

    def test_output_is_sorted_deterministically(self):
        entries = [
            make_entry(dataset="eval/eval_humaneval.py"),
            make_entry(dataset="eval/eval_gsm8k.py"),
        ]
        assert exp.merge_summaries(entries) == exp.merge_summaries(list(reversed(entries)))


class TestFilterConfigsForResume:
    def test_skips_already_successful(self, tmp_path):
        done = make_entry(dataset="eval/eval_gsm8k.py")
        pending = make_entry(dataset="eval/eval_humaneval.py")
        path = dump_summary(tmp_path, [done])

        selected = exp.filter_configs_for_resume(
            [done["config"], pending["config"]], path
        )

        assert len(selected) == 1
        assert selected[0]["eval_dataset"] == "eval/eval_humaneval.py"

    def test_failed_entry_is_retried(self, tmp_path):
        failed = make_entry(status="failed")
        path = dump_summary(tmp_path, [failed])
        assert len(exp.filter_configs_for_resume([failed["config"]], path)) == 1

    def test_no_resume_returns_everything(self):
        configs = [
            make_entry()["config"],
            make_entry(dataset="eval/eval_humaneval.py")["config"],
        ]
        assert exp.filter_configs_for_resume(configs) == configs

    def test_rerun_then_resume_converges(self, tmp_path):
        """重跑 -> 合并 -> 再按合并结果续跑，待跑数应为 0（幂等收敛）。"""
        base = [
            make_entry(dataset="eval/eval_gsm8k.py", status="failed", generated_tokens=0),
            make_entry(dataset="eval/eval_humaneval.py", status="success"),
        ]
        configs = [entry["config"] for entry in base]
        rerun = [make_entry(dataset="eval/eval_gsm8k.py", status="success")]
        merged = exp.merge_summaries(base, rerun)
        path = dump_summary(tmp_path, merged)

        assert exp.filter_configs_for_resume(configs, path) == []


class TestPerModeTransferTopK:
    """top-k 压缩按方法区分（docs/protocol.md §2 #8）。

    此前 transfer_top_k=300 一律传给所有方法，等于把"压缩传输"这项本该被
    验证的贡献免费送给基线：DSSD 原文拒绝下行是整词表 P_j(x)，DSD 上行是
    γ 个整词表分布。基线必须拿到"不压缩"的取值。
    """

    def test_ours_keeps_top_k_baselines_do_not(self):
        by_mode: dict = {}
        for config in exp.config_to_run:
            by_mode.setdefault(str(config["eval_mode"]), set()).add(
                config["transfer_top_k"]
            )

        assert by_mode["adaptive_tridecoding"] == {exp.TRANSFER_TOP_K_OURS}
        assert by_mode["dist_spec"] == {exp.TRANSFER_TOP_K_PAPER}
        assert by_mode["dist_split_spec"] == {exp.TRANSFER_TOP_K_PAPER}
        # CUHLM 走自己论文的式 (5)，不消费该值；保持 ours 档只为避免歧义
        assert by_mode["uncertainty_decoding"] == {exp.TRANSFER_TOP_K_OURS}
        # TK-SLT：K 是方法自身超参（top-K 稀疏 logits 上行），取论文
        # §VI-B Fig 4 的最优工作点 320
        assert by_mode["tk_slt"] == {exp.TRANSFER_TOP_K_TKSLT}

    def test_paper_value_really_turns_compression_off(self):
        """取值必须与计费函数的判定联动，而不是一个纯标记。

        `transfer()` 的 is_compressed = (k is not None and k > 0)；
        `reject_residual_payload_bytes` 的 `0 < k < vocab`。k=0 两个分支都落到整行。
        """
        assert exp.TRANSFER_TOP_K_PAPER == 0
        assert exp.TRANSFER_TOP_K_OURS > 0
        assert exp.TRANSFER_TOP_K_TKSLT > 0

    def test_only_modes_accepts_name_and_value(self):
        by_name = exp.filter_configs_by_mode(exp.config_to_run, "dsd,dssd")
        by_value = exp.filter_configs_by_mode(
            exp.config_to_run, "dist_spec,dist_split_spec"
        )
        # 2 方法 × 3 系列 × 3 数据集 × 带宽阶梯（2026-10-09 起 [46,10,5]）
        assert len(by_name) == 2 * 3 * 3 * len(exp.edge_cloud_bandwidth)
        assert by_name == by_value

    def test_only_modes_single_mode(self):
        selected = exp.filter_configs_by_mode(exp.config_to_run, "ceesd")
        assert len(selected) == 1 * 3 * 3 * len(exp.edge_cloud_bandwidth)
        assert {str(c["eval_mode"]) for c in selected} == {"adaptive_tridecoding"}

    def test_only_modes_rejects_unknown(self):
        with pytest.raises(ValueError):
            exp.filter_configs_by_mode(exp.config_to_run, "nope")

    def test_no_filter_returns_all(self):
        assert exp.filter_configs_by_mode(exp.config_to_run) == exp.config_to_run


class TestPerModeThreshold:
    """CUHLM 的不确定度阈值必须是非退化工作点（docs/protocol.md §3.2）。

    默认 0.8 让 CUHLM 在 Llama-2-13b/GSM8K 上掉到 0.0 准确率（同期 CEE-SD
    0.2375、DSD/DSSD 0.25），它的"快"来自几乎不调用云模型，与论文 Table V
    自称的 "baselines ≈target" 矛盾。0.08 是实测前沿上的等准确率点。
    """

    def test_cuhlm_gets_matched_accuracy_threshold(self):
        by_mode: dict = {}
        for config in exp.config_to_run:
            by_mode.setdefault(str(config["eval_mode"]), set()).add(
                config["uncertainty_threshold"]
            )

        assert by_mode["uncertainty_decoding"] == {exp.UNCERTAINTY_THRESHOLD_CUHLM}
        assert exp.UNCERTAINTY_THRESHOLD_CUHLM < exp.UNCERTAINTY_THRESHOLD_DEFAULT

    def test_other_methods_keep_default(self):
        """主表里只有 CUHLM 消费该参数，其余方法不该被带着改。"""
        by_mode: dict = {}
        for config in exp.config_to_run:
            by_mode.setdefault(str(config["eval_mode"]), set()).add(
                config["uncertainty_threshold"]
            )
        for mode in ("dist_spec", "dist_split_spec", "adaptive_tridecoding"):
            assert by_mode[mode] == {exp.UNCERTAINTY_THRESHOLD_DEFAULT}


class TestLoadSummary:
    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            exp.load_summary(tmp_path / "nope.json")

    def test_non_list_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
        with pytest.raises(ValueError):
            exp.load_summary(path)
