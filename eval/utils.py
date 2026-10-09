from src.metrics import DecodingMetrics
from src.metrics_dumper import ArgsLike
import json
import os
import random


def select_eval_data(data: list, args: ArgsLike) -> list:
    if getattr(args, "run_full_dataset", False):
        return data

    eval_data_num = getattr(args, "eval_data_num", None)
    if eval_data_num is None:
        return data

    sample_size = min(eval_data_num, len(data))
    if getattr(args, "random_sample", False):
        rng = random.Random(getattr(args, "sample_seed", 1234))
        return rng.sample(data, sample_size)

    return data[:sample_size]


class ExpPrint:
    analysis_metrics = (
        "little_entropy_history",
        "draft_entropy_history",
        "little_accept_rate_history",
        "draft_accept_rate_history",
        "little_accepted_vocab_rank_history",
        "draft_accepted_vocab_rank_history",
        "little_accepted_in_transfer_topk_history",
        "draft_accepted_in_transfer_topk_history",
        "little_accepted_transfer_topk_rank_history",
        "draft_accepted_transfer_topk_rank_history",
    )

    common_print_metrics = (
        "little_forward_times",
        "draft_forward_times",
        "target_forward_times",
        "little_computation_time",
        "draft_computation_time",
        "target_computation_time",
        "generated_tokens",
        "little_generated_tokens",
        "draft_generated_tokens",
        "little_accepted_tokens",
        "draft_accepted_tokens",
        "wall_time",
        "throughput",
        "communication_time",
        "computation_time",
        "edge_end_comm_time",
        "edge_cloud_data_bytes",
        "edge_end_data_bytes",
        "cloud_end_data_bytes",
        "loop_times",
        "each_loop_draft_tokens",
        "comm_energy",
        "connect_times",
        "accuracy",
        "queuing_time",
        "arp_overhead_time",
        "dra_overhead_time",
        "avg_top_k",
        "avg_draft_len",
    )

    def __init__(self, args: ArgsLike):
        self.args = args

    def _prepare_metrics(self, metrics: DecodingMetrics) -> DecodingMetrics:
        # 添加类型检查和默认值
        computation_time = metrics.get("computation_time", 0.0)
        if not isinstance(computation_time, (int, float)):
            metrics["computation_time"] = 0.0

        communication_time = metrics.get("communication_time", 0.0)
        if not isinstance(communication_time, (int, float)):
            metrics["communication_time"] = 0.0

        if metrics["wall_time"] != 0:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]

        return metrics

    def get_filtered_dict(self, metrics: DecodingMetrics) -> dict:
        metrics = self._prepare_metrics(metrics)
        key_to_dump = list(self.common_print_metrics) + list(self.analysis_metrics)
        if self.args.dump_network_stats:
            key_to_dump += [
                "edge_cloud_bandwidth_history",
                "edge_cloud_ntt_history",
                "edge_cloud_topk_history",
                "edge_cloud_draft_len_history",
            ]
        # 逐消息记账记录（[字节, 轮号]）：离线重放（scripts/rebill.py）的
        # 输入，带宽/NTT/口径敏感性不再占 GPU。与 dump_network_stats 的
        # 聚合历史不同，它是重放的完备输入，必须默认落盘。
        key_to_dump += ["comm_trace_edge_cloud"]
        dump_dict = {key: metrics.get(key) for key in key_to_dump}
        return dump_dict

    def get_printable_dict(self, metrics: DecodingMetrics) -> dict:
        return {k: v for k, v in metrics.items() if k in self.common_print_metrics}

    def dump_metrics(self, metrics: DecodingMetrics) -> str:
        return json.dumps(self.get_filtered_dict(metrics), indent=4)

    def get_printable_metrics(self, metrics: DecodingMetrics) -> str:
        res = json.dumps(self.get_printable_dict(metrics), indent=4)
        return f""" -------Decoding Metrics-------
         {res}
        -------Decoding Metrics-------"""

    def get_save_dict(self, metrics: DecodingMetrics) -> dict:
        eval_result = self.get_filtered_dict(metrics)
        eval_result["little_model"] = self.args.little_model
        eval_result["draft_model"] = self.args.draft_model
        eval_result["target_model"] = self.args.target_model
        eval_result["eval_mode"] = self.args.eval_mode
        eval_result["gamma"] = self.args.gamma if self.args.gamma is not None else -1
        eval_result["gamma1"] = self.args.gamma1 if self.args.gamma1 is not None else -1
        eval_result["gamma2"] = self.args.gamma2 if self.args.gamma2 is not None else -1
        # L1 口径收敛：每个结果工件自带通信计费口径与动态链路统计（可复现性）。
        # 2026-04 全表解钳：honest/legacy 只由计费两开关区分（残差 + 往返）；
        # transfer_top_k_cap 不再属于口径预设（cap=16 原是 ours 的方法设计），
        # 手动设 cap 只改变方法的 top-k 行为，不改变计费口径标签。
        charge = bool(getattr(self.args, "charge_residual_payload", False))
        mode = str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
        if charge and mode == "per_round":
            label = "honest"
        elif (not charge) and mode == "per_transfer":
            label = "legacy"
        else:
            label = "custom"
        eval_result["charge_residual_payload"] = charge
        eval_result["comm_round_trip_mode"] = mode
        eval_result["transfer_top_k_cap"] = int(
            getattr(self.args, "transfer_top_k_cap", 0) or 0
        )
        # §3.4 网络口径：发射时长模型 + 带宽地板（新旧口径的 run 只能靠这
        # 两个字段从工件区分，exp_name 里不可见）
        eval_result["comm_bw_model"] = str(
            getattr(self.args, "comm_bw_model", "instant")
        )
        eval_result["min_bandwidth_mbps"] = float(
            getattr(self.args, "min_bandwidth_mbps", 5.0)
        )
        # 机制层：记录**实际消费**而非仅标称值。2026-04 统一口径落地后
        # （docs/protocol.md §3）基线族与 adaptive 族消费同一套开关；CUHLM
        # 系不消费仓库开关，而是按 CU-HLM 论文自身的计费口径计费（式 (5)：
        # 上行 k·(b_prob+b_index) bits，token 索引 negligible）；无通信的
        # 纯本地模式标签无意义。标称口径若不被本模式消费，不得贴
        # honest/legacy 标签——否则标签会骗人（见 docs/param_ledger.md §2）。
        eval_result["protocol"] = str(getattr(self.args, "protocol", "none"))
        eval_result["protocol_deviations"] = list(
            getattr(self.args, "protocol_deviations", ())
        )
        try:
            from src.protocols import (
                _CUHLM_MODES,
                _NO_COMM_MODES,
                _TK_SLT_MODES,
                mode_consumption,
            )

            _cons = mode_consumption(str(self.args.eval_mode))
        except Exception:  # pragma: no cover - 诊断信息不应影响评测主流程
            _cons = None
        if _cons is not None:
            eval_result["comm_accounting_consumed"] = _cons.consumes_accounting
            eval_result["depth_keys"] = list(_cons.depth_keys)
            eval_result["consumption_source"] = _cons.source
            _mode = str(self.args.eval_mode)
            if (
                _cons.consumes_accounting
                and _cons.accounting == ("comm_round_trip_mode",)
                and (_mode in _CUHLM_MODES or _mode in _TK_SLT_MODES)
            ):
                # 2026-10-09 统一往返（docs/protocol.md §3.4）：CUHLM/TK-SLT
                # 系载荷字节按各自论文口径，但往返次数接入 comm_round_trip_mode
                # （per_round = 每次云端交互 1×NTT）。标 honest 会overclaim
                # 残差计费，标 paper 会漏掉往返统一——专用标签 paper_rt。
                label = "paper_rt"
            elif not _cons.consumes_accounting:
                if _mode in _NO_COMM_MODES:
                    label = "n/a"  # 无通信阶段：计费标签无意义
                elif _mode in _CUHLM_MODES:
                    # CU-HLM 论文自身口径（不是仓库统一开关的混合体）
                    label = "paper"
                elif _mode in _TK_SLT_MODES:
                    # TK-SLT 论文自身口径（γ·K·b_prob/FP16 上行，
                    # 索引与下行 negligible——docs/protocol.md §3.3）
                    label = "paper"
                else:
                    label = "mixed"  # 标称预设不描述本模式的实际行为
        eval_result["comm_accounting"] = label
        eval_result["stochastic_ntt"] = bool(
            getattr(self.args, "stochastic_ntt", False)
        )
        ntt_trace_file = getattr(self.args, "ntt_trace_file", "")
        eval_result["ntt_trace"] = (
            os.path.basename(ntt_trace_file) if ntt_trace_file else ""
        )
        if ntt_trace_file:
            eval_result["ntt_trace_scale"] = getattr(
                self.args, "ntt_trace_scale", 1.0
            )

        def _link_stats(key, scale=1.0):
            vals = [v * scale for v in (metrics.get(key) or [])]
            if not vals:
                return None
            return {"min": min(vals), "mean": sum(vals) / len(vals), "max": max(vals)}

        eval_result["edge_cloud_ntt_ms_stats"] = _link_stats(
            "edge_cloud_ntt_history"
        )
        eval_result["edge_cloud_bandwidth_mbps_stats"] = _link_stats(
            "edge_cloud_bandwidth_history"
        )
        return eval_result
