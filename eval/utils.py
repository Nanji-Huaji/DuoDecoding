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
        charge = bool(getattr(self.args, "charge_residual_payload", False))
        mode = str(getattr(self.args, "comm_round_trip_mode", "per_transfer"))
        cap = int(getattr(self.args, "transfer_top_k_cap", 0) or 0)
        if charge and mode == "per_round" and cap > 0:
            label = "honest"
        elif (not charge) and mode == "per_transfer" and cap == 0:
            label = "legacy"
        else:
            label = "custom"
        eval_result["comm_accounting"] = label
        eval_result["charge_residual_payload"] = charge
        eval_result["comm_round_trip_mode"] = mode
        eval_result["transfer_top_k_cap"] = cap
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
