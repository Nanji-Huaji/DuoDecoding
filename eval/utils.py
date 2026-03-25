from src.engine import DecodingMetrics, get_empty_metrics, INT_SIZE, ArgsLike
from typing import Protocol
import json


class ExpPrint:
    common_print_metrics = (
        "little_forward_times",
        "draft_forward_times",
        "target_forward_times",
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
        key_to_dump = list(self.common_print_metrics)
        if self.args.dump_network_stats:
            key_to_dump += [
                "edge_cloud_bandwidth_history",
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
        return eval_result
