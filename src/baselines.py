import warnings
from typing import List, Tuple, Dict, Any, TypedDict, Union, Optional

import torch
import torch.distributed as dist
import numpy as np
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    MaxLengthCriteria,
)
from accelerate import Accelerator
import llama_cpp
import draftretriever

import safetensors

import json


from .SpecDec_pp.specdec_pp.wrap_model import AcceptancePredictionHead

import os

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")

from .model_gpu import KVCacheModel
from .model_cpu import KVCacheCppModel
from .utils import seed_everything, norm_logits, sample, max_fn
from .model.rest.rest.model.utils import *
from .model.rest.rest.model.rest_model import RestModel
from .model.rest.rest.model.kv_cache import initialize_past_key_values
from .model.pld.pld import greedy_search_pld
from .communication import (
    CommunicationSimulator,
    CUHLM,
    PreciseCommunicationSimulator,
    PreciseCUHLM,
)
from .engine import Decoding, DecodingMetrics, get_empty_metrics, INT_SIZE


from .adapter import DecodingAdapter

from typing import Callable


from safetensors.torch import load_file

def get_decoding_fn(instance: "Baselines", name: str) -> Callable:
    if hasattr(instance, name):
        method = getattr(instance, name)
        if callable(method):
            return method
        else:
            raise ValueError(
                f"Attribute '{name}' in {instance.__class__.__name__} is not callable"
            )
    else:
        raise ValueError(
            f"Decoding method '{name}' not found in class {instance.__class__.__name__}"
        )


class Baselines(Decoding):
    """
    用于实验的方法。
    包含：
    - dssd
    - Uncertainty Decoding
    - dsd
    - Tridecoding
    """

    def __init__(self, args):
        super().__init__(args)
        if self.args.eval_mode == "adaptive_decoding":
            self.acc_head_path = args.acc_head_path
            self.acc_head = AcceptancePredictionHead.from_pretrained(
                self.acc_head_path,
            ).to("cuda:0")
            self.acc_head.eval()
            self.adapter = DecodingAdapter(
                self.acc_head, 0.8
            )

    @torch.no_grad()
    def dist_split_spec(
        self,
        prefix: torch.Tensor,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        串行传输 token 和 prob 的 speculative decoding
        - transfer_top_k: 用于传输的top-k压缩参数，None或0表示不压缩
        - use_precise_comm_sim: 是否使用物理的通信模拟器
        """
        if use_precise_comm_sim:
            comm_simulator = PreciseCommunicationSimulator(
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx: int = 0

        while prefix.shape[1] < max_tokens:

            prefix_len = prefix.shape[1]

            idx += 1

            # 确保不会生成超过max_tokens的token
            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            # 调整gamma以不超过剩余的token数量
            current_gamma = min(
                self.args.gamma, remaining_tokens - 1
            )  # 减1是为了留给最后的采样token
            if current_gamma <= 0:
                # 如果只剩1个token，直接用target model生成
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                total_accepted_tokens += 1
                self.num_acc_tokens.append(1)
                break

            x = approx_model_cache.generate(
                prefix.to(draft_device), current_gamma
            )
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            _ = target_model_cache.generate(x.to(target_device), 1)

            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            n = prefix_len + current_gamma - 1
            for i in range(current_gamma):
                # 检查索引是否合法
                draft_idx = prefix_len + i - 1
                target_idx = prefix_len + i - 1

                if draft_idx >= approx_model_cache._prob_history.shape[1]:
                    comm_simulator.send_reject_message("edge_cloud")
                    break
                if target_idx >= target_model_cache._prob_history.shape[1]:
                    comm_simulator.send_reject_message("edge_cloud")
                    break

                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                # 传输 token id 和 prob 用于 rejection sampling
                comm_simulator.transfer(
                    j,
                    approx_model_cache._prob_history[:, draft_idx, j],
                    "edge_cloud",
                )

                if r > (
                    target_model_cache._prob_history.to(draft_device)[
                        :, target_idx, j
                    ]
                ) / (approx_model_cache._prob_history[:, draft_idx, j]):
                    n = prefix_len + i - 1
                    comm_simulator.send_reject_message("edge_cloud")
                    break

            this_step_accepted_tokens = n - prefix_len + 1
            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            approx_model_cache.rollback(n + 1)

            # 检查是否还有空间添加一个token
            if prefix.shape[1] >= max_tokens:
                break

            if n < prefix_len + current_gamma - 1:
                # reject someone, sample from the pos n

                if transfer_top_k is not None and transfer_top_k > 0:
                    rebuild_probs = comm_simulator._apply_top_k_compression(
                        approx_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ],
                        transfer_top_k,
                    )
                    rebuild_probs = comm_simulator.rebuild_full_probs(
                        rebuild_probs
                    )
                    approx_model_cache._prob_history[
                        :, n, : self.vocab_size
                    ] = rebuild_probs  # 如果用了top-k压缩，先重建概率分布

                # 发生拒绝，传输被拒绝的 token 的 full prob 用于采样
                comm_simulator.transfer(
                    None,
                    approx_model_cache._prob_history[:, n, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )

                t = sample(
                    max_fn(
                        target_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ].to(draft_device)
                        - approx_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ]
                    )
                )

                new_generated_tokens = (
                    prefix.shape[1] - current_tokens.shape[1] + 1
                )

                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                target_model_cache.rollback(n + 2)

                new_generated_tokens = (
                    prefix.shape[1] - current_tokens.shape[1] + 1
                )

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)
                total_accepted_tokens += 1

            # 传输新生成的 token id
            comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = (
            generated_tokens
            / (elapsed_time + comm_simulator.edge_cloud_comm_time)
            if (elapsed_time + comm_simulator.edge_cloud_comm_time) > 0
            else 0
        )

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = (
            elapsed_time + comm_simulator.edge_cloud_comm_time
        )
        metrics["throughput"] = throughput
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        return prefix, metrics

    @torch.no_grad()
    def dist_spec(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if use_precise_comm_sim:
            comm_simulator = PreciseCommunicationSimulator(
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        self.color_print(f"Using transfer_top_k: {transfer_top_k}", 2)

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        start_event.record(stream=torch.cuda.current_stream())

        idx = 0

        while prefix.shape[1] < max_tokens:

            idx += 1

            prefix_len = prefix.shape[1]

            # 确保不会生成超过max_tokens的token
            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            if idx == 1:
                comm_simulator.transfer(
                    prefix, None, "edge_cloud"
                )  # 初始上下文传输

            # 调整gamma以不超过剩余的token数量
            current_gamma = min(
                self.args.gamma, remaining_tokens - 1
            )  # 减1是为了留给最后的采样token
            if current_gamma <= 0:
                # 如果只剩1个token，直接用target model生成
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                total_accepted_tokens += 1
                self.num_acc_tokens.append(1)
                break

            x = approx_model_cache.generate(
                prefix.to(draft_device), current_gamma
            )
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma
            comm_simulator.transfer(x, None, "edge_cloud")
            _ = target_model_cache.generate(x.to(target_device), 1)

            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            n = prefix_len + current_gamma - 1

            if transfer_top_k is not None and transfer_top_k > 0:
                approx_model_cache._prob_history[
                    :, -(1 + current_gamma) : -1, :
                ] = comm_simulator.compress_rebuild_probs(
                    approx_model_cache._prob_history[
                        :, -(1 + current_gamma) : -1, :
                    ],
                    transfer_top_k,
                )

            comm_simulator.transfer(
                None,
                approx_model_cache._prob_history[
                    :, -(1 + current_gamma) : -1, :
                ],
                "edge_cloud",
                transfer_top_k is not None and transfer_top_k > 0,
                transfer_top_k,
            )

            for i in range(current_gamma):
                # 检查索引是否合法
                draft_idx = prefix_len + i - 1
                target_idx = prefix_len + i - 1

                if draft_idx >= approx_model_cache._prob_history.shape[1]:
                    comm_simulator.send_reject_message("edge_cloud")
                    break
                if target_idx >= target_model_cache._prob_history.shape[1]:
                    comm_simulator.send_reject_message("edge_cloud")
                    break

                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                if r > (
                    target_model_cache._prob_history.to(draft_device)[
                        :, target_idx, j
                    ]
                ) / (approx_model_cache._prob_history[:, draft_idx, j]):
                    n = prefix_len + i - 1
                    comm_simulator.send_reject_message("edge_cloud")
                    break

            this_step_accepted_tokens = n - prefix_len + 1
            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            approx_model_cache.rollback(n + 1)

            # 检查是否还有空间添加一个token
            if prefix.shape[1] >= max_tokens:
                break

            if n < prefix_len + current_gamma - 1:
                # reject someone, sample from the pos n
                # # 压缩
                # if transfer_top_k is not None and transfer_top_k > 0:
                #     rebuild_probs = comm_simulator._apply_top_k_compression(
                #         approx_model_cache._prob_history[
                #             :, n, : self.vocab_size
                #         ],
                #         transfer_top_k,
                #     )
                #     rebuild_probs = comm_simulator.rebuild_full_probs(
                #         rebuild_probs
                #     )
                #     approx_model_cache._prob_history[
                #         :, n, : self.vocab_size
                #     ] = rebuild_probs  # 如果用了top-k压缩，先重建概率分布
                # comm_simulator.transfer(
                #     None,
                #     approx_model_cache._prob_history[:, n, : self.vocab_size],
                #     "edge_cloud",
                #     transfer_top_k is not None and transfer_top_k > 0,
                #     transfer_top_k,
                # )

                t = sample(
                    max_fn(
                        target_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ].to(draft_device)
                        - approx_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ]
                    )
                )
                new_generated_tokens = (
                    prefix.shape[1] - current_tokens.shape[1] + 1
                )

                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)

                new_generated_tokens = (
                    prefix.shape[1] - current_tokens.shape[1] + 1
                )

                target_model_cache.rollback(n + 2)

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)
                total_accepted_tokens += 1

            comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = (
            generated_tokens
            / (elapsed_time + comm_simulator.edge_cloud_comm_time)
            if (elapsed_time + comm_simulator.edge_cloud_comm_time) > 0
            else 0
        )

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = (
            elapsed_time + comm_simulator.edge_cloud_comm_time
        )
        metrics["throughput"] = throughput
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times

        return prefix, metrics

    @torch.no_grad()
    def uncertainty_decoding(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim=False,
        ntt_ms_edge_cloud: float = 200,
        ntt_ms_edge_end: float = 20,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        Implement of the method raised in "Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission"
        """
        if use_precise_comm_sim:
            comm_simulator = PreciseCUHLM(
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )
        else:
            comm_simulator = CUHLM(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                uncertainty_threshold=0.8,
                dimension="Mbps",
            )

        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        # Metrics Tracking
        target_forward_times = 0
        draft_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        loop_idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record(stream=torch.cuda.current_stream())

        input_len = prefix.shape[1]

        is_accepted_last_step = False

        while prefix.shape[1] < max_tokens:
            loop_idx += 1
            prefix_len = prefix.shape[1]

            # 传输 prompt
            if loop_idx == 1:
                comm_simulator.transfer(prefix, None, link_type="edge_cloud")

            # Sync
            x = approx_model_cache.generate(prefix.to(draft_device), 1)
            _ = target_model_cache.generate(x.to(target_device), 1)

            # 无论接受与否，都要传输起草的 token
            comm_simulator.transfer(x, None, link_type="edge_cloud")
            current_logit = approx_model_cache.logits_history[
                :, -1, : self.vocab_size
            ]
            assert (
                current_logit is not None
            ), "Logits history should not be None"
            uncertainty = comm_simulator.calculate_uncertainty(
                current_logit, M=20, theta_max=2.0, draft_token=x[0, -1].item()
            )
            should_transfer, vocab_size = (
                comm_simulator.determine_transfer_strategy(
                    uncertainty, current_logit
                )
            )

            draft_forward_times += 1
            if not is_accepted_last_step:
                target_forward_times += 1
            else:
                # 如果上一个token被接受了，等下一次没有被接受，这么做是为了实现简单
                target_forward_times += 0

            total_drafted_tokens += 1

            n = prefix_len + 1 - 1

            if not should_transfer:

                is_accepted_last_step = True

                # 接受draft token - 仿照接受所有token的情况
                accepted_token = x[:, -1:]  # draft token
                prefix = torch.cat((prefix, accepted_token), dim=1)

                comm_simulator.send_accept_message(
                    linktype="edge_cloud"
                )  # 发送消息告知应该接受

                # KVCache管理：仿照接受所有token的情况
                # 由于我们接受了draft token，需要从target model采样一个新token
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)

                # rollback target_model_cache，因为我们已经消费了它的输出
                # 这里n相当于prefix_len（接受了1个token）
                n = prefix_len  # 接受了位置为prefix_len的token
                target_model_cache.rollback(
                    n + 2
                )  # 等同于rollback(prefix_len + 2)

                total_accepted_tokens += 1

                # 将新采样的token添加到序列中
                if prefix.shape[1] < max_tokens:
                    prefix = torch.cat((prefix, t), dim=1)

                comm_simulator.transfer(
                    t, None, link_type="edge_cloud"
                )  # 传输接受的token和新采样的token

                continue

            is_accepted_last_step = False

            # 拒绝采样

            # 压缩
            current_probs = comm_simulator._get_current_probs(
                approx_model_cache._prob_history
            )
            compressed_prob = comm_simulator._apply_top_k_compression(
                current_probs, vocab_size
            )

            rebuild_probs = comm_simulator.rebuild_full_probs(compressed_prob)
            approx_model_cache._prob_history[:, -1, : self.vocab_size] = (
                rebuild_probs  # 完成概率的重建
            )

            r = torch.rand(1, device=draft_device)
            j = x[:, prefix_len]

            self.color_print(
                f"Uncertainty: {uncertainty:.4f}, Vocab size: {vocab_size}", 3
            )

            if r > (
                target_model_cache._prob_history.to(draft_device)[
                    :, prefix_len - 1, j
                ]
            ) / (approx_model_cache._prob_history[:, prefix_len - 1, j]):
                n = prefix_len - 1
                comm_simulator.send_reject_message(
                    linktype="edge_cloud"
                )  # 发送消息告知应该拒绝、
                comm_simulator.transfer(
                    None,  # 一开始已经传输过
                    approx_model_cache._prob_history[:, -1, : self.vocab_size],
                    link_type="edge_cloud",
                    is_compressed=True,
                    compressed_k=vocab_size,
                )

            total_accepted_tokens += n - prefix_len + 1

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            approx_model_cache.rollback(n + 1)

            if n < prefix_len:
                # reject someone, sample from the pos n
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ].to(draft_device)
                        - approx_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ]
                    )
                )
                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                target_model_cache.rollback(n + 2)

            comm_simulator.transfer(
                t, None, link_type="edge_cloud"
            )  # 传输新采样的token
            prefix = torch.cat((prefix, t), dim=1)

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        metrics = get_empty_metrics()

        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = prefix.shape[1] - input_len
        metrics["draft_generated_tokens"] = draft_forward_times
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = (
            elapsed_time + comm_simulator.edge_cloud_comm_time
        )
        metrics["throughput"] = (
            (prefix.shape[1] - input_len) / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        return prefix, metrics

    @torch.no_grad()
    def tridecoding(
        self,
        prefix,
        transfer_top_k=300,
        use_precise_comm_sim=False,
        ntt_ms_edge_cloud: float = 10,
        ntt_ms_edge_end: float = 1,
        **kwargs
    ):
        max_tokens = prefix.shape[1] + self.args.max_tokens
        little_device = self.little_model.device
        draft_device = self.draft_model.device
        target_device = self.target_model.device
        little_model_cache = KVCacheModel(
            self.little_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        little_model_cache.vocab_size = self.vocab_size
        draft_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        draft_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        if use_precise_comm_sim:
            comm_simulator = PreciseCommunicationSimulator(
                bandwidth_hz=1e7,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
                ntt_ms_edge_cloud=ntt_ms_edge_cloud,
                ntt_ms_edge_end=ntt_ms_edge_end,
            )

        # Metrics tracking
        little_model_forward_times = 0
        draft_model_forward_times = 0
        target_model_forward_times = 0
        total_little_model_generated_tokens = 0
        total_draft_model_generated_tokens = 0
        total_little_model_accepted_tokens = 0
        total_draft_model_accepted_tokens = 0
        wall_time = 0

        idx = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()  # 用于计算生成token数

        start_event.record(stream=torch.cuda.current_stream())

        comm_simulator.transfer(
            prefix, None, "edge_end"
        )  # 将 prompt 传输到 edge

        while prefix.shape[1] < max_tokens:

            idx += 1

            prefix_len = prefix.shape[1]

            # 第一层 speculative

            x = little_model_cache.generate(
                prefix.to(little_device), self.args.gamma2
            )
            _ = draft_model_cache.generate(x.to(draft_device), 1)

            little_model_forward_times += self.args.gamma2
            draft_model_forward_times += 1
            total_little_model_generated_tokens += self.args.gamma2

            n1: int = prefix_len + self.args.gamma2 - 1

            little_accepted_this_iter = 0
            for i in range(self.args.gamma2):
                r = torch.rand(1, device=little_device)
                j = x[:, prefix_len + i]

                # 传输 token id 和 prob 用于 reject sampling
                comm_simulator.transfer(
                    j,
                    little_model_cache._prob_history[:, prefix_len + i - 1, j],
                    "edge_end",
                )

                if r > (
                    draft_model_cache._prob_history.to(little_device)[
                        :, prefix_len + i - 1, j
                    ]
                ) / (
                    little_model_cache._prob_history[:, prefix_len + i - 1, j]
                ):
                    comm_simulator.send_reject_message("edge_end")
                    n1 = prefix_len + i - 1

                    break

                else:
                    little_accepted_this_iter += 1

            total_little_model_accepted_tokens += little_accepted_this_iter

            assert n1 >= prefix_len - 1, f"n {n1}, prefix_len {prefix_len}"
            prefix = x[:, : n1 + 1]

            little_model_cache.rollback(n1 + 1)

            if n1 < prefix_len + self.args.gamma2 - 1:
                # reject someone, sample from the pos n1
                rebuild_probs = comm_simulator.rebuild_full_probs(
                    little_model_cache._prob_history[:, n1, : self.vocab_size]
                )
                little_model_cache._prob_history[:, n1, : self.vocab_size] = (
                    rebuild_probs
                )

                comm_simulator.transfer(
                    None,
                    little_model_cache._prob_history[:, n1, : self.vocab_size],
                    "edge_end",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )

                t = sample(
                    max_fn(
                        draft_model_cache._prob_history[
                            :, n1, : self.vocab_size
                        ].to(little_device)
                        - little_model_cache._prob_history[
                            :, n1, : self.vocab_size
                        ]
                    )
                )

                draft_model_cache.rollback(n1 + 1)

            else:
                t = sample(
                    draft_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(little_device)

                draft_model_cache.rollback(n1 + 2)

            # 传输索引
            comm_simulator.simulate_transfer(INT_SIZE, "edge_end")
            comm_simulator.transfer(t, None, "edge_end")

            prefix = torch.cat((prefix, t), dim=1)
            new_generated_token = prefix[:, prefix_len:]

            # 第二层 speculative

            if idx == 1:
                comm_simulator.transfer(prefix, None, "edge_cloud")
            else:
                comm_simulator.transfer(new_generated_token, None, "edge_cloud")

            x = draft_model_cache.generate(
                prefix.to(draft_device), self.args.gamma1
            )

            _ = target_model_cache.generate(x.to(target_device), 1)

            draft_model_forward_times += self.args.gamma1
            target_model_forward_times += 1
            total_draft_model_generated_tokens += self.args.gamma1

            n2: int = (
                prefix_len + new_generated_token.shape[1] + self.args.gamma1 - 1
            )
            draft_accepted_this_iter = 0
            for i in range(
                new_generated_token.shape[1] + self.args.gamma1,
            ):
                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                # 传输 token id 和 prob 用于 reject sampling
                comm_simulator.transfer(
                    j,
                    draft_model_cache._prob_history[:, prefix_len + i - 1, j],
                    "edge_cloud",
                )

                if r > (
                    target_model_cache._prob_history.to(draft_device)[
                        :, prefix_len + i - 1, j
                    ]
                ) / (draft_model_cache._prob_history[:, prefix_len + i - 1, j]):
                    n2 = prefix_len + i - 1
                    comm_simulator.send_reject_message("edge_cloud")
                    break
                else:
                    draft_accepted_this_iter += 1
            total_draft_model_accepted_tokens += draft_accepted_this_iter

            assert n2 >= prefix_len - 1, f"n {n2}, prefix_len {prefix_len}"
            prefix = x[:, : n2 + 1]
            draft_model_cache.rollback(n2 + 1)
            if n2 <= little_model_cache.current_length:
                little_model_cache.rollback(n2 + 1)
            if n2 < prefix_len + self.args.gamma1 - 1:

                rebuild_probs = comm_simulator.rebuild_full_probs(
                    draft_model_cache._prob_history[:, n2, : self.vocab_size]
                )
                draft_model_cache._prob_history[:, n2, : self.vocab_size] = (
                    rebuild_probs
                )

                comm_simulator.transfer(
                    None,
                    draft_model_cache._prob_history[:, n2, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[
                            :, n2, : self.vocab_size
                        ].to(draft_device)
                        - draft_model_cache._prob_history[
                            :, n2, : self.vocab_size
                        ]
                    )
                )
                new_generated_token = prefix[:, prefix_len:]

                target_model_cache.rollback(n2 + 1)

            else:
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                new_generated_token = prefix[:, prefix_len:]

                target_model_cache.rollback(n2 + 2)

            prefix = torch.cat((prefix, t), dim=1)
            # 传输索引
            comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")
            comm_simulator.transfer(t, None, "edge_cloud")
            comm_simulator.simulate_transfer(INT_SIZE, "edge_end")
            comm_simulator.transfer(t, None, "edge_end")
            # 同步

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        wall_time += elapsed_time
        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        wall_time += (
            comm_simulator.edge_cloud_comm_time
            + comm_simulator.edge_end_comm_time
        )

        metrics = get_empty_metrics()
        metrics["little_forward_times"] = little_model_forward_times
        metrics["draft_forward_times"] = draft_model_forward_times
        metrics["target_forward_times"] = target_model_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["little_generated_tokens"] = total_little_model_generated_tokens
        metrics["draft_generated_tokens"] = total_draft_model_generated_tokens
        metrics["little_accepted_tokens"] = total_little_model_accepted_tokens
        metrics["draft_accepted_tokens"] = total_draft_model_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = (
            metrics["generated_tokens"] / wall_time if wall_time > 0 else 0
        )
        metrics["communication_time"] = (
            comm_simulator.edge_cloud_comm_time
            + comm_simulator.edge_end_comm_time
        )
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy
        metrics["connect_times"] = comm_simulator.connect_times
        return prefix, metrics

    @torch.no_grad()
    def adaptive_decoding(self, prefix, **kwargs) -> Tuple[torch.Tensor, DecodingMetrics]:
        max_tokens = prefix.shape[1] + self.args.max_tokens

        # TODO: 有质量问题

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = self.vocab_size
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = self.vocab_size

        draft_forward_times = 0
        target_forward_times = 0
        total_accepted_tokens = 0
        total_drafted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        current_tokens = prefix.clone()

        loop_idx = 0

        start_event.record(stream=torch.cuda.current_stream())

        while prefix.shape[1] < max_tokens:

            loop_idx += 1

            prefix_len = prefix.shape[1]

            draft_kvcache_length = approx_model_cache.current_length
            target_kvcache_length = target_model_cache.current_length

            least_generated_tokens = 3

            # 确保不会生成超过max_tokens的token
            remaining_tokens = max_tokens - prefix_len
            if remaining_tokens <= 0:
                break

            # 调整gamma以不超过剩余的token数量
            current_gamma = min(
                self.args.gamma, remaining_tokens - 1
            )  # 减1是为了留给最后的采样token
            if current_gamma <= 0:
                # 如果只剩1个token，直接用target model生成
                _ = target_model_cache.generate(prefix.to(target_device), 1)
                target_forward_times += 1
                if self.accelerator.is_main_process:
                    self.target_forward_times += 1

                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                total_accepted_tokens += 1
                self.num_acc_tokens.append(1)
                break

            # x = approx_model_cache.generate(
            #     prefix.to(draft_device), current_gamma
            # )

            actual_gamma = 0

            for _ in range(current_gamma):
                actual_gamma += 1
                x = approx_model_cache.generate(
                    x, 1
                ) if actual_gamma != 1 else approx_model_cache.generate(
                    prefix.to(draft_device), 1
                )
                prefix_len = prefix.shape[1]
                stop = self.adapter.predict(
                    approx_model_cache.hidden_states,
                ) and False
                if stop and x.shape[-1] - prefix_len > least_generated_tokens:
                    break



            actual_gamma = current_gamma

            current_gamma = actual_gamma
            print(f"Adaptive gamma: {current_gamma}")


            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            _ = target_model_cache.generate(x.to(target_device), 1)
            target_forward_times += 1


            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            n = prefix_len + current_gamma - 1
            for i in range(current_gamma):
                # 检查索引是否合法
                draft_idx = prefix_len + i - 1
                target_idx = prefix_len + i - 1

                if draft_idx >= approx_model_cache._prob_history.shape[1]:
                    break
                if target_idx >= target_model_cache._prob_history.shape[1]:
                    break

                r = torch.rand(1, device=draft_device)
                j = x[:, prefix_len + i]

                if r > (
                    target_model_cache._prob_history.to(draft_device)[
                        :, target_idx, j
                    ]
                ) / (approx_model_cache._prob_history[:, draft_idx, j]):
                    n = prefix_len + i - 1
                    break

            this_step_accepted_tokens = n - prefix_len + 1
            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"
            prefix = x[:, : n + 1]

            approx_model_cache.rollback(n + 1)

            # 检查是否还有空间添加一个token
            if prefix.shape[1] >= max_tokens:
                break

            if n < prefix_len + current_gamma - 1:
                # reject someone, sample from the pos n
                t = sample(
                    max_fn(
                        target_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ].to(draft_device)
                        - approx_model_cache._prob_history[
                            :, n, : self.vocab_size
                        ]
                    )
                )
                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                target_model_cache.rollback(n + 2)

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)
                total_accepted_tokens += 1

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]
        throughput = generated_tokens / elapsed_time if elapsed_time > 0 else 0

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = elapsed_time
        metrics["throughput"] = throughput
        metrics["loop_times"] = loop_idx
        metrics["each_loop_draft_tokens"] = (
            total_drafted_tokens / loop_idx if loop_idx > 0 else 0
        )


        return prefix, metrics
