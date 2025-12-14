import torch
import json
import torch.distributed as dist
import numpy as np
import os
import transformers
import warnings

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from transformers import AutoModelForCausalLM, AutoTokenizer
import llama_cpp
from abc import ABC, abstractmethod
from accelerate import Accelerator
from .model_gpu import KVCacheModel
from .model_cpu import KVCacheCppModel
from .utils import seed_everything, norm_logits, sample, max_fn
import time

from transformers import StoppingCriteriaList, MaxLengthCriteria

from .model.rest.rest.model.utils import *
from .model.rest.rest.model.rest_model import RestModel
from .model.rest.rest.model.kv_cache import initialize_past_key_values
import draftretriever

from typing import List, Tuple, Dict, Any, TypedDict, Union, Optional

from .communication import (
    CommunicationSimulator,
    CUHLM,
    PreciseCommunicationSimulator,
    PreciseCUHLM,
)

from typing import Literal

try:
    import flash_attn
except ImportError:
    pass

flash_attn_available = "flash_attn" in globals()

attn_impl = "sdpa" if not flash_attn_available else "flash_attention_2"

INT_SIZE = 4

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

class DecodingMetrics(TypedDict):
    """
    TypedDict class that defines metrics for tracking decoding performance and resource usage.

    This class serves as a type annotation for dictionaries containing comprehensive
    metrics about the decoding process, including forward pass counts, token statistics,
    timing information, communication overhead, and energy consumption.
    """

    little_forward_times: int
    draft_forward_times: int
    target_forward_times: int
    generated_tokens: int
    little_generated_tokens: int
    draft_generated_tokens: int
    little_accepted_tokens: int
    draft_accepted_tokens: int
    wall_time: float
    throughput: float
    communication_time: float
    computation_time: float
    edge_end_comm_time: float
    edge_cloud_data_bytes: int | float
    edge_end_data_bytes: int | float
    cloud_end_data_bytes: int | float
    loop_times: int
    each_loop_draft_tokens: float
    comm_energy: float
    connect_times: dict


def get_empty_metrics() -> DecodingMetrics:
    """
    Create and return an empty DecodingMetrics object with all fields initialized to zero.
    """
    return DecodingMetrics(
        little_forward_times=0,
        draft_forward_times=0,
        target_forward_times=0,
        generated_tokens=0,
        little_generated_tokens=0,
        draft_generated_tokens=0,
        little_accepted_tokens=0,
        draft_accepted_tokens=0,
        wall_time=0.0,
        throughput=0.0,
        communication_time=0.0,
        computation_time=0.0,
        edge_end_comm_time=0.0,
        edge_cloud_data_bytes=0,
        edge_end_data_bytes=0,
        cloud_end_data_bytes=0,
        loop_times=0,
        each_loop_draft_tokens=0,
        comm_energy=0.0,
        connect_times={"edge_end": 0, "cloud_end": 0, "edge_cloud": 0},
    )


class Decoding(ABC):
    def __init__(self, args):
        self.args = args
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            size = int(os.environ["WORLD_SIZE"])
            if "duodec" in args.eval_mode:
                dist.init_process_group(
                    "gloo", init_method="env://", rank=rank, world_size=size
                )
            else:
                dist.init_process_group(
                    "nccl", init_method="env://", rank=rank, world_size=size
                )
        self.accelerator = Accelerator()

        seed_everything(self.args.seed)
        self.seed = self.args.seed
        self.seed_set = set()

        # record metrics for report
        self.draft_forward_times = 0
        self.little_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.prob_with_flag = []

    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(
            f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}",
            3,
        )
        if self.args.eval_mode == "small":
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()
        elif self.args.eval_mode == "large":
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()
        elif self.args.eval_mode in [
            "sd",
            "dist_spec",
            "dist_split_spec",
            "uncertainty_decoding",
            "speculative_decoding_with_bandwidth",
            "speculative_decoding_with_bandwidth_full_prob",

        ]:
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="balanced_low_0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()

        elif self.args.eval_mode in ["para_sd", "para_sd_wo_1", "para_sd_wo_1"]:
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_model,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                    attn_implementation=attn_impl,
                ).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model,
                    device_map="balanced_low_0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                    attn_implementation=attn_impl,
                ).eval()

        elif self.args.eval_mode == "rc_para_sd":
            if self.accelerator.is_main_process:
                self.draft_model = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_model,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()
                self.draft_model_2 = AutoModelForCausalLM.from_pretrained(
                    self.args.draft_model,
                    device_map=f"cuda:{torch.cuda.device_count()-1}",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()

        elif self.args.eval_mode == "duodec":
            if self.accelerator.is_main_process:
                self.draft_model = llama_cpp.Llama(
                    model_path=self.args.draft_model,
                    n_ctx=4096,
                    verbose=False,
                    n_threads=16,
                )
            else:
                self.target_model = AutoModelForCausalLM.from_pretrained(
                    self.args.target_model,
                    device_map="cuda:0",
                    torch_dtype=torch.bfloat16,
                    trust_remote_code=True,
                    cache_dir="llama/.cache/huggingface",
                    local_files_only=True,
                ).eval()

        elif self.args.eval_mode == "lade":
            from .model.lade.utils import augment_all, config_lade
            from .model.lade.decoding import CONFIG_MAP

            if int(os.environ.get("USE_LADE", 0)):
                augment_all()
                config_lade(
                    LEVEL=self.args.level,
                    WINDOW_SIZE=self.args.window,
                    GUESS_SET_SIZE=self.args.guess,
                    DEBUG=0,
                    USE_FLASH=0,
                    DIST_WORKERS=len(
                        os.environ.get("CUDA_VISIBLE_DEVICES").split(",")
                    ),
                )
                print("lade activated config: ", CONFIG_MAP)
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
            ).eval()
        elif self.args.eval_mode == "rest":
            self.model = RestModel.from_pretrained(
                self.args.target_model,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
            )
            self.token_spans = list(range(2, self.args.max_token_span + 1))[
                ::-1
            ]
            self.datastore = draftretriever.Reader(
                index_file_path=self.args.datastore_path,
            )
        elif self.args.eval_mode in ["tridecoding", "tridecoding_with_bandwidth"]:
            self.little_model = AutoModelForCausalLM.from_pretrained(
                self.args.little_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()
        elif self.args.eval_mode == "adaptive_decoding":

            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="cuda:0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
                output_hidden_states=True,
            ).eval()
            
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="balanced_low_0",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()

        elif self.args.eval_mode == "adaptive_tridecoding":
            self.little_model = AutoModelForCausalLM.from_pretrained(
                self.args.little_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
                output_hidden_states=True,
            ).eval()
            self.draft_model = AutoModelForCausalLM.from_pretrained(
                self.args.draft_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
                output_hidden_states=True,
            ).eval()
            self.target_model = AutoModelForCausalLM.from_pretrained(
                self.args.target_model,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                cache_dir="llama/.cache/huggingface",
                local_files_only=True,
                attn_implementation=attn_impl,
            ).eval()

        self.vocab_size = self.args.vocab_size

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.target_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_model,
            trust_remote_code=True,
            cache_dir="llama/.cache/huggingface",
            local_files_only=True,
        )
        self.tokenizer.padding_side = "right"

        # for llama models
        self.tokenizer.pad_token_id = 2

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @torch.no_grad()
    def autoregressive_sampling(
        self, prefix, **kwargs
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if self.args.eval_mode == "small":
            model = self.draft_model
        elif self.args.eval_mode == "large":
            model = self.target_model
        else:
            raise RuntimeError(
                "Auto-Regressive Decoding can be used only in small / large eval mode!"
            )
        prefix = prefix.to(model.device)
        model = KVCacheModel(
            model, self.args.temp, self.args.top_k, self.args.top_p
        )
        model.vocab_size = self.args.vocab_size

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens

        x = prefix

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        target_forward_times = 0

        start_event.record(stream=torch.cuda.current_stream())
        while x.shape[1] < max_tokens:
            x = model.generate(x, 1)
            target_forward_times += 1

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert to seconds
        generated_tokens = x.shape[1] - prefix_len
        throughput = generated_tokens / elapsed_time if elapsed_time > 0 else 0

        metrics = get_empty_metrics()
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["wall_time"] = elapsed_time
        metrics["throughput"] = throughput

        return x, metrics

    @torch.no_grad()
    def speculative_decoding(
        self, prefix, transfer_top_k: int | None = 300
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
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

        loop_idx = 0

        start_event.record(stream=torch.cuda.current_stream())

        while prefix.shape[1] < max_tokens:

            loop_idx += 1

            prefix_len = prefix.shape[1]

            draft_kvcache_length = approx_model_cache.current_length
            target_kvcache_length = target_model_cache.current_length

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

    @torch.no_grad()
    def speculative_decoding_with_bandwidth(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if use_precise_comm_sim:
            comm_simulator = PreciseCommunicationSimulator(
                bandwidth_hz=1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
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

        return prefix, metrics

    @torch.no_grad()
    def speculative_decoding_with_bandwidth_full_prob(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim: bool = False,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        if use_precise_comm_sim:
            comm_simulator = PreciseCommunicationSimulator(
                bandwidth_hz=1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=float("inf"),
                bandwidth_cloud_end=float("inf"),
                dimension="Mbps",
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

        return prefix, metrics

    @torch.no_grad()
    def uncertainty_decoding_without_kvcache(
        self, prefix
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        comm_simulator = CUHLM(
            bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
            uncertainty_threshold=0.8,
            dimension="Mbps",
        )
        prefix_len = prefix.shape[1]
        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        draft_forwards = 0
        target_forwards = 0
        accepted_tokens = 0

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record(stream=torch.cuda.current_stream())

        while prefix.shape[1] < max_tokens:
            # ==== Draft model forward ====
            output = self.draft_model(prefix.to(draft_device))
            draft_logits = output.logits[:, -1, : self.vocab_size]
            draft_probs = norm_logits(
                draft_logits, self.args.temp, self.args.top_k, self.args.top_p
            )
            draft_token = sample(draft_probs)  # [batch, 1]

            draft_forwards += 1

            # ==== 计算不确定性 ====
            uncertainty = comm_simulator.calculate_uncertainty(
                draft_logits,
                M=20,
                theta_max=2.0,
                draft_token=draft_token[0, 0].item(),
            )
            should_transfer, vocab_size = (
                comm_simulator.determine_transfer_strategy(
                    uncertainty, draft_logits
                )
            )

            if not should_transfer:
                # ==== 直接接受draft token ====
                prefix = torch.cat(
                    (prefix, draft_token.to(prefix.device)), dim=1
                )
                accepted_tokens += 1
                continue

            # ==== Target model验证 ====
            extended_sequence = torch.cat(
                (prefix.to(target_device), draft_token.to(target_device)), dim=1
            )
            output = self.target_model(extended_sequence)
            target_logits = output.logits[
                :, -2, : self.vocab_size
            ]  # 最后一个位置（draft token位置）
            target_probs = norm_logits(
                target_logits, self.args.temp, self.args.top_k, self.args.top_p
            )

            target_forwards += 1

            # ==== Accept/Reject逻辑 ====
            r = torch.rand(1, device=draft_device)
            j = draft_token[0, 0].item()

            # 模拟通信
            comm_simulator.simulate_transfer(
                draft_token.numel() + vocab_size * draft_probs.element_size(),
                "edge_cloud",
            )

            # 计算接受概率
            target_prob_j = target_probs.to(draft_device)[0, j]
            draft_prob_j = draft_probs[0, j]
            accept_prob = min(1.0, target_prob_j / draft_prob_j)

            if r <= accept_prob:
                # ==== 接受draft token ====
                prefix = torch.cat(
                    (prefix, draft_token.to(prefix.device)), dim=1
                )
                accepted_tokens += 1
            else:
                # ==== 拒绝，从修正分布采样 ====
                corrected_probs = torch.clamp(
                    target_probs.to(draft_device) - draft_probs, min=0.0
                )

                if corrected_probs.sum() > 0:
                    corrected_probs = corrected_probs / corrected_probs.sum(
                        dim=-1, keepdim=True
                    )
                    corrected_token = sample(corrected_probs)
                else:
                    # 当修正分布全零时，直接从target分布采样
                    corrected_token = sample(target_probs.to(draft_device))

                prefix = torch.cat(
                    (prefix, corrected_token.to(prefix.device)), dim=1
                )
                accepted_tokens += 1

        end_event.record(stream=torch.cuda.current_stream())

        torch.cuda.synchronize()

        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert to seconds

        # ==== 更新统计 ====
        if self.accelerator.is_main_process:
            self.draft_forward_times += draft_forwards
            self.target_forward_times += target_forwards
        self.num_acc_tokens.extend([1] * accepted_tokens)

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forwards
        metrics["target_forward_times"] = target_forwards
        metrics["generated_tokens"] = prefix.shape[1] - prefix_len
        metrics["draft_generated_tokens"] = draft_forwards
        metrics["draft_accepted_tokens"] = accepted_tokens
        metrics["wall_time"] = (
            elapsed_time + comm_simulator.edge_cloud_comm_time
        )
        metrics["throughput"] = (
            (prefix.shape[1] - prefix_len) / metrics["wall_time"]
            if metrics["wall_time"] > 0
            else 0
        )
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["computation_time"] = elapsed_time
        metrics["edge_end_comm_time"] = comm_simulator.edge_end_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data
        metrics["edge_end_data_bytes"] = comm_simulator.edge_end_data
        metrics["cloud_end_data_bytes"] = comm_simulator.cloud_end_data

        return prefix, metrics

    @torch.no_grad()
    def uncertainty_decoding(
        self,
        prefix,
        transfer_top_k: Optional[int] = 300,
        use_precise_comm_sim=False,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        """
        Implement of the method raised in "Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission"
        """
        if use_precise_comm_sim:
            comm_simulator = PreciseCUHLM(
                bandwidth_hz=1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
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

        return prefix, metrics

    @torch.no_grad()
    def tridecoding_with_bandwidth(
        self, prefix, transfer_top_k=300, use_precise_comm_sim=False
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
                bandwidth_hz=1e6,
                channel_gain=1e-8,
                send_power_watt=0.5,
                noise_power_watt=1e-10,
            )
        else:
            comm_simulator = CommunicationSimulator(
                bandwidth_edge_cloud=self.args.edge_cloud_bandwidth,
                bandwidth_edge_end=self.args.edge_end_bandwidth,
                bandwidth_cloud_end=self.args.cloud_end_bandwidth,
                transfer_top_k=transfer_top_k,
                dimension="Mbps",
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
            (
                little_model_cache.rollback(n2 + 1)
                if n2 <= little_model_cache.current_length
                else None
            )
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
        return prefix, metrics

    @torch.no_grad()
    def lookahead_forward(self, prefix, **kwargs):
        input_ids = prefix.cuda()

        output_ids, idx, accept_length_list = self.target_model.generate(
            torch.as_tensor(input_ids).cuda(),
            do_sample=False,
            temperature=0.0,
            max_new_tokens=self.args.max_tokens,
            num_assistant_tokens_schedule="constant",
        )
        new_token = len(output_ids[0][len(input_ids[0]) :])
        return output_ids

    @torch.no_grad()
    def rest_forward(self, prefix, **kwargs):
        temperature = self.args.temp
        top_p = self.args.top_p
        num_draft = self.args.num_draft
        token_spans = self.token_spans
        max_steps = 512
        datastore = self.datastore
        max_new_tokens = self.args.max_tokens
        model = self.model
        tokenizer = self.tokenizer

        input_ids = prefix.cuda()
        assert input_ids.shape[0] == 1, "Only support batch size 1 for now!!"
        # Avoid modifying the input_ids in-place
        input_ids = input_ids.clone()
        accept_length_list = []

        # Initialize the past key and value states
        if hasattr(model, "past_key_values"):
            past_key_values = model.past_key_values
            past_key_values_data = model.past_key_values_data
            current_length_data = model.current_length_data
            # Reset the past key and value states
            current_length_data.zero_()
        else:
            (
                past_key_values,
                past_key_values_data,
                current_length_data,
            ) = initialize_past_key_values(model.base_model)
            model.past_key_values = past_key_values
            model.past_key_values_data = past_key_values_data
            model.current_length_data = current_length_data

        input_len = input_ids.shape[1]
        cur_length = input_len
        model.base_model.model.draft_mask = None
        logits = initialize_logits(input_ids, model, past_key_values)
        new_token = 0

        for idx in range(max_steps):
            candidates, tree_candidates, draft_buffers = (
                generate_candidates_and_draft_buffer(
                    logits,
                    input_ids,
                    datastore,
                    token_spans,
                    top_p,
                    temperature,
                    max_num_draft=num_draft,
                    device=model.base_model.device,
                )
            )
            model.base_model.model.draft_mask = draft_buffers["draft_attn_mask"]
            logits, outputs = tree_decoding(
                model,
                tree_candidates,
                past_key_values,
                draft_buffers["draft_position_ids"],
                input_ids,
                draft_buffers["retrieve_indices"],
            )
            best_candidate, accept_length = evaluate_posterior(
                logits, candidates, temperature, top_p
            )
            input_ids, logits, new_token = update_inference_inputs(
                input_ids,
                candidates,
                best_candidate,
                accept_length,
                draft_buffers["retrieve_indices"],
                outputs,
                logits,
                new_token,
                past_key_values_data,
                current_length_data,
            )
            accept_length_tree = input_ids.shape[1] - cur_length
            cur_length = accept_length_tree + cur_length
            accept_length_list.append(accept_length_tree)
            if tokenizer.eos_token_id in input_ids[0, input_len:].tolist():
                break
            if new_token > max_new_tokens:
                break
        return input_ids

    @torch.no_grad()
    def parallel_speculative_decoding(self, prefix, **kwargs):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheModel(
                self.draft_model,
                self.args.temp,
                self.args.top_k,
                self.args.top_p,
            )
            model.vocab_size = self.vocab_size
            device = self.draft_model.device
        else:
            model = KVCacheModel(
                self.target_model,
                self.args.temp,
                self.args.top_k,
                self.args.top_p,
            )
            model.vocab_size = self.vocab_size
            device = self.target_model.device

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0
        start_time = time.time()
        while prefix.shape[1] < max_tokens:
            # if self.accelerator.is_main_process:
            #     print(f"rank 0 verify time {time.time() - start_time}")
            # else:
            #     print(f"rank 1 verify time {time.time() - start_time}")
            prefix_len = prefix.shape[1]

            input_ids = prefix.to(device)
            if self.accelerator.is_main_process:
                x = model.generate(input_ids, self.args.gamma)
                prob = model._prob_history[
                    :,
                    prefix_len - self.args.gamma - 1 : prefix_len,
                    : self.vocab_size,
                ]
                prob[:, 0, 0] = -1
                prob[:, 0, 1 : self.args.gamma * 2] = x[
                    :,
                    prefix_len
                    - self.args.gamma
                    + 1 : prefix_len
                    + self.args.gamma,
                ]
                self.draft_forward_times += self.args.gamma
            else:
                x = model.generate(input_ids, 1)
                prob = model._prob_history[
                    :,
                    prefix_len - self.args.gamma - 1 : prefix_len,
                    : self.vocab_size,
                ]
                prob = prob.to("cuda:1")
                self.target_forward_times += 1

            self.accelerator.wait_for_everyone()
            start_time = time.time()

            # verification
            all_prob = self.accelerator.gather(prob).to(device)
            # dist.barrier()
            # if self.accelerator.is_main_process:
            #     print(f"rank 0 gather time {time.time() - start_time}")
            # else:
            #     print(f"rank 1 gather time {time.time() - start_time}")
            # with self.accelerator.main_process_first():
            #     print(f"rank {self.accelerator.process_index} {all_prob.shape}")
            #     print(prob)
            draft_ids = all_prob[0, [0], 1 : self.args.gamma * 2].int()
            draft_prob = all_prob[[0], 1:, :]
            target_prob = all_prob[[1], 1:, :]
            if cur_mode:
                first_token = draft_ids[:, -self.args.gamma]
                torch.manual_seed(self.seed + prefix_len)

                r = torch.rand(1, device=device)
                if (
                    r
                    > target_prob[:, -1, first_token]
                    / draft_prob[:, -1, first_token]
                ):
                    # reject the first token
                    t = sample(
                        max_fn(target_prob[:, -1, :] - draft_prob[:, -1, :])
                    )
                    prefix = torch.cat((input_ids, t), dim=1)

                    # record the number of accepted tokens
                    self.num_acc_tokens.append(num_acc_token)
                    num_acc_token = 0

                    if self.accelerator.is_main_process:
                        # rollback the small model kv cache
                        model.rollback(prefix_len)
                else:
                    # accept the first token, change the mode
                    cur_mode = False
                    prefix = torch.cat(
                        (input_ids, draft_ids[:, -self.args.gamma :]), dim=1
                    )
                    num_acc_token += 1

            else:
                n = self.args.gamma
                for i in range(self.args.gamma):
                    token = draft_ids[:, i]
                    torch.manual_seed(
                        self.seed + prefix_len - self.args.gamma + i
                    )
                    r = torch.rand(1, device=device)
                    if r > target_prob[:, i, token] / draft_prob[:, i, token]:
                        n = i
                        break
                if n == self.args.gamma:
                    # accept all guess tokens
                    prefix = torch.cat(
                        (input_ids, draft_ids[:, -self.args.gamma :]), dim=1
                    )
                    num_acc_token += self.args.gamma
                else:
                    # reject someone, change the mode
                    assert n < self.args.gamma
                    cur_mode = True
                    t = sample(
                        max_fn(target_prob[:, n, :] - draft_prob[:, n, :])
                    )

                    prefix = torch.cat(
                        (
                            input_ids[
                                :, : prefix_len - self.args.gamma + n + 1
                            ],
                            t,
                        ),
                        dim=1,
                    )
                    self.num_acc_tokens.append(num_acc_token + n)
                    num_acc_token = 0
                    # rollback both the large model and the small model kv cache
                    model.rollback(prefix_len - self.args.gamma + n + 1)
            # if self.accelerator.is_main_process:
            #     print(prefix.shape[1])
        return prefix

    @torch.no_grad()
    def duodecoding(self, prefix, **kwargs):
        # parallel speculative decoding
        if self.accelerator.is_main_process:
            model = KVCacheCppModel(
                self.draft_model,
                self.args.temp,
                self.args.top_k,
                self.args.top_p,
            )
            model.vocab_size = self.vocab_size
            device = "cpu"
        else:
            model = KVCacheModel(
                self.target_model,
                self.args.temp,
                self.args.top_k,
                self.args.top_p,
            )
            model.vocab_size = self.vocab_size
            device = self.target_model.device
            stop_signal = torch.ones(1, device="cpu")

        max_tokens = prefix.shape[1] + self.args.max_tokens

        # this flag is used to determine the current verify mode.
        cur_mode = True
        num_acc_token = 0
        total_time = 0

        comm_tensor = torch.tensor(
            [1, 1, 15000], dtype=torch.int, device="cpu"
        )  # cure_mode, accept_len, new_sampled_token_id

        draft_prob_comm = torch.zeros(self.args.gamma + 1, device="cpu")

        prev_k = 1
        prev_size = self.args.gamma

        prefilling = True
        gamma = self.args.gamma

        while prefix.shape[1] < max_tokens:
            if prefilling:
                self.args.gamma = 1
                prev_size = 1
                prefilling = False
            else:
                self.args.gamma = gamma

            prefix_len = prefix.shape[1]

            if self.accelerator.is_main_process:  # draft model
                input_ids = prefix.tolist()[0]

                flatten_draft_k_seq_ids, draft_k_seq_prob, cur_k = (
                    model.generate_k_seq(input_ids, self.args.gamma)
                )
                # draft_prob = model._prob_history[:, prefix_len - prev_size:prefix_len, :self.vocab_size]

                model.stop_signal[0] = 0.0

                # draft_prob_comm[0, 0, 0] = cur_k
                # draft_prob_comm[0, 0, 1:self.args.gamma + 1] = torch.tensor(flatten_draft_k_seq_ids)
                # draft_prob_comm[0, 1:prev_size+1, :] = torch.from_numpy(draft_prob)

                draft_prob_comm[0] = cur_k
                draft_prob_comm[1 : self.args.gamma + 1] = torch.tensor(
                    flatten_draft_k_seq_ids
                )

                self.draft_forward_times += self.args.gamma
                input_ids = prefix

            else:  # target model
                input_ids = prefix.to(device)
                x = model.generate(input_ids, 1)
                prob = model._prob_history[
                    :, prefix_len - prev_size : prefix_len, : self.vocab_size
                ]
                prob = prob.to(device)
                self.target_forward_times += 1

            if self.accelerator.is_main_process:
                dist.send(draft_prob_comm, dst=1)
            else:
                dist.recv(draft_prob_comm, src=0)

            if not self.accelerator.is_main_process:  # target model

                cur_k = int(draft_prob_comm[0].item())
                flatten_draft_k_seq_ids = draft_prob_comm[
                    1 : self.args.gamma + 1
                ]

                # prev_size = self.args.gamma / prev_k
                prev_ids_draft = prefix[
                    :, prefix_len - prev_size + 1 : prefix_len
                ]  # 两个进程的输入

                prev_prob_draft = []

                cur_size = int(self.args.gamma / cur_k)
                cur_ids_k_seq_draft = (
                    torch.tensor(flatten_draft_k_seq_ids, dtype=torch.int)
                    .reshape(cur_k, -1)
                    .to(device)
                )  # [k, cur_size]

                prob_target = prob

            else:
                cur_ids_k_seq_draft = (
                    torch.tensor(flatten_draft_k_seq_ids, dtype=torch.int)
                    .reshape(cur_k, -1)
                    .to(device)
                )
                cur_size = int(self.args.gamma / cur_k)
            if cur_mode:
                if not self.accelerator.is_main_process:

                    flag, resampled_token_id, chosen_draft_tokens_seq_idx = (
                        self.verify_first_token_for_k_seq(
                            cur_ids_k_seq_draft,
                            prev_prob_draft,
                            prob_target[:, [-1], :],
                        )
                    )

                    if flag == False:
                        # reject the first token in all k seq
                        prefix = torch.cat(
                            (input_ids, resampled_token_id), dim=1
                        )

                        # record the number of accepted tokens
                        self.num_acc_tokens.append(num_acc_token)
                        num_acc_token = 0

                        comm_tensor[0] = 1
                        comm_tensor[1] = 0
                        comm_tensor[2] = resampled_token_id
                    else:
                        # accept the first token, change the mode
                        cur_mode = False
                        prefix = torch.cat(
                            (
                                input_ids,
                                cur_ids_k_seq_draft[
                                    chosen_draft_tokens_seq_idx
                                ].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )
                        num_acc_token += 1
                        comm_tensor[0] = 0
                        comm_tensor[1] = chosen_draft_tokens_seq_idx
                    dist.send(comm_tensor, dst=0)
                else:  # draft model
                    dist.recv(comm_tensor, src=1)
                    cur_mode = comm_tensor[0].item()
                    if cur_mode:
                        prefix = torch.cat(
                            (input_ids, comm_tensor[2:3].unsqueeze(dim=0)),
                            dim=1,
                        )
                        # model.rollback(prefix_len)
                    else:
                        chosen_draft_tokens_seq_idx = comm_tensor[1].item()
                        prefix = torch.cat(
                            (
                                input_ids,
                                cur_ids_k_seq_draft[
                                    chosen_draft_tokens_seq_idx
                                ].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )

                        if cur_k != 1:
                            llama_cpp.llama_state_set_data(
                                model._model.ctx,
                                model.kv_cache[chosen_draft_tokens_seq_idx][0],
                                model.kv_cache[chosen_draft_tokens_seq_idx][3],
                            )
                            model._model.input_ids[
                                : model.kv_cache[chosen_draft_tokens_seq_idx][2]
                            ] = model.kv_cache[chosen_draft_tokens_seq_idx][1]

            else:
                if not self.accelerator.is_main_process:
                    n = prev_size - 1
                    for i in range(prev_size - 1):
                        token = prev_ids_draft[:, i]
                        r = torch.rand(1, device=device)

                        if r > prob_target[:, i, token]:
                            n = i
                            break
                    if n == prev_size - 1:
                        (
                            flag,
                            resampled_token_id,
                            chosen_draft_tokens_seq_idx,
                        ) = self.verify_first_token_for_k_seq(
                            cur_ids_k_seq_draft,
                            prev_prob_draft,
                            prob_target[:, [-1], :],
                        )

                        if flag == False:
                            cur_mode = True
                            comm_tensor[0] = 1
                            comm_tensor[1] = n
                            comm_tensor[2] = resampled_token_id
                            dist.send(comm_tensor, dst=0)
                            prefix = torch.cat(
                                (input_ids, resampled_token_id), dim=1
                            )
                            model.rollback(prefix_len)
                        else:
                            comm_tensor[0] = 0
                            comm_tensor[1] = chosen_draft_tokens_seq_idx
                            dist.send(comm_tensor, dst=0)
                            prefix = torch.cat(
                                (
                                    input_ids,
                                    cur_ids_k_seq_draft[
                                        chosen_draft_tokens_seq_idx
                                    ].unsqueeze(dim=0),
                                ),
                                dim=1,
                            )
                    else:
                        cur_mode = True
                        t = torch.where(prob_target[0, n, :] == 1)[0].unsqueeze(
                            0
                        )

                        comm_tensor[0] = 1
                        comm_tensor[1] = n
                        comm_tensor[2] = t
                        dist.send(comm_tensor, dst=0)
                        prefix = torch.cat(
                            (input_ids[:, : prefix_len - prev_size + n + 1], t),
                            dim=1,
                        )
                        model.rollback(prefix_len - prev_size + n + 1)

                else:  # draft model
                    dist.recv(comm_tensor, src=1)
                    cur_mode = comm_tensor[0].item()
                    n = comm_tensor[1].item()
                    t = comm_tensor[2].item()
                    if cur_mode == 0:
                        prefix = torch.cat(
                            (
                                input_ids,
                                cur_ids_k_seq_draft[n].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )

                        if cur_k != 1:
                            llama_cpp.llama_state_set_data(
                                model._model.ctx,
                                model.kv_cache[n][0],
                                model.kv_cache[n][3],
                            )
                            model._model.input_ids[: model.kv_cache[n][2]] = (
                                model.kv_cache[n][1]
                            )

                    else:  # reject someone
                        prefix = torch.cat(
                            (
                                input_ids[:, : prefix_len - prev_size + n + 1],
                                comm_tensor[2:3].unsqueeze(dim=0),
                            ),
                            dim=1,
                        )

            prev_k = cur_k
            prev_size = cur_size

        return prefix

    @torch.no_grad()
    def verify_first_token_for_k_seq(
        self, draft_tokens_k_seq, draft_prob_k_seq, target_prob, **kwargs
    ):
        flag = False  # if any accepted
        resampled_token_id = 0
        chosen_draft_tokens_seq_idx = 0

        first_token_k_seq = draft_tokens_k_seq[:, 0]

        r = torch.rand(1, device=target_prob.device)

        if r > target_prob[:, 0, first_token_k_seq[0]]:

            t = torch.where(target_prob[0, 0, :] == 1)[0].unsqueeze(0)
            resampled_token_id = t
            idx = 0
            for idx, first_token in enumerate(first_token_k_seq[1:], 1):
                if t == first_token:
                    flag = True
                    chosen_draft_tokens_seq_idx = idx
                    break
        else:
            flag = True
            chosen_draft_tokens_seq_idx = 0

        return flag, resampled_token_id, chosen_draft_tokens_seq_idx

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4):
        """print content with color. Some color numbers are listed: Gray: 0, Red: 1, Green: 2, Yellow: 3, Blue: 4."""
        if self.accelerator.is_main_process:
            print(f"\033[9{color_number}m{content}\033[0m")
