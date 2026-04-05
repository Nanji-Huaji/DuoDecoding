import os
import sys
import warnings
import logging

import torch
import torch.distributed as dist
import transformers

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, TypedDict, Literal, cast, Protocol

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .communication import (
    CUHLM,
    CommunicationSimulator,
    PreciseCommunicationSimulator,
    PreciseCUHLM,
)
from .decoding_ops import finalize_verification, verify_draft_sequence
from .decoding_types import AcceptanceResult, RollbackPlan, VerificationInputs
from .model_gpu import KVCacheModel
from .model_loading import (
    build_quant_config,
    get_model_size,
    load_causal_lm,
    log_dual_model_allocation,
    log_quantization_decision,
    log_tri_model_allocation,
    select_dual_model_devices,
    select_tri_model_devices,
)
from .register import Register
from .utils import (
    seed_everything,
    sample,
)
from .metrics import DecodingMetrics, get_empty_metrics, INT_SIZE
from .metrics_dumper import (
    MetricsDumpFactoryLike,
    ArgsLike,
    MetricsDumpLike,
    _load_default_metrics_dumper_factory,
)
from .debug_logs import (
    _log_sd_alignment_snapshot,
    _log_invalid_batch_details,
    _sd_alignment_debug_enabled,
    _format_cache_state,
)

try:
    import flash_attn  # type: ignore
except ImportError:
    pass

from functools import partial


flash_attn_available = "flash_attn" in globals()
logger = logging.getLogger(__name__)

attn_impl = "sdpa" if not flash_attn_available else "flash_attention_2"

INT_SIZE = 4


class Decoding(Register, ABC):
    def __init__(
        self,
        args,
        metrics_dumper_factory: Optional[MetricsDumpFactoryLike] = None,
    ):
        Register.__init__(self, args)
        self.args = args
        if metrics_dumper_factory is None:
            metrics_dumper_factory = _load_default_metrics_dumper_factory()
        self.metrics_dumper_factory = metrics_dumper_factory
        self.metrics_dumper = self.metrics_dumper_factory(args)
        if "RANK" in os.environ:
            rank = int(os.environ["RANK"])
            size = int(os.environ["WORLD_SIZE"])

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

        self.vocab_size: int = -1
        self.stop_tokens_matrix = None

    def _prepare_stop_tokens(self, stop_sequences: List[str]):
        """
        预处理停止词序列，将其转换为 GPU 上的张量矩阵，以便在生成过程中进行高效的广播检查。
        """
        if not stop_sequences or not getattr(self, "tokenizer", None):
            raise ValueError("Stop sequences provided but tokenizer is not available.")

        # 1. 获取完整的 ID 序列
        stop_ids_list = [
            self.tokenizer.encode(s, add_special_tokens=False) for s in stop_sequences
        ]

        if not stop_ids_list:
            self.stop_tokens_matrix = None
            return

        # 2. 找出最长的停止词长度
        max_len = max(len(ids) for ids in stop_ids_list)

        # 3. 填充并转为 Tensor (用 -1 填充左侧，方便右对齐比对)
        # 形状: [停止词个数, 最长长度]
        # 确保 device 正确，这里假设 self.target_model 已经加载
        device = (
            self.target_model.device
            if hasattr(self, "target_model") and self.target_model is not None
            else "cpu"
        )

        matrix = torch.full(
            (len(stop_ids_list), max_len),
            -1,
            dtype=torch.long,
            device=device,
        )
        for i, ids in enumerate(stop_ids_list):
            matrix[i, -len(ids) :] = torch.tensor(ids, device=device)

        self.stop_tokens_matrix = matrix

    def _should_stop(
        self,
        prefix: torch.Tensor,
        max_tokens: int,
        use_early_stopping: bool = False,
    ) -> bool:
        """
        Unified stopping criteria check.
        Prioritizes checks on GPU to minimize synchronization overhead.
        """
        # 1. Length check
        if prefix.shape[1] >= max_tokens:
            return True

        if not use_early_stopping:
            return False

        # 2. EOS check
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            # prefix: [batch, seq_len]
            if prefix[0, -1] == self.tokenizer.eos_token_id:
                return True

        # 3. Stop tokens matrix check
        if self.stop_tokens_matrix is not None:
            max_stop_len = self.stop_tokens_matrix.size(1)
            # Only check a reasonable window at the end
            check_window = max(64, max_stop_len + 10)

            # Get the trailing sequence
            seq = prefix[0, -check_window:]

            # If sequence is shorter than any stop token, skip
            if seq.size(0) < max_stop_len:
                return False

            # Use unfold to create sliding windows: [num_windows, max_stop_len]
            windows = seq.unfold(0, max_stop_len, 1)

            # Broadcasting comparison:
            # windows: [1, W, L]
            # matrix:  [S, 1, L]
            targets = windows.unsqueeze(0)
            stops = self.stop_tokens_matrix.unsqueeze(1)

            # Check matches, treating -1 in stops as always matching (padding)
            matches = (targets == stops) | (stops == -1)

            # Check if any full sequence matches in any window
            # dimensions: [S, W, L] -> all(dim=-1) -> [S, W] -> any() -> bool
            if matches.all(dim=-1).any():
                return True

        return False

    def _check_stopping_criteria(
        self, input_ids: torch.Tensor, stop_sequences: Optional[List[str]] = None
    ) -> bool:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return False

        # Check for EOS at the last position only
        if (
            input_ids.shape[1] > 0
            and input_ids[0, -1].item() == self.tokenizer.eos_token_id
        ):
            return True

        # Check for stop sequences
        if stop_sequences:
            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            for stop_seq in stop_sequences:
                if decoded_text.endswith(stop_seq):
                    return True
        return False

    @staticmethod
    def _get_available_gpu_count() -> int:
        """
        Get the number of available GPUs, considering CUDA_VISIBLE_DEVICES.

        Returns:
            Number of available GPUs
        """
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
            if visible_devices:
                # Count the number of devices in CUDA_VISIBLE_DEVICES
                return len([d for d in visible_devices.split(",") if d.strip()])

        # Fallback to torch.cuda.device_count()
        return torch.cuda.device_count()

    def load_model(self):
        # * load models according to different evaluation methods.
        self.color_print(
            f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}",
            3,
        )

        # Get available GPU count
        num_gpus = self._get_available_gpu_count()
        self.color_print(f"Available GPUs: {num_gpus}", 3)

        loader = partial(
            AutoModelForCausalLM.from_pretrained,
            local_files_only=False,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if self.args.eval_mode == "small":
            device_map = "cuda:0"
            self.color_print(f"Loading {self.args.draft_model} on {device_map}", 3)
            draft_quant = build_quant_config(self.args.draft_model)
            if draft_quant is not None:
                log_quantization_decision(self.color_print, self.args.draft_model)
            self.draft_model = load_causal_lm(
                loader,
                self.args.draft_model,
                device_map,
                quant_config=draft_quant,
            )

        elif self.args.eval_mode == "large":
            device_map = "cuda:0"
            self.color_print(f"Loading {self.args.target_model} on {device_map}", 3)
            target_quant = build_quant_config(self.args.target_model)
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                device_map,
                quant_config=target_quant,
            )

        elif self.args.eval_mode in [
            "sd",
            "dsd",
            "dssd",
            "dist_spec",
            "dist_split_spec",
            "uncertainty_decoding",
            "cuhlm",
            "speculative_decoding_with_bandwidth",
            "speculative_decoding_with_bandwidth_full_prob",
        ]:
            draft_device, target_device = select_dual_model_devices(
                self.args.draft_model,
                self.args.target_model,
                num_gpus,
            )
            log_dual_model_allocation(
                self.color_print,
                self.args.draft_model,
                self.args.target_model,
                draft_device,
                target_device,
            )

            self.color_print(f"Loading {self.args.draft_model} on {draft_device}", 3)
            draft_quant = build_quant_config(self.args.draft_model)
            if draft_quant is not None:
                log_quantization_decision(self.color_print, self.args.draft_model)
            self.draft_model = load_causal_lm(
                loader,
                self.args.draft_model,
                draft_device,
                quant_config=draft_quant,
            )
            self.color_print(f"Loading {self.args.target_model} on {target_device}", 3)
            target_quant = build_quant_config(self.args.target_model)
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                target_device,
                quant_config=target_quant,
            )

        elif self.args.eval_mode == "adaptive_decoding":
            draft_device, target_device = select_dual_model_devices(
                self.args.draft_model,
                self.args.target_model,
                num_gpus,
            )
            log_dual_model_allocation(
                self.color_print,
                self.args.draft_model,
                self.args.target_model,
                draft_device,
                target_device,
            )

            self.color_print(f"Loading {self.args.draft_model} on {draft_device}", 3)
            draft_quant = build_quant_config(self.args.draft_model)
            if draft_quant is not None:
                log_quantization_decision(self.color_print, self.args.draft_model)
            self.draft_model = load_causal_lm(
                loader,
                self.args.draft_model,
                draft_device,
                output_hidden_states=True,
                quant_config=draft_quant,
            )
            self.color_print(f"Loading {self.args.target_model} on {target_device}", 3)
            target_quant = build_quant_config(self.args.target_model)
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                target_device,
                quant_config=target_quant,
            )

        elif self.args.eval_mode in [
            "tridecoding",
            "adaptive_tridecoding",
            "cee_sd",
            "ceesd_without_arp",
            "ceesd_w/o_arp",
            "cee_cuhlm",
            "cee_dsd",
            "cee_dssd",
        ]:
            output_hidden_states = self.args.eval_mode in [
                "adaptive_tridecoding",
                "cee_sd",
                "cee_cuhlm",
            ]

            little_device, draft_device, target_device, model_sizes = (
                select_tri_model_devices(
                    self.args.little_model,
                    self.args.draft_model,
                    self.args.target_model,
                    num_gpus,
                )
            )
            log_tri_model_allocation(
                self.color_print,
                model_sizes,
                little_device,
                draft_device,
                target_device,
                num_gpus,
            )

            self.color_print(f"Loading {self.args.little_model} on {little_device}", 3)
            little_quant = build_quant_config(self.args.little_model)
            if little_quant is not None:
                log_quantization_decision(self.color_print, self.args.little_model)
            self.little_model = load_causal_lm(
                loader,
                self.args.little_model,
                little_device,
                output_hidden_states=output_hidden_states,
                quant_config=little_quant,
            )
            self.color_print(f"Loading {self.args.draft_model} on {draft_device}", 3)
            draft_quant = build_quant_config(self.args.draft_model)
            if draft_quant is not None:
                log_quantization_decision(self.color_print, self.args.draft_model)
            self.draft_model = load_causal_lm(
                loader,
                self.args.draft_model,
                draft_device,
                output_hidden_states=output_hidden_states,
                quant_config=draft_quant,
            )
            self.color_print(f"Loading {self.args.target_model} on {target_device}", 3)
            target_quant = build_quant_config(self.args.target_model)
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                target_device,
                quant_config=target_quant,
            )

        # # 从实际模型embedding层获取vocab_size

        # Seems fetching vocab size from model is unnecessary.
        self.vocab_size = int(self.args.vocab_size)

        # Print device allocation for loaded models
        self._print_model_device_info()

    def _print_model_device_info(self):
        """Print device allocation information for all loaded models."""
        self.color_print("=" * 60, 3)
        self.color_print("Model Device Allocation:", 3)
        self.color_print("=" * 60, 3)

        if hasattr(self, "little_model") and self.little_model is not None:
            self._print_single_model_device_info("Little Model", self.little_model)

        if hasattr(self, "draft_model") and self.draft_model is not None:
            self._print_single_model_device_info("Draft Model", self.draft_model)

        if hasattr(self, "target_model") and self.target_model is not None:
            self._print_single_model_device_info("Target Model", self.target_model)

        self.color_print("=" * 60, 3)

    def _print_single_model_device_info(self, model_name: str, model):
        """Print device allocation for a single model."""
        self.color_print(f"\n{model_name}:", 3)

        if hasattr(model, "hf_device_map"):
            device_map = model.hf_device_map
            device_summary = {}

            for layer_name, device in device_map.items():
                device_str = str(device)
                if device_str not in device_summary:
                    device_summary[device_str] = []
                device_summary[device_str].append(layer_name)

            for device, layers in sorted(device_summary.items()):
                self.color_print(f"  {device}: {len(layers)} layers", 3)

        elif hasattr(model, "device"):
            self.color_print(f"  Device: {model.device}", 3)
        else:
            # Try to get device from first parameter
            try:
                first_param = next(model.parameters())
                self.color_print(f"  Device: {first_param.device}", 3)
            except StopIteration:
                self.color_print("  Device: Unknown (no parameters)", 3)

    def load_tokenizer(self):
        # * load tokenizers
        self.color_print(f"Loading tokenizer of {self.args.target_model}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.target_model,
            trust_remote_code=True,
            local_files_only=False,
        )
        self.tokenizer.padding_side = "right"

        if self.tokenizer.pad_token_id is None:
            model_name = str(self.args.target_model).lower()
            if "llama" in model_name:
                self.tokenizer.pad_token_id = 2
            elif self.tokenizer.eos_token_id is not None:
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def preprocess(self, input_text):
        pass

    @abstractmethod
    def postprocess(self, input_text, output_text):
        pass

    @Register.register_decoding("large")
    @Register.register_decoding("small")
    @torch.inference_mode()
    def autoregressive_sampling(
        self,
        prefix,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
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
        model = KVCacheModel(model, self.args.temp, self.args.top_k, self.args.top_p)
        model.vocab_size = self.args.vocab_size

        prefix_len = prefix.shape[1]
        max_tokens = prefix_len + self.args.max_tokens

        x = prefix

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        target_forward_times = 0

        start_event.record(stream=torch.cuda.current_stream())
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)
        while x.shape[1] < max_tokens:
            queuing_time += batch_delay
            x = model.generate(x, 1)
            target_forward_times += 1

            if use_early_stopping and self._check_stopping_criteria(x, stop_sequences):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = (
            start_event.elapsed_time(end_event) / 1000.0
        )  # Convert to seconds
        generated_tokens = x.shape[1] - prefix_len

        metrics = get_empty_metrics()
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = elapsed_time + queuing_time
        metrics["throughput"] = (
            generated_tokens / metrics["wall_time"] if metrics["wall_time"] > 0 else 0
        )

        return x, metrics

    @Register.register_decoding("sd")
    @torch.inference_mode()
    def speculative_decoding(
        self,
        prefix,
        transfer_top_k: int | None = 300,
        use_early_stopping: bool = False,
        stop_sequences: Optional[List[str]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, DecodingMetrics]:
        max_tokens = prefix.shape[1] + self.args.max_tokens

        draft_device = self.draft_model.device
        target_device = self.target_model.device

        approx_model_cache = KVCacheModel(
            self.draft_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        approx_model_cache.vocab_size = int(self.vocab_size)
        target_model_cache = KVCacheModel(
            self.target_model, self.args.temp, self.args.top_k, self.args.top_p
        )
        target_model_cache.vocab_size = int(self.vocab_size)

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
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            x = approx_model_cache.generate(prefix.to(draft_device), current_gamma)
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            # Verification-time cache invariant: approx has forwarded states up to
            # x[:, :-1], while target forwards the full x and ends up one step ahead.
            _ = target_model_cache.generate(x.to(target_device), 1)
            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            this_step_accepted_tokens, n = verify_draft_sequence(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                transfer_mode="none",
                send_reject_message=False,
            )
            _log_sd_alignment_snapshot(
                "verify_exit",
                prefix_len,
                approx_model_cache,
                target_model_cache,
                x_len=x.shape[1],
                gamma=current_gamma,
                note=f"accepted={this_step_accepted_tokens}, n={n}",
            )

            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

            # 检查是否还有空间添加一个token（在接受序列过长时提前截断）
            if n + 1 >= max_tokens:
                prefix = x[:, :max_tokens]
                break

            prefix = finalize_verification(
                approx_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                n=n,
            )
            _log_sd_alignment_snapshot(
                "finalize_exit",
                prefix.shape[1],
                approx_model_cache,
                target_model_cache,
                x_len=prefix.shape[1],
                gamma=current_gamma,
            )

            if use_early_stopping and self._check_stopping_criteria(
                prefix, stop_sequences
            ):
                break

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]

        batch_delay = getattr(self.args, "batch_delay", 0)
        queuing_time = target_forward_times * batch_delay
        wall_time = elapsed_time + queuing_time

        throughput = generated_tokens / wall_time if wall_time > 0 else 0

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = throughput
        metrics["loop_times"] = loop_idx
        metrics["queuing_time"] = queuing_time
        metrics["each_loop_draft_tokens"] = (
            total_drafted_tokens / loop_idx if loop_idx > 0 else 0
        )

        return prefix, metrics

    @torch.inference_mode()
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
                    target_model_cache.prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                prefix = torch.cat((prefix, t), dim=1)
                self.num_acc_tokens.append(1)
                break

            x = approx_model_cache.generate(prefix.to(draft_device), current_gamma)
            draft_forward_times += current_gamma
            total_drafted_tokens += current_gamma

            _ = target_model_cache.generate(x.to(target_device), 1)

            target_forward_times += 1

            if self.accelerator.is_main_process:
                self.draft_forward_times += current_gamma
                self.target_forward_times += 1

            this_step_accepted_tokens, n = verify_draft_sequence(
                draft_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                comm_simulator=comm_simulator,
                comm_link="edge_cloud",
                transfer_mode="serial",
                send_reject_message=True,
            )

            total_accepted_tokens += this_step_accepted_tokens

            self.num_acc_tokens.append(this_step_accepted_tokens)

            assert n >= prefix_len - 1, f"n {n}, prefix_len {prefix_len}"

            # 检查是否还有空间添加一个token
            if n + 1 >= max_tokens:
                prefix = x[:, :max_tokens]
                break

            # 针对使用了通信压缩的传输逻辑
            if n < prefix_len + current_gamma - 1:
                if transfer_top_k is not None and transfer_top_k > 0:
                    rebuild_probs = comm_simulator._apply_top_k_compression(
                        approx_model_cache.prob_history[:, n, : self.vocab_size],
                        transfer_top_k,
                    )
                    rebuild_probs = comm_simulator.rebuild_full_probs(rebuild_probs)
                    approx_model_cache.prob_history[:, n, : self.vocab_size] = (
                        rebuild_probs
                    )

                comm_simulator.transfer(
                    None,
                    approx_model_cache.prob_history[:, n, : self.vocab_size],
                    "edge_cloud",
                    transfer_top_k is not None and transfer_top_k > 0,
                    transfer_top_k,
                )

            # finalize_verification自动回流状态与重新采样（或简单采样下一个字符）
            prefix = finalize_verification(
                approx_model_cache=approx_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=current_gamma,
                n=n,
            )

            # 传输新生成的 token id
            comm_simulator.simulate_transfer(INT_SIZE, "edge_cloud")

        end_event.record(stream=torch.cuda.current_stream())
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event) / 1000.0

        generated_tokens = prefix.shape[1] - current_tokens.shape[1]

        batch_delay = getattr(self.args, "batch_delay", 0)
        queuing_time = target_forward_times * batch_delay
        wall_time = elapsed_time + comm_simulator.edge_cloud_comm_time + queuing_time

        throughput = generated_tokens / wall_time if wall_time > 0 else 0

        metrics = get_empty_metrics()
        metrics["draft_forward_times"] = draft_forward_times
        metrics["target_forward_times"] = target_forward_times
        metrics["generated_tokens"] = generated_tokens
        metrics["draft_generated_tokens"] = total_drafted_tokens
        metrics["draft_accepted_tokens"] = total_accepted_tokens
        metrics["wall_time"] = wall_time
        metrics["throughput"] = throughput
        metrics["queuing_time"] = queuing_time
        metrics["communication_time"] = comm_simulator.edge_cloud_comm_time
        metrics["edge_cloud_data_bytes"] = comm_simulator.edge_cloud_data

        metrics["comm_energy"] = comm_simulator.total_comm_energy

        return prefix, metrics

    @torch.inference_mode()
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

    @abstractmethod
    def eval(self):
        pass

    def color_print(self, content: str, color_number: int = 4) -> None:
        """Print colorized output on the main process when supported."""
        if not self.accelerator.is_main_process:
            return

        text = str(content)
        color_map = {
            0: "90",
            1: "91",
            2: "92",
            3: "93",
            4: "94",
        }
        color_code = color_map.get(color_number)

        if color_code is None or not sys.stdout.isatty():
            print(text)
            return

        print(f"\033[{color_code}m{text}\033[0m")
