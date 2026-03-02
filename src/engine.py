import os
import warnings

import torch
import torch.distributed as dist
import transformers

transformers.utils.logging.set_verbosity(40)
warnings.filterwarnings("ignore")
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, TypedDict

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
from .model_gpu import KVCacheModel
from .register import Register
from .utils import max_fn, sample, seed_everything

try:
    import flash_attn
except ImportError:
    pass

from functools import partial

# from .register import Register

flash_attn_available = "flash_attn" in globals()

attn_impl = "sdpa" if not flash_attn_available else "flash_attention_2"

INT_SIZE = 4


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
    accuracy: Optional[Any]
    queuing_time: int | float
    arp_overhead_time: float
    dra_overhead_time: float
    avg_top_k: float
    avg_draft_len: float
    edge_cloud_bandwidth_history: List[float]
    edge_cloud_topk_history: List[int]
    edge_cloud_draft_len_history: List[int]


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
        each_loop_draft_tokens=0.0,
        comm_energy=0.0,
        connect_times={},
        accuracy=None,
        queuing_time=0.0,
        arp_overhead_time=0.0,
        dra_overhead_time=0.0,
        avg_top_k=0.0,
        avg_draft_len=0.0,
        edge_cloud_bandwidth_history=[],
        edge_cloud_topk_history=[],
        edge_cloud_draft_len_history=[],
    )


class Decoding(Register, ABC):
    def __init__(self, args):
        Register.__init__(self, args)
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

        self.vocab_size = -1

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

    def _get_device_map_strategy(self, model_name: str, num_gpus_available: int) -> str:
        """
        Determine the appropriate device_map strategy based on model size and available GPUs.

        新策略：避免单模型跨卡，大模型使用Q4量化放在单卡

        Args:
            model_name: Name of the model to load
            num_gpus_available: Number of GPUs available for model loading

        Returns:
            device_map strategy string
        """
        # Extract model size
        pattern = r"(\d+(?:\.\d+)?(?:[xX]\d+)?)[bB]"
        match = re.search(pattern, model_name)
        params = float(match.group(1)) if match else 0

        # 对于所有模型，优先使用单卡加载，避免跨卡通信
        # 大模型会在load_model中使用Q4量化
        return "auto"

    def _get_available_gpu_count(self) -> int:
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

        # Helper function to extract model size
        def get_model_size(model_name):
            pattern = r"(\d+(?:\.\d+)?(?:[xX]\d+)?)[bB]"
            match = re.search(pattern, model_name)
            return float(match.group(1)) if match else 0

        # Helper function to determine if model needs quantization
        # A6000 有 48GB 显存，约可容纳 ~24B 全精度或 ~16B bf16模型
        # 使用保守阈值：>20B 使用 Q4 量化
        def should_quantize(model_name):
            size = get_model_size(model_name)
            is_awq = "awq" in model_name.lower()
            return size > 20 and not is_awq

        # Helper function to get quantization config
        def get_quant_config(model_name):
            if should_quantize(model_name):
                print(
                    f"📦 Model {model_name} ({get_model_size(model_name)}B) will use 4-bit quantization"
                )
                return BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                )
            return None

        loader = partial(
            AutoModelForCausalLM.from_pretrained,
            local_files_only=False,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        if self.args.eval_mode == "small":
            device_map = "cuda:0"
            self.color_print(f"Loading draft model on {device_map}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=device_map,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=device_map,
                ).eval()

        elif self.args.eval_mode == "large":
            device_map = "cuda:0"
            self.color_print(f"Loading target model on {device_map}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=device_map,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=device_map,
                ).eval()

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
            # 双模型场景：大模型在一张卡，小模型在另一张卡
            draft_size = get_model_size(self.args.draft_model)
            target_size = get_model_size(self.args.target_model)

            # 将大模型放在GPU 0，小模型放在GPU 1（如果有多个GPU）
            if target_size > draft_size:
                draft_device = "cuda:1" if num_gpus > 1 else "cuda:0"
                target_device = "cuda:0"
                print(
                    f"🎯 Target ({target_size}B) -> GPU 0, Draft ({draft_size}B) -> GPU {1 if num_gpus > 1 else 0}"
                )
            else:
                draft_device = "cuda:0"
                target_device = "cuda:1" if num_gpus > 1 else "cuda:0"
                print(
                    f"🎯 Draft ({draft_size}B) -> GPU 0, Target ({target_size}B) -> GPU {1 if num_gpus > 1 else 0}"
                )

            self.color_print(f"Loading draft model on {draft_device}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                ).eval()

            self.color_print(f"Loading target model on {target_device}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                ).eval()

        elif self.args.eval_mode == "adaptive_decoding":
            # adaptive_decoding: 将大模型放GPU 0，小模型放GPU 1
            draft_size = get_model_size(self.args.draft_model)
            target_size = get_model_size(self.args.target_model)

            if target_size > draft_size:
                draft_device = "cuda:1" if num_gpus > 1 else "cuda:0"
                target_device = "cuda:0"
            else:
                draft_device = "cuda:0"
                target_device = "cuda:1" if num_gpus > 1 else "cuda:0"

            self.color_print(f"Loading draft model on {draft_device}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=True,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=True,
                ).eval()

            self.color_print(f"Loading target model on {target_device}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                ).eval()

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

            # 智能分配策略：
            # 1. 计算三个模型的大小
            # 2. 将最大的模型放在一张卡上（使用Q4量化如果需要）
            # 3. 将其余两个模型放在另一张卡上
            # 4. 避免单模型跨卡

            little_size = get_model_size(self.args.little_model)
            draft_size = get_model_size(self.args.draft_model)
            target_size = get_model_size(self.args.target_model)

            model_sizes = [
                (little_size, "little", self.args.little_model),
                (draft_size, "draft", self.args.draft_model),
                (target_size, "target", self.args.target_model),
            ]
            model_sizes.sort(reverse=True, key=lambda x: x[0])  # 按大小降序

            largest_model = model_sizes[0]
            print(
                f"🎯 Largest model: {largest_model[1]} ({largest_model[0]}B) -> GPU 0"
            )
            print(
                f"📍 Other models: {model_sizes[1][1]} ({model_sizes[1][0]}B), {model_sizes[2][1]} ({model_sizes[2][0]}B) -> GPU 1"
            )

            # 分配设备
            little_device = "cuda:1" if largest_model[1] != "little" else "cuda:0"
            draft_device = "cuda:1" if largest_model[1] != "draft" else "cuda:0"
            target_device = "cuda:1" if largest_model[1] != "target" else "cuda:0"

            # 如果只有1个GPU，全部放在cuda:0
            if num_gpus == 1:
                little_device = draft_device = target_device = "cuda:0"
                print("⚠️  Only 1 GPU available, all models will be loaded on cuda:0")

            self.color_print(f"Loading little model on {little_device}", 3)
            little_quant = get_quant_config(self.args.little_model)
            if little_quant:
                self.little_model = loader(
                    self.args.little_model,
                    device_map=little_device,
                    output_hidden_states=output_hidden_states,
                    quantization_config=little_quant,
                ).eval()
            else:
                self.little_model = loader(
                    self.args.little_model,
                    device_map=little_device,
                    output_hidden_states=output_hidden_states,
                ).eval()

            self.color_print(f"Loading draft model on {draft_device}", 3)
            draft_quant = get_quant_config(self.args.draft_model)
            if draft_quant:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=output_hidden_states,
                    quantization_config=draft_quant,
                ).eval()
            else:
                self.draft_model = loader(
                    self.args.draft_model,
                    device_map=draft_device,
                    output_hidden_states=output_hidden_states,
                ).eval()

            self.color_print(f"Loading target model on {target_device}", 3)
            target_quant = get_quant_config(self.args.target_model)
            if target_quant:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                    quantization_config=target_quant,
                ).eval()
            else:
                self.target_model = loader(
                    self.args.target_model,
                    device_map=target_device,
                ).eval()

        # 从实际模型embedding层获取vocab_size
        if hasattr(self, "target_model") and self.target_model is not None:
            actual_vocab_size = self.target_model.get_input_embeddings().weight.shape[0]
            self.vocab_size = actual_vocab_size
            print(f"✅ Using vocab_size from target model embedding: {self.vocab_size}")
        elif hasattr(self, "tokenizer") and self.tokenizer is not None:
            self.vocab_size = self.tokenizer.vocab_size
            print(f"⚠️  Using vocab_size from tokenizer: {self.vocab_size}")
        else:
            self.vocab_size = int(self.args.vocab_size)
            print(f"⚠️  Using vocab_size from args: {self.vocab_size}")

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
                    target_model_cache._prob_history.to(draft_device)[:, target_idx, j]
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
                        target_model_cache._prob_history[:, n, : self.vocab_size].to(
                            draft_device
                        )
                        - approx_model_cache._prob_history[:, n, : self.vocab_size]
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
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
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
                    target_model_cache._prob_history.to(draft_device)[:, target_idx, j]
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
                        approx_model_cache._prob_history[:, n, : self.vocab_size],
                        transfer_top_k,
                    )
                    rebuild_probs = comm_simulator.rebuild_full_probs(rebuild_probs)
                    approx_model_cache._prob_history[:, n, : self.vocab_size] = (
                        rebuild_probs  # 如果用了top-k压缩，先重建概率分布
                    )

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
                        target_model_cache._prob_history[:, n, : self.vocab_size].to(
                            draft_device
                        )
                        - approx_model_cache._prob_history[:, n, : self.vocab_size]
                    )
                )

                new_generated_tokens = prefix.shape[1] - current_tokens.shape[1] + 1

                target_model_cache.rollback(n + 1)
            else:
                # all approx model decoding accepted
                t = sample(
                    target_model_cache._prob_history[:, -1, : self.vocab_size]
                ).to(draft_device)
                target_model_cache.rollback(n + 2)

                new_generated_tokens = prefix.shape[1] - current_tokens.shape[1] + 1

            # 最后检查添加token后是否会超出限制
            if prefix.shape[1] < max_tokens:
                prefix = torch.cat((prefix, t), dim=1)

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
        queuing_time = 0
        batch_delay = getattr(self.args, "batch_delay", 0)

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
            queuing_time += batch_delay
            _ = target_model_cache.generate(x.to(target_device), 1)

            # 无论接受与否，都要传输起草的 token
            comm_simulator.transfer(x, None, link_type="edge_cloud")
            current_logit = approx_model_cache.logits_history[:, -1, : self.vocab_size]
            assert current_logit is not None, "Logits history should not be None"
            uncertainty = comm_simulator.calculate_uncertainty(
                current_logit, M=20, theta_max=2.0, draft_token=x[0, -1].item()
            )
            should_transfer, vocab_size = comm_simulator.determine_transfer_strategy(
                uncertainty, current_logit
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
                target_model_cache.rollback(n + 2)  # 等同于rollback(prefix_len + 2)

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
                target_model_cache._prob_history.to(draft_device)[:, prefix_len - 1, j]
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
                        target_model_cache._prob_history[:, n, : self.vocab_size].to(
                            draft_device
                        )
                        - approx_model_cache._prob_history[:, n, : self.vocab_size]
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
        metrics["queuing_time"] = queuing_time
        metrics["wall_time"] = (
            elapsed_time + queuing_time + comm_simulator.edge_cloud_comm_time
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

    @torch.inference_mode()
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
