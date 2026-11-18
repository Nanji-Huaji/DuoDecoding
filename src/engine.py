import os
import sys
import warnings
import logging

import torch
import torch.distributed as dist
import transformers

from typing import Any, List, Optional, Tuple, TypedDict, Literal, cast, Protocol

from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from . import autoregressive
from .comm_accounting import CommConfig, CommEnv
from .dual_decoding import SpeculativeDecodingMixin
from .legacy import bandwidth_sd
from .graph_decode import (
    acquire_graph_caches,
    kv_cache_construction_key,
)
from .model_gpu import KVCacheModel
from .mode_features import get_mode_spec
from .model_loading import (
    build_sharded_target_device_map,
    build_quant_config,
    resolve_model_dtype,
    estimate_model_reserve_gib,
    get_model_size,
    load_causal_lm,
    log_dual_model_allocation,
    log_quantization_decision,
    log_tri_model_allocation,
    select_dual_model_devices,
    select_tri_model_devices,
)
from .register import Register
from .utils import seed_everything
from .metrics_dumper import (
    MetricsDumpFactoryLike,
    ArgsLike,
    MetricsDumpLike,
    _load_default_metrics_dumper_factory,
)

try:
    import flash_attn  # type: ignore
except ImportError:
    pass

from functools import partial


flash_attn_available = "flash_attn" in globals()
logger = logging.getLogger(__name__)


def configure_verbosity() -> None:
    """跑真实评测前统一降噪（原先在 import 时无条件执行）。

    以前这两句写在模块顶部 ⇒ 任何人 `import src.engine` 都会全局关闭警告，
    连与评测无关的真实告警（依赖告警、transformers 的 skip/重试提示）也一并
    被掩盖。改为显式函数，只在真正加载模型跑评测时（load_model）调用一次：
    评测路径的日志与历史行为一致，而单纯 import 不再有全局副作用。
    """
    transformers.utils.logging.set_verbosity(40)
    warnings.filterwarnings("ignore")


attn_impl = "sdpa" if not flash_attn_available else "flash_attention_2"

# These are load-time settings, not task or decoding parameters. The evaluator
# uses the same rules when deciding whether existing weights are compatible.
TRI_HIDDEN_STATE_MODES = frozenset(
    {"adaptive_tridecoding", "cee_sd", "cee_sd_opportunistic", "cee_cuhlm"}
)


def tokenizer_source(mode, draft, target):
    # The Llama-2 70B tokenizer adds a pad id the smaller models cannot embed.
    target_name = str(target).lower()
    if (
        mode != "large" and "llama-2" in str(draft).lower()
        and "llama-2" in target_name and "70b" in target_name
    ):
        return draft
    return target


class Decoding(Register):
    # Direct callers and registry dispatch use the same implementations. Importing
    # engine still registers these modes without importing the Baselines family.
    autoregressive_sampling = autoregressive.autoregressive_sampling
    lookahead_forward = autoregressive.lookahead_forward
    speculative_decoding = SpeculativeDecodingMixin.speculative_decoding
    speculative_decoding_with_bandwidth = (
        bandwidth_sd.speculative_decoding_with_bandwidth
    )

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
        if "RANK" in os.environ and not dist.is_initialized():
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

    @property
    def comm_config(self) -> CommConfig:
        """一次运行生效的计费口径，首次访问时从 args 构造并缓存。

        字段名刻意与 CLI 开关名逐字一致：src/protocols.py 的
        live_mode_consumption 读方法代码对象里的名字来判断"消费了哪些
        开关"；属性访问会把名字留在 co_names 里
        （docs/comm_call_sites.md §4.1）。放在基类是因为所有注册解码方法
        （含 engine.py 里的历史方法）都从这里取同一份口径。
        """
        cfg = getattr(self, "_comm_config", None)
        if cfg is None:
            cfg = CommConfig.from_args(self.args)
            self._comm_config = cfg
        return cfg

    @property
    def comm_env(self) -> CommEnv:
        """当前 args 对应的通信数值层。

        **不缓存**：curriculum 训练会在同一个 decoder 上逐样本改写
        ``args.edge_cloud_bandwidth`` / ``edge_end_bandwidth``
        （eval/eval_mixed.py），缓存会让后续样本沿用第一个样本的条件，
        通信成本与 RL 奖励随之算错。构造只是 9 次 getattr，代价可忽略。
        """
        return CommEnv.from_args(self.args)

    def get_model_input_device(self, model) -> torch.device:
        if hasattr(model, "hf_device_map"):
            cuda_devices = sorted(
                {
                    str(device)
                    for device in model.hf_device_map.values()
                    if str(device).startswith("cuda")
                }
            )
            if cuda_devices:
                return torch.device(cuda_devices[0])
        model_device = getattr(model, "device", None)
        if model_device is not None:
            return torch.device(model_device)
        try:
            return next(model.parameters()).device
        except (StopIteration, AttributeError, TypeError) as exc:
            raise RuntimeError("Unable to determine model device") from exc

    def _get_model_embedding_vocab_size(self, model) -> int:
        embeddings = model.get_input_embeddings()
        return int(embeddings.weight.shape[0])

    def _get_model_config_vocab_size(self, model) -> int | None:
        config = getattr(model, "config", None)
        if config is None:
            return None
        if hasattr(config, "vocab_size"):
            return int(config.vocab_size)
        text_config = getattr(config, "text_config", None)
        if text_config is not None and hasattr(text_config, "vocab_size"):
            return int(text_config.vocab_size)
        return None

    def _runtime_models(self) -> List[Tuple[str, Any]]:
        models: List[Tuple[str, Any]] = []
        for name in ("little_model", "draft_model", "target_model"):
            model = getattr(self, name, None)
            if model is not None:
                models.append((name, model))
        return models

    def _validate_runtime_vocab_alignment(self) -> None:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return

        tokenizer_len = len(self.tokenizer)
        tokenizer_vocab_size = getattr(self.tokenizer, "vocab_size", tokenizer_len)
        tokenizer_limit = max(int(tokenizer_len), int(tokenizer_vocab_size))

        runtime_vocab_sizes: List[int] = []
        details: List[str] = [
            f"tokenizer_len={tokenizer_len}",
            f"tokenizer_vocab_size={tokenizer_vocab_size}",
        ]
        for model_name, model in self._runtime_models():
            embedding_vocab_size = self._get_model_embedding_vocab_size(model)
            config_vocab_size = self._get_model_config_vocab_size(model)
            runtime_vocab_sizes.append(embedding_vocab_size)
            details.append(
                f"{model_name}_embedding_vocab_size={embedding_vocab_size}"
            )
            if config_vocab_size is not None:
                details.append(f"{model_name}_config_vocab_size={config_vocab_size}")

        if not runtime_vocab_sizes:
            self.vocab_size = tokenizer_limit
            self.args.vocab_size = tokenizer_limit
            return

        shared_vocab_size = min(runtime_vocab_sizes)
        self.vocab_size = int(shared_vocab_size)
        self.args.vocab_size = int(shared_vocab_size)

        if tokenizer_limit > shared_vocab_size:
            raise ValueError(
                "Tokenizer/model vocab mismatch for shared decoding: "
                + ", ".join(details)
                + f", shared_vocab_size={shared_vocab_size}. "
                "The tokenizer can emit token ids that at least one loaded model cannot embed."
            )

        self.color_print(
            "Shared vocab alignment validated: "
            + ", ".join(details)
            + f", shared_vocab_size={shared_vocab_size}",
            3,
        )

    def validate_input_ids(self, input_ids: torch.Tensor, label: str = "input_ids"):
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)
        if input_ids.numel() == 0:
            return

        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0:
            raise ValueError(f"Negative token id at {label}: min={min_id}")

        model_limits = []
        for model_name, model in self._runtime_models():
            embedding_vocab_size = self._get_model_embedding_vocab_size(model)
            model_limits.append((model_name, embedding_vocab_size))
            if max_id >= embedding_vocab_size:
                raise ValueError(
                    f"Input token id out of range at {label}: max={max_id}, "
                    f"{model_name}_embedding_vocab_size={embedding_vocab_size}, "
                    f"self.vocab_size={self.vocab_size}, tokenizer_len={len(self.tokenizer)}"
                )

        if self.vocab_size > 0 and max_id >= self.vocab_size:
            limits = ", ".join(f"{name}={size}" for name, size in model_limits)
            raise ValueError(
                f"Input token id exceeds runtime shared vocab at {label}: max={max_id}, "
                f"self.vocab_size={self.vocab_size}, {limits}"
            )

    def _check_stopping_criteria(
        self,
        input_ids: torch.Tensor,
        stop_sequences: Optional[List[str]] = None,
        prompt_len: Optional[int] = None,
    ) -> bool:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return False

        # Check for EOS in the generated span. 此前只查最后一个位置：一轮追加
        # 多个 token 时，中部的 EOS 会被后续 token 埋掉、永不触发，继续解码
        # 到 max_tokens。给定 prompt_len 时扫描整个生成段（含中部），否则退回
        # 仅末位检查（单 token 轮次下两者等价）。
        if input_ids.shape[1] > 0:
            span = input_ids[0, prompt_len:] if prompt_len else input_ids[0, -1:]
            if bool((span == self.tokenizer.eos_token_id).any()):
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
        self.load_weights()

    def load_weights(self):
        """Load language-model weights without assembling task-specific policy heads."""
        # 真正跑评测的入口：统一降噪（原先在 import 时全局执行，见
        # configure_verbosity 的说明）。
        configure_verbosity()
        spec = get_mode_spec(self.args.eval_mode)

        # * load models according to different evaluation methods.
        self.color_print(
            f"Loading models:\n{self.args.draft_model}\n{self.args.target_model}",
            3,
        )

        # Get available GPU count
        num_gpus = self._get_available_gpu_count()
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible_devices is not None:
            self.color_print(f"CUDA_VISIBLE_DEVICES={visible_devices}", 3)
        else:
            self.color_print("CUDA_VISIBLE_DEVICES is not set", 3)
        self.color_print(f"Available GPUs: {num_gpus}", 3)

        # 计算精度由 --model_dtype 控制（默认 bf16，与历史行为一致）。
        model_dtype = resolve_model_dtype(
            getattr(self.args, "model_dtype", "bf16")
        )
        self.color_print(f"Model compute dtype: {model_dtype}", 3)
        loader = partial(
            AutoModelForCausalLM.from_pretrained,
            local_files_only=False,
            attn_implementation=attn_impl,
            trust_remote_code=True,
            torch_dtype=model_dtype,
        )
        if spec.models == "small":
            device_map = "cuda:0"
            self.color_print(f"Loading {self.args.draft_model} on {device_map}", 3)
            draft_quant = build_quant_config(
                self.args.draft_model,
                getattr(self.args, "draft_quantization", "auto"),
                compute_dtype=model_dtype,
            )
            if draft_quant is not None:
                log_quantization_decision(self.color_print, self.args.draft_model)
            self.draft_model = load_causal_lm(
                loader,
                self.args.draft_model,
                device_map,
                quant_config=draft_quant,
            )

        elif spec.models == "large":
            device_map = "cuda:0"
            self.color_print(f"Loading {self.args.target_model} on {device_map}", 3)
            target_quant = build_quant_config(
                self.args.target_model,
                getattr(self.args, "target_quantization", "auto"),
                compute_dtype=model_dtype,
            )
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                device_map,
                quant_config=target_quant,
            )

        elif spec.models == "dual" and not spec.uses_main_rl:
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
            draft_quant = build_quant_config(
                self.args.draft_model,
                getattr(self.args, "draft_quantization", "auto"),
                compute_dtype=model_dtype,
            )
            if draft_quant is not None:
                log_quantization_decision(self.color_print, self.args.draft_model)
            self.draft_model = load_causal_lm(
                loader,
                self.args.draft_model,
                draft_device,
                quant_config=draft_quant,
            )
            self.color_print(f"Loading {self.args.target_model} on {target_device}", 3)
            target_quant = build_quant_config(
                self.args.target_model,
                getattr(self.args, "target_quantization", "auto"),
                compute_dtype=model_dtype,
            )
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            target_max_memory = None
            if (
                target_quant is None
                and num_gpus >= 2
                and not getattr(self.args, "keep_target_on_single_gpu", False)
            ):
                reserve_gib = estimate_model_reserve_gib(
                    self.args.draft_model,
                    getattr(self.args, "draft_quantization", "auto"),
                )
                sharded_target = build_sharded_target_device_map(
                    num_gpus,
                    reserve_last_gpu_gib=reserve_gib,
                )
                if sharded_target is not None:
                    target_device, target_max_memory = sharded_target
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                target_device,
                quant_config=target_quant,
                max_memory=target_max_memory,
            )

        elif spec.models == "dual" and spec.uses_main_rl:
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
            draft_quant = build_quant_config(
                self.args.draft_model,
                getattr(self.args, "draft_quantization", "auto"),
                compute_dtype=model_dtype,
            )
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
            target_quant = build_quant_config(
                self.args.target_model,
                getattr(self.args, "target_quantization", "auto"),
                compute_dtype=model_dtype,
            )
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            target_max_memory = None
            if target_device == "auto":
                sharded_target = build_sharded_target_device_map(num_gpus)
                if sharded_target is not None:
                    target_device, target_max_memory = sharded_target
            elif target_quant is None and num_gpus >= 2:
                reserve_gib = estimate_model_reserve_gib(
                    self.args.draft_model,
                    getattr(self.args, "draft_quantization", "auto"),
                )
                sharded_target = build_sharded_target_device_map(
                    num_gpus,
                    reserve_last_gpu_gib=reserve_gib,
                )
                if sharded_target is not None:
                    target_device, target_max_memory = sharded_target
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                target_device,
                quant_config=target_quant,
                max_memory=target_max_memory,
            )

        elif spec.models == "tri":
            output_hidden_states = self.args.eval_mode in TRI_HIDDEN_STATE_MODES

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
            little_quant = build_quant_config(
                self.args.little_model,
                getattr(self.args, "little_quantization", "auto"),
                compute_dtype=model_dtype,
            )
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
            draft_quant = build_quant_config(
                self.args.draft_model,
                getattr(self.args, "draft_quantization", "auto"),
                compute_dtype=model_dtype,
            )
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
            target_quant = build_quant_config(
                self.args.target_model,
                getattr(self.args, "target_quantization", "auto"),
                compute_dtype=model_dtype,
            )
            if target_quant is not None:
                log_quantization_decision(self.color_print, self.args.target_model)
            target_max_memory = None
            if target_device == "auto":
                sharded_target = build_sharded_target_device_map(num_gpus)
                if sharded_target is not None:
                    target_device, target_max_memory = sharded_target
            self.target_model = load_causal_lm(
                loader,
                self.args.target_model,
                target_device,
                quant_config=target_quant,
                max_memory=target_max_memory,
            )

        self._validate_runtime_vocab_alignment()

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
        tokenizer_model_name = tokenizer_source(
            self.args.eval_mode,
            getattr(self.args, "draft_model", None),
            self.args.target_model,
        )

        self.color_print(f"Loading tokenizer of {tokenizer_model_name}...", 3)
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_name,
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

    def _acquire_autoregressive_cache(
        self, attr: str, graph_kw: dict, model
    ) -> KVCacheModel:
        """AR 单步循环的缓存入口：草稿模型开单步图，target 保持 eager。

        **跨样本复用缓存的理由**：本方法逐样本调用（每个数据集样本一次），而图
        捕获是绑在缓存实例上的 —— 每次重建 KVCacheModel 都要重捕获一次图
        （实测约 534 ms/样本，见 docs/graph_integration_status.md），单步图的
        收益会被捕获开销整个吃掉。这与 tridecoding / speculative_decoding 的既有
        接法一致：图开启时复用同一缓存对象，样本之间只做原地
        `reset_for_new_sample()`（StaticCache `zero_()`，地址不变 ⇒ 不重捕获）。

        只在 small 路径传非空 graph_kw（缓存的是草稿模型）；large 路径的缓存是
        target，按既有取舍保持 eager ⇒ graph_kw 为空，`acquire_graph_caches`
        每次重建且不挂属性，与接线前逐字节一致。
        """
        return acquire_graph_caches(
            self,
            attr,
            graph_kw,
            {
                "model": lambda: KVCacheModel(
                    model,
                    self.args.temp,
                    self.args.top_k,
                    self.args.top_p,
                    **graph_kw,
                )
            },
            cache_key=kv_cache_construction_key(
                model,
                temperature=self.args.temp,
                top_k=self.args.top_k,
                top_p=self.args.top_p,
                vocab_size=self.vocab_size,
                **graph_kw,
            ) if graph_kw else None,
        )["model"]

    def _acquire_speculative_caches(self, attr: str, graph_kw: dict, top_k: int):
        """投机解码的两条缓存：draft 走多 token 前向（图收益点），target 单步。

        B19：图开启时经 acquire_graph_caches 跨样本复用（draft 的定长验证图与
        历史缓冲不重捕获）；eager 模式每次重建，历史数字逐位不变。
        """
        return acquire_graph_caches(
            self,
            attr,
            graph_kw,
            {
                "draft": lambda: KVCacheModel(
                    self.draft_model,
                    self.args.temp,
                    top_k,
                    self.args.top_p,
                    **graph_kw,
                ),
                "target": lambda: KVCacheModel(
                    self.target_model,
                    self.args.temp,
                    top_k,
                    self.args.top_p,
                ),
            },
            cache_key={
                "draft": kv_cache_construction_key(
                    self.draft_model,
                    temperature=self.args.temp,
                    top_k=top_k,
                    top_p=self.args.top_p,
                    vocab_size=self.vocab_size,
                    **graph_kw,
                ),
                "target": kv_cache_construction_key(
                    self.target_model,
                    temperature=self.args.temp,
                    top_k=top_k,
                    top_p=self.args.top_p,
                    vocab_size=self.vocab_size,
                ),
            } if graph_kw else None,
        )


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
