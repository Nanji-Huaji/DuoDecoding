"""Load and place the language-model weights required by a decoding mode.

These factories take settings, not a decoder. Weights can be reused across
runs; neither policies, acceptance heads nor KV caches are attached here.
"""

from __future__ import annotations

import os
import warnings
from functools import partial
from typing import Any, Callable

import torch
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from .mode_features import ModeSpec, get_mode_spec
from .model_loading import (
    build_quant_config,
    build_sharded_target_device_map,
    estimate_model_reserve_gib,
    load_causal_lm,
    log_dual_model_allocation,
    log_quantization_decision,
    log_tri_model_allocation,
    resolve_model_dtype,
    select_dual_model_devices,
    select_tri_model_devices,
)

try:
    import flash_attn  # type: ignore # noqa: F401
except ImportError:
    pass


flash_attn_available = "flash_attn" in globals()

# color_print 的形状：(内容, 颜色号)，与 model_loading 各 log 函数的入参一致。
PrintFn = Callable[[str, int], None]

attn_impl = "sdpa" if not flash_attn_available else "flash_attention_2"


def configure_verbosity() -> None:
    """跑真实评测前统一降噪（原先在 import 时无条件执行）。

    以前这两句写在模块顶部 ⇒ 任何人 ``import src.engine`` 都会全局关闭警告，
    连与评测无关的真实告警（依赖告警、transformers 的 skip/重试提示）也一并
    被掩盖。改为显式函数，只在真正加载模型跑评测时（load_model_weights）
    调用一次：评测路径的日志与历史行为一致，而单纯 import 不再有全局副作用。
    """
    transformers.utils.logging.set_verbosity(40)
    warnings.filterwarnings("ignore")


# These are load-time settings, not task or decoding parameters. The evaluator
# uses the same rules when deciding whether existing weights are compatible.
TRI_HIDDEN_STATE_MODES = frozenset(
    {"adaptive_tridecoding", "cee_sd", "cee_sd_opportunistic", "cee_cuhlm"}
)


def tokenizer_source(mode, draft, target):
    # The Llama-2 70B tokenizer adds a pad id the smaller models cannot embed.
    target_name = str(target).lower()
    if (
        mode != "large"
        and "llama-2" in str(draft).lower()
        and "llama-2" in target_name
        and "70b" in target_name
    ):
        return draft
    return target


def get_available_gpu_count() -> int:
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


def _announce_model_load(
    print_fn: PrintFn,
    model_name: str,
    device: str,
    quantization: str,
    model_dtype: torch.dtype,
) -> BitsAndBytesConfig | None:
    """单个实例加载前的固定动作：设备横幅 → 量化配置 → 4-bit 决策日志。

    engine 时代这段在 small / large / dual / tri 共六个加载点逐字重复，
    合并于此；打印顺序与原实现一致。
    """
    print_fn(f"Loading {model_name} on {device}", 3)
    quant_config = build_quant_config(
        model_name, quantization, compute_dtype=model_dtype
    )
    if quant_config is not None:
        log_quantization_decision(print_fn, model_name)
    return quant_config


def _sharded_target_with_draft_reserve(
    num_gpus: int,
    draft_model_name: str,
    draft_quantization: str,
) -> tuple[str, dict[int | str, str]] | None:
    """dual 放置下 target 的跨卡 device_map：最后一张卡给 draft 留出显存。"""
    reserve_gib = estimate_model_reserve_gib(draft_model_name, draft_quantization)
    return build_sharded_target_device_map(num_gpus, reserve_last_gpu_gib=reserve_gib)


def _place_target(
    args: Any,
    spec: ModeSpec,
    num_gpus: int,
    device: str,
    quant_config: BitsAndBytesConfig | None,
) -> tuple[str, dict[int | str, str] | None]:
    """Resolve the target map *after* quantization is known, before loading it."""
    placement = None
    if device == "auto" and (spec.models == "tri" or spec.uses_main_rl):
        placement = build_sharded_target_device_map(num_gpus)
    elif (
        spec.models == "dual"
        and quant_config is None
        and num_gpus >= 2
        and (spec.uses_main_rl or not getattr(args, "keep_target_on_single_gpu", False))
    ):
        placement = _sharded_target_with_draft_reserve(
            num_gpus, args.draft_model, getattr(args, "draft_quantization", "auto")
        )
    return placement if placement is not None else (device, None)


def load_model_weights(
    args: Any,
    *,
    num_gpus: int,
    print_fn: PrintFn,
) -> dict[str, Any]:
    """按 eval_mode 加载语言模型权重，返回 ``{角色名: 模型}``。

    键只含当前模式真正加载的角色：small → ``draft_model``，large →
    ``target_model``，dual → ``draft_model``/``target_model``，tri →
    ``little_model``/``draft_model``/``target_model``。词表校验与设备信息
    打印不在这里——由调用方在把属性安装回 decoder 后执行。
    """
    # 真正跑评测的入口：统一降噪（原先在 import 时全局执行，见
    # configure_verbosity 的说明）。
    configure_verbosity()
    spec = get_mode_spec(args.eval_mode)

    # * load models according to different evaluation methods.
    print_fn(f"Loading models:\n{args.draft_model}\n{args.target_model}", 3)

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is not None:
        print_fn(f"CUDA_VISIBLE_DEVICES={visible_devices}", 3)
    else:
        print_fn("CUDA_VISIBLE_DEVICES is not set", 3)
    print_fn(f"Available GPUs: {num_gpus}", 3)

    # 计算精度由 --model_dtype 控制（默认 bf16，与历史行为一致）。
    model_dtype = resolve_model_dtype(getattr(args, "model_dtype", "bf16"))
    print_fn(f"Model compute dtype: {model_dtype}", 3)
    loader = partial(
        AutoModelForCausalLM.from_pretrained,
        local_files_only=False,
        attn_implementation=attn_impl,
        trust_remote_code=True,
        torch_dtype=model_dtype,
    )

    models: dict[str, Any] = {}

    def load_role(role: str, device: str, *, hidden_states: bool | None = None) -> None:
        key = f"{role}_model"
        name = getattr(args, key)
        quant = _announce_model_load(
            print_fn,
            name,
            device,
            getattr(args, f"{role}_quantization", "auto"),
            model_dtype,
        )
        kwargs: dict[str, Any] = {"quant_config": quant}
        if hidden_states is not None:
            kwargs["output_hidden_states"] = hidden_states
        if role == "target" and spec.models in {"dual", "tri"}:
            device, kwargs["max_memory"] = _place_target(
                args, spec, num_gpus, device, quant
            )
        models[key] = load_causal_lm(loader, name, device, **kwargs)

    if spec.models in {"small", "large"}:
        load_role("draft" if spec.models == "small" else "target", "cuda:0")
    elif spec.models == "dual":
        draft_device, target_device = select_dual_model_devices(
            args.draft_model, args.target_model, num_gpus
        )
        log_dual_model_allocation(
            print_fn,
            args.draft_model,
            args.target_model,
            draft_device,
            target_device,
        )

        load_role(
            "draft", draft_device, hidden_states=True if spec.uses_main_rl else None
        )
        load_role("target", target_device)

    elif spec.models == "tri":
        output_hidden_states = args.eval_mode in TRI_HIDDEN_STATE_MODES

        little_device, draft_device, target_device, model_sizes = (
            select_tri_model_devices(
                args.little_model,
                args.draft_model,
                args.target_model,
                num_gpus,
            )
        )
        log_tri_model_allocation(
            print_fn,
            model_sizes,
            little_device,
            draft_device,
            target_device,
            num_gpus,
        )

        load_role("little", little_device, hidden_states=output_hidden_states)
        load_role("draft", draft_device, hidden_states=output_hidden_states)
        load_role("target", target_device)

    return models


def load_tokenizer(args: Any, *, print_fn: PrintFn) -> Any:
    """加载共享 tokenizer 并应用历史 padding 规则，返回 tokenizer 实例。"""
    tokenizer_model_name = tokenizer_source(
        args.eval_mode,
        getattr(args, "draft_model", None),
        args.target_model,
    )

    print_fn(f"Loading tokenizer of {tokenizer_model_name}...", 3)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_model_name,
        trust_remote_code=True,
        local_files_only=False,
    )
    tokenizer.padding_side = "right"

    if tokenizer.pad_token_id is None:
        model_name = str(args.target_model).lower()
        if "llama" in model_name:
            tokenizer.pad_token_id = 2
        elif tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def get_model_input_device(model: Any) -> torch.device:
    """模型前向输入应放置的设备（shard 模型取第一个 cuda 分片）。"""
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


def embedding_vocab_size(model: Any) -> int:
    embeddings = model.get_input_embeddings()
    return int(embeddings.weight.shape[0])


def _config_vocab_size(model: Any) -> int | None:
    config = getattr(model, "config", None)
    if config is None:
        return None
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    return None


def shared_vocab_size(
    tokenizer: Any,
    models: list[tuple[str, Any]],
    print_fn: PrintFn,
) -> int:
    """共享解码词表上限：min(各模型嵌入行数)，且不得小于 tokenizer 词表。

    ``models`` 为 ``(角色名, 模型)`` 列表（engine ``_runtime_models()`` 的
    形状）；为空时退回 tokenizer 自身词表。tokenizer 能吐出任一模型嵌不进
    的 id 时抛 ValueError（与历史行为一致）。
    """
    tokenizer_len = len(tokenizer)
    tokenizer_vocab_size = getattr(tokenizer, "vocab_size", tokenizer_len)
    tokenizer_limit = max(int(tokenizer_len), int(tokenizer_vocab_size))

    runtime_vocab_sizes: list[int] = []
    details: list[str] = [
        f"tokenizer_len={tokenizer_len}",
        f"tokenizer_vocab_size={tokenizer_vocab_size}",
    ]
    for model_name, model in models:
        embedding_size = embedding_vocab_size(model)
        config_size = _config_vocab_size(model)
        runtime_vocab_sizes.append(embedding_size)
        details.append(f"{model_name}_embedding_vocab_size={embedding_size}")
        if config_size is not None:
            details.append(f"{model_name}_config_vocab_size={config_size}")

    if not runtime_vocab_sizes:
        return tokenizer_limit

    shared_size = min(runtime_vocab_sizes)

    if tokenizer_limit > shared_size:
        raise ValueError(
            "Tokenizer/model vocab mismatch for shared decoding: "
            + ", ".join(details)
            + f", shared_vocab_size={shared_size}. "
            "The tokenizer can emit token ids that at least one loaded "
            "model cannot embed."
        )

    print_fn(
        "Shared vocab alignment validated: "
        + ", ".join(details)
        + f", shared_vocab_size={shared_size}",
        3,
    )
    return int(shared_size)


_ROLE_DISPLAY_NAMES = {
    "little_model": "Little Model",
    "draft_model": "Draft Model",
    "target_model": "Target Model",
}


def log_model_devices(
    models: list[tuple[str, Any]],
    print_fn: PrintFn,
) -> None:
    """Print device allocation information for all loaded models."""
    print_fn("=" * 60, 3)
    print_fn("Model Device Allocation:", 3)
    print_fn("=" * 60, 3)

    for name, model in models:
        _log_single_model_devices(_ROLE_DISPLAY_NAMES.get(name, name), model, print_fn)

    print_fn("=" * 60, 3)


def _log_single_model_devices(
    model_name: str,
    model: Any,
    print_fn: PrintFn,
) -> None:
    """Print device allocation for a single model."""
    print_fn(f"\n{model_name}:", 3)

    if hasattr(model, "hf_device_map"):
        device_map = model.hf_device_map
        device_summary = {}

        for layer_name, device in device_map.items():
            device_str = str(device)
            if device_str not in device_summary:
                device_summary[device_str] = []
            device_summary[device_str].append(layer_name)

        for device, layers in sorted(device_summary.items()):
            print_fn(f"  {device}: {len(layers)} layers", 3)

    elif hasattr(model, "device"):
        print_fn(f"  Device: {model.device}", 3)
    else:
        # Try to get device from first parameter
        try:
            first_param = next(model.parameters())
            print_fn(f"  Device: {first_param.device}", 3)
        except StopIteration:
            print_fn("  Device: Unknown (no parameters)", 3)
