from __future__ import annotations

import re
from typing import Callable

import torch
from transformers import BitsAndBytesConfig


def get_model_size(model_name: str) -> float:
    pattern = r"(\d+(?:\.\d+)?(?:[xX]\d+)?)[bB]"
    match = re.search(pattern, model_name)
    return float(match.group(1)) if match else 0.0


def should_quantize(model_name: str) -> bool:
    size = get_model_size(model_name)
    is_awq = "awq" in model_name.lower()
    return size > 20 and not is_awq


def build_quant_config(model_name: str) -> BitsAndBytesConfig | None:
    if not should_quantize(model_name):
        return None

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def secondary_device(num_gpus: int) -> str:
    return "cuda:1" if num_gpus > 1 else "cuda:0"


def select_dual_model_devices(
    draft_model_name: str,
    target_model_name: str,
    num_gpus: int,
) -> tuple[str, str]:
    draft_size = get_model_size(draft_model_name)
    target_size = get_model_size(target_model_name)
    fallback_device = secondary_device(num_gpus)

    if target_size > draft_size:
        return fallback_device, "cuda:0"
    return "cuda:0", fallback_device


def select_tri_model_devices(
    little_model_name: str,
    draft_model_name: str,
    target_model_name: str,
    num_gpus: int,
) -> tuple[str, str, str, tuple[tuple[float, str, str], ...]]:
    model_sizes = [
        (get_model_size(little_model_name), "little", little_model_name),
        (get_model_size(draft_model_name), "draft", draft_model_name),
        (get_model_size(target_model_name), "target", target_model_name),
    ]
    model_sizes.sort(reverse=True, key=lambda x: x[0])
    ordered_sizes = tuple(model_sizes)

    if num_gpus == 1:
        return "cuda:0", "cuda:0", "cuda:0", ordered_sizes

    largest_model_role = ordered_sizes[0][1]
    little_device = "cuda:0" if largest_model_role == "little" else "cuda:1"
    draft_device = "cuda:0" if largest_model_role == "draft" else "cuda:1"
    target_device = "cuda:0" if largest_model_role == "target" else "cuda:1"
    return little_device, draft_device, target_device, ordered_sizes


def log_quantization_decision(
    print_fn: Callable[[str, int], None],
    model_name: str,
) -> None:
    print_fn(
        f"Model {model_name} ({get_model_size(model_name)}B) will use 4-bit quantization",
        3,
    )


def log_dual_model_allocation(
    print_fn: Callable[[str, int], None],
    draft_model_name: str,
    target_model_name: str,
    draft_device: str,
    target_device: str,
) -> None:
    draft_size = get_model_size(draft_model_name)
    target_size = get_model_size(target_model_name)
    print_fn(
        f"Dual-model placement: draft ({draft_size}B) -> {draft_device}, target ({target_size}B) -> {target_device}",
        3,
    )


def log_tri_model_allocation(
    print_fn: Callable[[str, int], None],
    model_sizes: tuple[tuple[float, str, str], ...],
    little_device: str,
    draft_device: str,
    target_device: str,
    num_gpus: int,
) -> None:
    if num_gpus == 1:
        print_fn("Only 1 GPU available, all models will be loaded on cuda:0", 3)
        return

    largest_model = model_sizes[0]
    print_fn(
        f"Largest model: {largest_model[1]} ({largest_model[0]}B) -> cuda:0",
        3,
    )
    print_fn(
        f"Tri-model placement: little -> {little_device}, draft -> {draft_device}, target -> {target_device}",
        3,
    )


def load_causal_lm(
    loader,
    model_name: str,
    device_map: str,
    *,
    output_hidden_states: bool = False,
    quant_config: BitsAndBytesConfig | None = None,
):
    load_kwargs: dict[str, object] = {
        "device_map": device_map,
    }
    if output_hidden_states:
        load_kwargs["output_hidden_states"] = True
    if quant_config is not None:
        load_kwargs["quantization_config"] = quant_config

    return loader(model_name, **load_kwargs).eval()
