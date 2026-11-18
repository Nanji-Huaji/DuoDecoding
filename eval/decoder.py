"""Keep one resident weight set, not the decoder that last used it."""

import gc
import os
from collections.abc import Mapping
from typing import NamedTuple

import torch

from src.baselines import Baselines
from src.comm_accounting import CommEnv
from src.communication import configure_ntt_trace
from src.engine import TRI_HIDDEN_STATE_MODES, Decoding, tokenizer_source
from src.mode_features import get_mode_spec
from src.model_loading import should_quantize
from src.model_zoo import resolve_model

_ROLES = {
    "small": ("draft",),
    "large": ("target",),
    "dual": ("draft", "target"),
    "tri": ("little", "draft", "target"),
}


class ModelKey(NamedTuple):
    # Each entry identifies a loaded role, its weights, and effective quantization.
    weights: tuple[tuple[str, str, str], ...]
    tokenizer: str
    dtype: str
    draft_hidden_states: bool
    keep_target_on_single_gpu: bool
    visible_devices: str | None
    num_gpus: int


def model_key(config) -> ModelKey:
    """Project a matrix config or parsed Namespace onto actual loader settings."""
    raw_config = isinstance(config, Mapping)
    fields = config if raw_config else vars(config)
    mode = fields["eval_mode"]
    spec = get_mode_spec(mode)
    paths = {}
    for role in ("little", "draft", "target"):
        name = fields.get(f"{role}_model")
        if name and raw_config:
            # Already-parsed args contain the paths actually passed to the loader.
            name, _, _ = resolve_model(name, fields.get("model_source", "auto"), {})
        paths[role] = name

    weights = []
    for role in _ROLES[spec.models]:
        name = paths[role]
        if not name:
            raise ValueError(f"{mode}: missing {role}_model")
        quant = fields.get(f"{role}_quantization", "auto")
        if quant == "auto":
            quant = "4bit" if should_quantize(name) else "none"
        weights.append((role, name, quant))
    tokenizer = tokenizer_source(mode, paths["draft"], paths["target"])
    if not tokenizer:
        raise ValueError(f"{mode}: missing tokenizer source")
    hidden = (
        spec.models == "dual" and spec.uses_main_rl
    ) or mode in TRI_HIDDEN_STATE_MODES
    return ModelKey(
        tuple(weights),
        tokenizer,
        fields.get("model_dtype", "bf16"),
        hidden,
        bool(
            spec.models == "dual"
            and not spec.uses_main_rl
            and fields.get("keep_target_on_single_gpu")
        ),
        os.environ.get("CUDA_VISIBLE_DEVICES"),
        Decoding._get_available_gpu_count(),
    )


class ModelStore:
    """Serial runs only. Drop the previous decoder before switching weight sets."""

    def __init__(self):
        self._key = None
        self._models = {}

    def new_decoder(self, args, task: str) -> Baselines:
        key = model_key(args)
        reuse = key == self._key
        if not reuse:
            # Release before allocating anything new; don't keep a local reference.
            self.close()
        if not CommEnv.from_args(args).ntt_trace_file:
            configure_ntt_trace([])  # Otherwise the preceding run's trace survives.
        decoder = Baselines(args)
        decoder.task = task
        decoder.load_tokenizer()
        if reuse:
            for name, model in self._models.items():
                setattr(decoder, name, model)
            decoder._validate_runtime_vocab_alignment()
            decoder.color_print(
                f"[ModelStore] reusing loaded weights: {key.weights}", 3
            )
        else:
            decoder.color_print(f"[ModelStore] loading weights: {key.weights}", 3)
            decoder.load_weights()
            models = {name: model for name, model in decoder._runtime_models()}
            if set(models) != {f"{role}_model" for role, _, _ in key.weights}:
                raise RuntimeError(
                    f"Incomplete model load: {list(models)} for {key.weights}"
                )
            self._models, self._key = models, key
        decoder.load_acc_head()
        return decoder

    def close(self):
        if self._key is not None:
            self._models.clear()
            self._key = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
