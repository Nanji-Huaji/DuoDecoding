"""Per-run state and compatibility entry points for registered decoders."""

import os
import sys
from typing import Any, Optional

import torch
import torch.distributed as dist
from accelerate import Accelerator

from . import autoregressive, model_runtime
from .comm_accounting import CommConfig, CommEnv
from .dual_decoding import SpeculativeDecodingMixin
from .graph_decode import acquire_graph_caches, kv_cache_construction_key
from .legacy import bandwidth_sd
from .metrics_dumper import MetricsDumpFactoryLike, _load_default_metrics_dumper_factory
from .model_gpu import KVCacheModel
from .register import Register
from .utils import seed_everything

# The evaluator and smoke tools import these loader settings from the engine.
TRI_HIDDEN_STATE_MODES = model_runtime.TRI_HIDDEN_STATE_MODES
tokenizer_source = model_runtime.tokenizer_source
attn_impl = model_runtime.attn_impl
configure_verbosity = model_runtime.configure_verbosity


class Decoding(Register):
    # Direct callers and registry dispatch use the same implementations. Importing
    # engine still registers these modes without importing the Baselines family.
    autoregressive_sampling = autoregressive.autoregressive_sampling
    lookahead_forward = autoregressive.lookahead_forward
    speculative_decoding = SpeculativeDecodingMixin.speculative_decoding
    speculative_decoding_with_bandwidth = (
        bandwidth_sd.speculative_decoding_with_bandwidth
    )

    _get_available_gpu_count = staticmethod(model_runtime.get_available_gpu_count)

    def __init__(
        self,
        args,
        metrics_dumper_factory: Optional[MetricsDumpFactoryLike] = None,
    ):
        super().__init__(args)
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

        seed_everything(args.seed)
        self.seed = args.seed
        self.seed_set = set()
        self.draft_forward_times = 0
        self.little_forward_times = 0
        self.target_forward_times = 0
        self.num_acc_tokens = []
        self.prob_with_flag = []
        self.vocab_size: int = -1

    @property
    def comm_config(self) -> CommConfig:
        """Billing switches are fixed for this run."""
        cfg = getattr(self, "_comm_config", None)
        if cfg is None:
            cfg = CommConfig.from_args(self.args)
            self._comm_config = cfg
        return cfg

    @property
    def comm_env(self) -> CommEnv:
        # Curriculum training changes bandwidth on the same decoder between samples.
        return CommEnv.from_args(self.args)

    def load_tokenizer(self):
        self.tokenizer = model_runtime.load_tokenizer(
            self.args, print_fn=self.color_print
        )

    def load_model(self):
        self.load_weights()

    def load_weights(self):
        """Install weights only. ModelStore can instead attach already resident models."""
        models = model_runtime.load_model_weights(
            self.args,
            num_gpus=self._get_available_gpu_count(),
            print_fn=self.color_print,
        )
        for name, model in models.items():
            setattr(self, name, model)
        self._validate_runtime_vocab_alignment()
        model_runtime.log_model_devices(self._runtime_models(), self.color_print)

    def _runtime_models(self) -> list[tuple[str, Any]]:
        models = []
        for name in ("little_model", "draft_model", "target_model"):
            model = getattr(self, name, None)
            if model is not None:
                models.append((name, model))
        return models

    def _validate_runtime_vocab_alignment(self) -> None:
        tokenizer = getattr(self, "tokenizer", None)
        if tokenizer is not None:
            self.vocab_size = self.args.vocab_size = model_runtime.shared_vocab_size(
                tokenizer, self._runtime_models(), self.color_print
            )

    get_model_input_device = staticmethod(model_runtime.get_model_input_device)
    _get_model_embedding_vocab_size = staticmethod(model_runtime.embedding_vocab_size)

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
        stop_sequences: Optional[list[str]] = None,
        prompt_len: Optional[int] = None,
    ) -> bool:
        if not hasattr(self, "tokenizer") or self.tokenizer is None:
            return False

        # Multi-token rounds can bury an EOS before the last token.
        if input_ids.shape[1] > 0:
            span = input_ids[0, prompt_len:] if prompt_len else input_ids[0, -1:]
            if bool((span == self.tokenizer.eos_token_id).any()):
                return True

        if stop_sequences:
            decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
            for stop_seq in stop_sequences:
                if decoded_text.endswith(stop_seq):
                    return True
        return False

    def _acquire_autoregressive_cache(
        self, attr: str, graph_kw: dict, model
    ) -> KVCacheModel:
        """Reuse the small draft graph across samples; large passes {} for eager."""
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
            )
            if graph_kw
            else None,
        )["model"]

    def _acquire_speculative_caches(self, attr: str, graph_kw: dict, top_k: int):
        """Only the draft captures graphs; the target always runs eager."""
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
            }
            if graph_kw
            else None,
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
