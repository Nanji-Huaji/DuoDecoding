import torch

from .utils import log_prob_tensor_if_invalid, norm_logits, sample

from collections.abc import Sequence
from typing import Any, Protocol, Iterator, TypeAlias, TypeGuard

class TextConfigLike(Protocol):
    vocab_size: int

class ConfigLike(Protocol):
    vocab_size: int
    text_config: TextConfigLike


KVPair: TypeAlias = tuple[torch.Tensor, torch.Tensor]
LegacyPastKeyValues: TypeAlias = tuple[KVPair, ...]


class EmbeddingLike(Protocol):
    weight: torch.Tensor


class CacheLike(Protocol):
    def crop(self, end_pos: int) -> None: ...
    def get_seq_length(self) -> int: ...
    def __iter__(self) -> Iterator[KVPair]: ...


PastKeyValues: TypeAlias = CacheLike | LegacyPastKeyValues | None


class ModelOutputLike(Protocol):
    logits: torch.Tensor | None
    past_key_values: PastKeyValues
    hidden_states: Sequence[torch.Tensor] | None


def _is_cache_like(value: PastKeyValues) -> TypeGuard[CacheLike]:
    return value is not None and hasattr(value, "get_seq_length")


def _is_legacy_past(value: PastKeyValues) -> TypeGuard[LegacyPastKeyValues]:
    return isinstance(value, tuple)

class CausalModel(Protocol):
    config: ConfigLike
    def __call__(self, *args: Any, **kwargs: Any) -> ModelOutputLike: ...
    def get_input_embeddings(self) -> EmbeddingLike: ...
    def parameters(self, recurse: bool = True) -> Iterator[torch.nn.Parameter]: ...


class KVCacheModel:
    def __init__(
        self,
        model: CausalModel,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
        return_hidden_states: bool = False,
        max_length: int = 16384,
    ) -> None:
        self._model: CausalModel = model
        self._past_key_values: PastKeyValues = None

        self._temperature: float = temperature
        self._top_k: int = top_k
        self._top_p: float = top_p
        self.max_length: int = max_length

        self.hidden_states: Sequence[torch.Tensor] | None = None

        if hasattr(model.config, "vocab_size"):
            self.vocab_size = int(model.config.vocab_size)
        elif hasattr(model.config, "text_config") and hasattr(
            model.config.text_config, "vocab_size"
        ):
            self.vocab_size = int(model.config.text_config.vocab_size)
        else:
            raise AttributeError("Vocab size not found in model config")

        # Pre-allocate buffers to eliminate O(N^2) memory allocations via torch.cat
        self._prob_buffer: torch.Tensor | None = None
        self._logits_buffer: torch.Tensor | None = None
        self._current_seq_len: int = 0

    @property
    def _prob_history(self) -> torch.Tensor | None:
        if self._prob_buffer is None:
            return None
        return self._prob_buffer[:, : self._current_seq_len, :]

    @_prob_history.setter
    def _prob_history(self, value):
        pass

    @property
    def logits_history(self) -> torch.Tensor | None:
        if self._logits_buffer is None:
            return None
        return self._logits_buffer[:, : self._current_seq_len, :]

    @logits_history.setter
    def logits_history(self, value):
        pass

    def _ensure_buffer_size(
        self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype
    ):
        # Dynamically resize buffers to prevent OOM on large context while maintaining contiguous memory access
        if self._prob_buffer is None:
            self.max_length = max(2048, seq_len + 1024)
            self._prob_buffer = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            self._logits_buffer = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            return

        if seq_len > self.max_length:
            old_len = self.max_length
            self.max_length = max(self.max_length * 2, seq_len + 1024)

            new_prob = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            new_prob[:, :old_len, :] = self._prob_buffer
            self._prob_buffer = new_prob

            new_logits = torch.empty(
                (batch_size, self.max_length, self.vocab_size),
                device=device,
                dtype=dtype,
            )
            if self._logits_buffer is not None:
                new_logits[:, :old_len, :] = self._logits_buffer
            self._logits_buffer = new_logits

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)

        actual_vocab_size = self._model.get_input_embeddings().weight.shape[0]
        input_ids = torch.clamp(input_ids, 0, actual_vocab_size - 1)

        batch_size = input_ids.shape[0]

        if self._past_key_values is None:
            seq_length = input_ids.shape[1]
            outputs = self._model(input_ids)
            logits = outputs.logits
            if logits is None:
                raise RuntimeError("Model returned logits=None in initial forward")

            self._ensure_buffer_size(
                batch_size, seq_length, logits.device, logits.dtype
            )

            sliced_logits = logits[..., : self.vocab_size]
            if self._logits_buffer is not None:
                self._logits_buffer[:, :seq_length, :] = sliced_logits

            probs = norm_logits(
                sliced_logits, self._temperature, self._top_k, self._top_p
            )
            log_prob_tensor_if_invalid(
                probs[:, -1, :],
                "KVCacheModel._forward_with_kvcache.initial_probs",
            )
            if self._prob_buffer is not None:
                self._prob_buffer[:, :seq_length, :] = probs

            self._current_seq_len = seq_length
            self._past_key_values = outputs.past_key_values
            self.hidden_states = outputs.hidden_states
            last_q = probs[:, -1, :]
        else:
            cached_len = self.current_length
            last_input_id = input_ids[:, cached_len:]

            if last_input_id.dim() == 1:
                last_input_id = last_input_id.unsqueeze(0)

            outputs = self._model(
                last_input_id, past_key_values=self._past_key_values, use_cache=True
            )
            logits = outputs.logits
            if logits is None:
                raise RuntimeError("Model returned logits=None in cached forward")

            new_len = last_input_id.shape[1]
            end_pos = self._current_seq_len + new_len

            self._ensure_buffer_size(
                batch_size, end_pos, logits.device, logits.dtype
            )

            sliced_logits = logits[..., : self.vocab_size]
            if self._logits_buffer is not None:
                self._logits_buffer[:, self._current_seq_len : end_pos, :] = sliced_logits

            probs = norm_logits(
                sliced_logits, self._temperature, self._top_k, self._top_p
            )
            log_prob_tensor_if_invalid(
                probs,
                "KVCacheModel._forward_with_kvcache.cached_probs",
            )
            if self._prob_buffer is not None:
                self._prob_buffer[:, self._current_seq_len : end_pos, :] = probs

            self._current_seq_len = end_pos
            self._past_key_values = outputs.past_key_values
            self.hidden_states = outputs.hidden_states
            last_q = probs[:, -1, :]

        return last_q

    @property
    def last_hidden_state(self) -> torch.Tensor:
        if self.hidden_states is None:
            raise ValueError("hidden_states is None")
        return self.hidden_states[-1]

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        x = prefix
        if x.dtype != torch.long:
            x = x.to(torch.long)

        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            next_tok = sample(q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            x = torch.cat((x, next_tok), dim=1)

        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        return self._generate_with_kvcache(input, gamma)

    @torch.no_grad()
    def rollback(self, end_pos: int):
        if self._past_key_values is None:
            return

        if _is_cache_like(self._past_key_values) and hasattr(
            self._past_key_values, "crop"
        ):
            self._past_key_values.crop(end_pos)
        else:
            assert _is_legacy_past(self._past_key_values)
            past_key_values_trimmed: list[KVPair] = []
            for kv in self._past_key_values:
                k, v = kv
                k = k[:, :, :end_pos, :]
                v = v[:, :, :end_pos, :]
                past_key_values_trimmed.append((k, v))
            self._past_key_values = tuple(past_key_values_trimmed)

        # Keep history length aligned with the real cache length. Rolling back beyond
        # the cache should not expose uninitialized rows from the preallocated buffers.
        self._current_seq_len = min(end_pos, self.current_length)

    @property
    def device(self) -> torch.device:
        return next(iter(self._model.parameters())).device

    @property
    def current_length(self) -> int:
        if self._past_key_values is None:
            return 0
        if _is_cache_like(self._past_key_values):
            return self._past_key_values.get_seq_length()
        assert _is_legacy_past(self._past_key_values)
        return self._past_key_values[0][0].shape[2]

    def debug_state(self) -> dict[str, int]:
        prob_history_len = 0 if self._prob_buffer is None else self._current_seq_len
        logits_history_len = 0 if self._logits_buffer is None else self._current_seq_len
        return {
            "current_length": self.current_length,
            "tracked_seq_len": self._current_seq_len,
            "prob_history_len": prob_history_len,
            "logits_history_len": logits_history_len,
            "max_length": self.max_length,
        }

    def debug_row_sums(self, start: int, end: int) -> list[float]:
        if self._prob_buffer is None or self._current_seq_len <= 0:
            return []

        start = max(0, min(start, self._current_seq_len))
        end = max(start, min(end, self._current_seq_len))
        if end <= start:
            return []

        row_sums = self._prob_buffer[0, start:end, :].detach().float().sum(dim=-1)
        return [float(v.item()) for v in row_sums]

    def __len__(self) -> int:
        return self.current_length
