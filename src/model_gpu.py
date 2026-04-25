import torch
from transformers.cache_utils import DynamicCache

from .proposal_utils import (
    build_topk_proposal_history_step,
    concat_topk_proposal_history,
)
from .utils import (
    log_prob_tensor_if_invalid,
    norm_logits,
    numeric_debug_checks_enabled,
    rebuild_topk_uniform_probs,
    sample,
)

from collections.abc import Sequence
from typing import Any, Protocol, Iterator, TypeAlias, TypeGuard, cast, Optional


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
    config: Any

    def __call__(self, *args: Any, **kwargs: Any) -> ModelOutputLike: ...
    def get_input_embeddings(self) -> torch.nn.Module: ...
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
        embeddings = cast(EmbeddingLike, model.get_input_embeddings())
        self.embedding_vocab_size = int(embeddings.weight.shape[0])

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

    def _new_dynamic_cache(self) -> DynamicCache:
        return DynamicCache(config=self._model.config)

    def _build_model_inputs(self, input_ids: torch.Tensor, *, use_cache: bool) -> dict:
        model_inputs: dict[str, object] = {
            "input_ids": input_ids,
            "use_cache": use_cache,
        }

        seq_len = input_ids.shape[1]
        device = input_ids.device
        past_seen_tokens = self.current_length if self._past_key_values is not None else 0
        attention_len = past_seen_tokens + seq_len if use_cache else seq_len
        attention_mask = torch.ones(
            (input_ids.shape[0], attention_len), dtype=torch.long, device=device
        )
        model_inputs["attention_mask"] = attention_mask

        return model_inputs

    def _prepare_generation_inputs(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: PastKeyValues,
    ) -> dict[str, object]:
        batch_size, seq_len = input_ids.shape
        cache_start = 0
        if past_key_values is not None and _is_cache_like(past_key_values):
            cache_start = past_key_values.get_seq_length()

        attention_mask = torch.ones(
            (batch_size, cache_start + seq_len),
            dtype=torch.long,
            device=input_ids.device,
        )
        cache_position = torch.arange(
            cache_start,
            cache_start + seq_len,
            dtype=torch.long,
            device=input_ids.device,
        )

        if hasattr(self._model, "prepare_inputs_for_generation"):
            prepared_inputs = self._model.prepare_inputs_for_generation(
                input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                cache_position=cache_position,
                use_cache=True,
            )
            prepared_inputs["use_cache"] = True
            return cast(dict[str, object], prepared_inputs)

        model_inputs = self._build_model_inputs(input_ids, use_cache=True)
        model_inputs["past_key_values"] = past_key_values
        model_inputs["cache_position"] = cache_position
        return model_inputs

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

    @property
    def prob_history(self) -> torch.Tensor:
        if self._prob_history is None:
            raise ValueError("Probability history buffer is not initialized")
        return self._prob_history

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

    def _raise_if_invalid_probs(self, probs: torch.Tensor, label: str) -> None:
        if not numeric_debug_checks_enabled():
            return
        if not log_prob_tensor_if_invalid(probs, label):
            return

        model_name = getattr(
            getattr(self._model, "config", None), "_name_or_path", "unknown"
        )
        probs_float = probs.detach().float()
        row_sums = probs_float.sum(dim=-1)
        raise ValueError(
            f"Invalid probability tensor before sampling for {model_name} at {label}: "
            f"shape={tuple(probs.shape)}, row_sum_min={float(row_sums.min().item())}, "
            f"row_sum_max={float(row_sums.max().item())}"
        )

    @torch.inference_mode()
    def _prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        input: (batch_size, seq_len)
        output: (batch_size, vocab_size) - probabilities for the next token after the entire input sequence
        """
        self._validate_input_ids(input_ids)
        seq_length = input_ids.shape[1]
        batch_size = input_ids.shape[0]
        self._past_key_values = self._new_dynamic_cache()
        outputs = self._model(
            **self._prepare_generation_inputs(
                input_ids,
                past_key_values=self._past_key_values,
            )
        )
        logits = outputs.logits
        if logits is None:
            raise RuntimeError("Model returned logits=None in prefill")

        self._ensure_buffer_size(batch_size, seq_length, logits.device, logits.dtype)
        sliced_logits = logits[..., : self.vocab_size]
        assert self._logits_buffer is not None, (
            "Logits buffer should not be None after ensuring buffer size"
        )
        self._logits_buffer[:, :seq_length, :] = sliced_logits

        probs = norm_logits(sliced_logits, self._temperature, self._top_k, self._top_p)
        log_prob_tensor_if_invalid(
            probs[:, -1, :],
            "KVCacheModel._forward_with_kvcache.initial_probs",
        )

        assert self._prob_buffer is not None, (
            "Probability buffer should not be None after ensuring buffer size"
        )
        self._prob_buffer[:, :seq_length, :] = probs
        self._current_seq_len = seq_length
        self._past_key_values = outputs.past_key_values
        self.hidden_states = outputs.hidden_states

        return probs[:, -1, :]

    # @torch.compile()
    @torch.inference_mode()
    def _decode_step(self, last_input_id: torch.Tensor) -> torch.Tensor:
        """
        Decode one cached step (or a short cached suffix) after prefill.
        """
        if last_input_id.dtype != torch.long:
            last_input_id = last_input_id.to(torch.long)

        if last_input_id.shape[1] == 0:
            if self._current_seq_len <= 0:
                raise RuntimeError(
                    "No new input provided for decode step and no cache available"
                )
            return self.prob_history[:, self._current_seq_len - 1, :]

        past_key_values = self._past_key_values
        if past_key_values is None:
            raise RuntimeError("Decode step called before cache initialization")

        batch_size = last_input_id.shape[0]
        new_len = last_input_id.shape[1]
        outputs = self._model(
            **self._prepare_generation_inputs(
                last_input_id,
                past_key_values=past_key_values,
            )
        )
        logits = outputs.logits

        if logits is None:
            raise RuntimeError("Model returned logits=None in decode step")

        new_past_key_values = outputs.past_key_values
        if new_past_key_values is None:
            raise RuntimeError("Model returned past_key_values=None in decode step")

        end_pos = self._current_seq_len + new_len

        self._ensure_buffer_size(batch_size, end_pos, logits.device, logits.dtype)

        sliced_logits = logits[..., : self.vocab_size]
        if self._logits_buffer is not None:
            self._logits_buffer[:, self._current_seq_len : end_pos, :] = sliced_logits

        probs = norm_logits(sliced_logits, self._temperature, self._top_k, self._top_p)
        log_prob_tensor_if_invalid(
            probs,
            "KVCacheModel._forward_with_kvcache.cached_probs",
        )
        if self._prob_buffer is not None:
            self._prob_buffer[:, self._current_seq_len : end_pos, :] = probs

        self._current_seq_len = end_pos
        self._past_key_values = new_past_key_values
        self.hidden_states = outputs.hidden_states

        return probs[:, -1, :]

    def _validate_input_ids(self, input_ids: torch.Tensor) -> None:
        if input_ids.numel() == 0:
            return

        min_id = int(input_ids.min().item())
        max_id = int(input_ids.max().item())
        if min_id < 0 or max_id >= self.embedding_vocab_size:
            model_name = getattr(
                getattr(self._model, "config", None), "_name_or_path", "unknown"
            )
            raise ValueError(
                "Input token id out of embedding range for model "
                f"{model_name}: min={min_id}, max={max_id}, "
                f"embedding_vocab_size={self.embedding_vocab_size}, "
                f"configured_vocab_size={self.vocab_size}"
            )

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dtype != torch.long:
            input_ids = input_ids.to(torch.long)

        if self._past_key_values is None:
            return self._prefill(input_ids)

        cached_len = self.current_length
        last_input_id = input_ids[:, cached_len:]

        if last_input_id.numel() == 0:
            if self._current_seq_len <= 0:
                raise RuntimeError(
                    "No new input provided for decode step and no cache available"
                )
            return self.prob_history[:, self._current_seq_len - 1, :]

        return self._decode_step(last_input_id)

    @property
    def last_hidden_state(self) -> torch.Tensor:
        if self.hidden_states is None:
            raise ValueError("hidden_states is None")
        return self.hidden_states[-1]

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        x = prefix
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x

        # First step: use _forward_with_kvcache to handle prefill or cache extension
        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(q, "KVCacheModel._generate_with_kvcache.q")
        next_tok = sample(q)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens: list[torch.Tensor] = [next_tok]

        # Subsequent steps: use _decode_step with only the new token to avoid
        # growing x with torch.cat on each iteration
        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(q, "KVCacheModel._generate_with_kvcache.q")
            next_tok = sample(q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        return torch.cat([x] + new_tokens, dim=1)

    def generate_with_rebuilt_topk(
        self,
        input: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        x = input
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x, None

        rebuilt_rows: list[torch.Tensor] = []
        new_tokens: list[torch.Tensor] = []

        # First step: use _forward_with_kvcache to handle prefill or cache extension
        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(q, "KVCacheModel.generate_with_rebuilt_topk.q")
        rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
        self._raise_if_invalid_probs(
            rebuilt_q, "KVCacheModel.generate_with_rebuilt_topk.rebuilt_q"
        )
        rebuilt_rows.append(rebuilt_q.unsqueeze(1))
        next_tok = sample(rebuilt_q)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens.append(next_tok)

        # Subsequent steps: use _decode_step with only the new token to avoid
        # growing x with torch.cat on each iteration
        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(
                q, "KVCacheModel.generate_with_rebuilt_topk.q"
            )
            rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
            self._raise_if_invalid_probs(
                rebuilt_q, "KVCacheModel.generate_with_rebuilt_topk.rebuilt_q"
            )
            rebuilt_rows.append(rebuilt_q.unsqueeze(1))
            next_tok = sample(rebuilt_q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        rebuilt_history = torch.cat(rebuilt_rows, dim=1)
        return torch.cat([x] + new_tokens, dim=1), rebuilt_history

    def generate_with_rebuilt_topk_metadata(
        self,
        input: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
    ):
        x = input
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x, None, None

        rebuilt_rows: list[torch.Tensor] = []
        proposal_steps = []
        new_tokens: list[torch.Tensor] = []

        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(
            q, "KVCacheModel.generate_with_rebuilt_topk_metadata.q"
        )
        rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
        self._raise_if_invalid_probs(
            rebuilt_q,
            "KVCacheModel.generate_with_rebuilt_topk_metadata.rebuilt_q",
        )
        rebuilt_rows.append(rebuilt_q.unsqueeze(1))
        proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
        if proposal_meta is not None:
            proposal_steps.append(proposal_meta)
        next_tok = sample(rebuilt_q)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens.append(next_tok)

        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(
                q, "KVCacheModel.generate_with_rebuilt_topk_metadata.q"
            )
            rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
            self._raise_if_invalid_probs(
                rebuilt_q,
                "KVCacheModel.generate_with_rebuilt_topk_metadata.rebuilt_q",
            )
            rebuilt_rows.append(rebuilt_q.unsqueeze(1))
            proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
            if proposal_meta is not None:
                proposal_steps.append(proposal_meta)
            next_tok = sample(rebuilt_q)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        rebuilt_history = torch.cat(rebuilt_rows, dim=1)
        rebuilt_history_meta = concat_topk_proposal_history(proposal_steps)
        return torch.cat([x] + new_tokens, dim=1), rebuilt_history, rebuilt_history_meta

    def _sample_from_topk_proposal(
        self,
        probs: torch.Tensor,
        proposal_top_k: Optional[int],
    ) -> torch.Tensor:
        proposal_meta = build_topk_proposal_history_step(probs, proposal_top_k)
        if proposal_meta is None:
            token = sample(probs)
            return token.to(torch.long) if token.dtype != torch.long else token

        topk_indices = proposal_meta.topk_indices[:, 0, :]
        topk_probs = proposal_meta.topk_probs[:, 0, :]
        tail_uniform_prob = proposal_meta.tail_uniform_prob[:, 0, :]
        topk_mass = topk_probs.sum(dim=-1, keepdim=True).clamp(0.0, 1.0)
        tail_mass = (1.0 - topk_mass).clamp_min(0.0)
        region_probs = torch.cat((topk_mass, tail_mass), dim=-1)
        region_choice = torch.multinomial(region_probs, num_samples=1)

        token = torch.empty(
            (probs.shape[0], 1),
            dtype=torch.long,
            device=probs.device,
        )

        topk_rows = region_choice.squeeze(-1) == 0
        if topk_rows.any():
            normalized_topk = topk_probs[topk_rows] / topk_mass[topk_rows].clamp_min(1e-12)
            topk_pick = torch.multinomial(normalized_topk, num_samples=1)
            token[topk_rows] = torch.gather(topk_indices[topk_rows], 1, topk_pick)

        tail_rows = ~topk_rows
        if tail_rows.any():
            tail_topk_indices = topk_indices[tail_rows]
            batch_size, _, = tail_topk_indices.shape
            vocab_size = probs.shape[-1]
            candidate_mask = torch.ones(
                (batch_size, vocab_size),
                dtype=torch.bool,
                device=probs.device,
            )
            candidate_mask.scatter_(1, tail_topk_indices, False)
            candidate_weights = candidate_mask.to(probs.dtype)
            tail_pick = torch.multinomial(candidate_weights, num_samples=1)
            token[tail_rows] = tail_pick

        return token

    def generate_with_topk_metadata_only(
        self,
        input: torch.Tensor,
        gamma: int,
        proposal_top_k: Optional[int],
    ):
        x = input
        if x.dtype != torch.long:
            x = x.to(torch.long)

        if gamma == 0:
            return x, None

        proposal_steps = []
        new_tokens: list[torch.Tensor] = []

        q = self._forward_with_kvcache(x)
        self._raise_if_invalid_probs(
            q, "KVCacheModel.generate_with_topk_metadata_only.q"
        )
        proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
        if proposal_meta is not None:
            proposal_steps.append(proposal_meta)
        next_tok = self._sample_from_topk_proposal(q, proposal_top_k)
        if next_tok.dtype != torch.long:
            next_tok = next_tok.to(torch.long)
        new_tokens.append(next_tok)

        for _ in range(gamma - 1):
            q = self._decode_step(new_tokens[-1])
            self._raise_if_invalid_probs(
                q, "KVCacheModel.generate_with_topk_metadata_only.q"
            )
            proposal_meta = build_topk_proposal_history_step(q, proposal_top_k)
            if proposal_meta is not None:
                proposal_steps.append(proposal_meta)
            next_tok = self._sample_from_topk_proposal(q, proposal_top_k)
            if next_tok.dtype != torch.long:
                next_tok = next_tok.to(torch.long)
            new_tokens.append(next_tok)

        rebuilt_history_meta = concat_topk_proposal_history(proposal_steps)
        return torch.cat([x] + new_tokens, dim=1), rebuilt_history_meta

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
