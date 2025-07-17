import torch
from . import model_gpu
from typing import Optional, Union, List, Dict, Any
import os, sys
from pathlib import Path

from .gpt_fast.model import Transformer
from .gpt_fast.generate import decode_one_token, decode_n_tokens, causal_mask, create_block_mask, prefill
from .utils import norm_logits, sample


class GPTFastWarpper(model_gpu.KVCacheModel):
    """
    GPTFastWarpper is a wrapper for the GPTFast model, providing methods to call gpt-fast models on duodecoding.
    """

    def __init__(
        self,
        model: Transformer,
        temperature: float = 1,
        top_k: int = 0,
        top_p: float = 0,
    ) -> None:
        self._model = model
        self._past_key_values = None
        self._prob_history = None

        self._temperature = temperature
        self._top_k = top_k
        self._top_p = top_p

        # gpt-fast specific attributes
        self._block_mask = None
        self._current_seq_len = 0

    @property
    def vocab_size(self):
        return self._model.config.vocab_size

    def _forward_with_kvcache(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward the model with KV cache"""
        if self._past_key_values is None:
            # First forward pass - initialize caches
            seq_len = input_ids.shape[1]
            max_seq_len = max(seq_len * 2, 1024)  # Reserve some extra space

            # Setup caches for gpt-fast model
            self._model.setup_caches(max_batch_size=input_ids.shape[0], max_seq_length=max_seq_len)

            # Create input position tensor
            input_pos = torch.arange(0, seq_len, device=input_ids.device)

            # Create block mask for causal attention
            self._block_mask = create_block_mask(causal_mask, 1, 1, max_seq_len, max_seq_len, device=input_ids.device)

            # Forward pass
            mask = create_block_mask(causal_mask, 1, 1, seq_len, max_seq_len, device=input_ids.device)
            outputs = self._model(mask, input_ids, input_pos)

            # Process logits and create prob_history
            self._prob_history = outputs[:, :, : self.vocab_size]
            for i in range(self._prob_history.shape[-2]):
                self._prob_history[:, i, :] = norm_logits(
                    self._prob_history[:, i, :],
                    self._temperature,
                    self._top_k,
                    self._top_p,
                )

            # Mark that we have cached data
            self._past_key_values = True  # gpt-fast manages its own cache internally
            self._current_seq_len = seq_len
            last_q = self._prob_history[:, -1, :]

        else:
            # Subsequent forward passes - use cached data
            cached_len = self._current_seq_len
            last_input_id = input_ids[:, cached_len:]

            if last_input_id.dim() == 1:
                last_input_id = torch.unsqueeze(last_input_id, 0)

            # Create input position for new tokens
            new_seq_len = last_input_id.shape[1]
            input_pos = torch.arange(cached_len, cached_len + new_seq_len, device=input_ids.device)

            # Use pre-computed block mask with appropriate slicing
            block_index = input_pos // self._block_mask.BLOCK_SIZE[0]
            mask = self._block_mask[:, :, block_index]
            mask.mask_mod = self._block_mask.mask_mod
            mask.seq_lengths = (new_seq_len, self._model.max_seq_length)

            # Forward pass for new tokens
            outputs = self._model(mask, last_input_id, input_pos)

            not_cached_q = outputs[:, :, : self.vocab_size]

            if not_cached_q.dim() == 2:
                not_cached_q = torch.unsqueeze(not_cached_q, 0)

            # Normalize new logits
            for i in range(not_cached_q.shape[-2]):
                not_cached_q[:, i, :] = norm_logits(not_cached_q[:, i, :], self._temperature, self._top_k, self._top_p)

            # Update prob_history
            self._prob_history = torch.cat([self._prob_history, not_cached_q], dim=1)
            self._current_seq_len += new_seq_len
            last_q = not_cached_q[:, -1, :]

        return last_q

    def _generate_with_kvcache(self, prefix: torch.Tensor, gamma: int) -> torch.Tensor:
        """Generate tokens with KV cache using gpt-fast's optimized decode functions

        Args:
            prefix (torch.Tensor): the prefix
            gamma (int): how many times approx guesses

        Returns:
            torch.Tensor: prefix+generated tokens
        """
        x = prefix.clone()

        if self._past_key_values is None:
            # Initialize cache with prefill
            seq_len = x.shape[1]
            max_seq_len = max(seq_len + gamma, 1024)
            self._model.setup_caches(max_batch_size=x.shape[0], max_seq_length=max_seq_len)

            # Use gpt-fast's prefill function
            input_pos = torch.arange(0, seq_len, device=x.device)
            next_token = prefill(
                self._model, x, input_pos, temperature=self._temperature, top_k=self._top_k if self._top_k > 0 else None
            )

            # Update state
            self._past_key_values = True
            self._current_seq_len = seq_len
            x = torch.cat([x, next_token], dim=1)
            gamma -= 1  # Already generated one token

        if gamma > 0:
            # Use gpt-fast's decode_n_tokens for batch generation
            input_pos = torch.tensor([self._current_seq_len], device=x.device)
            new_tokens, new_probs = decode_n_tokens(
                self._model,
                x[:, -1:],  # Last token
                input_pos,
                gamma,
                temperature=self._temperature,
                top_k=self._top_k if self._top_k > 0 else None,
            )

            # Concatenate new tokens
            if new_tokens:
                x = torch.cat([x] + new_tokens, dim=1)

            # Update prob_history if needed for compatibility
            if hasattr(self, "_prob_history") and new_probs:
                prob_tensor = torch.stack(new_probs, dim=1)  # [B, seq, vocab]
                if self._prob_history is None:
                    self._prob_history = prob_tensor
                else:
                    self._prob_history = torch.cat([self._prob_history, prob_tensor], dim=1)

            self._current_seq_len += gamma

        return x

    @torch.no_grad()
    def generate(self, input: torch.Tensor, gamma: int) -> torch.Tensor:
        """Generate tokens"""
        output = self._generate_with_kvcache(input, gamma)
        return output

    @torch.no_grad()
    def generate_cape(self, input_ids: torch.Tensor, gamma: int) -> torch.Tensor:
        """Generate tokens using CAPE method"""
        output = self._generate_with_kvcache(input_ids, gamma)
        return output

    @torch.no_grad()
    def rollback(self, end_pos: int):
        """Rollback the KV cache to a specific position"""
        if self._past_key_values is None:
            return

        # Rollback prob_history
        if self._prob_history is not None:
            self._prob_history = self._prob_history[:, :end_pos, :]

        # Update current sequence length
        self._current_seq_len = end_pos

        # For gpt-fast, we need to rollback the internal KV caches in each layer
        for layer in self._model.layers:
            if hasattr(layer.attention, "kv_cache") and layer.attention.kv_cache is not None:
                # Trim the k_cache and v_cache to end_pos
                kv_cache = layer.attention.kv_cache
                if hasattr(kv_cache, "k_cache") and hasattr(kv_cache, "v_cache"):
                    # Zero out the cache beyond end_pos
                    kv_cache.k_cache[:, :, end_pos:, :] = 0
                    kv_cache.v_cache[:, :, end_pos:, :] = 0
