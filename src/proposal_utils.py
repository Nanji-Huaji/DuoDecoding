from typing import Optional

import torch

from .model_gpu import KVCacheModel


def proposal_top_k(transfer_top_k: Optional[int]) -> Optional[int]:
    if transfer_top_k is None or transfer_top_k <= 0:
        return None
    return transfer_top_k


def build_draft_probs_override(
    cache: KVCacheModel,
    stage_start_len: int,
    rebuilt_draft_probs: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if rebuilt_draft_probs is None:
        return None
    return torch.cat(
        (
            cache.prob_history[:, : stage_start_len - 1, :],
            rebuilt_draft_probs,
        ),
        dim=1,
    )


def stage_prob_history(
    cache: KVCacheModel,
    stage_start_len: int,
    rebuilt_draft_probs: Optional[torch.Tensor],
) -> torch.Tensor:
    override = build_draft_probs_override(
        cache,
        stage_start_len,
        rebuilt_draft_probs,
    )
    return cache.prob_history if override is None else override


def stage_prob_batch(
    cache: KVCacheModel,
    stage_start_len: int,
    prefix_len: int,
    gamma: int,
    rebuilt_draft_probs: Optional[torch.Tensor],
) -> torch.Tensor:
    if gamma <= 0:
        return cache.prob_history[:, 0:0, :]

    history = cache.prob_history
    start_idx = prefix_len - 1
    end_idx = start_idx + gamma

    if rebuilt_draft_probs is not None:
        history_prefix_end = max(stage_start_len - 1, start_idx)
        history_prefix = history[:, start_idx:history_prefix_end, :]
        remaining_gamma = max(gamma - history_prefix.shape[1], 0)
        rebuilt_tail = rebuilt_draft_probs[:, :remaining_gamma, :]
        if history_prefix.shape[1] == 0:
            return rebuilt_tail
        if rebuilt_tail.shape[1] == 0:
            return history_prefix
        return torch.cat((history_prefix, rebuilt_tail), dim=1)

    return history[:, start_idx:end_idx, :]
