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
