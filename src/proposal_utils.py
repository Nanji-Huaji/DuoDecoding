from typing import Optional, TYPE_CHECKING

import torch

from .decoding_types import TopKProposalHistory

if TYPE_CHECKING:
    from .model_gpu import KVCacheModel


def proposal_top_k(transfer_top_k: Optional[int]) -> Optional[int]:
    if transfer_top_k is None or transfer_top_k <= 0:
        return None
    return transfer_top_k


def build_draft_probs_override(
    cache: "KVCacheModel",
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


def build_topk_proposal_history_step(
    probs: torch.Tensor,
    top_k: Optional[int],
) -> Optional[TopKProposalHistory]:
    if top_k is None or top_k <= 0 or probs.numel() == 0 or top_k >= probs.shape[-1]:
        return None

    top_k_values, top_k_indices = torch.topk(probs, top_k, dim=-1, sorted=True)
    tail_count = probs.shape[-1] - top_k
    tail_uniform_prob = (1.0 - top_k_values.sum(dim=-1, keepdim=True)).clamp_min(0.0)
    tail_uniform_prob = tail_uniform_prob / tail_count

    return TopKProposalHistory(
        topk_indices=top_k_indices.unsqueeze(1),
        topk_probs=top_k_values.unsqueeze(1),
        tail_uniform_prob=tail_uniform_prob.unsqueeze(1),
        vocab_size=probs.shape[-1],
    )


def concat_topk_proposal_history(
    steps: list[TopKProposalHistory],
) -> Optional[TopKProposalHistory]:
    if not steps:
        return None

    return TopKProposalHistory(
        topk_indices=torch.cat([step.topk_indices for step in steps], dim=1),
        topk_probs=torch.cat([step.topk_probs for step in steps], dim=1),
        tail_uniform_prob=torch.cat(
            [step.tail_uniform_prob for step in steps],
            dim=1,
        ),
        vocab_size=steps[0].vocab_size,
    )


def build_topk_proposal_history_from_probs(
    probs: torch.Tensor,
    top_k: Optional[int],
) -> Optional[TopKProposalHistory]:
    if probs.numel() == 0:
        return None

    if probs.dim() == 2:
        probs = probs.unsqueeze(1)

    steps = []
    for step_idx in range(probs.shape[1]):
        step = build_topk_proposal_history_step(probs[:, step_idx, :], top_k)
        if step is None:
            return None
        steps.append(step)
    return concat_topk_proposal_history(steps)


def build_stage_prefix_topk_history(
    probs: torch.Tensor,
    top_k: Optional[int],
) -> Optional[TopKProposalHistory]:
    return build_topk_proposal_history_from_probs(probs, top_k)


def merge_stage_topk_histories(
    prefix_history: Optional[TopKProposalHistory],
    draft_history: Optional[TopKProposalHistory],
) -> Optional[TopKProposalHistory]:
    if prefix_history is None:
        return draft_history
    if draft_history is None:
        return prefix_history
    if prefix_history.vocab_size != draft_history.vocab_size:
        raise ValueError(
            "Cannot merge top-k histories with different vocab sizes: "
            f"{prefix_history.vocab_size} != {draft_history.vocab_size}"
        )
    if prefix_history.topk_indices.shape[-1] != draft_history.topk_indices.shape[-1]:
        raise ValueError(
            "Cannot merge top-k histories with different top-k widths: "
            f"{prefix_history.topk_indices.shape[-1]} != {draft_history.topk_indices.shape[-1]}"
        )

    return TopKProposalHistory(
        topk_indices=torch.cat(
            (prefix_history.topk_indices, draft_history.topk_indices), dim=1
        ),
        topk_probs=torch.cat(
            (prefix_history.topk_probs, draft_history.topk_probs), dim=1
        ),
        tail_uniform_prob=torch.cat(
            (prefix_history.tail_uniform_prob, draft_history.tail_uniform_prob),
            dim=1,
        ),
        vocab_size=prefix_history.vocab_size,
    )


def stage_topk_proposal_history(
    history: Optional[TopKProposalHistory],
    gamma: int,
) -> Optional[TopKProposalHistory]:
    if history is None or gamma <= 0:
        return history

    return TopKProposalHistory(
        topk_indices=history.topk_indices[:, :gamma, :],
        topk_probs=history.topk_probs[:, :gamma, :],
        tail_uniform_prob=history.tail_uniform_prob[:, :gamma, :],
        vocab_size=history.vocab_size,
    )


def query_topk_proposal_token_probs(
    history: TopKProposalHistory,
    draft_tokens: torch.Tensor,
) -> torch.Tensor:
    if draft_tokens.numel() == 0:
        return draft_tokens.to(dtype=history.topk_probs.dtype)

    topk_indices = history.topk_indices[:, : draft_tokens.shape[1], :]
    topk_probs = history.topk_probs[:, : draft_tokens.shape[1], :]
    tail_uniform_prob = history.tail_uniform_prob[:, : draft_tokens.shape[1], :]

    token_matches = topk_indices == draft_tokens.unsqueeze(-1)
    match_mask = token_matches.any(dim=-1)
    match_index = token_matches.to(torch.int64).argmax(dim=-1, keepdim=True)
    gathered_topk_probs = torch.gather(topk_probs, 2, match_index).squeeze(-1)
    return torch.where(match_mask, gathered_topk_probs, tail_uniform_prob.squeeze(-1))


def stage_prob_history(
    cache: "KVCacheModel",
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
    cache: "KVCacheModel",
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
