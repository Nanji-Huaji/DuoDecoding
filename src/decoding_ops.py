from __future__ import annotations

from typing import Literal, Optional, Tuple

import torch

from .communication import CommunicationSimulator
from .debug_logs import _log_invalid_batch_details, _log_sd_alignment_snapshot
from .decoding_types import AcceptanceResult, RollbackPlan, VerificationInputs
from .metrics import DecodingMetrics
from .model_gpu import KVCacheModel
from .utils import (
    log_prob_tensor_if_invalid,
    log_ratio_if_invalid,
    max_fn,
    rebuild_topk_uniform_probs,
    sample,
)


def collect_verification_payload(
    prob_history: torch.Tensor,
    x: torch.Tensor,
    prefix_len: int,
    gamma: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if gamma <= 0:
        draft_tokens = x[:, 0:0]
        empty_probs = prob_history[:, 0:0, 0]
        return draft_tokens, empty_probs

    available_prob_steps = max(prob_history.shape[1] - (prefix_len - 1), 0)
    available_token_steps = max(x.shape[1] - prefix_len, 0)
    actual_gamma = min(gamma, available_prob_steps, available_token_steps)

    draft_tokens = x[:, prefix_len : prefix_len + actual_gamma]
    if actual_gamma <= 0:
        empty_probs = prob_history[:, 0:0, 0]
        return draft_tokens, empty_probs

    draft_prob_rows = prob_history[:, prefix_len - 1 : prefix_len + actual_gamma - 1, :]
    draft_token_probs = torch.gather(
        draft_prob_rows,
        2,
        draft_tokens.unsqueeze(-1),
    ).squeeze(-1)
    return draft_tokens, draft_token_probs


def prepare_verification_inputs(
    draft_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    x: torch.Tensor,
    prefix_len: int,
    gamma: int,
    draft_probs_override: Optional[torch.Tensor] = None,
) -> VerificationInputs:
    draft_device = draft_model_cache.device
    draft_probs = (
        draft_probs_override
        if draft_probs_override is not None
        else draft_model_cache.prob_history
    )
    if draft_probs is None or target_model_cache.prob_history is None:
        raise ValueError("Probability history is not initialized for verification")

    max_idx = min(
        prefix_len + gamma - 1,
        draft_probs.shape[1],
        target_model_cache.prob_history.shape[1],
    )
    actual_gamma = max_idx - (prefix_len - 1)
    if actual_gamma <= 0:
        empty_tokens = x[:, 0:0]
        empty_indices = empty_tokens.unsqueeze(-1)
        empty_probs = draft_probs[:, 0:0, :]
        return VerificationInputs(
            draft_probs_batch=empty_probs,
            target_probs_batch=target_model_cache.prob_history[:, 0:0, :].to(
                draft_device
            ),
            draft_tokens=empty_tokens,
            draft_token_indices=empty_indices,
            prefix_len=prefix_len,
            gamma=gamma,
            actual_gamma=0,
            max_idx=max_idx,
        )

    draft_probs_batch = draft_probs[:, prefix_len - 1 : max_idx, :]
    target_probs_batch = target_model_cache.prob_history[
        :, prefix_len - 1 : max_idx, :
    ].to(draft_device)
    draft_tokens = x[:, prefix_len : prefix_len + actual_gamma]
    draft_token_indices = draft_tokens.unsqueeze(-1)

    return VerificationInputs(
        draft_probs_batch=draft_probs_batch,
        target_probs_batch=target_probs_batch,
        draft_tokens=draft_tokens,
        draft_token_indices=draft_token_indices,
        prefix_len=prefix_len,
        gamma=gamma,
        actual_gamma=actual_gamma,
        max_idx=max_idx,
    )


def compute_acceptance_result(
    verification_inputs: VerificationInputs,
    *,
    r: Optional[torch.Tensor] = None,
) -> AcceptanceResult:
    if verification_inputs.actual_gamma <= 0:
        return AcceptanceResult(
            accepted_count=0,
            n=verification_inputs.prefix_len - 1,
            selected_draft_p=verification_inputs.draft_probs_batch[:, 0:0, 0],
            selected_target_p=verification_inputs.target_probs_batch[:, 0:0, 0],
            accept_mask=torch.zeros(
                (verification_inputs.draft_probs_batch.shape[0], 0),
                dtype=torch.bool,
                device=verification_inputs.draft_probs_batch.device,
            ),
        )

    selected_draft_p = torch.gather(
        verification_inputs.draft_probs_batch,
        2,
        verification_inputs.draft_token_indices,
    ).squeeze(-1)
    selected_target_p = torch.gather(
        verification_inputs.target_probs_batch,
        2,
        verification_inputs.draft_token_indices,
    ).squeeze(-1)

    if r is None:
        r = torch.rand(
            (selected_draft_p.shape[0], verification_inputs.actual_gamma),
            device=selected_draft_p.device,
        )

    accept_mask = r <= (selected_target_p / selected_draft_p)
    continuous_accept, _ = accept_mask.to(torch.int8).cummin(dim=1)
    accepted_count = int(continuous_accept[0].sum().item())
    n = verification_inputs.prefix_len + accepted_count - 1

    if accepted_count == verification_inputs.actual_gamma:
        n = verification_inputs.prefix_len + verification_inputs.actual_gamma - 1

    return AcceptanceResult(
        accepted_count=accepted_count,
        n=int(n),
        selected_draft_p=selected_draft_p,
        selected_target_p=selected_target_p,
        accept_mask=accept_mask,
    )


def compute_residual_distribution(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    return max_fn(target_probs - draft_probs)


def build_rollback_plan(prefix_len: int, gamma: int, n: int) -> RollbackPlan:
    all_accepted = n >= prefix_len + gamma - 1
    return RollbackPlan(
        draft_end_pos=n + 1,
        target_end_pos_reject=n + 1,
        target_end_pos_accept=n + 2,
        all_accepted=all_accepted,
    )


def apply_rollback(
    draft_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    rollback_plan: RollbackPlan,
) -> None:
    draft_model_cache.rollback(rollback_plan.draft_end_pos)
    if rollback_plan.all_accepted:
        target_model_cache.rollback(rollback_plan.target_end_pos_accept)
    else:
        target_model_cache.rollback(rollback_plan.target_end_pos_reject)


def sample_reject_token(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    if target_probs.device != draft_probs.device:
        target_probs = target_probs.to(draft_probs.device)

    residual_probs = compute_residual_distribution(target_probs, draft_probs)
    log_prob_tensor_if_invalid(
        residual_probs,
        "sample_reject_token.residual_probs",
    )
    token = sample(residual_probs)
    if output_device is not None and token.device != output_device:
        token = token.to(output_device)
    return token


def sample_accept_token(
    target_next_probs: torch.Tensor,
    output_device: Optional[torch.device] = None,
) -> torch.Tensor:
    log_prob_tensor_if_invalid(
        target_next_probs,
        "sample_accept_token.target_next_probs",
    )
    token = sample(target_next_probs)
    if output_device is not None and token.device != output_device:
        token = token.to(output_device)
    return token


def verify_draft_sequence(
    draft_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    x: torch.Tensor,
    prefix_len: int,
    gamma: int,
    comm_simulator: Optional[CommunicationSimulator] = None,
    comm_link: Literal["edge_cloud", "edge_end", "cloud_end"] = "edge_cloud",
    transfer_mode: Literal["none", "serial", "batch_before"] = "serial",
    send_reject_message: bool = True,
    draft_probs_override: Optional[torch.Tensor] = None,
    decoding_metrics: Optional[DecodingMetrics] = None,
) -> Tuple[int, int]:
    draft_device = draft_model_cache.device
    _log_sd_alignment_snapshot(
        "verify_enter",
        prefix_len,
        draft_model_cache,
        target_model_cache,
        x_len=x.shape[1],
        gamma=gamma,
    )

    verification_inputs = prepare_verification_inputs(
        draft_model_cache=draft_model_cache,
        target_model_cache=target_model_cache,
        x=x,
        prefix_len=prefix_len,
        gamma=gamma,
        draft_probs_override=draft_probs_override,
    )
    if verification_inputs.actual_gamma <= 0:
        return 0, prefix_len - 1

    draft_probs_batch = verification_inputs.draft_probs_batch
    target_probs_batch = verification_inputs.target_probs_batch
    invalid_draft_probs = log_prob_tensor_if_invalid(
        draft_probs_batch,
        "verify_draft_sequence.draft_probs_batch",
    )
    invalid_target_probs = log_prob_tensor_if_invalid(
        target_probs_batch,
        "verify_draft_sequence.target_probs_batch",
    )

    if transfer_mode == "batch_before" and comm_simulator is not None:
        batch_probs = torch.gather(
            draft_probs_batch,
            2,
            verification_inputs.draft_token_indices,
        ).squeeze(-1)
        comm_simulator.transfer(
            verification_inputs.draft_tokens,
            batch_probs,
            comm_link,
        )

    r = torch.rand(
        (x.shape[0], verification_inputs.actual_gamma),
        device=draft_device,
    )
    acceptance_result = compute_acceptance_result(verification_inputs, r=r)
    invalid_acceptance_ratio = log_ratio_if_invalid(
        acceptance_result.selected_target_p,
        acceptance_result.selected_draft_p,
        "verify_draft_sequence.acceptance_ratio",
    )
    if invalid_draft_probs or invalid_target_probs or invalid_acceptance_ratio:
        _log_invalid_batch_details(
            prefix_len=prefix_len,
            gamma=gamma,
            max_idx=verification_inputs.max_idx,
            actual_gamma=verification_inputs.actual_gamma,
            x=x,
            draft_model_cache=draft_model_cache,
            target_model_cache=target_model_cache,
            draft_probs_batch=draft_probs_batch,
            target_probs_batch=target_probs_batch,
            selected_draft_p=acceptance_result.selected_draft_p,
            selected_target_p=acceptance_result.selected_target_p,
        )
    accepted_counts = acceptance_result.accepted_count
    n = acceptance_result.n
    if (
        accepted_counts < verification_inputs.actual_gamma
        and send_reject_message
        and comm_simulator
    ):
        comm_simulator.send_reject_message(comm_link)

    if transfer_mode == "serial" and comm_simulator is not None:
        for i in range(
            accepted_counts
            + (1 if accepted_counts < verification_inputs.actual_gamma else 0)
        ):
            comm_simulator.transfer(
                verification_inputs.draft_token_indices[0, i, 0],
                draft_probs_batch[:, i, :].squeeze(0),
                comm_link,
            )

    if decoding_metrics is not None:
        decoding_metrics["draft_generated_tokens"] += gamma
        decoding_metrics["draft_accepted_tokens"] += int(n - prefix_len + 1)

    return acceptance_result.accepted_count, int(n)


def verify_draft_sequence_result(
    draft_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    x: torch.Tensor,
    prefix_len: int,
    gamma: int,
    *,
    draft_probs_override: Optional[torch.Tensor] = None,
    r: Optional[torch.Tensor] = None,
) -> Tuple[VerificationInputs, AcceptanceResult]:
    verification_inputs = prepare_verification_inputs(
        draft_model_cache=draft_model_cache,
        target_model_cache=target_model_cache,
        x=x,
        prefix_len=prefix_len,
        gamma=gamma,
        draft_probs_override=draft_probs_override,
    )
    acceptance_result = compute_acceptance_result(verification_inputs, r=r)
    return verification_inputs, acceptance_result


def resolve_stage_verification(
    proposer_cache: KVCacheModel,
    verifier_cache: KVCacheModel,
    x: torch.Tensor,
    prefix_len: int,
    gamma: int,
    *,
    output_device: torch.device,
    draft_probs_override: Optional[torch.Tensor] = None,
) -> Tuple[int, int, torch.Tensor, bool]:
    vocab_limit = min(proposer_cache.vocab_size, verifier_cache.vocab_size)
    verification_inputs, acceptance_result = verify_draft_sequence_result(
        draft_model_cache=proposer_cache,
        target_model_cache=verifier_cache,
        x=x,
        prefix_len=prefix_len,
        gamma=gamma,
        draft_probs_override=draft_probs_override,
    )
    n = acceptance_result.n
    rollback_plan = build_rollback_plan(
        prefix_len,
        verification_inputs.actual_gamma,
        n,
    )

    if rollback_plan.all_accepted:
        t = sample_accept_token(
            verifier_cache.prob_history[:, -1, : verifier_cache.vocab_size],
            output_device=output_device,
        )
    else:
        rejection_offset = n - (prefix_len - 1)
        t = sample_reject_token(
            verification_inputs.target_probs_batch[:, rejection_offset, :vocab_limit],
            verification_inputs.draft_probs_batch[:, rejection_offset, :vocab_limit],
            output_device=output_device,
        )

    apply_rollback(
        proposer_cache,
        verifier_cache,
        rollback_plan,
    )
    return acceptance_result.accepted_count, n, t, rollback_plan.all_accepted


def finalize_verification(
    approx_model_cache: KVCacheModel,
    target_model_cache: KVCacheModel,
    x: torch.Tensor,
    prefix_len: int,
    gamma: int,
    n: int,
    draft_probs_override: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    prefix = x[:, : n + 1]
    rollback_plan = build_rollback_plan(prefix_len, gamma, n)

    approx_model_cache.rollback(rollback_plan.draft_end_pos)

    draft_probs = (
        draft_probs_override
        if draft_probs_override is not None
        else approx_model_cache.prob_history
    )

    if not rollback_plan.all_accepted:
        target_prob_slice = target_model_cache.prob_history[
            :, n, : target_model_cache.vocab_size
        ]
        approx_prob_slice = draft_probs[:, n, : approx_model_cache.vocab_size]

        t = sample_reject_token(
            target_prob_slice,
            approx_prob_slice,
            output_device=prefix.device,
        )
        target_model_cache.rollback(rollback_plan.target_end_pos_reject)
    else:
        next_target_probs = target_model_cache.prob_history[
            :, -1, : target_model_cache.vocab_size
        ]
        t = sample_accept_token(
            next_target_probs,
            output_device=prefix.device,
        )
        target_model_cache.rollback(rollback_plan.target_end_pos_accept)

    return torch.cat((prefix, t), dim=1)
