"""Strict CEE numerical stages; no simulator, decoder or accounting state.

The inner verifier is the *decoded wire draft*, not raw D. Its original payloads
follow its output into the outer proposal. Cache histories remain model-owned.
"""

from dataclasses import dataclass

import torch

from .cee_proposal import CeeWireRow, proposal_from_wire
from .verification import (
    VerificationWindow,
    decide_verification,
    prepare_verification_window,
)


@dataclass(frozen=True, eq=False)
class PreparedCascadeStage:
    verification: VerificationWindow
    verifier_rows: torch.Tensor
    verifier_payloads: tuple[CeeWireRow, ...] | None


def prepare_inner_stage(
    proposal, draft, codec, *, selected_q=None
) -> PreparedCascadeStage:
    """Forward once, encode draft target rows, validate BEFORE acceptance RNG."""
    window = proposal.window
    draft.forward_context(window.tokens.to(draft.device))
    history = draft.prob_history
    if history.shape[1] != window.tokens.shape[1]:
        raise ValueError("CEE draft history must cover exactly the little window")
    start = window.prefix_len - 1
    payloads = tuple(
        codec.encode(history[:, i, :]) for i in range(start, history.shape[1])
    )
    rows = torch.stack([payload.dense() for payload in payloads], dim=1)
    effective_history = torch.cat((history[:, :start, : codec.vocab_size], rows), dim=1)
    prepared = prepare_verification_window(
        window, effective_history, selected_q=selected_q
    )
    return PreparedCascadeStage(prepared, rows, payloads)


def prepare_target_stage(proposal, target, *, selected_q=None) -> PreparedCascadeStage:
    """Final target stays uncompressed. Its bonus is sampled on its own device."""
    window = proposal.window
    target.forward_context(window.tokens.to(target.device))
    history = target.prob_history
    prepared = prepare_verification_window(window, history, selected_q=selected_q)
    rows = history[:, window.prefix_len - 1 :, : window.probabilities.shape[-1]].clone()
    return PreparedCascadeStage(prepared, rows, None)


def decide_stage(prepared, rng):
    window = prepared.verification
    count = window.proposal.count
    uniforms = (
        rng.rand((1, count), device=window.proposal.probabilities.device)
        if count
        else window.proposal.probabilities.new_empty((1, 0))
    )
    return decide_verification(window, uniforms)


def finish_stage(prepared, decision, rng):
    """Return accepted context plus one unforwarded continuation, never raw D."""
    proposal = prepared.verification.proposal
    law = (
        prepared.verifier_rows[:, decision.accepted_count, :]
        if decision.all_accepted
        else decision.continuation_probs
    )
    token = rng.sample(law).to(proposal.tokens.device)
    return torch.cat(
        (proposal.tokens[:, : decision.retained_context_end], token), dim=1
    )


def carry_inner_output(prepared, decision, tokens):
    """Keep the Q_D rows actually used, including the inner correction/bonus row.

    The row for the inner continuation is Q_D, not the conditional rejection
    residual: the *whole* inner accept/correct kernel targets Q_D. These payloads
    were captured before rollback or extra draft forwarding could change storage.
    """
    if prepared.verifier_payloads is None:
        raise ValueError("only an effective-draft stage can supply CEE carried rows")
    proposal = prepared.verification.proposal
    return proposal_from_wire(
        tokens,
        proposal.prefix_len,
        prepared.verifier_payloads[: decision.accepted_count + 1],
        proposal.probabilities.shape[-1],
        device=prepared.verifier_rows.device,
    )


def retain_cascade_context(caches, end):
    for cache in caches:
        cache.rollback(min(int(cache.current_length), int(end)))


@dataclass
class CascadeStats:
    rounds: int = 0
    little_forwards: int = 0
    draft_forwards: int = 0
    target_forwards: int = 0
    little_candidates: int = 0
    draft_candidates: int = 0
    little_accepted: int = 0
    draft_accepted: int = 0
