"""Strict, batch-one verification values with no model or protocol ownership.

These operations never draw randomness, sample a token, send a message or mutate
a cache. The caller validates a window, then supplies uniform draws explicitly,
and owns continuation sampling/communication/commit order. Raw tensor fields are
borrowed round-local values; producers must not mutate them during the round.
Raw probabilities preserve the sampler input (and a reconstructible full-row
payload). Internal weights capture the actual float32 sampler preprocessing;
normalization and stable residual arithmetic use float64, not raw BF16 rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from .sampling import sampling_weights


def _probability_rows(rows: torch.Tensor, label: str) -> None:
    if not rows.is_floating_point():
        raise ValueError(f"{label} must have a floating probability dtype")
    if not torch.isfinite(rows).all() or (rows < 0).any():
        raise ValueError(f"{label} must be finite and nonnegative")
    # Accumulate low-precision rows in float32, without normalizing an invalid law.
    accumulation_dtype = (
        torch.float64 if rows.dtype == torch.float64 else torch.float32
    )
    totals = rows.sum(dim=-1, dtype=accumulation_dtype)
    tolerance = max(1e-6, 4 * torch.finfo(rows.dtype).eps)
    if not torch.allclose(totals, torch.ones_like(totals), rtol=0, atol=tolerance):
        raise ValueError(f"{label} rows must sum to one")


def _selected_law(weights: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
    # Canonicalize the global scale first. Exactly proportional float32 rows
    # then yield identical double ratios, sums and selected scalars, including
    # at FP32 rounding midpoints; identical laws must never spuriously reject.
    precise = weights.double()
    relative = precise / precise.amax(dim=-1, keepdim=True)
    indices = tokens.to(weights.device).unsqueeze(-1)
    selected = torch.gather(relative, 2, indices).squeeze(-1)
    return selected / relative.sum(dim=-1)


@dataclass(frozen=True, eq=False)
class ProposalWindow:
    """Candidates and the raw sampler-input q rows that actually generated them."""

    tokens: torch.Tensor
    probabilities: torch.Tensor
    prefix_len: int
    sampling_weights: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, torch.Tensor) or not isinstance(
            self.probabilities, torch.Tensor
        ):
            raise ValueError("proposal tokens and probabilities must be tensors")
        if self.tokens.ndim != 2 or self.tokens.shape[0] != 1:
            raise ValueError("proposal supports exactly one token sequence")
        if self.tokens.dtype != torch.long:
            raise ValueError("proposal tokens must use torch.long")
        if (
            not isinstance(self.prefix_len, int)
            or isinstance(self.prefix_len, bool)
            or not 1 <= self.prefix_len <= self.tokens.shape[1]
        ):
            raise ValueError("proposal prefix must be nonempty and within tokens")
        q = self.probabilities
        if q.ndim != 3 or q.shape[:2] != (1, self.count) or q.shape[-1] < 1:
            raise ValueError("proposal probabilities must cover exactly its candidates")
        if (self.tokens < 0).any() or (self.tokens >= q.shape[-1]).any():
            raise ValueError("proposal token is outside its vocabulary")
        _probability_rows(q, "proposal")
        object.__setattr__(self, "sampling_weights", sampling_weights(q))
        if (self.selected_probabilities() <= 0).any():
            raise ValueError("proposal has non-positive selected support")

    def selected_probabilities(self) -> torch.Tensor:
        """Actual normalized selected-Q law; wire rounding is the caller's choice."""
        return _selected_law(self.sampling_weights, self.tokens[:, self.prefix_len:])

    @property
    def count(self) -> int:
        return int(self.tokens.shape[1]) - self.prefix_len


@dataclass(frozen=True, eq=False)
class VerificationWindow:
    """Prepared target rows, including the explicit bonus context as its last row."""

    proposal: ProposalWindow
    target_rows: torch.Tensor
    target_weights: torch.Tensor
    selected_q: torch.Tensor
    selected_p: torch.Tensor


@dataclass(frozen=True, eq=False)
class VerificationDecision:
    """Numeric decision and continuation law; no cache, policy or simulator state."""

    accepted_count: int
    accepted_end: int
    candidate_count: int
    continuation_probs: torch.Tensor
    target_row: torch.Tensor

    @property
    def all_accepted(self) -> bool:
        return self.accepted_count == self.candidate_count

    @property
    def retained_context_end(self) -> int:
        """Exclusive accepted context endpoint, not including the new sampled token."""
        return self.accepted_end + 1


def prepare_verification_window(
    proposal: ProposalWindow, target_history: torch.Tensor,
    *, selected_q: torch.Tensor | None = None,
) -> VerificationWindow:
    """Validate full forwarded context BEFORE the caller consumes acceptance RNG.

    Earlier prompt rows are not part of this round's probability window. Extra
    vocabulary padding may be sliced away, but extra/missing context rows are an
    error rather than a reason to truncate the candidate count silently.

    A supplied selected_q is the normalized scalar payload received by a split
    verifier. It alone drives that verifier's Q comparison (no hidden row sum);
    target scalars use the same wire precision. FP32 wire rounding is explicit,
    not an exact-real transmission claim. Full-row verification defaults to
    float64 arithmetic on the actual sampler weights.
    """
    vocab = proposal.probabilities.shape[-1]
    if (
        not isinstance(target_history, torch.Tensor)
        or target_history.ndim != 3
        or target_history.shape[0] != 1
        or target_history.shape[1] != proposal.tokens.shape[1]
        or target_history.shape[2] < vocab
    ):
        raise ValueError(
            "target history must cover exactly candidate and bonus context"
        )
    rows = target_history[:, proposal.prefix_len - 1:, :vocab].to(
        proposal.probabilities.device
    )
    _probability_rows(rows, "target")
    weights = sampling_weights(rows)
    if selected_q is None:
        selected_q = proposal.selected_probabilities()
    elif (
        not isinstance(selected_q, torch.Tensor)
        or selected_q.shape != (1, proposal.count)
        or selected_q.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(selected_q).all()
        or (selected_q <= 0).any()
        or (selected_q > 1).any()
    ):
        raise ValueError("selected q payload must contain positive normalized scalars")
    selected_q = selected_q.to(proposal.probabilities.device)
    selected_p = _selected_law(
        weights[:, :proposal.count], proposal.tokens[:, proposal.prefix_len:]
    ).to(selected_q.dtype)
    return VerificationWindow(proposal, rows, weights, selected_q, selected_p)


def decide_verification(
    window: VerificationWindow, uniforms: torch.Tensor,
) -> VerificationDecision:
    """Apply strict acceptance and return the residual or explicit bonus law.

    ``window`` must come from :func:`prepare_verification_window`. Uniforms are
    explicit values in [0, 1), including for deterministic boundary tests. Empty
    candidate windows use an empty draw tensor and return the target bonus row.
    """
    proposal = window.proposal
    q = proposal.probabilities
    if (
        not isinstance(uniforms, torch.Tensor)
        or uniforms.shape != (1, proposal.count)
        or not uniforms.is_floating_point()
        or not torch.isfinite(uniforms).all()
        or (uniforms < 0).any()
        or (uniforms >= 1).any()
    ):
        raise ValueError(
            "acceptance uniforms must have shape (1, count) and lie in [0, 1)"
        )
    if proposal.count:
        ratio = window.selected_p.double() / window.selected_q.double()
        mask = uniforms.to(q.device) < ratio
        accepted = int(mask.to(torch.int8).cummin(dim=1).values.sum().item())
    else:
        accepted = 0
    target_row = window.target_rows[:, accepted, :]
    if accepted == proposal.count:
        continuation = target_row
    else:
        p_weights = window.target_weights[:, accepted, :].double()
        q_weights = proposal.sampling_weights[:, accepted, :].double()
        pivot = q_weights.argmax(dim=-1, keepdim=True)
        p_pivot = torch.gather(p_weights, -1, pivot)
        q_pivot = torch.gather(q_weights, -1, pivot)
        # Remove the global weight scale BEFORE summing differences. Products
        # of our float32 weights are exact in float64 (at most 48 mantissa bits),
        # so the pivot difference is exactly zero, even for nearly proportional
        # rows with tiny tails. A plain p-q can lose those tails in sum(delta).
        delta = p_weights * q_pivot - q_weights * p_pivot
        # Centering now equals q[pivot] * sum(p) * (P-Q). The positive pivot
        # weight prevents a nonzero proportional component from cancelling all
        # positive mass. Normalize before casting a tiny residual to float32.
        correction = delta.sum(dim=-1, keepdim=True) / q_weights.sum(
            dim=-1, keepdim=True
        )
        residual = (delta - q_weights * correction).clamp_min(0)
        mass = residual.sum(dim=-1, keepdim=True)
        if not torch.isfinite(mass).all() or (mass <= 0).any():
            raise ValueError("rejected proposal has no positive finite residual mass")
        continuation = residual / mass
    return VerificationDecision(
        accepted_count=accepted,
        accepted_end=proposal.prefix_len + accepted - 1,
        candidate_count=proposal.count,
        continuation_probs=continuation,
        target_row=target_row,
    )
