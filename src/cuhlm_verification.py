"""Single-token CU-HLM verification: strict decision, reconstructed residual.

Paper II-A / IV-B and Algorithm 1 finalize exactly one draft token per round.
This module owns only the numeric contract of that round; it has no cache,
model, communication, tokenizer or metrics state:

* the **original proposal** is ``ProposalWindow`` (the raw draft row actually
  handed to the sampler).  Acceptance compares the target law with the
  normalized selected ``q(d)`` derived from those actual sampler weights;
* the **target law P** is the last row of an exact forward over the already
  finalized prefix.  There is no bonus row: only ``prefix_len`` history rows are
  accepted, so a missing or extra row is an error rather than a silent clip;
* the **reconstructed law qhat** is the codec output the server actually holds.
  It is validated as a raw probability row and normalized once by the sampler
  preprocessing, never replaced by the original ``q``.  On reject, the returned
  continuation is ``normalize((P - qhat)+)``; when that mass is zero the kernel
  raises instead of substituting epsilon, a uniform tail, argmax or ``q``.

The caller validates the whole window before drawing acceptance RNG, draws one
uniform in ``[0, 1)``, and only then asks for a decision.  An accepted decision
keeps the draft token and samples nothing; a rejected decision returns the
normalized residual law for the caller to sample once.  This is the paper's
intentional lossy protocol, so it is not claimed to be target-distribution
exact, and the original-proposal/P/reconstructed roles stay distinct.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .probability_laws import (
    positive_residual_law,
    selected_probability_law,
    validate_probability_rows,
)
from .sampling import sampling_weights
from .verification import ProposalWindow


@dataclass(frozen=True, eq=False)
class CuhlmVerificationWindow:
    """Validated single-token proposal, target row and reconstructed law."""

    proposal: ProposalWindow
    target_row: torch.Tensor
    target_weights: torch.Tensor
    reconstructed_weights: torch.Tensor
    selected_q: torch.Tensor
    selected_p: torch.Tensor


@dataclass(frozen=True, eq=False)
class CuhlmVerificationDecision:
    """Strict decision for one candidate; no cache, RNG or sampled token."""

    accepted: bool
    residual_probs: Optional[torch.Tensor]


def prepare_cuhlm_verification(
    proposal: ProposalWindow,
    target_prefix_history: torch.Tensor,
    reconstructed_probabilities: torch.Tensor,
) -> CuhlmVerificationWindow:
    """Validate one candidate, its target row and its reconstructed law.

    ``target_prefix_history`` must contain exactly the rows produced by the
    round's forward over the old prefix, shape ``(1, prefix_len, >= vocab)``.
    Row ``prefix_len - 1`` is ``P`` for the candidate; extra vocabulary padding
    is sliced away, but extra or missing context rows are rejected.

    ``reconstructed_probabilities`` is the raw codec output for that candidate,
    shape ``(1, 1, vocab)``.  It must be a valid probability row; it is then
    normalized exactly once (the same float32 sampler preprocessing used for
    ``P``), so a full-vocabulary reconstruction that equals the raw proposal
    reproduces the strict SD residual bit for bit.
    """
    if not isinstance(proposal, ProposalWindow):
        raise ValueError("CUHLM verification requires a ProposalWindow")
    if proposal.count != 1:
        raise ValueError("CUHLM verification requires exactly one candidate")

    vocab = proposal.probabilities.shape[-1]
    device = proposal.probabilities.device
    if (
        not isinstance(target_prefix_history, torch.Tensor)
        or target_prefix_history.ndim != 3
        or target_prefix_history.shape[0] != 1
        or target_prefix_history.shape[1] != proposal.prefix_len
        or target_prefix_history.shape[2] < vocab
    ):
        raise ValueError(
            "target history must cover exactly the forwarded prefix "
            f"({proposal.prefix_len} rows, at least {vocab} vocabulary entries)"
        )
    target_rows = target_prefix_history[:, proposal.prefix_len - 1:, :vocab].to(device)
    validate_probability_rows(target_rows, "target")
    target_weights = sampling_weights(target_rows)

    if (
        not isinstance(reconstructed_probabilities, torch.Tensor)
        or reconstructed_probabilities.ndim != 3
        or reconstructed_probabilities.shape != (1, 1, vocab)
    ):
        raise ValueError(
            "reconstructed probabilities must have shape (1, 1, vocab)"
        )
    reconstructed = reconstructed_probabilities.to(device)
    validate_probability_rows(reconstructed, "reconstructed")
    reconstructed_weights = sampling_weights(reconstructed)

    selected_q = proposal.selected_probabilities()
    selected_p = selected_probability_law(
        target_weights, proposal.tokens[:, proposal.prefix_len:]
    )
    if not torch.isfinite(selected_p).all() or (selected_p < 0).any():
        raise ValueError("target selected probability must be finite and nonnegative")
    return CuhlmVerificationWindow(
        proposal=proposal,
        target_row=target_rows,
        target_weights=target_weights,
        reconstructed_weights=reconstructed_weights,
        selected_q=selected_q,
        selected_p=selected_p,
    )


def decide_cuhlm_verification(
    window: CuhlmVerificationWindow, uniform: torch.Tensor
) -> CuhlmVerificationDecision:
    """Apply the strict ``uniform < P(d) / q(d)`` rule to one candidate.

    The uniform is an explicit value in ``[0, 1)``.  Equality rejects.  A zero
    target support is legal and rejects even at ``uniform == 0``.  The residual
    law is only required to have positive mass on rejection; the accepted branch
    never computes it, so ``P == qhat`` may accept with a zero residual.
    """
    if not isinstance(window, CuhlmVerificationWindow):
        raise ValueError("decision requires a prepared CUHLM window")
    if (
        not isinstance(uniform, torch.Tensor)
        or uniform.shape != (1, 1)
        or not uniform.is_floating_point()
        or not torch.isfinite(uniform).all()
        or (uniform < 0).any()
        or (uniform >= 1).any()
    ):
        raise ValueError(
            "acceptance uniform must have shape (1, 1) and lie in [0, 1)"
        )
    ratio = window.selected_p.double() / window.selected_q.double()
    accepted = bool((uniform.to(ratio.device) < ratio).item())
    if accepted:
        return CuhlmVerificationDecision(accepted=True, residual_probs=None)
    residual = positive_residual_law(
        window.target_weights[:, 0, :],
        window.reconstructed_weights[:, 0, :],
        label="CUHLM rejection under the reconstructed law",
    )
    return CuhlmVerificationDecision(accepted=False, residual_probs=residual)
