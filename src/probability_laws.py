"""Shared pure probability-law kernels for speculative verification.

One implementation of the numerically delicate pieces, used by both the strict
multi-token verifier (:mod:`src.verification`) and the single-token CU-HLM
verifier (:mod:`src.cuhlm_verification`):

* :func:`validate_probability_rows` - validate a raw probability law without
  normalizing an invalid one;
* :func:`selected_probability_law` - the normalized selected scalar under the
  actual float32 sampler weights, with the global scale removed first;
* :func:`positive_residual_law` - the stable ``normalize((P - Q)+)`` rejection
  law computed from the pivot/cross-product form.

These functions draw no randomness, sample no token, touch no cache or
communication state, and never mutate their inputs.  Callers validate raw rows
first; the residual kernel is then pure float64 arithmetic on already-validated
sampler weights.  Extraction from the strict verifier intentionally preserved
its operation order, dtype promotion (float32 sampler weights -> float64) and
scale invariance: changing those details changes accepted/rejected sets at
float32 rounding boundaries.
"""

from __future__ import annotations

import torch


def validate_probability_rows(rows: torch.Tensor, label: str) -> None:
    """Check approximate unit mass without changing the sampler's raw inputs.

    Softmax accumulates in at least FP32, then rounds its output dtype. Its mass
    error depends on vocabulary size, not only the output epsilon. Use the
    conservative positive-sum bound gamma_(V-1), reciprocal/product rounding and
    output subnormal rounding. This is an input-admissibility budget, never an
    epsilon added to acceptance or a zero residual. Measure mass in FP64 so the
    check does not introduce another FP32 reduction error.
    """
    if not rows.is_floating_point():
        raise ValueError(f"{label} must have a floating probability dtype")
    if not torch.isfinite(rows).all() or (rows < 0).any():
        raise ValueError(f"{label} must be finite and nonnegative")
    accumulation_dtype = (
        torch.float64 if rows.dtype == torch.float64 else torch.float32
    )
    info = torch.finfo(rows.dtype)
    unit = torch.finfo(accumulation_dtype).eps / 2
    vocab = rows.shape[-1]
    nu = max(0, vocab - 1) * unit
    if nu >= 0.5:
        raise ValueError(f"{label} vocabulary exceeds the mass-validation bound")
    gamma = nu / (1 - nu)
    roundoff = (1 + unit) ** 2 * (1 + info.eps / 2) / (1 - gamma) - 1
    # Include absolute output error when half-precision tail values are subnormal.
    subnormal = vocab * info.tiny * info.eps / 2
    tolerance = max(1e-6, 4 * info.eps, roundoff + subnormal)
    totals = rows.sum(dim=-1, dtype=torch.float64)
    if (
        not torch.isfinite(totals).all()
        or (totals <= 0).any()
        or not torch.allclose(totals, torch.ones_like(totals), rtol=0, atol=tolerance)
    ):
        raise ValueError(f"{label} rows must sum to one")


def selected_probability_law(
    weights: torch.Tensor, tokens: torch.Tensor
) -> torch.Tensor:
    """Return ``w(token) / sum(w)`` in float64 for already-preprocessed weights.

    The global scale is removed before summing.  Exactly proportional float32
    rows therefore yield identical double ratios, sums and selected scalars,
    including at FP32 rounding midpoints: identical sampler laws must never
    spuriously reject.
    """
    precise = weights.double()
    relative = precise / precise.amax(dim=-1, keepdim=True)
    indices = tokens.to(weights.device).unsqueeze(-1)
    selected = torch.gather(relative, 2, indices).squeeze(-1)
    return selected / relative.sum(dim=-1)


def positive_residual_law(
    target_weights: torch.Tensor,
    reference_weights: torch.Tensor,
    *,
    label: str = "rejected proposal",
) -> torch.Tensor:
    """Return normalized ``(P - Q)+`` from float32 sampler weights.

    ``P`` is the target law, ``Q`` the law the rejection residual is measured
    against (the proposal for exact SD, the reconstructed ``qhat`` for CU-HLM).
    Rows must already be validated and preprocessed, shape ``(batch, vocab)``.

    The pivot/cross-product form removes the global weight scale before summing
    differences.  Products of two float32 weights are exact in float64 (at most
    48 mantissa bits), so a pivot difference of exactly proportional rows is
    exactly zero, even for nearly proportional rows with tiny tails; a plain
    ``P - Q`` can lose those tails in ``sum(delta)``.  Centering is equivalent
    to ``q[pivot] * sum(P) * (P - Q)``: the positive pivot weight prevents a
    nonzero proportional component from cancelling all positive mass.

    A rejection with no positive finite residual mass raises instead of falling
    back to epsilon, a uniform tail, argmax or the original proposal; such a
    zero-residual reject cannot be sampled from the stated law.
    """
    p_weights = target_weights.double()
    q_weights = reference_weights.double()
    pivot = q_weights.argmax(dim=-1, keepdim=True)
    p_pivot = torch.gather(p_weights, -1, pivot)
    q_pivot = torch.gather(q_weights, -1, pivot)
    delta = p_weights * q_pivot - q_weights * p_pivot
    correction = delta.sum(dim=-1, keepdim=True) / q_weights.sum(
        dim=-1, keepdim=True
    )
    residual = (delta - q_weights * correction).clamp_min(0)
    mass = residual.sum(dim=-1, keepdim=True)
    if not torch.isfinite(mass).all() or (mass <= 0).any():
        raise ValueError(
            f"{label} has no positive finite residual mass; "
            "cannot draw a rejection residual"
        )
    return residual / mass
