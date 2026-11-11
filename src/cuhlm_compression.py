"""Pure CU-HLM compression codec and roundoff-aware compressed-vocabulary size.

The CU-HLM paper's Eq. (26) budget is a *proxy*: a beta fit and a softplus
denominator, not the protocol's global total-variation distance.  The old
solver minimized an ideal normalized tail (``top-k plus uniform residual``) and
ignored the actual low-precision codec.  On small vocabularies that ideal tail
can be exactly zero while the codec's returned row differs from the input law
by ``4e-8`` (FP32) or ``2.6e-3`` (BF16); for a tiny tail it can also erase the
dominant coordinate's compensating change unless the difference is centered.

This module therefore owns two things behind one small interface:

* :func:`compress_rebuild_probs` - the *actual* top-k plus uniform-tail codec,
  extracted verbatim from ``CommunicationSimulator`` and reused by both the
  solver check and the runtime path;
* :func:`select_compressed_vocab_size` - build one ``O(V log V)`` ideal screen,
  then verify at most two distinct candidates through the real codec. The
  initial candidate uses the ideal budget ``theta``; if it fails certification,
  one retry uses ``theta/2`` to leave headroom for low-precision codec rounding.
  Every actual-codec check still uses the original budget ``theta``. Each row
  gets one ``sampling_weights`` preprocessing; the verified proxy is its actual
  law's L1 difference divided by Eq. (26)'s beta/softplus denominator.

If neither candidate can be certified, the solver returns ``V``. ``V`` is always
legal and conservative: the codec returns the input unchanged and the proxy is
exactly zero. The solver never enumerates every ``k``; an unverified smaller
answer is not returned just to avoid full-vocabulary transmission.
"""

from __future__ import annotations

import math
import operator
import warnings
from typing import Optional

import torch

from .probability_laws import (
    law_l1_distance,
    selected_probability_law,
    validate_probability_rows,
)
from .sampling import sampling_weights

__all__ = [
    "compress_rebuild_probs",
    "select_compressed_vocab_size",
]


def compress_rebuild_probs(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Apply the CU-HLM codec: keep top-k values, fill the tail uniformly.

    Args:
        probs: shape ``(batch_size, seq_len, vocab_size)``.
        k: number of retained top-k entries.

    Returns:
        A tensor with the input shape and dtype.  ``k >= vocab_size`` returns
        the original object unchanged, exactly as the legacy
        ``CommunicationSimulator.compress_rebuild_probs`` did.

    The body is intentionally the extracted legacy implementation, including
    its FP32/BF16/FP16 arithmetic, explicit top-k mask (a zero value inside the
    top-k is not treated as tail), residual clamp and ``torch.where`` order.
    """
    if probs is None or probs.numel() == 0:
        warnings.warn("警告：probs为空，无法进行压缩重建")
        return torch.empty(0)

    if probs.dim() != 3:
        raise ValueError(f"probs维度应为3，实际为{probs.dim()}，无法进行压缩重建")

    if k >= probs.shape[-1]:
        return probs  # 无需压缩

    batch_size, seq_len, vocab_size = probs.shape

    # 向量化处理：reshape为(batch_size * seq_len, vocab_size)
    flat_probs = probs.view(-1, vocab_size)

    # 批量获取top-k
    top_k_values, top_k_indices = torch.topk(flat_probs, k, dim=-1, sorted=True)

    # 创建压缩的概率分布
    compressed_probs = torch.zeros_like(flat_probs)
    batch_indices = torch.arange(flat_probs.shape[0]).unsqueeze(1).expand(-1, k)
    compressed_probs[batch_indices, top_k_indices] = top_k_values

    # 重建概率分布（tail = 显式非 top-k 掩码；residual clamp 防负"概率"）
    top_k_sum = compressed_probs.sum(dim=-1, keepdim=True)
    residual_mass = (1.0 - top_k_sum).clamp_min(0.0)
    top_k_mask = torch.zeros_like(compressed_probs, dtype=torch.bool)
    top_k_mask[batch_indices, top_k_indices] = True
    tail_mask = ~top_k_mask
    tail_count = tail_mask.sum(dim=-1, keepdim=True)

    uniform_prob = torch.where(
        tail_count > 0,
        residual_mass / tail_count,
        torch.zeros_like(residual_mass),
    )

    # 恢复原始形状
    rebuilt_flat_probs = torch.where(tail_mask, uniform_prob, compressed_probs)
    return rebuilt_flat_probs.view(batch_size, seq_len, vocab_size)


def _validate_request(
    probabilities: torch.Tensor,
    *,
    uncertainty: float,
    theta: float,
    draft_token: Optional[int],
    vocab_size: Optional[int],
) -> tuple[torch.Tensor, int, int]:
    """Validate one full row and scalar parameters; return row, V and token."""
    if not isinstance(probabilities, torch.Tensor):
        raise ValueError(
            "CUHLM compression solver requires a full probability row tensor"
        )
    if probabilities.ndim == 0:
        raise ValueError("CUHLM compression solver requires a vocabulary axis")

    vocab = probabilities.shape[-1]
    if vocab_size is not None:
        try:
            declared = operator.index(vocab_size)
        except TypeError as exc:
            raise ValueError("vocab_size must be an integer") from exc
        if declared != vocab:
            raise ValueError(
                "CUHLM compression row length "
                f"({vocab}) does not match vocab_size ({declared})"
            )
    if probabilities.numel() != vocab:
        raise ValueError(
            "CUHLM compression solver requires exactly one full vocabulary row"
        )

    rows = probabilities.reshape(1, vocab)
    validate_probability_rows(rows, "CUHLM compression row")

    try:
        uncertainty_value = float(uncertainty)
    except (TypeError, ValueError) as exc:
        raise ValueError("uncertainty must be a finite number in [0, 1]") from exc
    if not math.isfinite(uncertainty_value) or not 0.0 <= uncertainty_value <= 1.0:
        raise ValueError("uncertainty must be a finite number in [0, 1]")

    try:
        theta_value = float(theta)
    except (TypeError, ValueError) as exc:
        raise ValueError("theta must be a finite nonnegative number") from exc
    if not math.isfinite(theta_value) or theta_value < 0.0:
        raise ValueError("theta must be a finite nonnegative number")

    if draft_token is None:
        return rows, vocab, -1
    try:
        token = operator.index(draft_token)
    except TypeError as exc:
        raise ValueError("draft_token must be an integer in [0, vocab)") from exc
    if not 0 <= token < vocab:
        raise ValueError("draft_token must be an integer in [0, vocab)")
    return rows, vocab, token


def _eq26_denominator(x_d: float, uncertainty: float) -> float:
    """Eq. (26) denominator proxy: ``(1-x_d)l(-1) + x_d l(-beta_d)``."""
    beta_d = max(0.0, min(1.0, 0.815 * uncertainty - 0.066))
    l_neg_1 = math.log1p(math.exp(-1.0))
    l_neg_beta = math.log1p(math.exp(-beta_d))
    return (1.0 - x_d) * l_neg_1 + x_d * l_neg_beta


def _ideal_proxies(
    raw_probabilities: torch.Tensor, denominator: float
) -> torch.Tensor:
    """Return ideal-codec proxies for k=1..V-1 with one O(V log V) screen.

    The screen works on the *raw* row the codec receives, not a separately
    normalized law: top-k values and the ``1 - top_k_sum`` uniform residual are
    computed in float64 from that raw row, then both sides are normalized for
    the distance.  This tracks the codec's raw-mass convention closely enough
    that an otherwise passing candidate is not rejected merely because the
    sampler's float32 normalization differs in the last few ulps.  It is still
    only a heuristic: :func:`select_compressed_vocab_size` always verifies the
    chosen candidate with the real codec.
    """
    vocab = raw_probabilities.numel()
    sorted_probs, _ = torch.sort(
        raw_probabilities.reshape(-1).double(), descending=True
    )
    prefix_sum = torch.cumsum(sorted_probs, dim=0)
    total = prefix_sum[vocab - 1]
    ks = torch.arange(1, vocab, device=sorted_probs.device)
    top_k_sum = prefix_sum[ks - 1]
    tail_sum = total - top_k_sum
    residual = (1.0 - top_k_sum).clamp_min(0.0)
    tail_count = vocab - ks
    uniform = residual / tail_count
    # The codec's raw top-k plus its uniform tail; normally exactly one, but
    # keep the general sum so a saturated (residual == 0) candidate is modeled.
    codec_mass = top_k_sum + tail_count * uniform

    top_contribution = top_k_sum * (1.0 / total - 1.0 / codec_mass).abs()
    threshold = uniform * total / codec_mass
    asc = -sorted_probs
    pos = torch.searchsorted(asc, -threshold, right=True)
    m = (pos - ks).clamp_min(0)
    idx = (ks + m - 1).clamp_max(vocab - 1)
    head = prefix_sum[idx] - top_k_sum
    tail_contribution = (
        2.0 * (head / total - m * uniform / codec_mass)
        + tail_count * uniform / codec_mass
        - tail_sum / total
    )
    numerator = (top_contribution + tail_contribution).clamp_min(0.0)
    return numerator / denominator


def select_compressed_vocab_size(
    probabilities: torch.Tensor,
    *,
    uncertainty: float,
    theta: float = 0.1,
    draft_token: Optional[int] = None,
    vocab_size: Optional[int] = None,
) -> int:
    """Choose a real-codec-certified compressed vocabulary size.

    ``probabilities`` must be exactly one full row of ``V`` finite,
    nonnegative, approximately unit-mass probabilities; leading singleton
    dimensions such as ``(1, V)`` or ``(1, 1, V)`` are accepted.  ``uncertainty``
    must lie in ``[0, 1]`` and ``theta`` must be finite and nonnegative.
    ``draft_token`` is the actual sampled draft token; when omitted, the top-1
    entry is used as an explicit legacy compatibility fallback, not as a claim
    that it is the actual candidate.  A selected probability of zero is valid
    for this proxy.

    Returns an integer in ``[1, V]``.  For a lossy ``k < V`` the candidate is
    rebuilt by :func:`compress_rebuild_probs`, both rows are preprocessed once
    with :func:`sampling_weights`, and
    ``L1(actual input law, actual codec law) / Eq.26 denominator`` must be at
    most ``theta``. At most two distinct candidates are checked: the first
    ideal hit at ``theta``, then (only if needed) the first hit at ``theta/2``.
    Both are certified against the original ``theta``. If neither passes,
    or ``V == 1``, the result is ``V``.

    The comparison is a direct float64 ``<= theta``; no arbitrary epsilon is
    added to the budget.  This certifies the proxy for the chosen row, not a
    global protocol TVD.
    """
    rows, vocab, token = _validate_request(
        probabilities,
        uncertainty=uncertainty,
        theta=theta,
        draft_token=draft_token,
        vocab_size=vocab_size,
    )
    if vocab == 1:
        return 1

    uncertainty_value = float(uncertainty)
    theta_value = float(theta)

    weights = sampling_weights(rows)
    if token < 0:
        token = int(weights.argmax(dim=-1).item())
    x_d = float(
        selected_probability_law(
            weights.unsqueeze(1),
            torch.tensor([[token]], dtype=torch.long, device=weights.device),
        ).item()
    )
    denominator = _eq26_denominator(x_d, uncertainty_value)
    if not math.isfinite(denominator) or denominator <= 0.0:
        return vocab

    ideal_proxies = _ideal_proxies(rows, denominator)
    previous_candidate = None
    # The first ideal hit hugs theta. FP16/BF16 codec rounding can put it just
    # over budget on almost every large row. Reuse the screen once at half the
    # budget to leave headroom, without relaxing the actual certification test.
    for screening_budget in (theta_value, theta_value / 2.0):
        hits = torch.nonzero(ideal_proxies <= screening_budget, as_tuple=False)
        if hits.numel() == 0:
            break
        candidate = int(hits[0, 0].item()) + 1
        if candidate == previous_candidate:
            break
        previous_candidate = candidate

        try:
            decoded = compress_rebuild_probs(rows.reshape(1, 1, vocab), candidate)
            decoded_rows = decoded.reshape(1, vocab)
            validate_probability_rows(decoded_rows, "CUHLM codec output")
            codec_weights = sampling_weights(decoded_rows)
            proxy = float(
                law_l1_distance(
                    weights,
                    codec_weights,
                    label="CUHLM compression row and codec output",
                ).item()
                / denominator
            )
        except (TypeError, ValueError):
            continue

        if math.isfinite(proxy) and proxy <= theta_value:
            return candidate
    return vocab
