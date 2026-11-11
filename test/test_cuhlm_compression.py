"""CU-HLM compression contract: real-codec k selection and stable law distance.

The old solver minimized an *ideal* normalized tail and ignored the actual
low-precision codec roundoff.  These tests use a small-vocabulary exhaustive
oracle built from ``Fraction`` values of the actual float32 sampler weights, so
they never reuse the production L1 helper or the old float32 brute-force loop.
"""

from fractions import Fraction
import math
import warnings

import pytest
import torch

from src.communication import CUHLM
from src.cuhlm_compression import (
    compress_rebuild_probs,
    select_compressed_vocab_size,
)
from src.probability_laws import law_l1_distance
from src.sampling import sampling_weights


# ---------------------------------------------------------------------------
# Independent small-vocabulary oracle (exact sampler laws, exhaustive over k)
# ---------------------------------------------------------------------------


def _exact_law(probs: torch.Tensor) -> list[Fraction]:
    weights = sampling_weights(probs.reshape(-1)).reshape(-1)
    exact = [Fraction(float(value)) for value in weights]
    mass = sum(exact)
    return [value / mass for value in exact]


def _exact_l1(probs: torch.Tensor, decoded: torch.Tensor) -> float:
    p = _exact_law(probs)
    q = _exact_law(decoded)
    return float(sum(abs(a - b) for a, b in zip(p, q)))


def _exact_proxy(probs: torch.Tensor, k: int, uncertainty: float, token: int) -> float:
    decoded = compress_rebuild_probs(probs.reshape(1, 1, -1), k)
    l1 = _exact_l1(probs, decoded)
    x_d = _exact_law(probs)[token]
    beta = max(0.0, min(1.0, 0.815 * uncertainty - 0.066))
    denominator = (1 - x_d) * math.log1p(math.exp(-1.0)) + x_d * math.log1p(
        math.exp(-beta)
    )
    return l1 / denominator


def _actual_l1(probs: torch.Tensor, k: int) -> float:
    decoded = compress_rebuild_probs(probs.reshape(1, 1, -1), k)
    p = sampling_weights(probs.reshape(1, -1))
    q = sampling_weights(decoded.reshape(1, -1))
    return float(law_l1_distance(p, q).item())


def _legacy_codec(probs: torch.Tensor, k: int) -> torch.Tensor:
    """Verbatim copy of the pre-extraction ``compress_rebuild_probs`` body."""
    if probs is None or probs.numel() == 0:
        warnings.warn("警告：probs为空，无法进行压缩重建")
        return torch.empty(0)
    if probs.dim() != 3:
        raise ValueError(f"probs维度应为3，实际为{probs.dim()}，无法进行压缩重建")
    if k >= probs.shape[-1]:
        return probs
    batch_size, seq_len, vocab_size = probs.shape
    flat_probs = probs.view(-1, vocab_size)
    top_k_values, top_k_indices = torch.topk(flat_probs, k, dim=-1, sorted=True)
    compressed_probs = torch.zeros_like(flat_probs)
    batch_indices = torch.arange(flat_probs.shape[0]).unsqueeze(1).expand(-1, k)
    compressed_probs[batch_indices, top_k_indices] = top_k_values
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
    rebuilt_flat_probs = torch.where(tail_mask, uniform_prob, compressed_probs)
    return rebuilt_flat_probs.view(batch_size, seq_len, vocab_size)


# Cases from the private codec oracle: every one needs k=V at theta=0 even
# though the ideal normalized tail is already zero at some k<V.
THETA_ZERO_FULL_V_CASES = [
    ("bf16_nonunit", [0.1, 0.2, 0.7], torch.bfloat16, 1),
    ("bf16_uniform", [1 / 3, 1 / 3, 1 / 3], torch.bfloat16, 1),
    ("fp32_uniform", [1 / 3, 1 / 3, 1 / 3], torch.float32, 1),
    ("fp32_peaked", [0.95, 0.04, 0.01], torch.float32, 1),
    ("fp32_tiny_tail", [1.0, 1e-30, 5e-31], torch.float32, 1),
]


# ---------------------------------------------------------------------------
# Codec extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("shape", [(1, 1, 7), (2, 3, 7)])
@pytest.mark.parametrize("k", [0, 1, 3, 7])
def test_codec_extraction_is_bitwise_identical(dtype, shape, k):
    torch.manual_seed(11)
    raw = torch.softmax(torch.randn(*shape), dim=-1).to(dtype)
    raw[:, :, -1] = 0  # force zero entries so the explicit top-k mask matters
    before = raw.clone()
    expected = _legacy_codec(raw, k)
    actual = compress_rebuild_probs(raw, k)
    if k >= shape[-1]:
        assert actual is raw
        assert expected is raw
    else:
        assert torch.equal(actual, expected)
        assert actual.dtype == raw.dtype
    assert torch.equal(raw, before)  # compare against the pre-call snapshot


def test_codec_keeps_original_k_ge_v_object_and_dtype():
    raw = torch.tensor([[[0.5, 0.3, 0.2]]], dtype=torch.bfloat16)
    assert CUHLM.compress_rebuild_probs(raw, 3) is raw
    assert CUHLM.compress_rebuild_probs(raw, 9) is raw
    assert compress_rebuild_probs(raw, 9) is raw


def test_codec_rejects_non_3d_input():
    with pytest.raises(ValueError, match="维度"):
        compress_rebuild_probs(torch.tensor([0.5, 0.5]), 1)


# ---------------------------------------------------------------------------
# Stable L1 helper
# ---------------------------------------------------------------------------


def test_stable_l1_keeps_the_dominant_coordinate_shift():
    raw = torch.tensor([1.0, 1e-30, 5e-31], dtype=torch.float32)
    decoded = compress_rebuild_probs(raw.reshape(1, 1, -1), 1)
    exact = _exact_l1(raw, decoded)
    stable = _actual_l1(raw, 1)
    # Naive double normalization drops the 1.5e-30 change of the dominant
    # coordinate and reports exactly half of the true distance.
    p = sampling_weights(raw.reshape(1, -1)).double()
    q = sampling_weights(decoded.reshape(1, -1)).double()
    naive = float(
        (p / p.sum(-1, keepdim=True) - q / q.sum(-1, keepdim=True)).abs().sum()
    )
    assert exact > 2.9e-30
    assert stable == pytest.approx(exact, rel=1e-12, abs=0.0)
    assert naive == pytest.approx(exact / 2, rel=1e-9, abs=0.0)


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_stable_l1_matches_fraction_oracle_on_small_rows(dtype):
    torch.manual_seed(7)
    probs = torch.softmax(torch.randn(5), dim=-1).to(dtype)
    for k in (1, 2, 4):
        decoded = compress_rebuild_probs(probs.reshape(1, 1, -1), k)
        exact = _exact_l1(probs, decoded)
        stable = _actual_l1(probs, k)
        assert stable == pytest.approx(exact, rel=1e-11, abs=1e-18)


# ---------------------------------------------------------------------------
# k selection contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,values,dtype,token", THETA_ZERO_FULL_V_CASES)
def test_theta_zero_requires_full_vocab_when_real_codec_rounds(
    name, values, dtype, token
):
    raw = torch.tensor(values, dtype=dtype)
    vocab = raw.numel()
    # Independent exhaustive oracle: every lossy k actually distorts the law.
    for k in range(1, vocab):
        assert _exact_proxy(raw, k, 0.9, token) > 0.0, (name, k)
    selected = select_compressed_vocab_size(
        raw, uncertainty=0.9, theta=0.0, draft_token=token
    )
    assert selected == vocab, name
    assert 1 <= selected <= vocab


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_theta_zero_can_still_certify_an_exactly_lossless_codec(dtype):
    raw = torch.tensor([0.5, 0.25, 0.25], dtype=dtype)
    assert _exact_proxy(raw, 1, 0.9, 0) == 0.0
    assert (
        select_compressed_vocab_size(
            raw, uncertainty=0.9, theta=0.0, draft_token=0
        )
        == 1
    )


def test_single_token_vocabulary_returns_one():
    raw = torch.tensor([1.0], dtype=torch.float32).reshape(1, 1, -1)
    assert (
        select_compressed_vocab_size(raw, uncertainty=0.9, theta=0.0)
        == 1
    )
    assert compress_rebuild_probs(raw, 1) is raw


def test_plain_fallback_is_v_not_a_fixed_constant():
    # Old code returned min(300, V // 100) on a shape mismatch and 0 when no
    # ideal k satisfied; both are replaced by explicit validation / V.
    raw = torch.full((32000,), 1 / 32000)
    selected = select_compressed_vocab_size(raw, uncertainty=0.9, theta=0.0)
    assert selected == 32000
    with pytest.raises(ValueError):
        select_compressed_vocab_size(raw, uncertainty=0.9, theta=0.0, vocab_size=32001)


def test_returned_k_satisfies_the_actual_proxy_in_ordinary_regime():
    torch.manual_seed(13)
    for dtype in (torch.float32, torch.bfloat16):
        for uncertainty, theta in ((0.3, 0.1), (0.9, 1e-6), (0.9, 0.01)):
            raw = torch.softmax(torch.randn(37), dim=-1).to(dtype)
            token = 3
            k = select_compressed_vocab_size(
                raw, uncertainty=uncertainty, theta=theta, draft_token=token
            )
            assert 1 <= k <= raw.numel()
            if k < raw.numel():
                proxy = _exact_proxy(raw, k, uncertainty, token)
                # Slack is float64 comparison noise only; the production check
                # itself is a direct <= theta with no added budget.
                assert proxy <= theta * (1 + 1e-12) + 1e-18


def test_ideal_candidate_is_verified_by_the_real_codec_then_full_v():
    raw = torch.tensor([1.0, 1e-30, 5e-31], dtype=torch.float32)
    # The ideal screen loses this tail in its prefix sums and proposes k=1.
    # Exact certification rejects it; a naive normalized subtraction would
    # underestimate L1 by half and wrongly pass the original 7e-30 budget.
    assert _exact_proxy(raw, 1, 0.9, 1) > 7e-30
    assert (
        select_compressed_vocab_size(raw, uncertainty=0.9, theta=7e-30, draft_token=1)
        == 3
    )


def test_selected_non_top1_token_is_used():
    raw = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    assert (
        select_compressed_vocab_size(raw, uncertainty=0.5, theta=0.25)
        == 1
    )
    assert (
        select_compressed_vocab_size(
            raw, uncertainty=0.5, theta=0.25, draft_token=1
        )
        == 2
    )


def test_selected_zero_probability_is_valid_for_the_proxy():
    raw = torch.tensor([0.6, 0.4, 0.0], dtype=torch.float32)
    k = select_compressed_vocab_size(
        raw, uncertainty=0.5, theta=0.5, draft_token=2
    )
    assert 1 <= k <= 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"theta": -1e-30},
        {"theta": float("nan")},
        {"theta": float("inf")},
        {"uncertainty": -0.01},
        {"uncertainty": 1.01},
        {"uncertainty": float("nan")},
        {"draft_token": -1},
        {"draft_token": 3},
        {"draft_token": 1.5},
    ],
)
def test_invalid_scalar_parameters_raise(kwargs):
    raw = torch.tensor([0.5, 0.3, 0.2], dtype=torch.float32)
    params = {"uncertainty": 0.5, "theta": 0.1}
    params.update(kwargs)
    with pytest.raises(ValueError):
        select_compressed_vocab_size(raw, **params)


@pytest.mark.parametrize(
    "raw",
    [
        torch.tensor([0.5, 0.3]),
        torch.tensor([[0.5, 0.3, 0.2], [0.5, 0.3, 0.2]]),
        torch.tensor([-0.1, 0.6, 0.5]),
        torch.tensor([float("nan"), 0.5, 0.5]),
        torch.tensor([2.0, -0.5, -0.5]),
        torch.tensor([1, 2, 3]),
        torch.tensor([], dtype=torch.float32),
    ],
)
def test_invalid_rows_raise_instead_of_falling_back(raw):
    with pytest.raises(ValueError):
        select_compressed_vocab_size(raw, uncertainty=0.9, theta=0.1)
    with pytest.raises(ValueError):
        CUHLM(20.0, vocab_size=3)._calculate_compressed_vocab_size(0.9, raw)


def test_none_and_empty_are_explicit_errors_in_the_pure_solver():
    with pytest.raises(ValueError):
        select_compressed_vocab_size(None, uncertainty=0.9, theta=0.1)
    with pytest.raises(ValueError):
        select_compressed_vocab_size(
            torch.empty(0), uncertainty=0.9, theta=0.1
        )


def test_shape_mismatch_and_raw_logits_raise_through_the_adapter():
    sim = CUHLM(20.0, vocab_size=3)
    with pytest.raises(ValueError):
        sim._calculate_compressed_vocab_size(0.9, torch.full((4,), 0.25))
    with pytest.raises(ValueError):
        sim._calculate_compressed_vocab_size(0.9, torch.randn(3) * 5)


def test_determine_transfer_strategy_keeps_skip_for_empty_input():
    sim = CUHLM(20.0, vocab_size=3, uncertainty_threshold=0.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert sim.determine_transfer_strategy(0.9, None) == (False, 0)
        assert sim.determine_transfer_strategy(0.9, torch.empty(0)) == (False, 0)


def test_solver_does_not_mutate_input_or_consume_rng():
    raw = torch.tensor([0.95, 0.04, 0.01], dtype=torch.float32)
    before = raw.clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1234)
        state = torch.random.get_rng_state().clone()
        select_compressed_vocab_size(raw, uncertainty=0.9, theta=0.05, draft_token=1)
        assert torch.equal(torch.random.get_rng_state(), state)
    assert torch.equal(raw, before)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("theta", [0.05, 0.1, 0.2])
def test_low_precision_large_vocab_does_not_degenerate_to_full_v(dtype, theta):
    generator = torch.Generator().manual_seed(0)
    raw = torch.softmax(torch.randn(32000, generator=generator), -1).to(dtype)
    # A real-codec witness establishes that a compressed answer is feasible.
    cumulative = raw.double().sort(descending=True).values.cumsum(0)
    witness = int(torch.searchsorted(cumulative, 0.99).item()) + 1
    assert witness < raw.numel()
    assert _exact_proxy(raw, witness, 0.9, 1) < theta

    k = select_compressed_vocab_size(
        raw, uncertainty=0.9, theta=theta, draft_token=1
    )
    assert 1 <= k < raw.numel()
    assert _exact_proxy(raw, k, 0.9, 1) <= theta * (1 + 1e-12)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_large_vocab_rebuilds_at_most_two_distinct_candidates(monkeypatch, dtype):
    import src.cuhlm_compression as module

    calls = []
    original = module.compress_rebuild_probs

    def counting(probs, k):
        calls.append(k)
        return original(probs, k)

    monkeypatch.setattr(module, "compress_rebuild_probs", counting)
    generator = torch.Generator().manual_seed(11)
    raw = torch.softmax(torch.randn(4096, generator=generator), dim=-1).to(dtype)
    k = select_compressed_vocab_size(
        raw, uncertainty=0.9, theta=0.1, draft_token=5
    )
    assert 1 <= k <= 4096
    assert len(calls) <= 2
    assert len(calls) == len(set(calls))
    if k < 4096:
        assert calls[-1] == k


def test_terminal_prob_uses_the_main_codec_semantics(monkeypatch):
    sim = CUHLM(20.0, vocab_size=4, uncertainty_threshold=0.0)
    probs = torch.tensor([0.5, 0.3, 0.0, 0.0], dtype=torch.float32)
    seen = {}

    def determine(uncertainty, current_probs, draft_token=None):
        seen["draft_token"] = draft_token
        return True, 3

    monkeypatch.setattr(sim, "calculate_uncertainty", lambda *a, **k: 0.9)
    monkeypatch.setattr(sim, "determine_transfer_strategy", determine)
    rebuilt = sim.terminal_prob(probs, logits=torch.zeros(4))
    # Main codec uses the explicit top-k mask: the zero inside top-3 stays 0
    # and the whole residual goes to the single true tail slot.
    expected = compress_rebuild_probs(probs.reshape(1, 1, -1), 3).reshape(-1)
    assert seen["draft_token"] is None  # legacy top-1 compatibility preserved
    assert torch.equal(rebuilt, expected)
    # Whichever zero the top-k tie kept, the explicit mask leaves one tail slot
    # at exactly 0 and puts the whole residual in the other; the old two-step
    # >0 mask split it evenly and also promoted the kept zero into the tail.
    assert rebuilt[2:].min().item() == 0.0
    assert rebuilt[2:].max().item() == pytest.approx(0.2)
    legacy = sim.rebuild_full_probs(sim._apply_top_k_compression(probs, 3))
    assert not torch.equal(rebuilt, legacy)


def test_adapter_still_accepts_plain_and_leading_singleton_rows():
    sim = CUHLM(20.0, vocab_size=3)
    raw = torch.tensor([0.5, 0.25, 0.25], dtype=torch.float32)
    for shaped in (raw, raw.reshape(1, -1), raw.reshape(1, 1, -1)):
        assert (
            sim._calculate_compressed_vocab_size(0.9, shaped, theta=0.0) == 1
        )
