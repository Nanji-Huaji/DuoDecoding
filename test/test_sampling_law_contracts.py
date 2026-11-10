"""Actual sampler weights, low-precision roundoff, and stable rejection laws."""

from fractions import Fraction

import pytest
import torch

from src.sampling import sample
from src.verification import (
    ProposalWindow,
    decide_verification,
    prepare_verification_window,
)


def _decision(p, q, token, draw=0.0):
    proposal = ProposalWindow(torch.tensor([[0, token]]), q.reshape(1, 1, -1), 1)
    history = torch.stack((p, p)).unsqueeze(0)
    return decide_verification(
        prepare_verification_window(proposal, history),
        torch.tensor([[draw]], dtype=torch.float64),
    )


def test_bfloat16_rows_follow_the_law_used_by_the_sampler():
    p = torch.tensor([0.333984375, 0.333984375, 0.330078125], dtype=torch.bfloat16)
    q = torch.tensor([0.333984375, 0.333984375, 0.33203125], dtype=torch.bfloat16)
    assert (p <= q).all() and p.float().sum() != q.float().sum()
    decision = _decision(p, q, token=2, draw=0.999)
    assert not decision.all_accepted
    torch.testing.assert_close(
        decision.continuation_probs.double(),
        torch.tensor([[0.5, 0.5, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


@pytest.mark.parametrize("tail", [1e-8, 1e-20, 1e-38, 1e-44])
def test_tiny_positive_tail_does_not_become_a_false_zero_residual(tail):
    p, q = torch.tensor([0.0, 1.0]), torch.tensor([tail, 1.0])
    assert q[0] > 0 and q.sum() == 1  # ordinary normalization loses the tiny mass
    decision = _decision(p, q, token=0)
    assert not decision.all_accepted
    torch.testing.assert_close(
        decision.continuation_probs.double(),
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("tail", [1e-8, 1e-20, 1e-38, 1e-44])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("tail_first", [False, True])
def test_global_weight_scale_does_not_erase_a_tiny_residual(tail, split, tail_first):
    p = torch.tensor([0.3333333432674408] * 3 + [0.0])
    q = torch.tensor([0.333333283662796] * 3 + [tail])
    expected = torch.tensor([[1 / 3, 1 / 3, 1 / 3, 0]], dtype=torch.float64)
    token = 3
    if tail_first:
        p, q, expected = p.roll(1), q.roll(1), expected.roll(1, dims=-1)
        token = 0
    proposal = ProposalWindow(torch.tensor([[0, token]]), q.reshape(1, 1, -1), 1)
    payload = proposal.selected_probabilities().float() if split else None
    window = prepare_verification_window(
        proposal, torch.stack((p, p)).unsqueeze(0), selected_q=payload,
    )
    decision = decide_verification(window, torch.zeros(1, 1))
    assert not decision.all_accepted
    torch.testing.assert_close(
        decision.continuation_probs, expected, rtol=0, atol=1e-12,
    )


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("draw_dtype", [torch.float32, torch.float64])
def test_proportional_sampler_laws_cannot_split_at_a_scalar_rounding_midpoint(
    split, draw_dtype,
):
    exponents = torch.tensor([
        0, 3, 4, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19,
        26, 28, 30, 31, 33, 40, 41, 44, 45, 46, 47, 48,
    ])
    base = torch.tensor(0.822862982749939)
    above = torch.nextafter(base, torch.tensor(float("inf")))
    p = torch.ldexp(base.expand_as(exponents), -exponents)
    q = torch.ldexp(above.expand_as(exponents), -exponents)
    proposal = ProposalWindow(torch.tensor([[0, 0]]), q.reshape(1, 1, -1), 1)
    wire = proposal.selected_probabilities().float() if split else None
    window = prepare_verification_window(
        proposal, torch.stack((p, p)).unsqueeze(0), selected_q=wire,
    )
    # Both actual sampler rows are one scalar times the exact same powers of
    # two. Their categorical laws are identical, independent of that scale.
    wp, wq = window.target_weights[0, 0], proposal.sampling_weights[0, 0]
    wp_exact = [Fraction.from_float(float(value)) for value in wp]
    wq_exact = [Fraction.from_float(float(value)) for value in wq]
    assert [x / sum(wp_exact) for x in wp_exact] == [
        x / sum(wq_exact) for x in wq_exact
    ]
    one = torch.ones(1, 1, dtype=draw_dtype)
    draw = torch.nextafter(one, torch.zeros_like(one))
    assert decide_verification(window, draw).all_accepted
    assert torch.equal(window.selected_p, window.selected_q)


def test_bonus_is_preprocessed_once_like_target_autoregression(monkeypatch):
    raw = torch.tensor(
        [
            0.208984375,
            0.26171875,
            0.050537109375,
            0.0267333984375,
            0.0164794921875,
            0.224609375,
            0.2119140625,
        ],
        dtype=torch.bfloat16,
    )
    proposal = ProposalWindow(torch.tensor([[0]]), torch.empty(1, 0, 7), 1)
    window = prepare_verification_window(proposal, raw.reshape(1, 1, -1))
    decision = decide_verification(window, torch.empty(1, 0))
    seen = []

    def capture(weights, num_samples):
        seen.append(weights.clone())
        return torch.zeros(1, num_samples, dtype=torch.long)

    monkeypatch.setattr(torch, "multinomial", capture)
    sample(raw[None])
    sample(decision.continuation_probs)
    assert decision.continuation_probs.dtype == raw.dtype
    assert torch.equal(seen[0], seen[1])
    # This fixture would detect a naive prepared-weights -> sample second pass.
    sample(seen[0])
    assert not torch.equal(seen[0], seen[2])


def test_split_verifier_consumes_the_actual_fp32_selected_q_payload():
    q = torch.tensor([0.2, 0.3, 0.5])
    p = torch.tensor([0.1, 0.4, 0.5])
    proposal = ProposalWindow(torch.tensor([[0, 0]]), q.reshape(1, 1, -1), 1)
    history = torch.stack((p, p)).unsqueeze(0)
    full = prepare_verification_window(proposal, history)
    wire_q = proposal.selected_probabilities().float()
    split = prepare_verification_window(proposal, history, selected_q=wire_q)
    # Float64-normalized P/Q is just above .5; the declared FP32 scalar rule
    # rounds both selected scalars and has ratio exactly .5. This distinguishes
    # consuming the transmitted value from silently consulting the full q row.
    assert decide_verification(full, torch.tensor([[0.5]])).all_accepted
    assert not decide_verification(split, torch.tensor([[0.5]])).all_accepted
    assert torch.equal(split.selected_q, wire_q)
    assert split.selected_p.dtype == split.selected_q.dtype == torch.float32


@pytest.mark.parametrize(
    "payload",
    [
        torch.tensor([[0.0]]),
        torch.tensor([[float("nan")]]),
        torch.tensor([[1.1]]),
        torch.tensor([[0.5]], dtype=torch.bfloat16),
        torch.tensor([0.5]),
    ],
)
def test_invalid_selected_q_payload_fails_before_rng(payload, monkeypatch):
    proposal = ProposalWindow(torch.tensor([[0, 0]]), torch.full((1, 1, 2), 0.5), 1)
    monkeypatch.setattr(torch, "rand", lambda *args, **kw: pytest.fail("consumed RNG"))
    with pytest.raises(ValueError, match="selected q payload"):
        prepare_verification_window(
            proposal, torch.full((1, 2, 2), 0.5), selected_q=payload
        )


def test_selected_support_is_validated_after_real_sampler_dtype_conversion():
    with pytest.raises(ValueError, match="selected support"):
        ProposalWindow(
            torch.tensor([[0, 0]]),
            torch.tensor([[[1e-300, 1.0]]], dtype=torch.float64),
            1,
        )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_sampler_refactor_preserves_preprocessing_and_rng(dtype, monkeypatch):
    raw = torch.tensor([[0.25, 0.75], [float("nan"), -1.0]], dtype=dtype)
    expected = torch.tensor([[0.25, 0.75], [1.0, 0.0]])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(137)
        expected_tokens = torch.multinomial(expected, 1)
        expected_rng = torch.random.get_rng_state().clone()
        torch.manual_seed(137)
        original = torch.multinomial

        def inspect_weights(weights, num_samples):
            torch.testing.assert_close(weights, expected, rtol=0, atol=0)
            return original(weights, num_samples)

        monkeypatch.setattr(torch, "multinomial", inspect_weights)
        assert torch.equal(sample(raw), expected_tokens)
        assert torch.equal(torch.random.get_rng_state(), expected_rng)
