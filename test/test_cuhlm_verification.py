"""CU-HLM single-token verification kernel: laws, boundaries and fail-closed cases.

These tests exercise the independent ``cuhlm_verification`` interface only.  It
never samples, never draws a uniform, and never mutates its inputs: the caller
validates a proposal, a reconstructed law and one target row, then asks for a
strict ``<`` decision.  The accepted branch keeps the original proposal token
(no target bonus); the rejected branch returns one normalized ``(P - qhat)+``
residual law.

Two independent counterexamples drive this file:

* a BF16 row and the float32 weights the sampler actually used describe the
  same law, so a raw-q comparison can spuriously reject;
* compressing ``q=[.95,.04,.01]`` with ``k=1`` yields ``qhat=[.95,.025,.025]``.
  If the target law equals ``qhat``, a draft token 1 can still be rejected, but
  ``(P - qhat)+`` is exactly zero.  That is a real degenerate case of the lossy
  paper protocol; this kernel fails closed instead of substituting an epsilon,
  a uniform tail, argmax, or the original ``q``.
"""

import pytest
import torch

from src.communication import CUHLM
from src.cuhlm_verification import (
    CuhlmVerificationWindow,
    decide_cuhlm_verification,
    prepare_cuhlm_verification,
)
from src.sampling import sampling_weights
from src.verification import (
    ProposalWindow,
    decide_verification,
    prepare_verification_window,
)


def _proposal(raw_q, token, prefix_len=1):
    q = raw_q.reshape(1, 1, -1)
    tokens = torch.cat(
        (
            torch.zeros(1, prefix_len, dtype=torch.long),
            torch.tensor([[token]], dtype=torch.long),
        ),
        dim=1,
    )
    return ProposalWindow(tokens, q, prefix_len)


def _target_row(p_row):
    return p_row.reshape(1, 1, -1)


def _window(proposal, p_row, qhat_row):
    return prepare_cuhlm_verification(
        proposal, _target_row(p_row), qhat_row.reshape(1, 1, -1)
    )


def _draw(value):
    return torch.tensor([[value]], dtype=torch.float64)


# ---------------------------------------------------------------------------
# Counterexample 1: the law the sampler used, not the raw low-precision row.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float16, torch.float32]
)
def test_sampler_law_roundoff_is_not_rejected_for_a_matching_target(dtype):
    raw_q = torch.tensor([0.1, 0.3, 0.6], dtype=dtype)
    p_row = sampling_weights(raw_q.reshape(1, 1, -1))
    window = _window(_proposal(raw_q, token=2), p_row, raw_q)

    ratio = float(window.selected_p[0, 0] / window.selected_q[0, 0])
    # The old raw-q ratio (0.300048... / 0.30078125) was below .999 and rejected.
    assert ratio > 0.999
    decision = decide_cuhlm_verification(window, _draw(0.999))
    assert decision.accepted
    assert decision.residual_probs is None


def test_bf16_row_is_compared_as_the_float32_sampler_law():
    raw_q = torch.tensor([0.1, 0.3, 0.6], dtype=torch.bfloat16)
    assert float(raw_q.float().sum()) != 1.0
    p_row = sampling_weights(raw_q.reshape(1, 1, -1))
    window = _window(_proposal(raw_q, token=2), p_row, raw_q)
    assert torch.equal(window.target_weights, window.proposal.sampling_weights)


# ---------------------------------------------------------------------------
# Counterexample 2: a genuine zero-residual compression degeneracy.
# ---------------------------------------------------------------------------
def test_zero_reconstructed_residual_fails_closed_instead_of_argmax():
    raw_q = torch.tensor([0.95, 0.04, 0.01])
    qhat = CUHLM.compress_rebuild_probs(raw_q.reshape(1, 1, 3), 1)
    assert torch.allclose(
        qhat, torch.tensor([[[0.95, 0.025, 0.025]]]), rtol=0, atol=1e-7
    )
    window = _window(_proposal(raw_q, token=1), qhat, qhat)

    ratio = float(window.selected_p[0, 0] / window.selected_q[0, 0])
    assert ratio == pytest.approx(0.025 / 0.04, abs=1e-6)
    # Rejection is genuinely reachable in [0, 1) ...
    assert ratio < 0.9
    # ... but (P - qhat)+ has no mass, so no residual law exists.
    assert torch.equal(window.target_weights, window.reconstructed_weights)
    with pytest.raises(ValueError, match="positive finite residual mass"):
        decide_cuhlm_verification(window, _draw(0.9))
    # Failing closed on reject does not wrongly reject below the boundary.
    assert decide_cuhlm_verification(window, _draw(0.1)).accepted


# ---------------------------------------------------------------------------
# Full-vocabulary reconstruction degenerates to the strict SD residual.
# ---------------------------------------------------------------------------
def test_full_vocabulary_qhat_degenerates_to_the_strict_sd_residual():
    raw_q = torch.tensor([0.5, 0.3, 0.2])
    p_row = torch.tensor([0.1, 0.5, 0.4])
    cu = _window(_proposal(raw_q, token=0), p_row, raw_q)
    decision = decide_cuhlm_verification(cu, _draw(0.5))
    assert not decision.accepted
    # The inputs are float32 rows, so the represented law is 1e-8 off the
    # decimal literals; exact equality with the strict verifier is asserted
    # immediately below on the same represented rows.
    torch.testing.assert_close(
        decision.residual_probs,
        torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float64),
        rtol=0,
        atol=1e-7,
    )

    strict = decide_verification(
        prepare_verification_window(
            ProposalWindow(torch.tensor([[0, 0]]), raw_q.reshape(1, 1, 3), 1),
            torch.stack((p_row, p_row)).unsqueeze(0),
        ),
        _draw(0.5),
    )
    torch.testing.assert_close(
        decision.residual_probs, strict.continuation_probs, rtol=0, atol=0
    )


def test_lossy_qhat_keeps_its_own_rejection_law_not_the_strict_one():
    raw_q = torch.tensor([0.5, 0.2, 0.3])
    qhat = torch.tensor([0.1, 0.2, 0.7])
    p_row = torch.tensor([0.3, 0.0, 0.7])
    decision = decide_cuhlm_verification(
        _window(_proposal(raw_q, token=0), p_row, qhat), _draw(0.7)
    )
    assert not decision.accepted
    # (P - qhat)+ = [.2, 0, 0]; the strict (P - q)+ would be [0, 0, .4].
    torch.testing.assert_close(
        decision.residual_probs,
        torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    strict = decide_verification(
        prepare_verification_window(
            ProposalWindow(torch.tensor([[0, 0]]), raw_q.reshape(1, 1, 3), 1),
            torch.stack((p_row, p_row)).unsqueeze(0),
        ),
        _draw(0.7),
    )
    torch.testing.assert_close(
        strict.continuation_probs,
        torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    assert not torch.equal(decision.residual_probs, strict.continuation_probs)


# ---------------------------------------------------------------------------
# Acceptance boundaries and degenerate supports.
# ---------------------------------------------------------------------------
def test_acceptance_is_strict_and_equal_ratio_rejects():
    proposal = _proposal(torch.tensor([0.5, 0.5]), token=0)
    p_row = torch.tensor([0.25, 0.75])
    qhat = torch.tensor([0.5, 0.5])
    window = _window(proposal, p_row, qhat)

    assert decide_cuhlm_verification(window, _draw(0.499)).accepted
    rejected = decide_cuhlm_verification(window, _draw(0.5))
    assert not rejected.accepted
    torch.testing.assert_close(
        rejected.residual_probs,
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_zero_target_support_rejects_even_at_zero_uniform():
    window = _window(
        _proposal(torch.tensor([0.5, 0.5]), token=0),
        torch.tensor([0.0, 1.0]),
        torch.tensor([0.5, 0.5]),
    )
    decision = decide_cuhlm_verification(window, _draw(0.0))
    assert not decision.accepted
    torch.testing.assert_close(
        decision.residual_probs,
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_accept_allows_a_zero_residual_when_ratio_is_one():
    law = torch.tensor([0.4, 0.35, 0.25])
    window = _window(_proposal(law, token=0), law, law)
    decision = decide_cuhlm_verification(window, _draw(0.999999))
    assert decision.accepted
    assert decision.residual_probs is None


@pytest.mark.parametrize("tail", [1e-8, 1e-20, 1e-38, 1e-44])
def test_tiny_positive_reconstruction_tail_keeps_its_residual(tail):
    raw_q = torch.tensor([tail, 1.0])
    p_row = torch.tensor([0.0, 1.0])
    decision = decide_cuhlm_verification(
        _window(_proposal(raw_q, token=0), p_row, raw_q), _draw(0.0)
    )
    assert not decision.accepted
    torch.testing.assert_close(
        decision.residual_probs,
        torch.tensor([[0.0, 1.0]], dtype=torch.float64),
        rtol=0,
        atol=0,
    )


def test_proportional_sampler_laws_accept_at_the_uniform_midpoint():
    exponents = torch.tensor(
        [
            0, 3, 4, 6, 7, 8, 12, 13, 15, 16, 17, 18, 19,
            26, 28, 30, 31, 33, 40, 41, 44, 45, 46, 47, 48,
        ]
    )
    base = torch.tensor(0.822862982749939)
    above = torch.nextafter(base, torch.tensor(float("inf")))
    p_row = torch.ldexp(base.expand_as(exponents), -exponents)
    raw_q = torch.ldexp(above.expand_as(exponents), -exponents)
    window = _window(_proposal(raw_q, token=24), p_row, raw_q)

    assert torch.equal(window.selected_p, window.selected_q)
    one = torch.ones(1, 1, dtype=torch.float64)
    draw = torch.nextafter(one, torch.zeros_like(one))
    assert decide_cuhlm_verification(window, draw).accepted


def test_only_the_forwarded_prefix_row_is_the_target_law():
    raw_q = torch.tensor([0.2, 0.3, 0.5])
    history = torch.tensor(
        [
            [
                [0.5, 0.25, 0.25],
                [0.1, 0.1, 0.8],
                [0.25, 0.25, 0.5],
            ]
        ]
    )
    proposal = _proposal(raw_q, token=0, prefix_len=3)
    window = prepare_cuhlm_verification(proposal, history, raw_q.reshape(1, 1, 3))
    torch.testing.assert_close(window.target_row, history[:, 2:3])
    # target token 0 has p=.25, q=.2 -> ratio 1.25; the draw decides.
    assert decide_cuhlm_verification(window, _draw(0.9)).accepted


# ---------------------------------------------------------------------------
# Invalid inputs fail before the caller consumes acceptance RNG.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "history",
    [
        torch.zeros(1, 0, 3),
        torch.full((1, 2, 3), 1 / 3),
        torch.zeros(2, 1, 3),
        torch.zeros(1, 3),
    ],
)
def test_target_history_must_cover_exactly_the_forwarded_prefix(history):
    proposal = _proposal(torch.tensor([0.2, 0.3, 0.5]), token=0)
    with pytest.raises(ValueError, match="exactly the forwarded prefix"):
        prepare_cuhlm_verification(
            proposal, history, torch.tensor([[[0.2, 0.3, 0.5]]])
        )


def test_extra_target_history_row_is_not_silently_truncated():
    proposal = _proposal(torch.tensor([0.2, 0.3, 0.5]), token=0)
    with pytest.raises(ValueError, match="exactly the forwarded prefix"):
        prepare_cuhlm_verification(
            proposal,
            torch.full((1, 2, 3), 1 / 3),
            torch.tensor([[[0.2, 0.3, 0.5]]]),
        )


@pytest.mark.parametrize(
    "qhat",
    [
        torch.tensor([0.2, 0.3, 0.5]),
        torch.tensor([[[0.5, 0.5]]]),
        torch.tensor([[[0.2, 0.3, 0.5], [0.2, 0.3, 0.5]]]),
        torch.tensor([[[float("nan"), 0.5, 0.5]]]),
        torch.tensor([[[-0.1, 0.6, 0.5]]]),
        torch.tensor([[[0.2, 0.3, 0.4]]]),
        torch.tensor([[[float("inf"), 0.0, 0.0]]]),
        torch.zeros(1, 1, 3, dtype=torch.long),
    ],
)
def test_invalid_reconstructed_law_fails_explicitly(qhat):
    proposal = _proposal(torch.tensor([0.2, 0.3, 0.5]), token=0)
    with pytest.raises(ValueError, match="reconstructed"):
        prepare_cuhlm_verification(
            proposal, torch.tensor([[[0.2, 0.3, 0.5]]]), qhat
        )


@pytest.mark.parametrize(
    "uniform",
    [
        torch.tensor([[1.0]]),
        torch.tensor([[-0.1]]),
        torch.tensor([[float("nan")]]),
        torch.tensor([[0.5, 0.5]]),
        0.5,
    ],
)
def test_invalid_uniform_fails_explicitly(uniform):
    window = _window(
        _proposal(torch.tensor([0.2, 0.3, 0.5]), token=0),
        torch.tensor([0.2, 0.3, 0.5]),
        torch.tensor([0.2, 0.3, 0.5]),
    )
    with pytest.raises(ValueError, match="uniform"):
        decide_cuhlm_verification(window, uniform)


def test_proposal_with_multiple_candidates_is_not_accepted():
    proposal = ProposalWindow(
        torch.tensor([[0, 0, 1]]),
        torch.full((1, 2, 3), 1 / 3),
        1,
    )
    with pytest.raises(ValueError, match="exactly one candidate"):
        prepare_cuhlm_verification(
            proposal,
            torch.tensor([[[0.2, 0.3, 0.5]]]),
            torch.tensor([[[0.2, 0.3, 0.5]]]),
        )


def test_invalid_reconstruction_fails_before_any_uniform_is_drawn(monkeypatch):
    proposal = _proposal(torch.tensor([0.2, 0.3, 0.5]), token=0)
    monkeypatch.setattr(
        torch, "rand", lambda *a, **k: pytest.fail("consumed acceptance RNG")
    )
    with pytest.raises(ValueError, match="reconstructed"):
        prepare_cuhlm_verification(
            proposal,
            torch.tensor([[[0.2, 0.3, 0.5]]]),
            torch.tensor([[[0.2, 0.3, 0.4]]]),
        )


def test_kernel_draws_no_randomness_and_does_not_mutate_its_inputs(monkeypatch):
    raw_q = torch.tensor([0.2, 0.3, 0.5])
    p_row = torch.tensor([0.1, 0.6, 0.3])
    qhat = torch.tensor([0.3, 0.3, 0.4])
    proposal = _proposal(raw_q, token=0)
    copies = [raw_q.clone(), p_row.clone(), qhat.clone()]
    rng = torch.random.get_rng_state().clone()

    monkeypatch.setattr(
        torch, "rand", lambda *a, **k: pytest.fail("implicit acceptance RNG")
    )
    monkeypatch.setattr(
        torch, "multinomial", lambda *a, **k: pytest.fail("implicit token sample")
    )
    window = prepare_cuhlm_verification(
        proposal, p_row.reshape(1, 1, -1), qhat.reshape(1, 1, -1)
    )
    assert isinstance(window, CuhlmVerificationWindow)
    decision = decide_cuhlm_verification(window, _draw(0.9))
    assert not decision.accepted
    for original, copy in zip((raw_q, p_row, qhat), copies, strict=True):
        assert torch.equal(original, copy)
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_real_large_vocab_softmax_survives_all_verification_roles(dtype):
    with torch.random.fork_rng():
        torch.manual_seed(0)
        if dtype == torch.float32:
            logits = torch.randn(151936) * 3
        else:
            logits = torch.full((151936,), -11.9921875, dtype=dtype)
            logits[0] = 0
        raw = torch.softmax(logits, dim=-1)
    token = int(raw.argmax())
    proposal = _proposal(raw, token)
    decision = decide_cuhlm_verification(_window(proposal, raw, raw), _draw(.999))
    assert decision.accepted and decision.residual_probs is None
    # The shared strict verifier must admit the same real model row, too.
    history = raw.reshape(1, 1, -1).expand(1, 2, -1)
    strict = prepare_verification_window(proposal, history)
    assert decide_verification(strict, _draw(.999)).all_accepted


@pytest.mark.parametrize("mass", [0., .5, 1.25])
def test_large_vocab_roundoff_budget_still_rejects_invalid_mass(mass):
    raw = torch.full((151936,), mass / 151936)
    with pytest.raises(ValueError, match="sum to one"):
        _proposal(raw, token=0)


def test_small_vocab_mass_check_is_not_globally_relaxed():
    with pytest.raises(ValueError, match="sum to one"):
        _proposal(torch.tensor([.5, .501]), token=0)
