"""Independent toy laws for strict round values, not old-output snapshots."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import pytest
import torch

from src.verification import (
    ProposalWindow,
    decide_verification,
    prepare_verification_window,
)


def _proposal(tokens=(0, 0), q=((0.5, 0.5),), prefix_len=1):
    return ProposalWindow(
        torch.tensor([tokens], dtype=torch.long),
        torch.tensor([q], dtype=torch.float64),
        prefix_len,
    )


def _window(proposal=None, rows=((0.25, 0.75), (0.9, 0.1))):
    return prepare_verification_window(
        proposal or _proposal(), torch.tensor([rows], dtype=torch.float64)
    )


def test_zero_target_probability_rejects_even_at_zero_uniform():
    window = _window(rows=((0.0, 1.0), (0.9, 0.1)))
    decision = decide_verification(window, torch.zeros(1, 1))
    assert decision.accepted_count == 0
    assert decision.accepted_end == 0
    assert decision.retained_context_end == 1
    assert not decision.all_accepted
    torch.testing.assert_close(
        decision.continuation_probs, torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    )


def test_acceptance_boundary_is_strict_and_bonus_has_explicit_row():
    window = _window()  # token 0 has p/q = 0.5, bonus differs from both p and q.
    accepted = decide_verification(window, torch.tensor([[0.499]], dtype=torch.float64))
    rejected = decide_verification(window, torch.tensor([[0.5]], dtype=torch.float64))
    assert accepted.accepted_count == 1 and accepted.all_accepted
    assert accepted.retained_context_end == 2  # The bonus is not forwarded.
    torch.testing.assert_close(
        accepted.continuation_probs, torch.tensor([[0.9, 0.1]], dtype=torch.float64)
    )
    assert rejected.accepted_count == 0
    torch.testing.assert_close(
        rejected.continuation_probs, torch.tensor([[0.0, 1.0]], dtype=torch.float64)
    )


def test_only_contiguous_accepted_prefix_counts():
    proposal = _proposal(
        tokens=(0, 0, 1, 0), q=((0.5, 0.5),) * 3,
    )
    window = _window(
        proposal, rows=((0.8, 0.2), (0.8, 0.2), (0.8, 0.2), (0.1, 0.9)),
    )
    # First succeeds, second fails, third would succeed but must not be committed.
    decision = decide_verification(window, torch.tensor([[0.9, 0.9, 0.0]]))
    assert decision.accepted_count == 1
    assert decision.retained_context_end == 2
    torch.testing.assert_close(
        decision.target_row, torch.tensor([[0.8, 0.2]], dtype=torch.float64)
    )
    torch.testing.assert_close(
        decision.continuation_probs, torch.tensor([[1.0, 0.0]], dtype=torch.float64)
    )


def test_zero_candidates_still_has_bonus_and_no_rng_or_token_sampling():
    proposal = ProposalWindow(
        torch.tensor([[0, 1, 0]]), torch.empty(1, 0, 2), prefix_len=3,
    )
    history = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.3, 0.7]]])
    rng = torch.random.get_rng_state().clone()
    with patch("torch.rand", side_effect=AssertionError("implicit RNG")), patch(
        "torch.multinomial", side_effect=AssertionError("token sampling")
    ):
        decision = decide_verification(
            prepare_verification_window(proposal, history), torch.empty(1, 0)
        )
    assert decision.accepted_count == 0 and decision.all_accepted
    assert decision.retained_context_end == 3
    torch.testing.assert_close(decision.continuation_probs, history[:, 2])
    assert torch.equal(rng, torch.random.get_rng_state())


@pytest.mark.parametrize("length", [1, 3])
def test_missing_or_stale_target_context_is_not_silently_truncated(length):
    proposal = _proposal()
    with pytest.raises(ValueError, match="exactly candidate and bonus"):
        prepare_verification_window(proposal, torch.full((1, length, 2), 0.5))


def test_prefix_offset_not_last_arbitrary_history_row_selects_window():
    proposal = _proposal(tokens=(1, 0, 1, 0), prefix_len=3)
    history = torch.tensor([[[0.0, 0.0], [0.0, 0.0], [0.25, 0.75], [0.9, 0.1]]])
    window = prepare_verification_window(proposal, history)
    torch.testing.assert_close(window.target_rows, history[:, 2:4])
    decision = decide_verification(window, torch.zeros(1, 1))
    assert decision.retained_context_end == 4
    torch.testing.assert_close(decision.continuation_probs, history[:, 3])


@pytest.mark.parametrize(
    "q, message",
    [
        (((0.0, 1.0),), "selected support"),
        (((-0.1, 1.1),), "finite and nonnegative"),
        (((float("nan"), 1.0),), "finite and nonnegative"),
        (((float("inf"), 0.0),), "finite and nonnegative"),
        (((0.25, 0.25),), "sum to one"),
    ],
)
def test_invalid_proposal_law_fails_before_caller_draws(q, message):
    with patch("torch.rand", side_effect=AssertionError("implicit RNG")):
        with pytest.raises(ValueError, match=message):
            _proposal(q=q)


@pytest.mark.parametrize("row", [(float("nan"), 0.0), (-0.1, 1.1), (0.2, 0.2)])
def test_invalid_target_law_is_not_normalized_behind_callers_back(row):
    with pytest.raises(ValueError):
        _window(rows=(row, (0.9, 0.1)))


@pytest.mark.parametrize(
    "draws",
    [torch.tensor([[1.0]]), torch.tensor([[-0.1]]), torch.tensor([[float("nan")]]),
     torch.tensor([[0]]), torch.zeros(1, 2)],
)
def test_invalid_uniform_values_and_shapes_fail_explicitly(draws):
    with pytest.raises(ValueError, match="uniforms"):
        decide_verification(_window(), draws)


def test_proposal_window_length_batch_dtype_and_token_support_are_explicit():
    with pytest.raises(ValueError, match="exactly its candidates"):
        _proposal(tokens=(0, 1, 0))
    with pytest.raises(ValueError, match="one token sequence"):
        ProposalWindow(
            torch.zeros(2, 2, dtype=torch.long), torch.full((2, 1, 2), 0.5), 1,
        )
    with pytest.raises(ValueError, match="torch.long"):
        ProposalWindow(torch.zeros(1, 2), torch.full((1, 1, 2), 0.5), 1)
    with pytest.raises(ValueError, match="outside its vocabulary"):
        _proposal(tokens=(0, 2))
    with pytest.raises(ValueError, match="nonempty"):
        _proposal(prefix_len=0)


def test_values_frozen_and_tensor_equality_not_auto_generated():
    proposal = _proposal()
    other = _proposal()
    assert proposal != other
    with pytest.raises(FrozenInstanceError):
        proposal.prefix_len = 2


@pytest.mark.parametrize(
    "p, q",
    [
        ([0.1, 0.6, 0.3], [0.5, 0.2, 0.3]),
        ([0.0, 0.25, 0.75], [0.2, 0.6, 0.2]),
        ([0.2, 0.3, 0.5], [0.2, 0.3, 0.5]),
    ],
)
@pytest.mark.parametrize("dtype", [
    torch.float16, torch.bfloat16, torch.float32, torch.float64,
])
def test_enumerated_one_step_output_law_equals_target(p, q, dtype):
    """Enumerate actual sampler laws at every source precision; keep 1e-12 bound."""
    p_input = torch.tensor(p, dtype=dtype)
    q_input = torch.tensor(q, dtype=dtype)
    # Independent specification of the valid-row sampler preprocessing. The
    # categorical law is normalized *multinomial weights*, not the raw cache.
    p_weights = p_input.float() / p_input.float().sum()
    q_weights = q_input.float() / q_input.float().sum()
    p = p_weights.double() / p_weights.double().sum()
    q = q_weights.double() / q_weights.double().sum()
    result = torch.zeros_like(p)
    for token in range(len(q)):
        probability_accept = min(1.0, float(p[token] / q[token]))
        proposal = ProposalWindow(
            torch.tensor([[0, token]]), q_input.reshape(1, 1, -1), 1,
        )
        history = torch.stack((
            p_input, torch.full_like(p_input, 1.0 / len(p_input)),
        )).unsqueeze(0)
        window = prepare_verification_window(proposal, history)
        if 0 < probability_accept < 1:
            # First check the represented scalar ratio against the independent
            # law. Then probe strict < at THAT representation's last bit, not
            # at a differently ordered normalization's adjacent float64 value.
            represented = float(window.selected_p[0, 0] / window.selected_q[0, 0])
            assert represented == pytest.approx(probability_accept, rel=1e-15, abs=0)
            boundary = torch.tensor([[represented]], dtype=torch.float64)
            below = torch.nextafter(boundary, torch.zeros_like(boundary))
            assert decide_verification(window, below).all_accepted
            assert not decide_verification(window, boundary).all_accepted
        boundaries = sorted({0.0, probability_accept, 1.0})
        for lower, upper in zip(boundaries[:-1], boundaries[1:], strict=True):
            draw = (lower + upper) / 2
            decision = decide_verification(
                window, torch.tensor([[draw]], dtype=torch.float64),
            )
            # Integrate BOTH actual SUT branches; do not inject accepted mass
            # analytically while testing only its residual branch.
            assert decision.all_accepted == (draw < probability_accept)
            if decision.all_accepted:
                first_token_law = torch.zeros_like(p)
                first_token_law[token] = 1.0
            else:
                first_token_law = decision.continuation_probs[0]
            result += q[token] * (upper - lower) * first_token_law
    torch.testing.assert_close(result, p, rtol=0, atol=1e-12)
