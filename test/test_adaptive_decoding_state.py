"""The adaptive decoder's internal values expose data, not runtime owners."""

from dataclasses import FrozenInstanceError

import pytest
import torch

from src.baselines._adaptive_state import AdaptiveDecodeStats, AdaptiveProposal


def test_proposal_keeps_the_sampled_rows_and_fixed_positions():
    tokens = torch.tensor([[0, 1, 2]])
    probabilities = torch.ones(1, 2, 4) / 4
    proposal = AdaptiveProposal(tokens, probabilities, prefix_len=1, gamma=2)
    assert proposal.tokens is tokens
    assert proposal.probabilities is probabilities
    with pytest.raises(FrozenInstanceError):
        proposal.gamma = 3
    # Equality must not attempt an elementwise Tensor comparison.
    assert proposal != AdaptiveProposal(tokens.clone(), probabilities.clone(), 1, 2)


def test_run_statistics_are_independent_numeric_aggregates():
    first = AdaptiveDecodeStats()
    second = AdaptiveDecodeStats()
    first.draft_forward_times += 3
    first.queuing_time += 0.25
    assert second == AdaptiveDecodeStats()
    assert first.draft_forward_times == 3
    assert first.queuing_time == 0.25
    assert all(type(value) in (int, float) for value in vars(first).values())


def test_value_construction_does_not_consume_rng():
    tokens = torch.tensor([[0, 1]])
    probabilities = torch.ones(1, 1, 4) / 4
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(73)
        before = torch.get_rng_state().clone()
        AdaptiveDecodeStats()
        AdaptiveProposal(tokens, probabilities, 1, 1)
        assert torch.equal(before, torch.get_rng_state())
