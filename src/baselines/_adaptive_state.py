"""Private values for the two-model adaptive decoder.

The proposal preserves the exact probability rows used to verify one draft.
Statistics contain numeric run aggregates only: caches, policy, communication,
and stopping decisions remain owned by the decoder.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class AdaptiveProposal:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    prefix_len: int
    gamma: int


@dataclass
class AdaptiveDecodeStats:
    draft_forward_times: int = 0
    target_forward_times: int = 0
    accepted_tokens: int = 0
    drafted_tokens: int = 0
    draft_steps: int = 0
    sum_draft_len: float = 0.0
    sum_top_k: float = 0.0
    queuing_time: float = 0.0
