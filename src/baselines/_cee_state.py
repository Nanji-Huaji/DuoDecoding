"""Private CEE-SD round values, with no model, policy or communication ownership.

Proposal rows are local to the proposed tokens (row zero predicts the token at
``prefix_len``). They are captured before wire quantization. In lossless mode a
corrected inner stage has its verifier's law; lossy wire verification is explicitly
an approximation and must not be described as unbiased target sampling.
"""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, eq=False)
class CeeProposal:
    tokens: torch.Tensor
    probabilities: torch.Tensor
    prefix_len: int
    entropy_source: torch.Tensor | None = None
    computation_time: float = 0.0

    @property
    def count(self) -> int:
        return self.tokens.shape[1] - self.prefix_len


@dataclass(frozen=True, eq=False)
class CeeVerification:
    accepted_count: int
    accepted_end: int
    token: torch.Tensor
    all_accepted: bool
    carried_probabilities: torch.Tensor
    wire_probabilities: torch.Tensor
    skipped: bool = False


@dataclass
class CeeDecodeStats:
    little_forward_times: int = 0
    draft_forward_times: int = 0
    target_forward_times: int = 0
    little_generated_tokens: int = 0
    draft_generated_tokens: int = 0
    little_accepted_tokens: int = 0
    draft_accepted_tokens: int = 0
    little_computation_time: float = 0.0
    draft_computation_time: float = 0.0
    target_computation_time: float = 0.0
    queuing_time: float = 0.0
    dra_overhead_time: float = 0.0
