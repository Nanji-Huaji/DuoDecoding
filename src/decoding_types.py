from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class VerificationInputs:
    draft_probs_batch: torch.Tensor
    target_probs_batch: torch.Tensor
    draft_tokens: torch.Tensor
    draft_token_indices: torch.Tensor
    prefix_len: int
    gamma: int
    actual_gamma: int
    max_idx: int


@dataclass
class AcceptanceResult:
    accepted_count: int
    n: int
    selected_draft_p: torch.Tensor
    selected_target_p: torch.Tensor
    accept_mask: torch.Tensor


@dataclass
class RollbackPlan:
    draft_end_pos: int
    target_end_pos_reject: int
    target_end_pos_accept: int
    all_accepted: bool
