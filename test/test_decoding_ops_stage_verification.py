import unittest
from unittest.mock import patch

import torch

from src.decoding_ops import resolve_stage_verification
from src.decoding_types import AcceptanceResult, VerificationInputs


class _FakeCache:
    def __init__(self, probs: torch.Tensor, vocab_size: int):
        self.prob_history = probs.clone()
        self.vocab_size = vocab_size
        self.rollback_calls: list[int] = []

    def rollback(self, end_pos: int):
        self.rollback_calls.append(end_pos)


class ResolveStageVerificationTests(unittest.TestCase):
    def test_reject_sampling_is_limited_to_effective_vocab_size(self):
        proposer = _FakeCache(
            probs=torch.tensor(
                [
                    [
                        [0.6, 0.4, 0.0, 0.0],
                        [0.6, 0.4, 0.0, 0.0],
                    ]
                ],
                dtype=torch.float,
            ),
            vocab_size=2,
        )
        verifier = _FakeCache(
            probs=torch.tensor(
                [
                    [
                        [0.1, 0.2, 0.3, 0.4],
                        [0.1, 0.2, 0.3, 0.4],
                    ]
                ],
                dtype=torch.float,
            ),
            vocab_size=4,
        )
        verification_inputs = VerificationInputs(
            draft_probs_batch=torch.tensor(
                [[[0.6, 0.4, 0.0, 0.0]]],
                dtype=torch.float,
            ),
            target_probs_batch=torch.tensor(
                [[[0.1, 0.2, 0.3, 0.4]]],
                dtype=torch.float,
            ),
            draft_tokens=torch.tensor([[1]], dtype=torch.long),
            draft_token_indices=torch.tensor([[[1]]], dtype=torch.long),
            prefix_len=1,
            gamma=1,
            actual_gamma=1,
            max_idx=1,
        )
        acceptance_result = AcceptanceResult(
            accepted_count=0,
            n=0,
            selected_draft_p=torch.tensor([[0.4]], dtype=torch.float),
            selected_target_p=torch.tensor([[0.2]], dtype=torch.float),
            accept_mask=torch.tensor([[False]]),
        )
        captured = {}

        def fake_sample_reject(target_probs, draft_probs, output_device=None):
            captured["target_shape"] = tuple(target_probs.shape)
            captured["draft_shape"] = tuple(draft_probs.shape)
            captured["target_probs"] = target_probs.clone()
            captured["draft_probs"] = draft_probs.clone()
            return torch.tensor([[1]], dtype=torch.long)

        with (
            patch(
                "src.decoding_ops.verify_draft_sequence_result",
                return_value=(verification_inputs, acceptance_result),
            ),
            patch(
                "src.decoding_ops.sample_reject_token",
                side_effect=fake_sample_reject,
            ),
        ):
            accepted_count, n, t, all_accepted = resolve_stage_verification(
                proposer_cache=proposer,
                verifier_cache=verifier,
                x=torch.tensor([[0, 1]], dtype=torch.long),
                prefix_len=1,
                gamma=1,
                output_device=torch.device("cpu"),
            )

        self.assertEqual(accepted_count, 0)
        self.assertEqual(n, 0)
        self.assertFalse(all_accepted)
        self.assertEqual(int(t.item()), 1)
        self.assertEqual(captured["target_shape"], (1, 2))
        self.assertEqual(captured["draft_shape"], (1, 2))
        self.assertTrue(
            torch.equal(captured["target_probs"], torch.tensor([[0.1, 0.2]]))
        )
        self.assertTrue(
            torch.equal(captured["draft_probs"], torch.tensor([[0.6, 0.4]]))
        )
        self.assertEqual(proposer.rollback_calls, [1])
        self.assertEqual(verifier.rollback_calls, [1])


if __name__ == "__main__":
    unittest.main()
