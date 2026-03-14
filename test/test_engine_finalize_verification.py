import unittest

import torch

from src.engine import Decoding


class _FakeCache:
    def __init__(self, probs: torch.Tensor):
        self._prob_history = probs.clone()
        self.vocab_size = probs.shape[-1]
        self.rollback_calls: list[int] = []

    def rollback(self, end_pos: int):
        self.rollback_calls.append(end_pos)


class FinalizeVerificationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.tensor([[10, 11, 12, 13]])
        self.approx_probs = torch.tensor(
            [[[0.7, 0.3], [0.8, 0.2], [0.6, 0.4], [0.5, 0.5]]], dtype=torch.float
        )
        self.target_probs = torch.tensor(
            [
                [
                    [0.6, 0.4],
                    [0.9, 0.1],
                    [0.2, 0.8],
                    [0.3, 0.7],
                    [0.1, 0.9],
                ]
            ],
            dtype=torch.float,
        )

    def test_rejection_path_rolls_back_target_to_rejection_position(self):
        approx = _FakeCache(self.approx_probs)
        target = _FakeCache(self.target_probs)

        output = Decoding._finalize_verification(
            approx_model_cache=approx,
            target_model_cache=target,
            x=self.x,
            prefix_len=2,
            gamma=2,
            n=1,
        )

        self.assertEqual(approx.rollback_calls, [2])
        self.assertEqual(target.rollback_calls, [2])
        self.assertEqual(tuple(output.shape), (1, 3))

    def test_all_accepted_path_keeps_next_target_distribution_then_rolls_back(self):
        approx = _FakeCache(self.approx_probs)
        target = _FakeCache(self.target_probs)

        output = Decoding._finalize_verification(
            approx_model_cache=approx,
            target_model_cache=target,
            x=self.x,
            prefix_len=2,
            gamma=2,
            n=3,
        )

        self.assertEqual(approx.rollback_calls, [4])
        self.assertEqual(target.rollback_calls, [5])
        self.assertEqual(tuple(output.shape), (1, 5))


if __name__ == "__main__":
    unittest.main()
