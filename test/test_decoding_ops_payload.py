import unittest

import torch

from src.decoding_ops import collect_verification_payload


class CollectVerificationPayloadTests(unittest.TestCase):
    def test_collect_verification_payload_truncates_to_available_prob_rows(self):
        prob_history = torch.zeros((1, 1, 4), dtype=torch.float)
        prob_history[0, 0, 2] = 1.0
        x = torch.tensor([[0, 2, 3]], dtype=torch.long)

        draft_tokens, draft_probs = collect_verification_payload(
            prob_history=prob_history,
            x=x,
            prefix_len=1,
            gamma=2,
        )

        self.assertTrue(
            torch.equal(draft_tokens, torch.tensor([[2]], dtype=torch.long))
        )
        self.assertTrue(torch.equal(draft_probs, torch.tensor([[1.0]])))

    def test_collect_verification_payload_truncates_to_available_tokens(self):
        prob_history = torch.zeros((1, 3, 4), dtype=torch.float)
        prob_history[0, 0, 1] = 1.0
        prob_history[0, 1, 2] = 1.0
        prob_history[0, 2, 3] = 1.0
        x = torch.tensor([[0, 1, 2]], dtype=torch.long)

        draft_tokens, draft_probs = collect_verification_payload(
            prob_history=prob_history,
            x=x,
            prefix_len=1,
            gamma=5,
        )

        self.assertTrue(
            torch.equal(draft_tokens, torch.tensor([[1, 2]], dtype=torch.long))
        )
        self.assertTrue(torch.equal(draft_probs, torch.tensor([[1.0, 1.0]])))

    def test_collect_verification_payload_returns_empty_when_no_overlap(self):
        prob_history = torch.zeros((1, 1, 4), dtype=torch.float)
        x = torch.tensor([[0]], dtype=torch.long)

        draft_tokens, draft_probs = collect_verification_payload(
            prob_history=prob_history,
            x=x,
            prefix_len=1,
            gamma=3,
        )

        self.assertEqual(tuple(draft_tokens.shape), (1, 0))
        self.assertEqual(tuple(draft_probs.shape), (1, 0))


if __name__ == "__main__":
    unittest.main()
