import unittest

import torch

from src.baselines import Baselines


class DssduplinkPayloadTests(unittest.TestCase):
    def test_collect_dssd_uplink_payload_returns_scalar_token_probs(self):
        prob_history = torch.tensor(
            [
                [
                    [0.1, 0.2, 0.7],
                    [0.3, 0.4, 0.3],
                    [0.5, 0.1, 0.4],
                    [0.25, 0.5, 0.25],
                ]
            ],
            dtype=torch.float,
        )
        x = torch.tensor([[10, 2, 0, 1]])

        draft_tokens, draft_token_probs = Baselines._collect_dssd_uplink_payload(
            prob_history=prob_history,
            x=x,
            prefix_len=1,
            gamma=3,
        )

        self.assertTrue(torch.equal(draft_tokens, torch.tensor([[2, 0, 1]])))
        self.assertEqual(tuple(draft_token_probs.shape), (1, 3))
        self.assertTrue(
            torch.allclose(
                draft_token_probs,
                torch.tensor([[0.7, 0.3, 0.1]], dtype=torch.float),
            )
        )


if __name__ == "__main__":
    unittest.main()
