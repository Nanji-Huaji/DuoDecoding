import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.decoding_ops import (
    resolve_stage_verification,
    verify_draft_sequence,
    verify_draft_sequence_result,
)
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
            selected_draft_p=torch.tensor([[0.4]], dtype=torch.float),
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
            accepted_count=torch.tensor([0], dtype=torch.int64),
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


class VerifyDraftSequenceBatchGuardTests(unittest.TestCase):
    """B43：整条 verify_draft_sequence 路径是 bs=1 形状（accepted_count 与
    serial 下行计费都只取 batch 0）。bs>1 必须在入口响亮失败。"""

    def test_batch_size_above_one_fails_loudly(self):
        with self.assertRaisesRegex(AssertionError, "batch_size=1"):
            verify_draft_sequence(
                draft_model_cache=SimpleNamespace(device=torch.device("cpu")),
                target_model_cache=SimpleNamespace(device=torch.device("cpu")),
                x=torch.zeros((2, 8), dtype=torch.long),
                prefix_len=4,
                gamma=2,
            )

    def test_result_variant_also_rejects_batch_size_above_one(self):
        # 同一条 B43 假设的另一入口（resolve_stage_verification 的上游）：
        # 守卫必须两条路径都有，否则 cee_dsd/adaptive_tridecoding 的
        # 分层验证在 bs>1 下仍会静默错计。
        with self.assertRaisesRegex(AssertionError, "batch_size=1"):
            verify_draft_sequence_result(
                draft_model_cache=SimpleNamespace(device=torch.device("cpu")),
                target_model_cache=SimpleNamespace(device=torch.device("cpu")),
                x=torch.zeros((2, 8), dtype=torch.long),
                prefix_len=4,
                gamma=2,
            )


if __name__ == "__main__":
    unittest.main()
