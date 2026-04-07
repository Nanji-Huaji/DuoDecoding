import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.decoding_ops import (
    compute_acceptance_result,
    finalize_verification,
    resolve_stage_verification,
    sample_reject_token,
)
from src.decoding_types import VerificationInputs
from src.model_gpu import KVCacheModel
from src.utils import (
    norm_logits,
    rebuild_topk_probs,
    rebuild_topk_uniform_probs,
    sample,
)


class TemperatureSamplingTests(unittest.TestCase):
    def test_norm_logits_with_temperature_produces_valid_distribution(self):
        logits = torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float)

        cold_probs = norm_logits(logits, temperature=0.5, top_k=0, top_p=0.0)
        hot_probs = norm_logits(logits, temperature=2.0, top_k=0, top_p=0.0)

        self.assertTrue(torch.allclose(cold_probs.sum(dim=-1), torch.ones(1)))
        self.assertTrue(torch.allclose(hot_probs.sum(dim=-1), torch.ones(1)))
        self.assertTrue(torch.all(cold_probs >= 0))
        self.assertTrue(torch.all(hot_probs >= 0))

        # Lower temperature should sharpen the distribution around the max logit.
        self.assertGreater(cold_probs[0, 2].item(), hot_probs[0, 2].item())
        self.assertLess(cold_probs[0, 0].item(), hot_probs[0, 0].item())

    def test_sample_with_temperature_distribution_can_pick_non_argmax_token(self):
        probs = torch.tensor([[0.7, 0.2, 0.1]], dtype=torch.float)

        with patch(
            "torch.multinomial", return_value=torch.tensor([[1]])
        ) as multinomial:
            sampled = sample(probs)

        multinomial.assert_called_once()
        self.assertEqual(int(sampled.item()), 1)

    def test_kv_cache_model_prefill_with_temperature_stores_probabilities(self):
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
        )
        model = LlamaForCausalLM(config)
        model.eval()

        cache = KVCacheModel(model, temperature=1.0, top_k=0, top_p=0.0)
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        last_probs = cache._prefill(prefix)

        self.assertEqual(tuple(last_probs.shape), (1, config.vocab_size))
        self.assertEqual(cache.prob_history.shape[1], prefix.shape[1])
        self.assertTrue(
            torch.allclose(
                cache.prob_history[0].sum(dim=-1),
                torch.ones(prefix.shape[1]),
                atol=1e-5,
            )
        )
        self.assertTrue(torch.all(cache.prob_history >= 0))

    def test_compute_acceptance_result_handles_stochastic_probabilities(self):
        verification_inputs = VerificationInputs(
            draft_probs_batch=torch.tensor(
                [[[0.6, 0.4], [0.3, 0.7]]],
                dtype=torch.float,
            ),
            target_probs_batch=torch.tensor(
                [[[0.3, 0.7], [0.8, 0.2]]],
                dtype=torch.float,
            ),
            draft_tokens=torch.tensor([[1, 1]], dtype=torch.long),
            draft_token_indices=torch.tensor([[[1], [1]]], dtype=torch.long),
            prefix_len=1,
            gamma=2,
            actual_gamma=2,
            max_idx=2,
        )
        r = torch.tensor([[0.5, 0.5]], dtype=torch.float)

        result = compute_acceptance_result(verification_inputs, r=r)

        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.n, 1)
        self.assertTrue(torch.equal(result.accept_mask, torch.tensor([[True, False]])))
        self.assertTrue(
            torch.allclose(result.selected_draft_p, torch.tensor([[0.4, 0.7]]))
        )
        self.assertTrue(
            torch.allclose(result.selected_target_p, torch.tensor([[0.7, 0.2]]))
        )

    def test_sample_reject_token_uses_residual_distribution_under_temperature(self):
        target_probs = torch.tensor([[0.5, 0.3, 0.2]], dtype=torch.float)
        draft_probs = torch.tensor([[0.2, 0.6, 0.2]], dtype=torch.float)

        captured = {}

        def fake_sample(residual_probs, num_samples=1):
            captured["residual_probs"] = residual_probs.clone()
            return torch.tensor([[0]], dtype=torch.long)

        with patch("src.decoding_ops.sample", side_effect=fake_sample):
            token = sample_reject_token(target_probs, draft_probs)

        self.assertEqual(int(token.item()), 0)
        self.assertTrue(
            torch.allclose(
                captured["residual_probs"],
                torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float),
            )
        )

    def test_norm_logits_applies_top_k_after_temperature(self):
        logits = torch.tensor([[0.1, 0.2, 3.0, 2.5]], dtype=torch.float)

        probs = norm_logits(logits, temperature=1.0, top_k=2, top_p=0.0)

        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(1)))
        self.assertEqual(probs[0, 0].item(), 0.0)
        self.assertEqual(probs[0, 1].item(), 0.0)
        self.assertGreater(probs[0, 2].item(), 0.0)
        self.assertGreater(probs[0, 3].item(), 0.0)

    def test_norm_logits_applies_top_p_after_temperature(self):
        logits = torch.tensor([[3.0, 2.0, 0.5, -1.0]], dtype=torch.float)

        probs = norm_logits(logits, temperature=1.0, top_k=0, top_p=0.8)

        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(1)))
        self.assertGreater(probs[0, 0].item(), 0.0)
        self.assertGreater(probs[0, 1].item(), 0.0)
        self.assertEqual(probs[0, 2].item(), 0.0)
        self.assertEqual(probs[0, 3].item(), 0.0)

    def test_rebuild_topk_uniform_probs_preserves_topk_and_uniform_tail(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float)

        rebuilt = rebuild_topk_uniform_probs(probs, top_k=2)

        self.assertTrue(torch.allclose(rebuilt.sum(dim=-1), torch.ones(1)))
        self.assertTrue(torch.allclose(rebuilt[0, :2], torch.tensor([0.5, 0.3])))
        self.assertTrue(torch.allclose(rebuilt[0, 2:], torch.tensor([0.1, 0.1])))

    def test_rebuild_topk_probs_uniform_matches_legacy_helper(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float)

        rebuilt = rebuild_topk_probs(probs, top_k=2, strategy="uniform")
        legacy = rebuild_topk_uniform_probs(probs, top_k=2)

        self.assertTrue(torch.allclose(rebuilt, legacy))

    def test_rebuild_topk_probs_rejects_unknown_strategy(self):
        probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]], dtype=torch.float)

        with self.assertRaises(ValueError):
            rebuild_topk_probs(probs, top_k=2, strategy="tail")

    def test_compute_acceptance_result_zero_draft_probability_currently_accepts(self):
        verification_inputs = VerificationInputs(
            draft_probs_batch=torch.tensor([[[1.0, 0.0]]], dtype=torch.float),
            target_probs_batch=torch.tensor([[[0.25, 0.75]]], dtype=torch.float),
            draft_tokens=torch.tensor([[1]], dtype=torch.long),
            draft_token_indices=torch.tensor([[[1]]], dtype=torch.long),
            prefix_len=1,
            gamma=1,
            actual_gamma=1,
            max_idx=1,
        )

        result = compute_acceptance_result(
            verification_inputs,
            r=torch.tensor([[0.5]], dtype=torch.float),
        )

        # Current implementation computes target/draft directly, so division by zero
        # yields inf and the token is treated as accepted.
        self.assertEqual(result.accepted_count, 1)
        self.assertEqual(result.n, 1)
        self.assertTrue(torch.equal(result.accept_mask, torch.tensor([[True]])))

    def test_sample_reject_token_falls_back_when_residual_mass_is_zero(self):
        target_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float)
        draft_probs = torch.tensor([[0.2, 0.3, 0.5]], dtype=torch.float)

        token = sample_reject_token(target_probs, draft_probs)

        # With zero residual mass, max_fn falls back to argmax of the all-zero
        # residual row, which deterministically selects index 0 in the current
        # implementation.
        self.assertEqual(int(token.item()), 0)

    def test_resolve_stage_verification_uses_truncated_vocab_on_reject(self):
        class _FakeCache:
            def __init__(self, probs: torch.Tensor, vocab_size: int):
                self.prob_history = probs.clone()
                self.vocab_size = vocab_size
                self.rollback_calls = []

            def rollback(self, end_pos: int):
                self.rollback_calls.append(end_pos)

        proposer = _FakeCache(
            probs=torch.tensor([[[0.55, 0.45]]], dtype=torch.float),
            vocab_size=2,
        )
        verifier = _FakeCache(
            probs=torch.tensor([[[0.1, 0.2, 0.3]]], dtype=torch.float),
            vocab_size=3,
        )
        verification_inputs = VerificationInputs(
            draft_probs_batch=torch.tensor([[[0.55, 0.45, 0.0]]], dtype=torch.float),
            target_probs_batch=torch.tensor([[[0.1, 0.2, 0.7]]], dtype=torch.float),
            draft_tokens=torch.tensor([[1]], dtype=torch.long),
            draft_token_indices=torch.tensor([[[1]]], dtype=torch.long),
            prefix_len=1,
            gamma=1,
            actual_gamma=1,
            max_idx=1,
        )
        captured = {}

        def fake_sample_reject(target_probs, draft_probs, output_device=None):
            captured["target_probs"] = target_probs.clone()
            captured["draft_probs"] = draft_probs.clone()
            return torch.tensor([[1]], dtype=torch.long)

        with (
            patch(
                "src.decoding_ops.verify_draft_sequence_result",
                return_value=(
                    verification_inputs,
                    type(
                        "Acceptance",
                        (),
                        {
                            "accepted_count": 0,
                            "n": 0,
                            "selected_draft_p": torch.tensor([[0.45]]),
                            "selected_target_p": torch.tensor([[0.2]]),
                            "accept_mask": torch.tensor([[False]]),
                        },
                    )(),
                ),
            ),
            patch(
                "src.decoding_ops.sample_reject_token", side_effect=fake_sample_reject
            ),
        ):
            accepted_count, n, token, all_accepted = resolve_stage_verification(
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
        self.assertEqual(int(token.item()), 1)
        self.assertTrue(
            torch.equal(captured["target_probs"], torch.tensor([[0.1, 0.2]]))
        )
        self.assertTrue(
            torch.equal(captured["draft_probs"], torch.tensor([[0.55, 0.45]]))
        )

    def test_finalize_verification_uses_draft_override_on_reject(self):
        class _FakeCache:
            def __init__(self, probs: torch.Tensor):
                self._prob_history = probs.clone()
                self.vocab_size = probs.shape[-1]
                self.rollback_calls = []

            @property
            def prob_history(self) -> torch.Tensor:
                return self._prob_history

            def rollback(self, end_pos: int):
                self.rollback_calls.append(end_pos)

        approx = _FakeCache(
            torch.tensor([[[0.7, 0.3], [0.8, 0.2], [0.6, 0.4]]], dtype=torch.float)
        )
        target = _FakeCache(
            torch.tensor([[[0.4, 0.6], [0.9, 0.1], [0.2, 0.8]]], dtype=torch.float)
        )
        draft_override = torch.tensor(
            [[[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]],
            dtype=torch.float,
        )
        captured = {}

        def fake_sample_reject(target_probs, draft_probs, output_device=None):
            captured["draft_probs"] = draft_probs.clone()
            return torch.tensor([[1]], dtype=torch.long)

        with patch(
            "src.decoding_ops.sample_reject_token", side_effect=fake_sample_reject
        ):
            output = finalize_verification(
                approx_model_cache=approx,
                target_model_cache=target,
                x=torch.tensor([[10, 11, 12]], dtype=torch.long),
                prefix_len=2,
                gamma=1,
                n=1,
                draft_probs_override=draft_override,
            )

        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertTrue(
            torch.equal(captured["draft_probs"], torch.tensor([[0.2, 0.8]]))
        )


if __name__ == "__main__":
    unittest.main()
