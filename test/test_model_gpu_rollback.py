import unittest

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.model_gpu import KVCacheModel


class KVCacheRollbackTests(unittest.TestCase):
    def test_rollback_does_not_extend_history_past_real_cache_length(self):
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

        cache = KVCacheModel(model, temperature=0, top_k=0, top_p=0)
        cache.vocab_size = config.vocab_size

        prefix = torch.tensor([[1, 2, 3, 4, 5]])
        x = cache.generate(prefix, 2)

        self.assertEqual(tuple(x.shape), (1, 7))
        self.assertEqual(cache.current_length, 6)
        self.assertEqual(cache._current_seq_len, 6)

        cache.rollback(7)

        self.assertEqual(cache.current_length, 6)
        self.assertEqual(cache._current_seq_len, 6)
        self.assertEqual(cache.prob_history.shape[1], 6)

    def test_decode_step_updates_cache_and_history_after_prefill(self):
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

        cache = KVCacheModel(model, temperature=1.0, top_k=0, top_p=0)
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        prefill_q = cache._prefill(prefix)

        self.assertEqual(tuple(prefill_q.shape), (1, config.vocab_size))
        self.assertEqual(cache.current_length, prefix.shape[1])
        self.assertEqual(cache._current_seq_len, prefix.shape[1])
        self.assertEqual(cache.prob_history.shape[1], prefix.shape[1])
        self.assertEqual(cache.logits_history.shape[1], prefix.shape[1])

        next_input = torch.tensor([[5]], dtype=torch.long)
        decode_q = cache._decode_step(next_input)

        self.assertEqual(tuple(decode_q.shape), (1, config.vocab_size))
        self.assertEqual(cache.current_length, prefix.shape[1] + 1)
        self.assertEqual(cache._current_seq_len, prefix.shape[1] + 1)
        self.assertEqual(cache.prob_history.shape[1], prefix.shape[1] + 1)
        self.assertEqual(cache.logits_history.shape[1], prefix.shape[1] + 1)

    def test_decode_step_with_empty_suffix_reuses_last_probability_row(self):
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

        cache = KVCacheModel(model, temperature=1.0, top_k=0, top_p=0)
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        cache._prefill(prefix)
        cache._decode_step(torch.tensor([[5]], dtype=torch.long))

        cached_q = cache._decode_step(torch.empty((1, 0), dtype=torch.long))

        self.assertTrue(torch.equal(cached_q, cache.prob_history[:, -1, :]))
        self.assertEqual(cache.current_length, 5)
        self.assertEqual(cache._current_seq_len, 5)


if __name__ == "__main__":
    unittest.main()
