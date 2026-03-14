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
        self.assertEqual(cache._prob_history.shape[1], 6)


if __name__ == "__main__":
    unittest.main()
