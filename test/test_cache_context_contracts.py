"""A3 的 CPU 契约测试：不采样上下文前向 + 图缓存有效构造身份。

覆盖两件事（均不需要 GPU）：

1. `KVCacheModel.forward_context()` —— 用真实 `KVCacheModel` + 小型 CPU
   Llama 后端验证 prefill/extend/等长复用/缓存领先拒绝、单 batch 边界、输入
   校验、hidden/history 保留，以及**不采样、不消耗 torch RNG**；
2. `acquire_graph_caches()` —— 图配置冻结比较、显式 `cache_key` 身份变化重建、
   外部可变 dict 不别名、等价身份只 reset、eager 每次新建、旧的无身份缓存不
   被静默复用。

限制：CUDA Graph 的捕获/回放无法在 CPU 上执行；这里只验证 KVCacheModel 的
eager 分支与其复用的内部 prefill/decode 路由，以及 acquire 层的身份/生命周期
语义。真实图执行仍需 A5 的 GPU 验收。
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from src.baselines.operations import SharedOperations
from src.graph_decode import acquire_graph_caches
from src.model_gpu import KVCacheModel
from src.utils import sample

VOCAB_SIZE = 32
PROB_DTYPE = torch.float32


def _tiny_config() -> LlamaConfig:
    return LlamaConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=64,
        output_hidden_states=True,
    )


def _tiny_model() -> LlamaForCausalLM:
    torch.manual_seed(0)
    model = LlamaForCausalLM(_tiny_config())
    model.eval()
    return model


class _CountingKVCache(KVCacheModel):
    """真实 KVCacheModel + reset 计数，用于证明“等价身份才 reset”。"""

    def __init__(self, model: LlamaForCausalLM, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.reset_calls = 0

    def reset_for_new_sample(self) -> None:
        self.reset_calls += 1
        super().reset_for_new_sample()


class ForwardContextContractTests(unittest.TestCase):
    """真实 KVCacheModel 的公开不采样前向契约。"""

    def setUp(self) -> None:
        self.model = _tiny_model()

    def _cache(self) -> KVCacheModel:
        return KVCacheModel(self.model, temperature=1.0, top_k=0, top_p=0.0)

    def test_prefill_matches_private_prefill_and_initializes_history(self):
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        public = self._cache()
        private = self._cache()

        got = public.forward_context(prefix)
        want = private._prefill(prefix)

        self.assertEqual(tuple(got.shape), (1, VOCAB_SIZE))
        self.assertEqual(got.dtype, PROB_DTYPE)
        torch.testing.assert_close(got, want)
        self.assertAlmostEqual(float(got.sum()), 1.0, places=5)

        for cache in (public, private):
            self.assertEqual(cache.current_length, prefix.shape[1])
            self.assertEqual(cache._current_seq_len, prefix.shape[1])
            self.assertEqual(
                tuple(cache.prob_history.shape), (1, prefix.shape[1], VOCAB_SIZE)
            )
            self.assertEqual(
                tuple(cache.logits_history.shape), (1, prefix.shape[1], VOCAB_SIZE)
            )
            self.assertIsNotNone(cache.hidden_states)

        torch.testing.assert_close(public.prob_history, private.prob_history)
        torch.testing.assert_close(public.logits_history, private.logits_history)
        torch.testing.assert_close(public.last_hidden_state, private.last_hidden_state)

    def test_extend_existing_context_matches_private_decode(self):
        prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
        suffix = torch.tensor([[4, 5, 6]], dtype=torch.long)
        full = torch.cat([prefix, suffix], dim=1)

        public = self._cache()
        public.forward_context(prefix)
        hidden_after_prefill = public.hidden_states
        got = public.forward_context(full)

        private = self._cache()
        private._prefill(prefix)
        want = private._decode_step(suffix)

        self.assertEqual(tuple(got.shape), (1, VOCAB_SIZE))
        torch.testing.assert_close(got, want)
        self.assertEqual(public.current_length, full.shape[1])
        self.assertEqual(public._current_seq_len, full.shape[1])
        self.assertEqual(
            tuple(public.prob_history.shape), (1, full.shape[1], VOCAB_SIZE)
        )
        torch.testing.assert_close(public.prob_history, private.prob_history)
        torch.testing.assert_close(public.logits_history, private.logits_history)
        self.assertIsNot(public.hidden_states, hidden_after_prefill)

    def test_repeated_single_token_extension_keeps_prefix_semantics(self):
        cache = self._cache()
        ids = [1, 2, 3]
        prefix = torch.tensor([ids], dtype=torch.long)
        first = cache.forward_context(prefix)
        self.assertEqual(cache.current_length, 3)

        for token in (7, 8):
            ids.append(token)
            step = cache.forward_context(torch.tensor([ids], dtype=torch.long))
            self.assertEqual(step.shape, (1, VOCAB_SIZE))
            self.assertEqual(cache.current_length, len(ids))
            self.assertEqual(cache._current_seq_len, len(ids))

        # 与逐 token 的私有 decode 路径数值一致（同算子序列，避免多 token
        # 并行前向与单步前向的 bf16/fp32 舍入差被当成契约失败）
        reference = self._cache()
        reference._prefill(prefix)
        reference._decode_step(torch.tensor([[7]], dtype=torch.long))
        reference._decode_step(torch.tensor([[8]], dtype=torch.long))
        torch.testing.assert_close(cache.prob_history, reference.prob_history)
        self.assertEqual(tuple(first.shape), (1, VOCAB_SIZE))

    def test_equal_length_input_reuses_without_forward_or_mutation(self):
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        cache = self._cache()
        first = cache.forward_context(prefix)

        calls = []

        def _count_forward(module, args, kwargs):
            calls.append(1)

        handle = self.model.register_forward_pre_hook(_count_forward, with_kwargs=True)
        try:
            prob_before = cache.prob_history.clone()
            logits_before = cache.logits_history.clone()
            hidden_before = cache.hidden_states
            entropy_before = cache._last_entropy_t
            second = cache.forward_context(prefix.clone())
        finally:
            handle.remove()

        self.assertEqual(calls, [])
        torch.testing.assert_close(second, first)
        self.assertEqual(cache.current_length, prefix.shape[1])
        self.assertEqual(cache._current_seq_len, prefix.shape[1])
        torch.testing.assert_close(cache.prob_history, prob_before)
        torch.testing.assert_close(cache.logits_history, logits_before)
        self.assertIs(cache.hidden_states, hidden_before)
        self.assertIs(cache._last_entropy_t, entropy_before)

        # 等长复用没有破坏 KV/历史：随后 extend 与“只 prefill 后直接 extend”一致
        extended = torch.cat([prefix, torch.tensor([[9]], dtype=torch.long)], dim=1)
        got_extended = cache.forward_context(extended)
        reference = self._cache()
        reference.forward_context(prefix)
        want_extended = reference.forward_context(extended)
        torch.testing.assert_close(got_extended, want_extended)
        torch.testing.assert_close(cache.prob_history, reference.prob_history)

    def test_cache_leading_input_is_rejected_without_state_change(self):
        prefix = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        cache = self._cache()
        cache.forward_context(prefix)
        prob_before = cache.prob_history.clone()
        hidden_before = cache.hidden_states

        with self.assertRaisesRegex(ValueError, "缓存领先"):
            cache.forward_context(prefix[:, :3])

        self.assertEqual(cache.current_length, prefix.shape[1])
        self.assertEqual(cache._current_seq_len, prefix.shape[1])
        torch.testing.assert_close(cache.prob_history, prob_before)
        self.assertIs(cache.hidden_states, hidden_before)

    def test_equal_length_reuse_rejects_inconsistent_tracked_history(self):
        prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
        cache = self._cache()
        cache.forward_context(prefix)
        cache._current_seq_len = 2  # 模拟内部状态损坏

        with self.assertRaisesRegex(RuntimeError, "不一致"):
            cache.forward_context(prefix)

    def test_extend_rejects_inconsistent_history_before_changing_state(self):
        prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
        cache = self._cache()
        cache.forward_context(prefix)
        cache._current_seq_len = 2  # Same corruption fixture as equal-length reuse.
        prob_before = cache.prob_history.clone()
        logits_before = cache.logits_history.clone()
        hidden_before = cache.hidden_states
        extended = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)

        with self.assertRaisesRegex(RuntimeError, "不一致"):
            cache.forward_context(extended)

        self.assertEqual(cache.current_length, 3)
        self.assertEqual(cache._current_seq_len, 2)
        torch.testing.assert_close(cache.prob_history, prob_before)
        torch.testing.assert_close(cache.logits_history, logits_before)
        self.assertIs(cache.hidden_states, hidden_before)

    def test_forward_context_consumes_no_sampling_rng_or_sample(self):
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        extended = torch.cat([prefix, torch.tensor([[5]], dtype=torch.long)], dim=1)
        cache = self._cache()

        torch.manual_seed(1234)
        rng_before = torch.random.get_rng_state().clone()
        with patch(
            "src.model_gpu.sample",
            side_effect=AssertionError("forward_context 不允许调用 sample()"),
        ):
            cache.forward_context(prefix)
            cache.forward_context(extended)
            cache.forward_context(extended)  # 等长复用分支同样不采样
        rng_after = torch.random.get_rng_state()

        self.assertTrue(
            torch.equal(rng_before, rng_after),
            "forward_context 改变了 torch RNG 状态（应完全不消耗采样 RNG）",
        )
        self.assertEqual(cache.current_length, extended.shape[1])
        # 返回的是概率而不是 token：形状必须保留 vocab 维
        probs = cache.forward_context(extended)
        self.assertEqual(tuple(probs.shape), (1, VOCAB_SIZE))

    def test_real_proposal_helper_uses_public_forward_and_only_its_own_samples(self):
        cache = self._cache()
        prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
        observations = []

        def predict(hidden):
            self.assertIsNotNone(hidden)
            observations.append(cache.current_length)
            return len(observations) == 2

        with (
            patch.object(
                cache, "forward_context", wraps=cache.forward_context
            ) as forward,
            patch.object(cache, "_forward_with_kvcache", side_effect=AssertionError(
                "the shared proposal must not use the private API"
            )),
            patch("src.model_gpu.sample", side_effect=AssertionError(
                "context forwarding must not sample"
            )),
            patch("src.baselines.operations.sample", wraps=sample) as sampler,
        ):
            tokens, rows, metadata, q = (
                SharedOperations._generate_with_optional_rebuilt_proposal(
                    SimpleNamespace(), cache, prefix, 3, 4,
                    adapter=SimpleNamespace(predict=predict), need_topk_metadata=True,
                )
            )
        self.assertEqual(forward.call_count, 2)
        self.assertEqual(sampler.call_count, 2)
        self.assertEqual(observations, [3, 4])  # sample then adapter, not vice versa
        self.assertEqual(tokens.shape, (1, 5))
        self.assertEqual(cache.current_length, 4)  # final sampled token not forwarded
        self.assertEqual(rows.shape, (1, 2, VOCAB_SIZE))
        self.assertIsNotNone(metadata)
        torch.testing.assert_close(q, cache.prob_history[:, -1])

    def test_reset_then_forward_context_prefills_again(self):
        prefix = torch.tensor([[1, 2, 3]], dtype=torch.long)
        cache = self._cache()
        cache.forward_context(prefix)
        cache.reset_for_new_sample()
        self.assertEqual(cache.current_length, 0)

        got = cache.forward_context(prefix)
        self.assertEqual(tuple(got.shape), (1, VOCAB_SIZE))
        self.assertEqual(cache.current_length, prefix.shape[1])
        self.assertEqual(cache._current_seq_len, prefix.shape[1])

    def test_accepts_non_long_integer_dtype(self):
        prefix = torch.tensor([[1, 2, 3]], dtype=torch.int32)
        int_cache = self._cache()
        long_cache = self._cache()

        torch.testing.assert_close(
            int_cache.forward_context(prefix),
            long_cache.forward_context(prefix.to(torch.long)),
        )
        self.assertEqual(int_cache.current_length, 3)

    def test_rejects_batch_greater_than_one(self):
        cache = self._cache()
        with self.assertRaisesRegex(ValueError, "batch=1"):
            cache.forward_context(torch.tensor([[1, 2], [3, 4]], dtype=torch.long))
        self.assertEqual(cache.current_length, 0)

    def test_rejects_empty_prefix(self):
        cache = self._cache()
        with self.assertRaisesRegex(ValueError, "非空"):
            cache.forward_context(torch.empty((1, 0), dtype=torch.long))
        self.assertEqual(cache.current_length, 0)

    def test_rejects_non_2d_shapes(self):
        cache = self._cache()
        for bad in (
            torch.tensor([1, 2, 3], dtype=torch.long),
            torch.zeros((1, 1, 1), dtype=torch.long),
        ):
            with self.assertRaisesRegex(ValueError, "二维"):
                cache.forward_context(bad)
        self.assertEqual(cache.current_length, 0)

    def test_rejects_float_bool_and_non_tensor_inputs(self):
        cache = self._cache()
        for bad in (
            torch.tensor([[1.0, 2.0]]),
            torch.tensor([[True, False]]),
        ):
            with self.assertRaisesRegex(TypeError, "整型"):
                cache.forward_context(bad)
        with self.assertRaises(TypeError):
            cache.forward_context([1, 2, 3])  # type: ignore[arg-type]
        self.assertEqual(cache.current_length, 0)

    def test_rejects_out_of_vocab_token_ids(self):
        cache = self._cache()
        with self.assertRaisesRegex(ValueError, "embedding range"):
            cache.forward_context(torch.tensor([[1, VOCAB_SIZE]], dtype=torch.long))
        self.assertEqual(cache.current_length, 0)


class AcquireGraphCachesIdentityTests(unittest.TestCase):
    """图缓存复用身份：冻结比较、显式身份、重建与 reset 边界。"""

    def setUp(self) -> None:
        self.model = _tiny_model()

    def _graph_kw(self, **overrides) -> dict:
        kw = {
            "use_cuda_graph": True,
            "verify_graph_sizes": [4, 8],
            "graph_len_budget": 512,
        }
        kw.update(overrides)
        return kw

    def _builders(self, graph_kw, sink):
        def build():
            cache = _CountingKVCache(
                self.model,
                temperature=1.0,
                top_k=0,
                top_p=0.0,
                **graph_kw,
            )
            sink.append(cache)
            return cache

        return {"a": build, "b": build}

    def test_equal_identity_reuses_and_resets_despite_new_lambdas(self):
        holder = SimpleNamespace()
        sink = []

        graph_kw_1 = self._graph_kw()
        builders_1 = self._builders(graph_kw_1, sink)
        first = acquire_graph_caches(holder, "_caches", graph_kw_1, builders_1)

        graph_kw_2 = self._graph_kw()
        builders_2 = self._builders(graph_kw_2, sink)
        self.assertIsNot(builders_1["a"], builders_2["a"])
        second = acquire_graph_caches(holder, "_caches", graph_kw_2, builders_2)

        self.assertIs(first, second)
        self.assertEqual(len(sink), 2)  # 两次 builder 调用，第二次没有新建
        self.assertEqual([c.reset_calls for c in sink], [1, 1])
        self.assertIs(holder._caches, first)

    def test_graph_config_change_rebuilds_and_drops_old_caches(self):
        holder = SimpleNamespace()
        sink = []
        first = acquire_graph_caches(
            holder, "_caches", self._graph_kw(), self._builders(self._graph_kw(), sink)
        )
        old_caches = list(sink)

        second = acquire_graph_caches(
            holder,
            "_caches",
            self._graph_kw(graph_len_budget=768),
            self._builders(self._graph_kw(graph_len_budget=768), sink),
        )

        self.assertIsNot(first, second)
        self.assertIsNot(first["a"], second["a"])
        self.assertEqual(len(sink), 4)
        self.assertTrue(all(cache.reset_calls == 0 for cache in old_caches))
        self.assertIs(holder._caches, second)

    def test_explicit_cache_key_change_rebuilds_even_with_same_graph_kw(self):
        holder = SimpleNamespace()
        sink = []
        graph_kw = self._graph_kw()

        first = acquire_graph_caches(
            holder,
            "_caches",
            graph_kw,
            self._builders(graph_kw, sink),
            cache_key=("model-a", 1.0, 0, 0.0, 1024),
        )
        second = acquire_graph_caches(
            holder,
            "_caches",
            self._graph_kw(),
            self._builders(self._graph_kw(), sink),
            cache_key=("model-b", 1.0, 0, 0.0, 1024),
        )
        self.assertIsNot(first, second)
        self.assertEqual(len(sink), 4)

        third = acquire_graph_caches(
            holder,
            "_caches",
            self._graph_kw(),
            self._builders(self._graph_kw(), sink),
            cache_key=("model-b", 1.0, 0, 0.0, 1024),
        )
        self.assertIs(second, third)
        self.assertEqual(len(sink), 4)
        self.assertTrue(all(cache.reset_calls == 1 for cache in sink[2:]))

    def test_graph_kw_is_snapshotted_not_aliased(self):
        holder = SimpleNamespace()
        sink = []
        graph_kw = self._graph_kw()
        builders = self._builders(graph_kw, sink)
        first = acquire_graph_caches(holder, "_caches", graph_kw, builders)
        self.assertEqual(len(sink), 2)

        # 调用方原地改自己的 dict：已存身份必须是快照，不能被一起改掉
        graph_kw["verify_graph_sizes"].append(16)
        fresh_equal = {
            "verify_graph_sizes": [4, 8],  # 顺序/插入顺序不同也应等价
            "graph_len_budget": 512,
            "use_cuda_graph": True,
        }
        second = acquire_graph_caches(
            holder, "_caches", fresh_equal, self._builders(fresh_equal, sink)
        )

        self.assertIs(second, first)  # 等价配置仍命中 ⇒ 存的是冻结快照
        self.assertEqual(len(sink), 2)

        # 真正被改过的配置必须重建，而不是复用
        mutated = {
            "use_cuda_graph": True,
            "verify_graph_sizes": [4, 8, 16],
            "graph_len_budget": 512,
        }
        third = acquire_graph_caches(
            holder, "_caches", mutated, self._builders(mutated, sink)
        )
        self.assertIsNot(third, second)
        self.assertEqual(len(sink), 4)
        # 前一批缓存只在第二次等价命中时 reset 过一次；配置真正变化后它们被
        # 丢弃，不会因重建再被 reset（sink[2:] 是重建出的新缓存）。
        self.assertTrue(all(cache.reset_calls == 1 for cache in sink[:2]))
        self.assertTrue(all(cache.reset_calls == 0 for cache in sink[2:]))

    def test_eager_mode_always_rebuilds_even_with_same_cache_key(self):
        holder = SimpleNamespace()
        sink = []
        empty: dict = {}

        first = acquire_graph_caches(
            holder, "_caches", empty, self._builders(empty, sink), cache_key="same"
        )
        second = acquire_graph_caches(
            holder, "_caches", {}, self._builders({}, sink), cache_key="same"
        )

        self.assertEqual(len(sink), 4)
        self.assertIsNot(first["a"], second["a"])
        self.assertFalse(hasattr(holder, "_caches"))
        self.assertFalse(hasattr(holder, "_caches__graph_cache_identity"))

    def test_unrecorded_legacy_caches_are_not_reused(self):
        legacy = _CountingKVCache(self.model, temperature=1.0, top_k=0, top_p=0.0)
        holder = SimpleNamespace()
        holder._caches = {"a": legacy}

        sink = []
        rebuilt = acquire_graph_caches(
            holder, "_caches", self._graph_kw(), self._builders(self._graph_kw(), sink)
        )

        self.assertIsNot(rebuilt["a"], legacy)
        self.assertEqual(legacy.reset_calls, 0)
        self.assertEqual(len(sink), 2)

    def test_externally_replaced_cache_mapping_is_not_trusted(self):
        holder = SimpleNamespace()
        sink = []
        graph_kw = self._graph_kw()
        first = acquire_graph_caches(
            holder, "_caches", graph_kw, self._builders(graph_kw, sink)
        )
        holder._caches = dict(first)  # 同一批对象，但不是记录的 dict

        rebuilt = acquire_graph_caches(
            holder,
            "_caches",
            self._graph_kw(),
            self._builders(self._graph_kw(), sink),
        )

        self.assertIsNot(rebuilt, first)
        self.assertEqual(len(sink), 4)

    def test_builder_name_set_change_rebuilds(self):
        holder = SimpleNamespace()
        sink = []
        kw = self._graph_kw()

        def build():
            cache = _CountingKVCache(
                self.model, temperature=1.0, top_k=0, top_p=0.0, **kw
            )
            sink.append(cache)
            return cache

        first = acquire_graph_caches(holder, "_caches", kw, {"a": build})
        self.assertEqual(set(first), {"a"})

        second = acquire_graph_caches(holder, "_caches", kw, {"a": build, "b": build})
        self.assertEqual(set(second), {"a", "b"})
        self.assertIsNot(second["a"], first["a"])
        self.assertEqual(len(sink), 3)
        self.assertEqual(first["a"].reset_calls, 0)

    def test_unfreezable_identity_value_fails_instead_of_using_object_id(self):
        holder = SimpleNamespace()
        with self.assertRaisesRegex(TypeError, "稳定的值语义"):
            acquire_graph_caches(
                holder,
                "_caches",
                {"use_cuda_graph": True, "bad": object()},
                {"a": lambda: None},
            )
        self.assertFalse(hasattr(holder, "_caches"))

    def test_non_mapping_graph_kw_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "Mapping"):
            acquire_graph_caches(
                SimpleNamespace(), "_caches", ["use_cuda_graph"], {"a": lambda: None}
            )


if __name__ == "__main__":
    unittest.main()
