"""B19 CUDA Graph 接线的 CPU 侧回归测试。

图本身只能在 GPU 上捕获，但真正出过事故的是**接线**：构造 KVCacheModel 时
漏透传 use_cuda_graph / verify_graph_sizes / graph_len_budget，以及忘了跨样本
复用导致每样本重捕获（~534ms/样本）。这两件事都不需要 GPU 就能验证——这里用
假 cache 记录构造参数与 reset 调用，锁住：

1. `graph_mode_cache_kwargs` 的开关语义、档位阶梯与桶长公式；
2. `acquire_graph_caches` 的复用/重建语义；
3. 三层缓存入口（tridecoding 与三个 CEE 方法共用）确实透传图参数并复用；
4. 投机解码入口（engine 两个方法共用）draft 开图、target 保持 eager 并复用；
5. AR 入口（engine `autoregressive_sampling`）small/草稿开图并跨样本复用、
   large/target 保持 eager（kwargs 与接线前逐字节一致）；
6. `import src.engine` 不再有全局降噪副作用，而 `configure_verbosity()`
   仍同时压制 transformers 日志与 warnings。
"""

import subprocess
import sys
import textwrap
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.engine import Decoding
from src.graph_decode import acquire_graph_caches, graph_mode_cache_kwargs


def _graph_args(**overrides):
    base = dict(use_cuda_graph=True, graph_verify_sizes=None, max_tokens=128)
    base.update(overrides)
    return Namespace(**base)


class _FakeCache:
    """记录构造参数与 reset 次数，替代 KVCacheModel。"""

    created: list = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.kwargs = kwargs
        self.vocab_size = None
        self.resets = 0
        type(self).created.append(self)

    def reset_for_new_sample(self):
        self.resets += 1


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class _TestDecoding(Decoding):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class GraphModeCacheKwargsTests(unittest.TestCase):
    def test_disabled_returns_empty_kwargs(self):
        self.assertEqual(
            graph_mode_cache_kwargs(_graph_args(use_cuda_graph=False), cap=8), {}
        )
        # 最小 Namespace（无 use_cuda_graph 字段）同样按关闭处理
        self.assertEqual(graph_mode_cache_kwargs(Namespace(max_tokens=8), cap=8), {})

    def test_enabled_returns_use_graph_sizes_and_budget(self):
        kwargs = graph_mode_cache_kwargs(_graph_args(max_tokens=64), cap=20)
        self.assertTrue(kwargs["use_cuda_graph"])
        self.assertEqual(kwargs["graph_len_budget"], 64 + 256)
        sizes = kwargs["verify_graph_sizes"]
        self.assertEqual(sizes, sorted(sizes))
        self.assertTrue(all(2 <= size <= 128 for size in sizes))
        self.assertGreaterEqual(sizes[-1], 20)

    def test_explicit_ladder_respected_and_topped_up_to_cap(self):
        kwargs = graph_mode_cache_kwargs(
            _graph_args(graph_verify_sizes="3, 6, 9"), cap=12
        )
        self.assertEqual(kwargs["verify_graph_sizes"], [3, 6, 9, 16])


class AcquireGraphCachesTests(unittest.TestCase):
    def test_eager_mode_rebuilds_and_stores_nothing(self):
        holder = Namespace()
        built = []
        builders = {"a": lambda: built.append(_FakeCache("m", 1.0, 0, 0.0)) or built[-1]}

        first = acquire_graph_caches(holder, "_caches", {}, builders)
        second = acquire_graph_caches(holder, "_caches", {}, builders)

        self.assertEqual(len(built), 2)
        self.assertIsNot(first["a"], second["a"])
        self.assertFalse(hasattr(holder, "_caches"))

    def test_graph_mode_reuses_same_objects_and_resets(self):
        holder = Namespace()
        _FakeCache.created = []
        graph_kw = {"use_cuda_graph": True}
        builders = {"a": lambda: _FakeCache("m", 1.0, 0, 0.0)}

        first = acquire_graph_caches(holder, "_caches", graph_kw, builders)
        second = acquire_graph_caches(holder, "_caches", graph_kw, builders)

        self.assertIs(first["a"], second["a"])
        self.assertEqual(len(_FakeCache.created), 1)
        self.assertEqual(second["a"].resets, 1)


class ThreeLayerCacheWiringTests(unittest.TestCase):
    """tridecoding 与 ceesd_without_arp / cee_dssd / cee_dsd 共用这一入口。"""

    def _instance(self, use_cuda_graph):
        instance = object.__new__(_TestBaselines)
        instance.args = Namespace(
            temp=1.0,
            top_k=0,
            top_p=0.0,
            max_tokens=64,
            gamma1=4,
            gamma2=8,
            use_cuda_graph=use_cuda_graph,
            graph_verify_sizes=None,
        )
        instance.little_model = "little"
        instance.draft_model = "draft"
        instance.target_model = "target"
        instance.vocab_size = 32
        return instance

    def test_graph_off_builds_fresh_caches_without_graph_kwargs(self):
        instance = self._instance(False)
        with patch("src.baselines.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            first = instance.build_adaptive_tridecoding_caches(5)
            second = instance.build_adaptive_tridecoding_caches(5)

        self.assertEqual(len(_FakeCache.created), 6)
        self.assertIsNot(first["little"], second["little"])
        for cache in first.values():
            self.assertNotIn("use_cuda_graph", cache.kwargs)
            self.assertEqual(cache.vocab_size, 32)

    def test_graph_on_passes_graph_kwargs_and_reuses(self):
        instance = self._instance(True)
        with patch("src.baselines.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            first = instance.build_adaptive_tridecoding_caches(5)
            built_first_call = len(_FakeCache.created)
            second = instance.build_adaptive_tridecoding_caches(5)

        self.assertEqual(built_first_call, 3)
        self.assertEqual(len(_FakeCache.created), 3)  # 第二次没新建
        self.assertIs(first["little"], second["little"])
        for cache in second.values():
            self.assertEqual(cache.resets, 1)
        for cache in first.values():
            self.assertTrue(cache.kwargs["use_cuda_graph"])
            self.assertEqual(cache.kwargs["graph_len_budget"], 64 + 256)
            # γ1+γ2+4 = 16 必须被档位覆盖
            self.assertGreaterEqual(cache.kwargs["verify_graph_sizes"][-1], 16)
        self.assertEqual(first["little"].top_k, 5)
        self.assertEqual(first["draft"].top_k, 5)
        self.assertEqual(first["target"].top_k, 0)
        self.assertEqual(first["target"].top_p, 0)


class SpeculativeCacheWiringTests(unittest.TestCase):
    """engine 的 speculative_decoding / _with_bandwidth 共用这一入口。"""

    def _instance(self):
        instance = object.__new__(_TestDecoding)
        instance.args = Namespace(temp=1.0, top_k=3, top_p=0.5)
        instance.draft_model = "draft"
        instance.target_model = "target"
        instance.vocab_size = 32
        return instance

    def test_graph_off_builds_two_fresh_eager_caches(self):
        instance = self._instance()
        with patch("src.engine.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            first = instance._acquire_speculative_caches("_spec_caches", {}, 3)
            second = instance._acquire_speculative_caches("_spec_caches", {}, 3)

        self.assertEqual(len(_FakeCache.created), 4)
        self.assertIsNot(first["draft"], second["draft"])
        self.assertNotIn("use_cuda_graph", first["draft"].kwargs)

    def test_graph_on_reuses_and_keeps_target_eager(self):
        instance = self._instance()
        graph_kw = graph_mode_cache_kwargs(_graph_args(max_tokens=64), cap=9)
        with patch("src.engine.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            first = instance._acquire_speculative_caches(
                "_spec_caches", graph_kw, 3
            )
            second = instance._acquire_speculative_caches(
                "_spec_caches", graph_kw, 3
            )

        self.assertEqual(len(_FakeCache.created), 2)
        self.assertIs(first["draft"], second["draft"])
        self.assertTrue(first["draft"].kwargs["use_cuda_graph"])
        self.assertEqual(first["draft"].kwargs["graph_len_budget"], 64 + 256)
        # target 每步只做单 token 前向，保持 eager（不开图、不占显存）
        self.assertNotIn("use_cuda_graph", first["target"].kwargs)
        self.assertEqual(first["target"].resets, 1)


class DraftTargetCacheWiringTests(unittest.TestCase):
    """adaptive_decoding 的缓存入口：草稿开图、目标 eager、跨样本复用。"""

    def _instance(self, use_cuda_graph):
        instance = object.__new__(_TestBaselines)
        instance.args = Namespace(
            temp=1.0,
            top_k=0,
            top_p=0.0,
            max_tokens=64,
            gamma=5,
            use_cuda_graph=use_cuda_graph,
            graph_verify_sizes=None,
        )
        instance.draft_model = "draft"
        instance.target_model = "target"
        instance.vocab_size = 32
        return instance

    def test_graph_off_builds_two_fresh_eager_caches(self):
        instance = self._instance(False)
        with patch("src.baselines.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            first_draft, _ = instance._acquire_draft_target_caches(
                "_ad_caches", {}, 5, 0, 0.0
            )
            second_draft, _ = instance._acquire_draft_target_caches(
                "_ad_caches", {}, 5, 0, 0.0
            )

        self.assertEqual(len(_FakeCache.created), 4)
        self.assertIsNot(first_draft, second_draft)
        self.assertNotIn("use_cuda_graph", first_draft.kwargs)

    def test_graph_on_draft_graphed_target_eager_and_reused(self):
        instance = self._instance(True)
        graph_kw = graph_mode_cache_kwargs(_graph_args(max_tokens=64), cap=9)
        with patch("src.baselines.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            first_draft, first_target = instance._acquire_draft_target_caches(
                "_ad_caches", graph_kw, 5, 0, 0.0
            )
            second_draft, second_target = instance._acquire_draft_target_caches(
                "_ad_caches", graph_kw, 5, 0, 0.0
            )

        self.assertEqual(len(_FakeCache.created), 2)
        self.assertIs(first_draft, second_draft)
        self.assertTrue(first_draft.kwargs["use_cuda_graph"])
        self.assertEqual(first_draft.kwargs["graph_len_budget"], 64 + 256)
        self.assertGreaterEqual(first_draft.kwargs["verify_graph_sizes"][-1], 9)
        self.assertNotIn("use_cuda_graph", first_target.kwargs)
        self.assertEqual(first_target.resets, 1)
        self.assertIs(first_target, second_target)

    def test_compression_applies_to_draft_only(self):
        instance = self._instance(True)
        with patch("src.baselines.KVCacheModel", _FakeCache):
            _FakeCache.created = []
            draft, target = instance._acquire_draft_target_caches(
                "_ad_caches", {}, 7, 0, 0.0
            )

        self.assertEqual(draft.top_k, 7)
        self.assertEqual(target.top_k, 0)
        self.assertEqual(target.top_p, 0.0)


class _FakeEmbedding:
    def __init__(self, vocab_size):
        self.weight = torch.zeros((vocab_size, 4))


class _FakeModel:
    """AR 接线测试所需的最小模型接口（device + 词嵌入）。"""

    device = "cpu"

    def __init__(self, vocab_size=32):
        self._embeddings = _FakeEmbedding(vocab_size)

    def get_input_embeddings(self):
        return self._embeddings


class _FakeCudaEvent:
    """CPU 上替代 torch.cuda.Event（测试只关心 cache 怎么被构造）。"""

    def __init__(self, enable_timing=False):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        return None

    def elapsed_time(self, end):
        return 0.0


class AutoregressiveCacheWiringTests(unittest.TestCase):
    """engine 的 autoregressive_sampling：草稿开单步图并复用，target 保持 eager。"""

    def _instance(self, eval_mode, use_cuda_graph):
        instance = object.__new__(_TestDecoding)
        instance.args = Namespace(
            eval_mode=eval_mode,
            temp=1.0,
            top_k=3,
            top_p=0.5,
            max_tokens=0,
            use_cuda_graph=use_cuda_graph,
            graph_verify_sizes=None,
        )
        instance.draft_model = _FakeModel()
        instance.target_model = _FakeModel()
        instance.vocab_size = 32
        return instance

    def _run(self, instance, prefix):
        with (
            patch("src.engine.KVCacheModel", _FakeCache),
            patch("src.engine.torch.cuda.Event", _FakeCudaEvent),
            patch("src.engine.torch.cuda.current_stream"),
            patch("src.engine.torch.cuda.synchronize"),
        ):
            return instance.autoregressive_sampling(prefix)

    def test_small_mode_graphed_and_reused_across_samples(self):
        instance = self._instance("small", True)
        prefix = torch.tensor([[0, 1, 2]])
        _FakeCache.created = []
        first, metrics = self._run(instance, prefix)
        second, metrics_again = self._run(instance, prefix)

        self.assertEqual(len(_FakeCache.created), 1)  # 第二次复用，不重捕获
        cache = _FakeCache.created[0]
        self.assertIs(first, second)
        self.assertEqual(cache.resets, 1)
        self.assertTrue(cache.kwargs["use_cuda_graph"])
        self.assertEqual(cache.kwargs["graph_len_budget"], 256)
        self.assertGreaterEqual(cache.kwargs["verify_graph_sizes"][-1], 5)
        # small 模式的前向次数键仍是 draft（B39）
        self.assertEqual(metrics["draft_forward_times"], 0)
        self.assertEqual(metrics_again["draft_forward_times"], 0)

    def test_large_mode_keeps_target_eager(self):
        instance = self._instance("large", True)
        _FakeCache.created = []
        self._run(instance, torch.tensor([[0, 1, 2]]))

        cache = _FakeCache.created[0]
        # target 保持 eager：kwargs 里绝不能出现图参数
        self.assertNotIn("use_cuda_graph", cache.kwargs)
        self.assertEqual(cache.top_k, 3)
        self.assertEqual(cache.top_p, 0.5)

    def test_graph_disabled_matches_legacy_construction(self):
        instance = self._instance("small", False)
        _FakeCache.created = []
        self._run(instance, torch.tensor([[0, 1, 2]]))
        self._run(instance, torch.tensor([[0, 1, 2]]))

        # 关图红线：每次重建，且传给 KVCacheModel 的 kwargs 为空
        self.assertEqual(len(_FakeCache.created), 2)
        self.assertEqual(_FakeCache.created[0].kwargs, {})


class VerbosityTests(unittest.TestCase):
    """问题 1：import 不再全局降噪，configure_verbosity() 仍执行原两句。"""

    def test_import_quietly_defers_to_configure_verbosity(self):
        script = textwrap.dedent(
            """
            import importlib
            import warnings

            import transformers

            import src.engine as engine

            calls = []
            transformers.utils.logging.set_verbosity = (
                lambda *a, **k: calls.append("verbosity")
            )
            warnings.filterwarnings = lambda *a, **k: calls.append("filter")

            importlib.reload(engine)
            print(",".join(calls), end="|")

            del calls[:]
            engine.configure_verbosity()
            print(",".join(sorted(set(calls))))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        on_import, on_configure = result.stdout.strip().split("|")
        self.assertEqual(on_import, "")
        self.assertEqual(on_configure, "filter,verbosity")


if __name__ == "__main__":
    unittest.main()
