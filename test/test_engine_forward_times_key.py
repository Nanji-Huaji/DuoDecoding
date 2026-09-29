"""B39 回归测试：autoregressive_sampling 的前向计数键随实际模型身份。

small 模式跑 draft 模型，必须记 ``draft_forward_times``；large 模式跑 target
模型，记 ``target_forward_times``。此前两者都写 ``target_forward_times``，
在 small 基线上把 draft 前向误标成 target 前向。
"""

import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from src.engine import Decoding


class _TestDecoding(Decoding):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class _FakeEvent:
    def __init__(self, enable_timing=True):
        del enable_timing

    def record(self, stream=None):
        del stream

    def elapsed_time(self, other):
        del other
        return 0.0


class _FakeModel:
    def __init__(self):
        self.device = torch.device("cpu")


class _FakeCacheModel:
    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        del model, temperature, top_k, top_p, kwargs
        self.vocab_size = 8
        self.device = torch.device("cpu")

    def generate(self, x, n):
        pad = torch.zeros((x.shape[0], n), dtype=x.dtype)
        return torch.cat((x, pad), dim=1)


class ForwardTimesKeyTests(unittest.TestCase):
    def _run(self, eval_mode):
        instance = object.__new__(_TestDecoding)
        instance.args = Namespace(
            eval_mode=eval_mode,
            temp=0.0,
            top_k=0,
            top_p=1.0,
            max_tokens=3,
            batch_delay=0,
        )
        instance.vocab_size = 8
        instance.draft_model = _FakeModel()
        instance.target_model = _FakeModel()
        instance.validate_input_ids = lambda *a, **k: None

        prefix = torch.tensor([[1]], dtype=torch.long)
        with (
            patch("src.engine.KVCacheModel", _FakeCacheModel),
            patch.object(torch.cuda, "Event", _FakeEvent),
            patch.object(torch.cuda, "current_stream", lambda *a, **k: None),
            patch.object(torch.cuda, "synchronize", lambda *a, **k: None),
        ):
            _, metrics = instance.autoregressive_sampling(prefix)

        return metrics

    def test_small_mode_reports_draft_forward_times(self):
        metrics = self._run("small")
        self.assertEqual(metrics["draft_forward_times"], 3)
        self.assertEqual(metrics["target_forward_times"], 0)

    def test_large_mode_reports_target_forward_times(self):
        metrics = self._run("large")
        self.assertEqual(metrics["target_forward_times"], 3)
        self.assertEqual(metrics["draft_forward_times"], 0)


if __name__ == "__main__":
    unittest.main()
