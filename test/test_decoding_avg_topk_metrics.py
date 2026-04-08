import unittest
from argparse import Namespace
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.utils import rebuild_topk_uniform_probs, sample


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None

    def color_print(self, *args, **kwargs):
        return None


class _FakeCudaEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        return None

    def elapsed_time(self, other):
        return 0.0


class _FakeCommSimulator:
    def __init__(self, *args, **kwargs):
        self.edge_cloud_comm_time = 0.0
        self.edge_end_comm_time = 0.0
        self.edge_cloud_data = 0
        self.edge_end_data = 0
        self.cloud_end_data = 0
        self.total_comm_energy = 0.0
        self.connect_times = {}
        self.edge_cloud_bandwidth_history = []
        self.edge_cloud_topk_history = []
        self.edge_cloud_draft_len_history = []
        self.bandwidth_edge_cloud = 10
        self.bandwidth_edge_end = 10
        self.ntt_edge_cloud = 0
        self.ntt_edge_end = 0

    def transfer(self, tokens, probs, link_type="edge_cloud", *args, **kwargs):
        return 0.0

    def simulate_transfer(self, size, link_type="edge_cloud", **kwargs):
        return 0.0

    def send_reject_message(self, link_type):
        return None

    def send_accept_message(self, link_type):
        return None


class _FakeAdapter:
    def __init__(self):
        self.device = torch.device("meta")
        self.threshold = 0.5
        self.step_acc_probs = []

    def reset_step(self):
        self.step_acc_probs = []

    def predict(self, hidden_states):
        self.step_acc_probs.append(1.0)
        return True


class _FakeCache:
    def __init__(self, model, temperature, top_k, top_p):
        self.model = model
        self.device = model.device
        self.vocab_size = 4
        self.current_length = 0
        self.prob_history = None
        self.logits_history = None
        self.hidden_states = torch.zeros(1, 1, 1)

    def _uniform_probs(self):
        probs = torch.zeros((1, self.vocab_size), dtype=torch.float)
        probs[:, 1] = 1.0
        return probs

    def _uniform_logits(self):
        logits = torch.full((1, self.vocab_size), -8.0, dtype=torch.float)
        logits[:, 1] = 8.0
        return logits

    def _set_uniform_history(self, x):
        steps = max(x.shape[1], 1)
        self.prob_history = torch.zeros((1, steps, self.vocab_size), dtype=torch.float)
        self.prob_history[:, :, 1] = 1.0
        self.logits_history = torch.full(
            (1, steps, self.vocab_size), -8.0, dtype=torch.float
        )
        self.logits_history[:, :, 1] = 8.0

    def _forward_with_kvcache(self, x):
        self.current_length = x.shape[1] + 1
        probs = self._uniform_probs()
        logits = self._uniform_logits()
        if self.prob_history is None:
            self.prob_history = probs.unsqueeze(1)
            self.logits_history = logits.unsqueeze(1)
        else:
            self.prob_history = torch.cat(
                (self.prob_history, probs.unsqueeze(1)), dim=1
            )
            self.logits_history = torch.cat(
                (self.logits_history, logits.unsqueeze(1)), dim=1
            )
        return probs

    def generate(self, prefix, gamma):
        self.current_length = prefix.shape[1] + gamma
        if gamma > 0:
            extra = torch.ones(
                (prefix.shape[0], gamma), dtype=prefix.dtype, device=prefix.device
            )
            prefix = torch.cat((prefix, extra), dim=1)
        self._set_uniform_history(prefix)
        return prefix

    def generate_with_rebuilt_topk(self, prefix, gamma, proposal_top_k):
        if gamma <= 0:
            return prefix.clone(), None
        x = prefix.clone()
        rebuilt_rows = []
        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
            rebuilt_rows.append(rebuilt_q.unsqueeze(1))
            next_tok = sample(rebuilt_q)
            x = torch.cat((x, next_tok), dim=1)
        prompt_steps = max(prefix.shape[1] - 1, 0)
        if prompt_steps > 0:
            prompt_probs = torch.zeros(
                (prefix.shape[0], prompt_steps, self.vocab_size), dtype=torch.float
            )
            prompt_probs[:, :, 1] = 1.0
            prompt_logits = torch.full(
                (prefix.shape[0], prompt_steps, self.vocab_size),
                -8.0,
                dtype=torch.float,
            )
            prompt_logits[:, :, 1] = 8.0
            self.prob_history = torch.cat((prompt_probs, self.prob_history), dim=1)
            self.logits_history = torch.cat((prompt_logits, self.logits_history), dim=1)
        self._set_uniform_history(x)
        rebuilt = None if not rebuilt_rows else torch.cat(rebuilt_rows, dim=1)
        return x, rebuilt

    def rollback(self, end_pos):
        self.current_length = end_pos


class DecodingAvgTopKMetricTests(unittest.TestCase):
    def setUp(self):
        args = Namespace(
            eval_mode="unit",
            edge_cloud_bandwidth=10,
            edge_end_bandwidth=10,
            cloud_end_bandwidth=10,
            max_tokens=2,
            temp=1.0,
            top_k=0,
            top_p=0.0,
            batch_delay=0.0,
            seed=0,
            exp_name="test",
            eval_dataset="unit",
            little_model="little",
            draft_model="draft",
            target_model="target",
            gamma=1,
            gamma1=1,
            gamma2=1,
            use_early_stopping=False,
            dump_network_stats=False,
            disable_rl_update=True,
        )
        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = 4
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.little_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "little"}
        )()
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()
        self.instance.adapter = _FakeAdapter()
        self.instance.small_draft_adapter = _FakeAdapter()
        self.instance.draft_target_adapter = _FakeAdapter()
        self.instance.rl_adapter = None
        self.instance.little_rl_adapter = None

    def _enter_common_patches(self, stack: ExitStack):
        stack.enter_context(patch("src.baselines.KVCacheModel", _FakeCache))
        stack.enter_context(
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator)
        )
        stack.enter_context(patch("src.baselines.torch.cuda.Event", _FakeCudaEvent))
        stack.enter_context(
            patch("src.baselines.torch.cuda.current_stream", return_value=None)
        )
        stack.enter_context(
            patch("src.baselines.torch.cuda.synchronize", return_value=None)
        )

    def test_dssd_reports_average_active_top_k(self):
        prefix = torch.tensor([[0]], dtype=torch.long)

        with ExitStack() as stack:
            self._enter_common_patches(stack)
            output, metrics = self.instance.dist_split_spec(prefix, transfer_top_k=5)

        self.assertGreater(output.shape[1], prefix.shape[1])
        self.assertEqual(metrics["avg_top_k"], 5)
        self.assertEqual(metrics["avg_draft_len"], 1)

    def test_adaptive_decoding_reports_average_active_top_k(self):
        prefix = torch.tensor([[0]], dtype=torch.long)

        with ExitStack() as stack:
            self._enter_common_patches(stack)
            output, metrics = self.instance.adaptive_decoding(prefix, transfer_top_k=7)

        self.assertGreater(output.shape[1], prefix.shape[1])
        self.assertEqual(metrics["avg_top_k"], 7)
        self.assertEqual(metrics["avg_draft_len"], 1)

    def test_tridecoding_reports_average_active_top_k(self):
        prefix = torch.tensor([[0]], dtype=torch.long)

        def fake_verify(
            *,
            draft_model_cache,
            target_model_cache,
            x,
            prefix_len,
            gamma,
            draft_probs_override=None,
        ):
            actual_gamma = max(x.shape[1] - prefix_len, 0)
            acceptance = SimpleNamespace(
                n=prefix_len + actual_gamma - 1, accepted_count=actual_gamma
            )
            inputs = SimpleNamespace(
                actual_gamma=actual_gamma,
                draft_probs_batch=torch.ones(
                    (1, max(actual_gamma, 1), self.instance.vocab_size),
                    dtype=torch.float,
                ),
            )
            return inputs, acceptance

        with ExitStack() as stack:
            self._enter_common_patches(stack)
            stack.enter_context(
                patch(
                    "src.baselines.verify_draft_sequence_result",
                    side_effect=fake_verify,
                )
            )
            output, metrics = self.instance.tridecoding(prefix, transfer_top_k=9)

        self.assertGreater(output.shape[1], prefix.shape[1])
        self.assertEqual(metrics["avg_top_k"], 9)
        self.assertEqual(metrics["avg_draft_len"], 1)


if __name__ == "__main__":
    unittest.main()
