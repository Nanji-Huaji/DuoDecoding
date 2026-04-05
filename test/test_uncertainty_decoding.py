import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from src.baselines import Baselines


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class _FakeCudaEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        return None

    def elapsed_time(self, other):
        return 0.0


class _FakeCommSimulator:
    instances = []

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
        self.transfer_calls = []
        self.accept_messages = 0
        self.reject_messages = 0
        _FakeCommSimulator.instances.append(self)

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "tokens": None if tokens is None else tokens.clone(),
                "probs": None if probs is None else probs.clone(),
                "link_type": link_type,
                "kwargs": kwargs,
            }
        )
        return 0.0

    def calculate_uncertainty(self, logits, M=20, theta_max=2.0, draft_token=None):
        return 1.0

    def determine_transfer_strategy(self, uncertainty, current_probs):
        return True, 2

    def _get_current_probs(self, prob_history):
        return prob_history[0, -1, :]

    def _apply_top_k_compression(self, probs, k):
        return probs

    def send_accept_message(self, linktype):
        self.accept_messages += 1

    def send_reject_message(self, linktype):
        self.reject_messages += 1


class _FakeCache:
    instances = []

    def __init__(self, model, temperature, top_k, top_p):
        self.model = model
        self.device = model.device
        self.vocab_size = 2
        self.rollback_calls = []
        self.generate_calls = []
        self.prob_history = None
        self.logits_history = None
        _FakeCache.instances.append(self)

    def generate(self, prefix, gamma):
        self.generate_calls.append((prefix.clone(), gamma))
        if self.model.kind == "draft":
            self.prob_history = torch.tensor(
                [[[0.9, 0.1], [1.0, 0.0]]], dtype=torch.float
            )
            self.logits_history = torch.tensor(
                [[[0.0, 0.0], [6.0, -6.0]]], dtype=torch.float
            )
            return torch.cat(
                (prefix, torch.tensor([[0]], dtype=prefix.dtype, device=prefix.device)),
                dim=1,
            )

        self.prob_history = torch.tensor([[[0.1, 0.9], [0.0, 1.0]]], dtype=torch.float)
        self.logits_history = torch.tensor(
            [[[0.0, 0.0], [-6.0, 6.0]]], dtype=torch.float
        )
        return prefix

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)


class UncertaintyDecodingTests(unittest.TestCase):
    def setUp(self):
        _FakeCache.instances = []
        _FakeCommSimulator.instances = []
        args = Namespace(
            eval_mode="uncertainty_decoding",
            edge_cloud_bandwidth=10,
            max_tokens=1,
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
        )

        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = 2
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()

    def test_uncertainty_decoding_rejection_path_uses_helper_flow(self):
        prefix = torch.tensor([[7]], dtype=torch.long)

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CUHLM", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
        ):
            output, metrics = self.instance.uncertainty_decoding(prefix)

        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.equal(output[:, :1], prefix))
        self.assertEqual(metrics["draft_generated_tokens"], 1)
        self.assertEqual(metrics["draft_accepted_tokens"], 0)
        self.assertEqual(metrics["target_forward_times"], 1)
        self.assertEqual(self.instance.num_acc_tokens, [])

        approx_cache, target_cache = _FakeCache.instances
        self.assertEqual(approx_cache.rollback_calls, [1])
        self.assertEqual(target_cache.rollback_calls, [1])
        self.assertEqual(len(approx_cache.generate_calls), 1)
        self.assertEqual(len(target_cache.generate_calls), 1)

        comm_simulator = _FakeCommSimulator.instances[0]
        self.assertEqual(comm_simulator.reject_messages, 1)
        self.assertEqual(comm_simulator.accept_messages, 0)


if __name__ == "__main__":
    unittest.main()
