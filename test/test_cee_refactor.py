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

    def calculate_uncertainty(self, logits, M=20, theta_max=2.0, draft_token=None):
        return 1.0

    def determine_transfer_strategy(self, uncertainty, current_probs):
        return True, 2


class _FakeCache:
    def __init__(self, model, temperature, top_k, top_p):
        self.model = model
        self.device = model.device
        self.vocab_size = 4
        self.current_length = 0
        self.prob_history = None
        self.logits_history = None
        self.hidden_states = torch.zeros(1, 1, 1)

    def _forward_with_kvcache(self, x):
        self.current_length = x.shape[1] + 1
        probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float)
        logits = torch.tensor([[0.0, 8.0, -8.0, -8.0]], dtype=torch.float)
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
        steps = max(prefix.shape[1], 1)
        self.prob_history = torch.zeros((1, steps, self.vocab_size), dtype=torch.float)
        self.prob_history[:, :, 1] = 1.0
        self.logits_history = torch.zeros(
            (1, steps, self.vocab_size), dtype=torch.float
        )
        self.logits_history[:, :, 1] = 8.0
        return prefix

    def rollback(self, end_pos):
        self.current_length = end_pos


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


class CeeRefactorTests(unittest.TestCase):
    def _make_instance(self, eval_mode):
        args = Namespace(
            eval_mode=eval_mode,
            edge_cloud_bandwidth=10,
            edge_end_bandwidth=10,
            cloud_end_bandwidth=10,
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
            disable_rl_update=True,
        )
        instance = object.__new__(_TestBaselines)
        instance.args = args
        instance.accelerator = type("Accel", (), {"is_main_process": True})()
        instance.vocab_size = 4
        instance.num_acc_tokens = []
        instance.draft_forward_times = 0
        instance.target_forward_times = 0
        instance.little_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "little"}
        )()
        instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()
        instance.small_draft_adapter = _FakeAdapter()
        instance.draft_target_adapter = _FakeAdapter()
        instance.rl_adapter = None
        instance.little_rl_adapter = None
        return instance

    def test_ceesd_without_arp_uses_stage_helper_twice(self):
        instance = self._make_instance("ceesd_without_arp")
        prefix = torch.tensor([[0]], dtype=torch.long)
        stage_calls = []

        def fake_stage_verify(
            *, proposer_cache, verifier_cache, x, prefix_len, gamma, output_device
        ):
            stage_calls.append(
                (proposer_cache.model.kind, verifier_cache.model.kind, gamma)
            )
            return (
                gamma,
                prefix_len + gamma - 1,
                torch.tensor([[1]], dtype=torch.long),
                True,
            )

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
            patch(
                "src.baselines.resolve_stage_verification",
                side_effect=fake_stage_verify,
            ),
        ):
            instance.ceesd_without_arp(prefix)

        self.assertEqual(stage_calls[0][:2], ("little", "draft"))
        self.assertEqual(stage_calls[1][:2], ("draft", "target"))

    def test_cee_dssd_uses_stage_helper_twice(self):
        instance = self._make_instance("cee_dssd")
        prefix = torch.tensor([[0]], dtype=torch.long)
        stage_calls = []

        def fake_stage_verify(
            *, proposer_cache, verifier_cache, x, prefix_len, gamma, output_device
        ):
            stage_calls.append(
                (proposer_cache.model.kind, verifier_cache.model.kind, gamma)
            )
            return (
                gamma,
                prefix_len + gamma - 1,
                torch.tensor([[1]], dtype=torch.long),
                True,
            )

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
            patch(
                "src.baselines.resolve_stage_verification",
                side_effect=fake_stage_verify,
            ),
        ):
            instance.cee_dssd(prefix)

        self.assertEqual(stage_calls[0][:2], ("little", "draft"))
        self.assertEqual(stage_calls[1][:2], ("draft", "target"))

    def test_cee_dsd_uses_stage_helper_twice(self):
        instance = self._make_instance("cee_dsd")
        prefix = torch.tensor([[0]], dtype=torch.long)
        stage_calls = []

        def fake_stage_verify(
            *, proposer_cache, verifier_cache, x, prefix_len, gamma, output_device
        ):
            stage_calls.append(
                (proposer_cache.model.kind, verifier_cache.model.kind, gamma)
            )
            return (
                gamma,
                prefix_len + gamma - 1,
                torch.tensor([[1]], dtype=torch.long),
                True,
            )

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
            patch(
                "src.baselines.resolve_stage_verification",
                side_effect=fake_stage_verify,
            ),
        ):
            instance.cee_dsd(prefix)

        self.assertEqual(stage_calls[0][:2], ("little", "draft"))
        self.assertEqual(stage_calls[1][:2], ("draft", "target"))


if __name__ == "__main__":
    unittest.main()
