import unittest
from argparse import Namespace
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
        self.uncertainty_threshold = 0.8

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

    def _set_uniform_history(self, x):
        steps = max(x.shape[1], 1)
        self.prob_history = torch.zeros((1, steps, self.vocab_size), dtype=torch.float)
        self.prob_history[:, :, 1] = 1.0
        self.logits_history = torch.zeros(
            (1, steps, self.vocab_size), dtype=torch.float
        )
        self.logits_history[:, :, 1] = 8.0

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
            prompt_logits = torch.zeros(
                (prefix.shape[0], prompt_steps, self.vocab_size), dtype=torch.float
            )
            prompt_logits[:, :, 1] = 8.0
            self.prob_history = torch.cat((prompt_probs, self.prob_history), dim=1)
            self.logits_history = torch.cat((prompt_logits, self.logits_history), dim=1)
        self._set_uniform_history(x)
        rebuilt = None if not rebuilt_rows else torch.cat(rebuilt_rows, dim=1)
        return x, rebuilt

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


class _UnexpectedRLAdapter:
    def select_config(self, *args, **kwargs):
        raise AssertionError("RL adapter should not be used in cee_cuhlm")

    def step(self, reward):
        raise AssertionError("RL adapter step should not be used in cee_cuhlm")

    def save(self, throughput):
        return None


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
            uncertainty_threshold=0.8,
            small_draft_threshold=0.8,
            draft_target_threshold=0.8,
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
            *,
            proposer_cache,
            verifier_cache,
            x,
            prefix_len,
            gamma,
            output_device,
            draft_probs_override=None,
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
            *,
            proposer_cache,
            verifier_cache,
            x,
            prefix_len,
            gamma,
            output_device,
            draft_probs_override=None,
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
            *,
            proposer_cache,
            verifier_cache,
            x,
            prefix_len,
            gamma,
            output_device,
            draft_probs_override=None,
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

    def test_cee_cuhlm_does_not_use_stage_helper(self):
        instance = self._make_instance("cee_cuhlm")
        prefix = torch.tensor([[0]], dtype=torch.long)

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CUHLM", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
            patch(
                "src.baselines.resolve_stage_verification",
                side_effect=AssertionError(
                    "resolve_stage_verification should not be used in cee_cuhlm"
                ),
            ),
        ):
            output, metrics = instance.cee_cuhlm(prefix)

        self.assertGreater(output.shape[1], prefix.shape[1])
        self.assertEqual(metrics["little_accepted_tokens"], 0)
        self.assertEqual(metrics["draft_accepted_tokens"], 0)

    def test_cee_cuhlm_ignores_rl_adapter_hooks(self):
        instance = self._make_instance("cee_cuhlm")
        prefix = torch.tensor([[0]], dtype=torch.long)
        instance.rl_adapter = _UnexpectedRLAdapter()
        instance.little_rl_adapter = _UnexpectedRLAdapter()

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CUHLM", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
        ):
            output, _ = instance.cee_cuhlm(prefix)

        self.assertGreater(output.shape[1], prefix.shape[1])


if __name__ == "__main__":
    unittest.main()
