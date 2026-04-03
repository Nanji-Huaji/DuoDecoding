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
        self.transfer_calls = []
        self.simulate_transfer_calls = []

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

    def simulate_transfer(self, size, link_type="edge_cloud", **kwargs):
        self.simulate_transfer_calls.append(
            {"size": size, "link_type": link_type, "kwargs": kwargs}
        )
        return 0.0


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
    instances = []

    def __init__(self, model, temperature, top_k, top_p):
        self.model = model
        self.device = model.device
        self.vocab_size = 4
        self.current_length = 0
        self.prob_history = None
        self.hidden_states = torch.zeros(1, 1, 1)
        self.rollback_calls = []
        _FakeCache.instances.append(self)

    def _forward_with_kvcache(self, x):
        self.current_length = x.shape[1] + 1
        self.hidden_states = torch.zeros(1, 1, 1)
        probs = torch.tensor([[0.0, 1.0, 0.0, 0.0]], dtype=torch.float)
        if self.prob_history is None:
            self.prob_history = probs.unsqueeze(1)
        else:
            self.prob_history = torch.cat(
                (self.prob_history, probs.unsqueeze(1)), dim=1
            )
        return probs

    def generate(self, prefix, gamma):
        self.current_length = prefix.shape[1] + gamma
        steps = max(prefix.shape[1], 1) + gamma
        self.prob_history = torch.zeros((1, steps, self.vocab_size), dtype=torch.float)
        self.prob_history[:, :, 1] = 1.0
        return prefix

    def rollback(self, end_pos):
        self.current_length = end_pos
        self.rollback_calls.append(end_pos)


class AdaptiveTriDecodingTests(unittest.TestCase):
    def setUp(self):
        _FakeCache.instances = []
        args = Namespace(
            eval_mode="adaptive_tridecoding",
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
        self.instance.small_draft_adapter = _FakeAdapter()
        self.instance.draft_target_adapter = _FakeAdapter()
        self.instance.rl_adapter = None
        self.instance.little_rl_adapter = None

    def test_adaptive_tridecoding_uses_stage_verification_helper_for_both_layers(self):
        prefix = torch.tensor([[0]], dtype=torch.long)
        stage_calls = []

        def fake_stage_verify(
            proposer_cache,
            verifier_cache,
            x,
            prefix_len,
            gamma,
            *,
            output_device,
        ):
            stage_calls.append(
                {
                    "proposer": proposer_cache.model.kind,
                    "verifier": verifier_cache.model.kind,
                    "prefix_len": prefix_len,
                    "gamma": gamma,
                }
            )
            token = torch.tensor([[1]], dtype=torch.long, device=output_device)
            return gamma, prefix_len + gamma - 1, token, True

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator),
            patch(
                "src.baselines.resolve_stage_verification",
                side_effect=fake_stage_verify,
            ),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
        ):
            output, metrics = self.instance.adaptive_tridecoding(prefix)

        self.assertEqual(output.shape[0], 1)
        self.assertGreater(output.shape[1], prefix.shape[1])
        self.assertEqual(metrics["little_accepted_tokens"], 1)
        self.assertGreaterEqual(metrics["draft_accepted_tokens"], 1)
        self.assertEqual(len(stage_calls), 2)
        self.assertEqual(
            stage_calls[0],
            {
                "proposer": "little",
                "verifier": "draft",
                "prefix_len": 1,
                "gamma": 1,
            },
        )
        self.assertEqual(stage_calls[1]["proposer"], "draft")
        self.assertEqual(stage_calls[1]["verifier"], "target")
        self.assertEqual(stage_calls[1]["prefix_len"], 1)
        self.assertGreaterEqual(stage_calls[1]["gamma"], 1)

    def test_adaptive_tridecoding_transfers_vectorized_payloads(self):
        prefix = torch.tensor([[0]], dtype=torch.long)
        fake_comm = _FakeCommSimulator()

        def fake_stage_verify(
            proposer_cache,
            verifier_cache,
            x,
            prefix_len,
            gamma,
            *,
            output_device,
        ):
            token = torch.tensor([[1]], dtype=torch.long, device=output_device)
            return gamma, prefix_len + gamma - 1, token, True

        with (
            patch("src.baselines.KVCacheModel", _FakeCache),
            patch("src.baselines.CommunicationSimulator", return_value=fake_comm),
            patch(
                "src.baselines.resolve_stage_verification",
                side_effect=fake_stage_verify,
            ),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
        ):
            self.instance.adaptive_tridecoding(prefix)

        payload_calls = [
            call for call in fake_comm.transfer_calls if call["probs"] is not None
        ]
        self.assertEqual(len(payload_calls), 2)

        edge_end_payload = payload_calls[0]
        self.assertEqual(edge_end_payload["link_type"], "edge_end")
        self.assertTrue(torch.equal(edge_end_payload["tokens"], torch.tensor([[1]])))
        self.assertTrue(torch.equal(edge_end_payload["probs"], torch.tensor([[1.0]])))

        edge_cloud_payload = payload_calls[1]
        self.assertEqual(edge_cloud_payload["link_type"], "edge_cloud")
        self.assertTrue(
            torch.equal(edge_cloud_payload["tokens"], torch.tensor([[1, 1, 1]]))
        )
        self.assertTrue(
            torch.equal(edge_cloud_payload["probs"], torch.tensor([[1.0, 1.0, 1.0]]))
        )


if __name__ == "__main__":
    unittest.main()
