import unittest
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import patch

import torch

from eval.eval_mixed_adaptive import EvalMixedAdaptive


class _FakeTokenizer:
    def __init__(self):
        self.vocab_size = 32
        self.pad_token_id = 0

    def __call__(self, prompt, return_tensors=None):
        del prompt, return_tensors
        return SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]], dtype=torch.long))

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return messages[-1]["content"]

    def encode(self, prompt, add_special_tokens=True):
        del prompt, add_special_tokens
        return [1, 2, 3]


class _FakeConversation:
    def __init__(self):
        self.roles = ("user", "assistant")
        self.messages = []

    def append_message(self, role, content):
        self.messages.append((role, content))

    def get_prompt(self):
        return "\n".join(
            f"{role}: {content}"
            for role, content in self.messages
            if content is not None
        )


class EvalMixedAdaptiveTests(unittest.TestCase):
    def test_eval_uses_adaptive_decoding_and_sets_task_from_flattened_pool(self):
        args = Namespace(
            eval_data_num=2,
            edge_cloud_bandwidth=10.0,
            edge_end_bandwidth=10.0,
            ntt_ms_edge_cloud=0.0,
            ntt_ms_edge_end=0.0,
            transfer_top_k=4,
            use_precise=False,
            use_stochastic_comm=True,
            num_shots=0,
        )

        instance = object.__new__(EvalMixedAdaptive)
        instance.args = args
        instance.device = torch.device("cpu")
        instance.accelerator = SimpleNamespace(device=torch.device("cpu"))
        instance.tokenizer = _FakeTokenizer()
        instance.vocab_size = instance.tokenizer.vocab_size
        instance.pad_token_id = instance.tokenizer.pad_token_id
        instance.model_id = "vicuna"
        instance.all_data = {
            "gsm8k": [{"question": "1+1?"}],
            "xsum": [{"document": "news"}],
        }
        instance.flattened_data = [
            ("gsm8k", {"question": "1+1?"}),
            ("xsum", {"document": "news"}),
        ]
        instance.color_print = lambda *args, **kwargs: None

        calls = []

        def fake_adaptive_decoding(input_ids, **kwargs):
            calls.append(
                {"task": instance.task, "shape": tuple(input_ids.shape), **kwargs}
            )
            return input_ids, {
                "throughput": 1.0,
                "draft_accepted_tokens": 1,
                "draft_generated_tokens": 1,
                "wall_time": 0.1,
                "avg_top_k": kwargs["transfer_top_k"],
            }

        instance.adaptive_decoding = fake_adaptive_decoding

        with (
            patch(
                "eval.eval_mixed_adaptive.random.choice",
                side_effect=instance.flattened_data,
            ),
            patch(
                "eval.eval_mixed_adaptive.random.uniform",
                side_effect=[25.0, 1.0, 30.0, 2.0],
            ),
            patch(
                "eval.eval_mixed_adaptive._get_conversation_template",
                return_value=_FakeConversation(),
            ),
        ):
            instance.eval()

        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["task"], "gsm8k")
        self.assertEqual(calls[1]["task"], "xsum")
        self.assertEqual(calls[0]["transfer_top_k"], 4)
        self.assertTrue(all(call["shape"] == (1, 3) for call in calls))

    def test_build_input_ids_uses_scalar_encode_for_vicuna_style_models(self):
        instance = object.__new__(EvalMixedAdaptive)
        instance.tokenizer = _FakeTokenizer()
        instance.model_id = "vicuna"

        input_ids = instance.build_input_ids("prompt")

        self.assertTrue(torch.equal(input_ids, torch.tensor([[1, 2, 3]])))


if __name__ == "__main__":
    unittest.main()
