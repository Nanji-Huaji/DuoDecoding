import unittest
from argparse import Namespace
from unittest.mock import call, patch

from exp import EvalMode, create_config
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


class AdaptiveDecodingResourceResolutionTests(unittest.TestCase):
    def test_create_config_resolves_single_stage_head_and_main_rl(self):
        with (
            patch(
                "exp.resolve_acc_head_path", return_value="resolved/head"
            ) as resolve_head,
            patch("exp.get_rl_agent_spec") as get_spec,
        ):
            get_spec.return_value = type(
                "Spec",
                (),
                {
                    "latest_path": "resolved/main/latest.pth",
                    "best_path": "resolved/main/best.pth",
                },
            )()

            config = create_config(
                eval_mode=EvalMode.adaptive_decoding,
                draft_model="llama-68m",
                target_model="llama-2-13b",
                little_model="llama-68m",
                use_rl_adapter=True,
            )

        self.assertEqual(config["acc_head_path"], "resolved/head")
        self.assertEqual(config["main_rl_path"], "resolved/main/latest.pth")
        self.assertEqual(config["main_rl_best_path"], "resolved/main/best.pth")
        self.assertEqual(config["small_draft_acc_head_path"], "")
        self.assertEqual(config["draft_target_acc_head_path"], "")
        self.assertEqual(config["little_rl_path"], "")
        self.assertEqual(config["little_rl_best_path"], "")
        resolve_head.assert_called_once_with("llama-68m", "llama-2-13b")
        get_spec.assert_called_once()
        _, kwargs = get_spec.call_args
        self.assertEqual(kwargs["little_model"], None)
        self.assertEqual(kwargs["draft_model"], "llama-68m")
        self.assertEqual(kwargs["target_model"], "llama-2-13b")

    def test_create_config_keeps_explicit_main_rl_best_path(self):
        with (
            patch("exp.resolve_acc_head_path", return_value="resolved/head"),
            patch("exp.get_rl_agent_spec") as get_spec,
        ):
            config = create_config(
                eval_mode=EvalMode.adaptive_decoding,
                draft_model="llama-68m",
                target_model="llama-2-13b",
                main_rl_path="custom/latest.pth",
                main_rl_best_path="custom/best.pth",
            )

        self.assertEqual(config["main_rl_path"], "custom/latest.pth")
        self.assertEqual(config["main_rl_best_path"], "custom/best.pth")
        get_spec.assert_not_called()


class AdaptiveDecodingRLInitTests(unittest.TestCase):
    def test_adaptive_decoding_only_initializes_main_rl_adapter(self):
        args = Namespace(
            eval_mode="adaptive_decoding",
            use_rl_adapter=True,
            little_model="llama-68m",
            draft_model="llama-68m",
            target_model="llama-2-13b",
            main_rl_path="main/latest.pth",
            main_rl_best_path="main/best.pth",
            little_rl_path="little/latest.pth",
            little_rl_best_path="little/best.pth",
        )

        with (
            patch("src.baselines.Decoding.__init__", return_value=None),
            patch("src.baselines.get_rl_agent_spec") as get_spec,
            patch("src.baselines.RLNetworkAdapter") as rl_adapter,
            patch("src.baselines.resolve_legacy_rl_agent_load_path", return_value=None),
        ):
            get_spec.return_value = type(
                "Spec",
                (),
                {
                    "latest_path": "resolved/latest.pth",
                    "best_path": "resolved/best.pth",
                    "agent_name": "rl_adapter_main",
                    "threshold_candidates": [0.1, 0.2],
                },
            )()

            instance = _TestBaselines(args)

        self.assertIsNotNone(instance.rl_adapter)
        self.assertIsNone(instance.little_rl_adapter)
        self.assertEqual(get_spec.call_count, 1)
        self.assertEqual(rl_adapter.call_count, 1)


class AdaptiveTriDecodingAcceptanceHeadInitTests(unittest.TestCase):
    def test_cee_sd_initializes_both_acceptance_adapters(self):
        args = Namespace(
            eval_mode="cee_sd",
            small_draft_threshold=0.1,
            draft_target_threshold=0.2,
            small_draft_acc_head_path="small-draft-head",
            draft_target_acc_head_path="draft-target-head",
        )

        with (
            patch("src.baselines.Decoding.__init__", return_value=None),
            patch("src.baselines.load_acceptance_prediction_head") as load_head,
            patch("src.baselines.DecodingAdapter") as adapter,
        ):
            instance = _TestBaselines(args)
            instance.args = args
            instance.load_acc_head()

        self.assertIs(instance.small_draft_adapter, adapter.return_value)
        self.assertIs(instance.draft_target_adapter, adapter.return_value)
        self.assertEqual(
            load_head.call_args_list,
            [call("small-draft-head"), call("draft-target-head")],
        )
        self.assertEqual(
            adapter.call_args_list,
            [
                call(load_head.return_value, 0.1),
                call(load_head.return_value, 0.2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
