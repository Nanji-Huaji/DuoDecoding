from argparse import Namespace
from unittest.mock import patch

from src import utils
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


def _load_dsd_models(*, keep_target_on_single_gpu: bool):
    loader_calls: list[tuple[str, str, dict]] = []
    decoder = object.__new__(_TestDecoding)
    decoder.args = Namespace(
        eval_mode="dsd",
        draft_model="tiny-llama-1.1b",
        target_model="llama-2-13b",
        draft_quantization="none",
        target_quantization="none",
        keep_target_on_single_gpu=keep_target_on_single_gpu,
    )
    decoder.color_print = lambda *_args: None

    def load_causal_lm(_loader, model_name, device_map, **kwargs):
        loader_calls.append((model_name, device_map, kwargs))
        return None

    with (
        patch.object(Decoding, "_get_available_gpu_count", return_value=2),
        patch("src.engine.build_quant_config", return_value=None),
        patch("src.engine.load_causal_lm", side_effect=load_causal_lm),
        patch(
            "src.engine.build_sharded_target_device_map",
            return_value=("auto", {0: "20GiB", 1: "17GiB"}),
        ) as build_sharded_target_device_map,
    ):
        decoder.load_model()

    return loader_calls, build_sharded_target_device_map


def test_dsd_keeps_target_on_selected_gpu_when_requested():
    # Given: an unquantized DSD pair with two visible GPUs and the placement flag.
    # When: the evaluator loads both models.
    loader_calls, build_sharded_target_device_map = _load_dsd_models(
        keep_target_on_single_gpu=True
    )

    # Then: the target stays on cuda:0 and is not converted to auto-sharding.
    assert loader_calls == [
        ("tiny-llama-1.1b", "cuda:1", {"quant_config": None}),
        ("llama-2-13b", "cuda:0", {"quant_config": None, "max_memory": None}),
    ]
    build_sharded_target_device_map.assert_not_called()


def test_dsd_auto_shards_target_by_default():
    # Given: the same unquantized DSD pair without the placement flag.
    # When: the evaluator loads both models.
    loader_calls, build_sharded_target_device_map = _load_dsd_models(
        keep_target_on_single_gpu=False
    )

    # Then: the established automatic target-sharding behavior is preserved.
    assert loader_calls == [
        ("tiny-llama-1.1b", "cuda:1", {"quant_config": None}),
        (
            "llama-2-13b",
            "auto",
            {"quant_config": None, "max_memory": {0: "20GiB", 1: "17GiB"}},
        ),
    ]
    build_sharded_target_device_map.assert_called_once_with(
        2,
        reserve_last_gpu_gib=8,
    )


def test_keep_target_on_single_gpu_flag_defaults_to_false(monkeypatch):
    # Given: the normal evaluator CLI without the placement flag.
    monkeypatch.setattr("sys.argv", ["prog"])

    # When: arguments are parsed without model resolution side effects.
    with patch("src.utils.model_zoo"):
        args = utils.parse_arguments()

    # Then: automatic target sharding remains the default behavior.
    assert args.keep_target_on_single_gpu is False


def test_keep_target_on_single_gpu_flag_enables_single_gpu_target(monkeypatch):
    # Given: the normal evaluator CLI with the placement flag.
    monkeypatch.setattr("sys.argv", ["prog", "--keep_target_on_single_gpu"])

    # When: arguments are parsed without model resolution side effects.
    with patch("src.utils.model_zoo"):
        args = utils.parse_arguments()

    # Then: the DSD placement guard is enabled.
    assert args.keep_target_on_single_gpu is True
