import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src import fixed_prefix_probe_runtime as runtime
from src.distribution_probe import (
    categorical_metrics,
    first_token_distribution,
    metric_comparisons,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "scripts" / "probe_fixed_prefix_distribution.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"


def test_categorical_metrics_are_symmetric_and_zero_for_identical_inputs() -> None:
    given = torch.tensor([0.2, 0.3, 0.5])

    when = categorical_metrics(given, given)

    assert when.total_variation == pytest.approx(0.0)
    assert when.jensen_shannon == pytest.approx(0.0)
    assert when.maximum_absolute_difference == pytest.approx(0.0)


def test_categorical_metrics_match_known_disjoint_distribution_values() -> None:
    given_left = torch.tensor([1.0, 0.0])
    given_right = torch.tensor([0.0, 1.0])

    when = categorical_metrics(given_left, given_right)

    assert when.total_variation == pytest.approx(1.0)
    assert when.jensen_shannon == pytest.approx(0.6931471805599453)
    assert when.maximum_absolute_difference == pytest.approx(1.0)


def test_first_token_distribution_counts_only_first_returned_token() -> None:
    given_full_cee_returns = torch.tensor(
        [
            [10, 1, 2, 4],
            [10, 2, 0, 0],
            [10, 1, 3, 3],
        ]
    )

    when = first_token_distribution(given_full_cee_returns, prefix_len=1, vocab_size=5)

    assert when.counts.tolist() == [0, 2, 1, 0, 0]
    assert when.probabilities.tolist() == pytest.approx([0.0, 2 / 3, 1 / 3, 0.0, 0.0])


def test_metric_comparisons_use_empirical_inputs_not_duplicate_target_logits() -> None:
    given_cee_empirical = torch.tensor([1.0, 0.0])
    given_ar_empirical = torch.tensor([0.0, 1.0])
    given_target_exact = torch.tensor([0.5, 0.5])

    when = metric_comparisons(
        cee_empirical=given_cee_empirical,
        ar_empirical=given_ar_empirical,
        target_exact=given_target_exact,
    )

    assert when.cee_sd_vs_ar.total_variation == pytest.approx(1.0)
    assert when.cee_sd_vs_target_exact.total_variation == pytest.approx(0.5)
    assert when.ar_vs_target_exact.total_variation == pytest.approx(0.5)


def test_cee_trials_reuse_loaded_decoder_and_reset_rng_per_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    given_decoder = SimpleNamespace(
        args=SimpleNamespace(
            transfer_top_k=300,
            max_tokens=1,
            gamma1=1,
            gamma2=1,
        ),
        get_decoding_method=lambda: (
            lambda prefix, transfer_top_k: (
                torch.cat((prefix, torch.tensor([[2, 4]])), dim=1),
                {},
            )
        ),
    )
    given_config = SimpleNamespace(trials=3, seed=11)
    seeded: list[int] = []
    monkeypatch.setattr(runtime, "_seed_everything", seeded.append)

    when, average_length = runtime._cee_trial_outputs(
        given_decoder,
        given_config,
        torch.tensor([[9]]),
    )

    assert seeded == [runtime._cee_seed(11, trial) for trial in range(3)]
    assert when.tolist() == [[9, 2], [9, 2], [9, 2]]
    assert average_length == pytest.approx(2.0)
    assert given_decoder.args.probe_cache_max_length == 4


def test_measure_loads_decoder_once_per_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> SimpleNamespace:
            return SimpleNamespace(input_ids=torch.tensor([[1]]))

        def decode(self, token_ids: list[int]) -> str:
            return str(token_ids[0])

    given_config = SimpleNamespace(
        temperatures=(0.2,),
        prompts=("first", "second"),
        trials=2,
        seed=3,
        top_k_output=1,
    )
    loads: list[float] = []
    monkeypatch.setattr(
        runtime.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(runtime, "_target_model_path", lambda config: "target")
    monkeypatch.setattr(
        runtime,
        "_new_decoder",
        lambda config, temperature: loads.append(temperature) or SimpleNamespace(),
    )
    monkeypatch.setattr(
        runtime,
        "_target_probabilities",
        lambda decoder, prefix, temperature: torch.tensor([0.0, 1.0, 0.0]),
    )
    monkeypatch.setattr(
        runtime,
        "_cee_trial_outputs",
        lambda decoder, config, prefix: (torch.tensor([[1, 1], [1, 2]]), 1.0),
    )
    monkeypatch.setattr(
        runtime,
        "_ar_trial_outputs",
        lambda decoder, config, prefix, temperature: torch.tensor([[1, 1], [1, 1]]),
    )

    when = runtime.measure(given_config)

    assert loads == [0.2]
    assert [result["prefix"]["text"] for result in when] == ["first", "second"]


def test_measure_releases_each_completed_temperature_before_loading_the_next(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Tokenizer:
        def __call__(self, prompt: str, *, return_tensors: str) -> SimpleNamespace:
            return SimpleNamespace(input_ids=torch.tensor([[1]]))

        def decode(self, token_ids: list[int]) -> str:
            return str(token_ids[0])

    given_config = SimpleNamespace(
        temperatures=(0.2, 0.8),
        prompts=("first",),
        trials=1,
        seed=3,
        top_k_output=1,
    )
    events: list[tuple[str, float | None]] = []
    monkeypatch.setattr(
        runtime.AutoTokenizer,
        "from_pretrained",
        lambda *args, **kwargs: _Tokenizer(),
    )
    monkeypatch.setattr(runtime, "_target_model_path", lambda config: "target")
    monkeypatch.setattr(
        runtime,
        "_new_decoder",
        lambda config, temperature: (
            events.append(("load", temperature))
            or SimpleNamespace(temperature=temperature)
        ),
    )
    monkeypatch.setattr(
        runtime,
        "_release_cuda_resources",
        lambda: events.append(("release", None)),
    )
    monkeypatch.setattr(
        runtime,
        "_target_probabilities",
        lambda decoder, prefix, temperature: torch.tensor([0.0, 1.0, 0.0]),
    )
    monkeypatch.setattr(
        runtime,
        "_cee_trial_outputs",
        lambda decoder, config, prefix: (torch.tensor([[1, 1]]), 1.0),
    )
    monkeypatch.setattr(
        runtime,
        "_ar_trial_outputs",
        lambda decoder, config, prefix, temperature: torch.tensor([[1, 1]]),
    )

    runtime.measure(given_config)

    assert events == [
        ("load", 0.2),
        ("release", None),
        ("load", 0.8),
        ("release", None),
    ]


def test_new_decoder_forwards_target_quantization_to_engine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    given_config = SimpleNamespace(
        little_model="little",
        draft_model="draft",
        target_model="target",
        target_quantization="4bit",
        seed=3,
    )
    loaded_args: list[SimpleNamespace] = []

    class _Decoder:
        def __init__(self, args: SimpleNamespace) -> None:
            loaded_args.append(args)
            self.target_model = SimpleNamespace()

        def load_model(self) -> None:
            pass

        def _get_model_embedding_vocab_size(self, model: SimpleNamespace) -> int:
            return 42

    monkeypatch.setattr(runtime, "model_zoo", lambda args: None)
    monkeypatch.setattr(runtime, "resolve_acc_head_path", lambda *args: "head")
    monkeypatch.setattr(runtime, "_seed_everything", lambda seed: None)
    monkeypatch.setattr(runtime, "FixedPrefixProbeRunner", _Decoder)

    runtime._new_decoder(given_config, temperature=0.2)

    assert loaded_args[0].target_quantization == "4bit"


def test_target_probability_cache_has_one_token_probe_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacities: list[int | None] = []

    class _Cache:
        def __init__(self, model: object, **kwargs: object) -> None:
            capacities.append(kwargs.get("max_length"))

        def _forward_with_kvcache(self, prefix: torch.Tensor) -> tuple[torch.Tensor]:
            return (torch.zeros((1, 3)),)

    monkeypatch.setattr(runtime, "KVCacheModel", _Cache)
    given_prefix = torch.tensor([[1, 2]])
    given_decoder = SimpleNamespace(
        target_model=SimpleNamespace(),
        get_model_input_device=lambda model: torch.device("cpu"),
    )

    runtime._target_probabilities(given_decoder, given_prefix, temperature=0.2)

    assert capacities == [3]


def test_ar_trial_cache_has_one_token_probe_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capacities: list[int | None] = []

    class _Cache:
        def __init__(self, model: object, **kwargs: object) -> None:
            capacities.append(kwargs.get("max_length"))

        def generate(self, prefix: torch.Tensor, max_tokens: int) -> torch.Tensor:
            return torch.cat((prefix, torch.tensor([[3]])), dim=1)

    monkeypatch.setattr(runtime, "KVCacheModel", _Cache)
    given_config = SimpleNamespace(trials=2, seed=1)
    given_prefix = torch.tensor([[1, 2]])
    given_decoder = SimpleNamespace(
        target_model=SimpleNamespace(),
        get_model_input_device=lambda model: torch.device("cpu"),
    )

    runtime._ar_trial_outputs(
        given_decoder, given_config, given_prefix, temperature=0.2
    )

    assert capacities == [3, 3]


def test_dry_run_writes_structured_probe_plan_without_model_loading(
    tmp_path: Path,
) -> None:
    given_output_path = tmp_path / "probe.json"
    command = [
        str(PYTHON),
        str(DRIVER),
        "--dry-run",
        "--target-model",
        "llama-2-13b",
        "--target-quantization",
        "4bit",
        "--temperatures",
        "0.2",
        "0.8",
        "--trials",
        "17",
        "--seed",
        "29",
        "--prompt",
        "Fixed prefix",
        "--top-k-output",
        "3",
        "--output-path",
        str(given_output_path),
    ]

    when = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert when.returncode == 0, when.stderr
    then = json.loads(given_output_path.read_text(encoding="utf-8"))
    assert then["measurement"].startswith("Empirical first-output-token")
    assert then["dry_run"] is True
    assert then["configuration"]["trials"] == 17
    assert then["configuration"]["seed"] == 29
    assert then["configuration"]["target_quantization"] == "4bit"
    assert then["configuration"]["disable_rl_update"] is True
    assert then["configuration"]["use_stochastic_comm"] is False
    temperatures = [entry["temperature"] for entry in then["results"]]
    assert temperatures == [0.2, 0.8]
    assert all(entry["prefix"]["text"] == "Fixed prefix" for entry in then["results"])
    assert all(entry["cee_sd_vs_ar"] is None for entry in then["results"])
