#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, TypedDict

REPO_ROOT = Path(__file__).resolve().parent.parent

MEASUREMENT = (
    "Empirical first-output-token distributions from repeated independent "
    "CEE-SD and target AR trials at identical tokenized prefixes; not an "
    "end-to-end sequence-distribution comparison."
)


class PrefixJson(TypedDict):
    identity: str
    text: str
    token_count: int | None


class MetricsJson(TypedDict):
    total_variation: float
    jensen_shannon: float
    maximum_absolute_difference: float


class TokenProbabilityJson(TypedDict):
    token_id: int
    token: str
    probability: float


class ResultJson(TypedDict):
    temperature: float
    prefix: PrefixJson
    trials: int
    cee_sd_counts: list[int] | None
    ar_counts: list[int] | None
    cee_sd_vs_ar: MetricsJson | None
    cee_sd_vs_target_exact: MetricsJson | None
    ar_vs_target_exact: MetricsJson | None
    cee_sd_top_tokens: list[TokenProbabilityJson] | None
    ar_top_tokens: list[TokenProbabilityJson] | None
    target_exact_top_tokens: list[TokenProbabilityJson] | None
    cee_sd_average_appended_length: float | None


class ProbeJson(TypedDict):
    measurement: str
    empirical: bool
    dry_run: bool
    configuration: dict[str, str | int | float | bool | list[float]]
    results: list[ResultJson]


@dataclass(frozen=True, slots=True)
class ProbeConfig:
    little_model: str
    draft_model: str
    target_model: str
    target_quantization: str
    temperatures: tuple[float, ...]
    prompts: tuple[str, ...]
    trials: int
    seed: int
    top_k_output: int
    output_path: Path
    dry_run: bool


def parse_args(argv: Sequence[str] | None = None) -> ProbeConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--little-model", default="llama-68m")
    parser.add_argument("--draft-model", default="tiny-llama-1.1b")
    parser.add_argument("--target-model", default="llama-2-13b")
    parser.add_argument(
        "--target-quantization",
        choices=["auto", "4bit", "none"],
        default="auto",
        help="Quantization mode for the target model.",
    )
    parser.add_argument("--temperatures", type=float, nargs="+", default=[0.2])
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt")
    prompt_group.add_argument("--prompt-file", type=Path)
    parser.add_argument("--trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--top-k-output", type=int, default=10)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=REPO_ROOT
        / "experiment_results"
        / "fixed_prefix_distribution_probe.json",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.trials <= 0:
        parser.error("--trials must be positive")
    if args.top_k_output <= 0:
        parser.error("--top-k-output must be positive")
    return ProbeConfig(
        little_model=args.little_model,
        draft_model=args.draft_model,
        target_model=args.target_model,
        target_quantization=args.target_quantization,
        temperatures=tuple(args.temperatures),
        prompts=_read_prompts(args.prompt, args.prompt_file),
        trials=args.trials,
        seed=args.seed,
        top_k_output=args.top_k_output,
        output_path=args.output_path,
        dry_run=args.dry_run,
    )


def build_probe_json(config: ProbeConfig) -> ProbeJson:
    results = _planned_results(config) if config.dry_run else _measured_results(config)
    return ProbeJson(
        measurement=MEASUREMENT,
        empirical=True,
        dry_run=config.dry_run,
        configuration={
            "little_model": config.little_model,
            "draft_model": config.draft_model,
            "target_model": config.target_model,
            "target_quantization": config.target_quantization,
            "temperatures": list(config.temperatures),
            "trials": config.trials,
            "seed": config.seed,
            "top_k_output": config.top_k_output,
            "eval_mode": "cee_sd",
            "max_tokens": 1,
            "gamma1": 1,
            "gamma2": 1,
            "transfer_top_k": 300,
            "top_k": 0,
            "top_p": 0.0,
            "batch_delay": 0.0,
            "disable_rl_update": True,
            "use_rl_adapter": False,
            "use_stochastic_comm": False,
            "target_cache_top_k": 0,
            "target_cache_top_p": 0.0,
        },
        results=results,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(
        json.dumps(build_probe_json(config), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(config.output_path)
    return 0


def _read_prompts(prompt: str | None, prompt_file: Path | None) -> tuple[str, ...]:
    if prompt is not None:
        return (prompt,)
    assert prompt_file is not None
    prompts = tuple(
        line for line in prompt_file.read_text(encoding="utf-8").splitlines() if line
    )
    if not prompts:
        raise ValueError("Prompt file must contain at least one non-empty line")
    return prompts


def _planned_results(config: ProbeConfig) -> list[ResultJson]:
    return [
        _result(temperature, prompt, None)
        for temperature in config.temperatures
        for prompt in config.prompts
    ]


def _measured_results(config: ProbeConfig) -> list[ResultJson]:
    sys.path.insert(0, str(REPO_ROOT))
    from src.fixed_prefix_probe_runtime import measure

    return measure(config)


def _result(temperature: float, prompt: str, token_count: int | None) -> ResultJson:
    return ResultJson(
        temperature=temperature,
        prefix=PrefixJson(
            identity=hashlib.sha256(prompt.encode()).hexdigest(),
            text=prompt,
            token_count=token_count,
        ),
        trials=0,
        cee_sd_counts=None,
        ar_counts=None,
        cee_sd_vs_ar=None,
        cee_sd_vs_target_exact=None,
        ar_vs_target_exact=None,
        cee_sd_top_tokens=None,
        ar_top_tokens=None,
        target_exact_top_tokens=None,
        cee_sd_average_appended_length=None,
    )


if __name__ == "__main__":
    raise SystemExit(main())
