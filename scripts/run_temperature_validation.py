#!/usr/bin/env python3
"""Run matched GSM8K temperature comparisons for CEE-SD and target AR.

The current runtime exposes one shared ``--temp`` argument per decoding run.
This driver validates equal CEE-SD and target-autoregressive temperatures; it
does not validate independent per-model temperatures.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
EVAL_SCRIPT = REPO_ROOT / "eval" / "eval_gsm8k.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "experiment_results"
MODES = ("cee_sd", "large")


@dataclass(frozen=True)
class RunSpec:
    mode: str
    temperature: float
    seed: int
    exp_name: str
    command: list[str]
    stdout_path: str
    stderr_path: str


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--temperatures", nargs="+", type=float, default=[0.2])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1234])
    parser.add_argument("--eval-data-num", type=int, default=4)
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--little-model", default="llama-68m")
    parser.add_argument("--draft-model", default="tiny-llama-1.1b")
    parser.add_argument("--target-model", default="llama-2-13b")
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--cuda-visible-devices",
        help="Set CUDA_VISIBLE_DEVICES for all child evaluations.",
    )
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Use a matched random GSM8K subset for every matrix cell.",
    )
    parser.add_argument("--sample-seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def build_run_specs(args: argparse.Namespace, run_dir: Path) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for mode in MODES:
        for temperature in args.temperatures:
            for seed in args.seeds:
                name = f"gsm8k_{mode}_temp{temperature:g}_seed{seed}"
                command = [
                    str(args.python),
                    str(EVAL_SCRIPT),
                    "--eval_mode",
                    mode,
                    "--target_model",
                    args.target_model,
                    "--draft_model",
                    args.draft_model,
                    "--temp",
                    str(temperature),
                    "--seed",
                    str(seed),
                    "--eval_data_num",
                    str(args.eval_data_num),
                    "--num_shots",
                    str(args.num_shots),
                    "--max_tokens",
                    str(args.max_tokens),
                    "--top_k",
                    str(args.top_k),
                    "--top_p",
                    str(args.top_p),
                    "--exp_name",
                    name,
                ]
                if mode == "cee_sd":
                    command.extend(
                        [
                            "--little_model",
                            args.little_model,
                        ]
                    )
                if args.random_sample:
                    command.extend(
                        ["--random_sample", "--sample_seed", str(args.sample_seed)]
                    )
                specs.append(
                    RunSpec(
                        mode=mode,
                        temperature=temperature,
                        seed=seed,
                        exp_name=name,
                        command=command,
                        stdout_path=str(run_dir / f"{name}.stdout.log"),
                        stderr_path=str(run_dir / f"{name}.stderr.log"),
                    )
                )
    return specs


def write_manifest(path: Path, manifest: dict[str, object]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.output_root / f"temperature_validation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    specs = build_run_specs(args, run_dir)
    parameters = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    manifest: dict[str, object] = {
        "created_at_utc": timestamp,
        "dry_run": args.dry_run,
        "environment": {
            "cwd": str(REPO_ROOT),
            "python": str(args.python),
            "cuda_visible_devices": args.cuda_visible_devices,
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        },
        "parameters": parameters,
        "temperature_scope": (
            "Each run applies one common --temp value. Independent per-model "
            "temperatures are not exposed by the current runtime."
        ),
        "runs": [
            {**asdict(spec), "status": "planned", "exit_code": None} for spec in specs
        ],
    }
    manifest_path = run_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    if args.dry_run:
        print(f"Dry run written to {manifest_path}")
        return 0

    environment = os.environ.copy()
    if args.cuda_visible_devices is not None:
        environment["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    first_failure_code = 0
    runs = manifest["runs"]
    assert isinstance(runs, list)
    for spec, record in zip(specs, runs, strict=True):
        assert isinstance(record, dict)
        with (
            Path(spec.stdout_path).open("w", encoding="utf-8") as stdout,
            Path(spec.stderr_path).open("w", encoding="utf-8") as stderr,
        ):
            completed = subprocess.run(
                spec.command,
                cwd=REPO_ROOT,
                env=environment,
                stdout=stdout,
                stderr=stderr,
                check=False,
            )
        record["exit_code"] = completed.returncode
        record["status"] = "success" if completed.returncode == 0 else "failed"
        write_manifest(manifest_path, manifest)
        print(f"{spec.exp_name}: {record['status']} ({completed.returncode})")
        if completed.returncode != 0 and first_failure_code == 0:
            first_failure_code = completed.returncode
    print(f"Manifest: {manifest_path}")
    return first_failure_code


if __name__ == "__main__":
    raise SystemExit(main())
