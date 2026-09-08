#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.10.8", "pydantic>=2.12.5"]
# ///
"""Plot DSD aggregate latency components as a pie chart.

Run with:
uv run scripts/plot_dsd_latency_composition.py --metrics METRICS.json \
    --output latency.png
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict

plt.switch_backend("Agg")


class AggregateMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    wall_time: float
    draft_computation_time: float
    target_computation_time: float
    communication_time: float
    queuing_time: float


class CompositionSidecar(BaseModel):
    model_config = ConfigDict(frozen=True)

    metrics_path: str
    wall_time_seconds: float
    components_seconds: dict[str, float]


@dataclass(frozen=True, slots=True)
class PlotArguments:
    metrics: Path
    output: Path
    sidecar: Path | None

@dataclass(frozen=True, slots=True)
class LatencyComposition:
    computation: float
    communication: float

    def as_dict(self) -> dict[str, float]:
        return {
            "Computation": self.computation,
            "Communication": self.communication,
        }


def parse_metrics(metrics_path: Path) -> AggregateMetrics:
    return AggregateMetrics.model_validate_json(
        metrics_path.read_text(encoding="utf-8")
    )


def calculate_composition(metrics: AggregateMetrics) -> LatencyComposition:
    return LatencyComposition(
        computation=(
            metrics.draft_computation_time + metrics.target_computation_time
        ),
        communication=metrics.communication_time,
    )


def write_chart(composition: LatencyComposition, output_path: Path) -> None:
    components = composition.as_dict()
    figure, axis = plt.subplots(figsize=(7, 7))
    wedges, _, _ = axis.pie(
        components.values(),
        autopct="%1.1f%%",
        colors=["#4C78A8", "#F58518"],
        startangle=90,
    )
    total_seconds = sum(components.values())
    legend_labels = [
        f"{label}: {seconds:.2f} s ({seconds / total_seconds:.1%})"
        for label, seconds in components.items()
    ]
    axis.legend(
        wedges,
        legend_labels,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )
    axis.set_title("DSD Latency Composition")
    axis.axis("equal")
    figure.savefig(output_path, bbox_inches="tight", dpi=180, transparent=True)
    plt.close(figure)


def write_sidecar(sidecar: CompositionSidecar, sidecar_path: Path) -> None:
    sidecar_path.write_text(
        sidecar.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


def parse_args() -> PlotArguments:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sidecar", type=Path)
    parsed = parser.parse_args()
    return PlotArguments(
        metrics=parsed.metrics,
        output=parsed.output,
        sidecar=parsed.sidecar,
    )


def main() -> None:
    args = parse_args()
    metrics = parse_metrics(args.metrics)
    composition = calculate_composition(metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_chart(composition, args.output)
    sidecar_path = args.sidecar or args.output.with_suffix(".json")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    write_sidecar(
        CompositionSidecar(
            metrics_path=str(args.metrics),
            wall_time_seconds=metrics.wall_time,
            components_seconds=composition.as_dict(),
        ),
        sidecar_path,
    )


if __name__ == "__main__":
    main()
