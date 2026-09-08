import importlib.util
import json
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "plot_dsd_latency_composition.py"
)
SPEC = importlib.util.spec_from_file_location(
    "plot_dsd_latency_composition", SCRIPT_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
plotter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(plotter)


def test_composition_when_metrics_have_other_timing_values():
    # Given: aggregate metrics with timing values outside the requested categories.
    metrics = plotter.AggregateMetrics(
        wall_time=10.0,
        draft_computation_time=2.0,
        target_computation_time=3.0,
        communication_time=1.5,
        queuing_time=0.5,
    )

    # When: the latency composition is calculated.
    composition = plotter.calculate_composition(metrics)

    # Then: computation combines draft and target time; only communication remains.
    assert composition.as_dict() == {
        "Computation": 5.0,
        "Communication": 1.5,
    }


def test_composition_when_wall_time_is_less_than_selected_categories():
    # Given: a wall time that is smaller than the requested timing categories.
    metrics = plotter.AggregateMetrics(
        wall_time=1.0,
        draft_computation_time=0.5,
        target_computation_time=0.5,
        communication_time=0.1,
        queuing_time=0.0,
    )

    # When: the latency composition is calculated.
    composition = plotter.calculate_composition(metrics)

    # Then: it still reports only the requested categories, independent of wall time.
    assert composition.as_dict() == {
        "Computation": 1.0,
        "Communication": 0.1,
    }


def test_chart_when_written_uses_percentages_and_a_detailed_legend(
    tmp_path: Path, monkeypatch
):
    # Given: a composition containing the two requested categories.
    composition = plotter.LatencyComposition(computation=3.0, communication=1.0)
    output_path = tmp_path / "latency.png"
    captured = {}

    class FakeAxis:
        def pie(self, values, **kwargs):
            captured["values"] = list(values)
            captured["pie_kwargs"] = kwargs
            return (["computation", "communication"], [], [])

        def legend(self, handles, labels, **kwargs):
            captured["legend_handles"] = handles
            captured["legend_labels"] = labels
            captured["legend_kwargs"] = kwargs

        def set_title(self, title):
            captured["title"] = title

        def axis(self, value):
            captured["axis"] = value

    class FakeFigure:
        def savefig(self, path, **kwargs):
            captured["output_path"] = path
            captured["savefig_kwargs"] = kwargs

    fake_figure = FakeFigure()
    monkeypatch.setattr(
        plotter.plt,
        "subplots",
        lambda **kwargs: (fake_figure, FakeAxis()),
    )
    monkeypatch.setattr(plotter.plt, "close", lambda figure: None)

    # When: the chart is rendered.
    plotter.write_chart(composition, output_path)

    # Then: the pie shows only percentages, while the legend names every value.
    assert captured["values"] == [3.0, 1.0]
    assert captured["pie_kwargs"] == {
        "autopct": "%1.1f%%",
        "colors": ["#4C78A8", "#F58518"],
        "startangle": 90,
    }
    assert captured["legend_labels"] == [
        "Computation: 3.00 s (75.0%)",
        "Communication: 1.00 s (25.0%)",
    ]
    assert captured["title"] == "DSD Latency Composition"
    assert captured["axis"] == "equal"
    assert captured["savefig_kwargs"] == {
        "bbox_inches": "tight",
        "dpi": 180,
        "transparent": True,
    }


def test_cli_when_given_metrics_writes_png_and_sidecar(tmp_path: Path, monkeypatch):
    # Given: a valid aggregate-metrics JSON file and a requested PNG destination.
    metrics_path = tmp_path / "metrics.json"
    output_path = tmp_path / "latency.png"
    metrics_path.write_text(
        json.dumps(
            {
                "wall_time": 4.0,
                "draft_computation_time": 1.0,
                "target_computation_time": 1.5,
                "communication_time": 0.75,
                "queuing_time": 0.25,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            str(SCRIPT_PATH),
            "--metrics",
            str(metrics_path),
            "--output",
            str(output_path),
        ],
    )

    # When: the plotter CLI runs end to end.
    plotter.main()

    # Then: it produces a nonempty PNG and the default JSON sidecar.
    sidecar_path = output_path.with_suffix(".json")
    assert output_path.is_file()
    assert output_path.stat().st_size > 0
    assert json.loads(sidecar_path.read_text(encoding="utf-8")) == {
        "components_seconds": {
            "Communication": 0.75,
            "Computation": 2.5,
        },
        "metrics_path": str(metrics_path),
        "wall_time_seconds": 4.0,
    }
