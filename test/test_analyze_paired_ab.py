"""paired A/B 汇总工具的离线测试。"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import analyze_paired_ab as ab  # noqa: E402


def _write_run(root, rep, slot, name, metrics, tokens):
    d = root / f"rep{rep}" / slot
    d.mkdir(parents=True)
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (d / "mode_name.txt").write_text(name, encoding="utf-8")
    (d / "run.jsonl").write_text(
        "\n".join(json.dumps({"gen_ids": t}) for t in tokens) + "\n", encoding="utf-8"
    )


def test_summarize_and_markdown(tmp_path):
    a = {
        "throughput": 10.0,
        "communication_time": 20.0,
        "generated_tokens": 100,
        "target_forward_times": 50,
        "draft_forward_times": 60,
        "little_forward_times": 70,
        "edge_cloud_data_bytes": 1000,
        "accuracy": 0.25,
    }
    b = dict(a)
    b.update(
        throughput=9.0,
        communication_time=18.0,
        target_forward_times=40,
        draft_forward_times=80,
        little_forward_times=90,
        edge_cloud_data_bytes=1100,
    )
    _write_run(tmp_path, 1, "A", "cee_sd", a, [[1, 2], [3]])
    _write_run(tmp_path, 1, "B", "cee_sd_no_arp", b, [[1, 2], [4]])

    data = ab.collect(tmp_path)
    s = ab.summarize(data)
    assert s["names"] == {"A": "cee_sd", "B": "cee_sd_no_arp"}
    assert s["modes"]["A"]["throughput"] == 10.0
    assert s["paired_ratio_median"]["ratio_throughput"] == 0.9
    assert s["pairs"][0]["token_match"] == 1
    md = ab.render_markdown(s)
    assert "cee_sd_no_arp" in md and "target_forward_times/token" in md


def test_skip_incomplete_rep(tmp_path):
    a = {
        "throughput": 10.0,
        "communication_time": 20.0,
        "generated_tokens": 100,
        "target_forward_times": 50,
        "draft_forward_times": 60,
        "little_forward_times": 70,
        "edge_cloud_data_bytes": 1000,
        "accuracy": 0.25,
    }
    _write_run(tmp_path, 1, "A", "a", a, [[1]])
    s = ab.summarize(ab.collect(tmp_path))
    assert s["pairs"] == []
