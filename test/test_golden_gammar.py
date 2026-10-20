"""Phase 0.2：golden 固件完整性与执法工具的正确性。

锁定的内容：
- 7 个 gammar tag 的固件齐全、manifest 命令完整、checkpoint 指纹与
  worktree 当前文件一致（checkpoint 漂移 = golden 失效，必须红灯）
- compare 工具的判定逻辑：自比对通过 / 锁定字段扰动失败 / timing
  扰动容忍
"""

import json
import re
import sys
from pathlib import Path

import pytest

GOLDEN_DIR = Path(__file__).parent / "golden"
GAMMAR_DIR = GOLDEN_DIR / "gammar"

sys.path.insert(0, str(GOLDEN_DIR))
import compare  # noqa: E402  (test/ 非包，路径注入加载)

MANIFEST = json.loads((GAMMAR_DIR / "manifest.json").read_text(encoding="utf-8"))


def _load(tag: str) -> dict:
    return json.loads((GAMMAR_DIR / tag / "metrics.json").read_text(encoding="utf-8"))


def test_manifest_and_fixtures_in_sync():
    fixture_tags = sorted(d.name for d in GAMMAR_DIR.iterdir() if d.is_dir())
    assert fixture_tags == sorted(MANIFEST["tags"]), (
        f"manifest 与固件目录不同步: {fixture_tags} vs {sorted(MANIFEST['tags'])}"
    )


@pytest.mark.parametrize("tag", sorted(MANIFEST["tags"]))
def test_self_comparison_passes(tag):
    metrics = _load(tag)
    ok, lines = compare.compare_metrics(metrics, json.loads(json.dumps(metrics)))
    assert ok, f"{tag} 自比对失败: {lines}"


@pytest.mark.parametrize("tag", sorted(MANIFEST["tags"]))
def test_manifest_command_is_complete(tag):
    cmd = MANIFEST["tags"][tag]["command"]
    for required in (
        "--eval_mode adaptive_tridecoding",
        "--protocol paper_table5",
        "--comm_accounting honest",
        "--temp 0.0",
        "--sample_seed 1234",
        "--random_sample",
        "--disable_rl_update",
        "--use_cuda_graph",
        f"--exp_name gammar/{tag}_",
    ):
        assert required in cmd, f"{tag} 命令缺少 {required}"


@pytest.mark.parametrize("tag", sorted(MANIFEST["tags"]))
def test_checkpoint_fingerprint_matches_worktree(tag):
    """golden 绑定 checkpoint 状态：v3 快照被改动 → 这里红灯（golden 失效）。"""
    import hashlib

    for role, spec in MANIFEST["tags"][tag]["checkpoints"].items():
        digest = hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
        assert digest[:16] == spec["sha256_16"], (
            f"{tag}/{role}: checkpoint {spec['path']} 与 golden 记录的指纹不符——"
            f"checkpoint 已漂移，golden 需要重新收割"
        )


def test_locked_field_perturbation_fails():
    metrics = _load("q15_free_g5")
    perturbed = dict(metrics, generated_tokens=metrics["generated_tokens"] + 1)
    ok, lines = compare.compare_metrics(metrics, perturbed)
    assert not ok
    assert any("generated_tokens" in ln for ln in lines)


def test_timing_field_perturbation_tolerated():
    metrics = _load("q15_free_g5")
    perturbed = dict(metrics, wall_time=metrics["wall_time"] * 1.05)
    ok, _ = compare.compare_metrics(metrics, perturbed)
    assert ok


def test_missing_locked_field_fails():
    metrics = _load("q15_free_g5")
    perturbed = {k: v for k, v in metrics.items() if k != "accuracy"}
    ok, _ = compare.compare_metrics(metrics, perturbed)
    assert not ok


def test_new_field_is_informational():
    metrics = _load("q15_free_g5")
    perturbed = dict(metrics, git_sha="0" * 40)
    ok, lines = compare.compare_metrics(metrics, perturbed)
    assert ok
    assert any("git_sha" in ln and "NEW" in ln for ln in lines)


def test_timing_classification_covers_observed_timing_fields():
    """固件里出现的 *_time / throughput 必须有明确分类——模拟量（锁定）
    与实测计时（TIMING）都不得漏登记。Phase 0.1 裁决：arp/dra_overhead
    为墙钟测量，归 TIMING。"""
    for tag in MANIFEST["tags"]:
        for field in _load(tag):
            if field.endswith("_time") or field == "throughput":
                if field in ("communication_time", "queuing_time",
                             "edge_end_comm_time"):
                    continue  # 模拟量：确定性，锁定（校准确认逐位稳定）
                assert field in compare.TIMING_FIELDS, (
                    f"{tag}: 字段 {field} 疑似实测计时但不在 TIMING 表——"
                    f"校准双跑后请修订分类"
                )


def test_golden_recorded_under_snapshot_code_state():
    """golden 产出的代码状态必须等于快照提交（否则复现对照无意义）。"""
    note = MANIFEST["_meta"]["code_state"]
    assert "6d66c72" in note
