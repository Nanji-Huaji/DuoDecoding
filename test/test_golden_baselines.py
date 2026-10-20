"""7 个非 adaptive 方法的 golden 固件（cee_dsd/cee_dssd/dist_spec/tridecoding 等）。

为什么单独一套：gammar 固件只覆盖 `adaptive_tridecoding`（γ 扫描），
H1/H2 之后的下一步（CEE 公共核心）要动 `cee_dsd`/`cee_dssd`，必须先有
这两个方法的数字与 token 基线。本套固件在 `refactor/baselines-h1@2aa225c`
上重跑，20 样本、同一 q15 模型与 paper_table5 协议。

用法（worktree 根目录）：

    python test/golden/compare.py  # 只服务 gammar；本套直接调 compare_metrics

重跑后对照：

    python - <<'PY'
    import sys, json
    sys.path.insert(0, "test/golden")
    import compare
    from pathlib import Path
    base = json.loads(Path("test/golden/baselines/q15_cee_dsd_g5/metrics.json").read_text())
    new = json.loads(Path("exp/.../cee_dsd_gsm8k_metrics.json").read_text())
    print(compare.compare_metrics(base, new))
    PY
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden"
SUITE = GOLDEN / "baselines"
sys.path.insert(0, str(GOLDEN))
import compare  # noqa: E402

MANIFEST = json.loads((SUITE / "manifest.json").read_text(encoding="utf-8"))
TAGS = sorted(MANIFEST["tags"])


def _metrics(tag: str) -> dict:
    return json.loads((SUITE / tag / "metrics.json").read_text(encoding="utf-8"))


def _tokens(tag: str) -> dict:
    return json.loads((SUITE / tag / "tokens.json").read_text(encoding="utf-8"))


def test_manifest_and_fixtures_in_sync():
    fixture_tags = sorted(d.name for d in SUITE.iterdir() if d.is_dir())
    assert fixture_tags == TAGS


@pytest.mark.parametrize("tag", TAGS)
def test_self_comparison_passes(tag):
    metrics = _metrics(tag)
    ok, lines = compare.compare_metrics(metrics, json.loads(json.dumps(metrics)))
    assert ok, f"{tag} 自比对失败: {lines}"


@pytest.mark.parametrize("tag", TAGS)
def test_token_fixture_is_complete(tag):
    fixture = _tokens(tag)
    assert len(fixture["samples"]) == MANIFEST["tags"][tag]["samples"]
    assert all(sample["gen_ids"] for sample in fixture["samples"])
    records = [{"gen_ids": s["gen_ids"]} for s in fixture["samples"]]
    ok, lines = compare.compare_tokens(fixture, records)
    assert ok, f"{tag} token 自比对失败: {lines}"


@pytest.mark.parametrize("tag", TAGS)
def test_manifest_command_is_complete(tag):
    spec = MANIFEST["tags"][tag]
    cmd = spec["command"]
    for required in (
        f"--eval_mode {spec['eval_mode']}",
        "--protocol paper_table5",
        "--use_rl_adapter",
        "--disable_rl_update",
        "--use_cuda_graph",
        f"--exp_name gammar/{Path(spec['original_run_dir']).name}",
    ):
        assert required in cmd, f"{tag} 命令缺少 {required!r}"


@pytest.mark.parametrize("tag", TAGS)
def test_checkpoint_fingerprint_matches_worktree(tag):
    """固件绑定 checkpoint；checkpoint 漂移即 golden 失效。"""
    for role, spec in MANIFEST["tags"][tag]["checkpoints"].items():
        digest = hashlib.sha256(Path(spec["path"]).read_bytes()).hexdigest()
        assert digest[:16] == spec["sha256_16"], (
            f"{tag}/{role}: {spec['path']} 指纹不符，golden 需重新收割"
        )


def test_locked_field_perturbation_fails():
    metrics = _metrics("q15_cee_dsd_g5")
    perturbed = dict(metrics, generated_tokens=metrics["generated_tokens"] + 1)
    ok, lines = compare.compare_metrics(metrics, perturbed)
    assert not ok
    assert any("generated_tokens" in ln for ln in lines)


@pytest.mark.parametrize("tag", TAGS)
def test_fixture_matches_original_run_when_available(tag):
    """固件必须与 manifest 记录的 original_run_dir 产物一致（防止收割错文件）。"""
    spec = MANIFEST["tags"][tag]
    run = Path(__file__).resolve().parents[1] / spec["original_run_dir"]
    if not run.is_dir():
        pytest.skip(f"原始 run 不在本工作区: {run}")
    mode = spec["eval_mode"]
    assert _metrics(tag) == json.loads(
        (run / f"{mode}_gsm8k_metrics.json").read_text(encoding="utf-8")
    )
    jsonl = next(run.glob("*.jsonl"))
    records = [
        {"gen_ids": json.loads(line)["gen_ids"]}
        for line in jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ok, lines = compare.compare_tokens(_tokens(tag), records)
    assert ok, lines
