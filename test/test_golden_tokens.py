"""token 级 golden：聚合指标之外，锁住"每步选了什么"。

metrics.json 锁的是口径（token 总数、接受数、通信字节、accuracy）。两份
不同的 token 序列完全可能撞出同一组指标，所以这里直接把原始 run 的
gen_ids 收成固件——它们本来就在主树 exp/ 里，收割零成本。

重跑 golden 配置后：

    python test/golden/compare.py --tokens <tag> <new.jsonl>
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "test" / "golden"
sys.path.insert(0, str(GOLDEN))

import compare  # noqa: E402

GAMMAR = GOLDEN / "gammar"
MANIFEST = json.loads((GAMMAR / "manifest.json").read_text(encoding="utf-8"))
TAGS = sorted(MANIFEST["tags"])


def _tokens(tag: str) -> dict:
    return json.loads((GAMMAR / tag / "tokens.json").read_text(encoding="utf-8"))


def _records(fixture: dict) -> list[dict]:
    return [{"gen_ids": sample["gen_ids"]} for sample in fixture["samples"]]


def test_every_tag_has_token_fixture() -> None:
    missing = [tag for tag in TAGS if not (GAMMAR / tag / "tokens.json").exists()]
    assert not missing, f"缺 token 固件: {missing}；跑 tools/harvest_tokens.py"


@pytest.mark.parametrize("tag", TAGS)
def test_fixture_size_and_total_match_manifest(tag: str) -> None:
    fixture = _tokens(tag)
    metrics = json.loads((GAMMAR / tag / "metrics.json").read_text(encoding="utf-8"))
    expected_samples = MANIFEST["tags"][tag]["samples"]

    assert len(fixture["samples"]) == expected_samples
    assert sum(len(sample["gen_ids"]) for sample in fixture["samples"]) == metrics[
        "generated_tokens"
    ]
    assert fixture["_meta"]["total_tokens"] == metrics["generated_tokens"]


@pytest.mark.parametrize("tag", TAGS)
def test_self_comparison_passes(tag: str) -> None:
    fixture = _tokens(tag)
    ok, lines = compare.compare_tokens(fixture, _records(fixture))
    assert ok, lines


def test_single_token_perturbation_is_detected() -> None:
    fixture = _tokens("q15_free_g5")
    records = _records(fixture)
    mutated = list(records)
    first = dict(mutated[0])
    gen_ids = list(first["gen_ids"])
    gen_ids[3] = (gen_ids[3] + 1) % 32000
    first["gen_ids"] = gen_ids
    mutated[0] = first

    ok, lines = compare.compare_tokens(fixture, mutated)
    assert not ok
    assert any("样本 0" in line and "token[3]" in line for line in lines)


def test_missing_sample_is_detected() -> None:
    fixture = _tokens("q15_free_g5")
    ok, lines = compare.compare_tokens(fixture, _records(fixture)[:-1])
    assert not ok
    assert any("样本数" in line for line in lines)


def test_truncated_sequence_is_detected() -> None:
    fixture = _tokens("q15_free_g5")
    records = _records(fixture)
    records[-1] = {"gen_ids": records[-1]["gen_ids"][:-1]}
    ok, lines = compare.compare_tokens(fixture, records)
    assert not ok
    assert any("长度" in line for line in lines)
