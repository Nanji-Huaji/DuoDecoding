"""分布一致性检查的离线测试。"""

import json
import sys

import pytest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import compare_distributions as cd  # noqa: E402


def _write(tmp_path, name, records):
    p = tmp_path / name
    p.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    return p


def _rec(tokens, correct=True):
    return {"gen_ids": tokens, "correct": correct}


def test_identical_passes(tmp_path):
    base = [_rec([1, 2, 3]), _rec([4, 5])]
    new = [_rec([1, 2, 3]), _rec([4, 5])]
    r = cd.compare(base, new)
    assert r["pass"]
    assert r["length_ks"] == 0
    assert r["unigram_cosine"] == pytest.approx(1.0)
    assert r["token_match"] == 2


def test_length_shift_fails(tmp_path):
    base = [_rec([1] * 10), _rec([1] * 10)]
    new = [_rec([1] * 100), _rec([1] * 100)]
    r = cd.compare(base, new)
    assert not r["checks"]["length_ks"]


def test_token_distribution_shift_fails():
    base = [_rec([1, 1, 1, 1])]
    new = [_rec([2, 2, 2, 2])]
    r = cd.compare(base, new)
    assert not r["checks"]["unigram_cosine"] or not r["checks"]["unigram_tv"]


def test_accuracy_difference_fails():
    base = [_rec([1], correct=True), _rec([1], correct=True)]
    new = [_rec([1], correct=False), _rec([1], correct=False)]
    r = cd.compare(base, new)
    assert not r["checks"]["accuracy"]
