"""Tests for the curriculum-based network condition sampler used by the
RL training loop (eval/eval_mixed.py)."""

import math
import random

import pytest

from src.utils import parse_range_spec, sample_curriculum_condition


BW_START = (20.0, 50.0)
BW_END = (1.0, 50.0)
NTT_START = (0.0, 5.0)
NTT_END = (0.0, 100.0)


def _sample_many(step, total, n=2000, sampling="loguniform"):
    random.seed(1234)
    return [
        sample_curriculum_condition(
            step, total, BW_START, BW_END, NTT_START, NTT_END, sampling=sampling
        )
        for _ in range(n)
    ]


class TestCurriculumSampling:
    def test_start_step_uses_start_ranges(self):
        conds = _sample_many(0, 100)
        for bw, ntt in conds:
            assert BW_START[0] <= bw <= BW_START[1]
            assert NTT_START[0] <= ntt <= NTT_START[1]

    def test_final_step_uses_end_ranges(self):
        conds = _sample_many(99, 100)
        for bw, ntt in conds:
            assert BW_END[0] <= bw <= BW_END[1]
            assert NTT_END[0] <= ntt <= NTT_END[1]

    def test_midpoint_bounds_are_geometric(self):
        # At progress 0.5 the low bound interpolates geometrically:
        # 20 * (1/20)^0.5 = sqrt(20) ≈ 4.47
        expected_low = 20.0 * (1.0 / 20.0) ** 0.5
        conds = _sample_many(50, 101)  # progress = 50/100 = 0.5
        for bw, ntt in conds:
            assert expected_low - 1e-9 <= bw <= 50.0
            assert 0.0 <= ntt <= 5.0 + (100.0 - 5.0) * 0.5

    def test_loguniform_spreads_mass_logarithmically(self):
        conds = _sample_many(99, 100, n=5000)
        geo_mean = math.sqrt(BW_END[0] * BW_END[1])
        below = sum(1 for bw, _ in conds if bw < geo_mean) / len(conds)
        assert 0.4 < below < 0.6

    def test_uniform_sampling_respects_range(self):
        random.seed(7)
        for _ in range(500):
            bw, ntt = sample_curriculum_condition(
                0, 100, BW_START, BW_END, NTT_START, NTT_END, sampling="uniform"
            )
            assert 20.0 <= bw <= 50.0 and 0.0 <= ntt <= 5.0

    def test_progress_monotonicity_of_low_bound(self):
        # The deterministic bound interpolation should move towards the
        # harder end-range as training progresses (check the min of many
        # samples shrinks over time).
        mins = []
        for step in [0, 50, 99]:
            conds = _sample_many(step, 100, n=2000)
            mins.append(min(bw for bw, _ in conds))
        assert mins[0] >= mins[1] >= mins[2]


class TestParseRangeSpec:
    def test_valid(self):
        assert parse_range_spec("20,50") == (20.0, 50.0)
        assert parse_range_spec(" 1.5 , 3 ") == (1.5, 3.0)

    def test_missing_comma(self):
        with pytest.raises(ValueError):
            parse_range_spec("20")

    def test_non_numeric(self):
        with pytest.raises(ValueError):
            parse_range_spec("abc,def")

    def test_low_exceeds_high(self):
        with pytest.raises(ValueError):
            parse_range_spec("50,20")
