"""ODLD policy contracts through the existing public scalar/selection APIs."""

import math
import subprocess
import sys
from decimal import ROUND_DOWN, Inexact, localcontext
from fractions import Fraction
from pathlib import Path

import pytest

from src.tk_slt_policy import (
    _lambert_w_minus_one,
    tk_slt_optimal_draft_length,
    tk_slt_select_speculative,
    tk_slt_speedup_ratio,
)


@pytest.mark.parametrize(
    "alpha,cost,expected",
    [(0.5, 0.001, 8), (0.5, 0.0005, 9), (math.nextafter(1.0, 0.0), 0.999999, 64)],
)
def test_valid_parameters_choose_the_bounded_optimum(alpha, cost, expected):
    # Independently enumerated with exact rational arithmetic / Decimal120.
    gamma, score = tk_slt_optimal_draft_length(alpha, cost, 0.0, 64)
    assert gamma == expected
    assert score > 1


@pytest.mark.parametrize(
    "alpha,cost",
    [
        (1e-300, 0.0),
        (math.nextafter(0.5, 1.0), 0.5),
        (0.5, math.nextafter(0.5, 0.0)),
    ],
)
def test_as2_detects_real_gain_even_when_reported_score_rounds_to_one(alpha, cost):
    # S(1)>1 iff alpha>cost; here the positive gain is below float resolution.
    use, _, score = tk_slt_select_speculative(alpha, cost, 0.0, 64)
    assert score == 1.0
    assert use


@pytest.mark.parametrize(
    "z,expected",
    [
        (-0.2, -2.5426413577735265),
        (-1e-300, -697.3227762954601),
        (-math.ulp(0.0), -751.0615595398791),
        (math.nextafter(-1 / math.e, 0.0), -1.0000000153042543),
    ],
)
def test_exported_lambert_helper_covers_tiny_and_branchpoint_inputs(z, expected):
    # Independent mpmath160 evaluation at the exact binary64 input.
    actual = _lambert_w_minus_one(z)
    assert abs(actual - expected) <= math.ulp(expected)


def _exhaustive_scores(alpha, cost, cap):
    # Independent definition: sum the geometric terms and enumerate every
    # candidate. No production sign transform, binary search or Lambert root.
    a, cost_fraction = Fraction(alpha), Fraction(cost)
    power = total = Fraction(1)
    scores = []
    for gamma in range(1, cap + 1):
        power *= a
        total += power
        scores.append(total / (1 + gamma * cost_fraction))
    return scores


@pytest.mark.parametrize(
    "alpha", [0.0, 1e-100, 0.1, 0.4, 0.5, 0.8, 0.9, math.nextafter(1.0, 0.0), 1.0]
)
@pytest.mark.parametrize("cost", [0.0, 1e-300, 0.0005, 0.001, 0.1, 0.4, 1.0, 2.0])
@pytest.mark.parametrize("cap", [1, 8, 64])
def test_matches_exhaustive_exact_rational_objective(alpha, cost, cap):
    scores = _exhaustive_scores(alpha, cost, cap)
    expected = max(range(1, cap + 1), key=lambda g: scores[g - 1])
    use, gamma, score = tk_slt_select_speculative(alpha, cost, 0.0, cap)
    assert gamma == expected  # max keeps the earliest *exact* tie
    assert use == (scores[expected - 1] > 1)
    expected_score = float(scores[expected - 1])
    assert abs(score - expected_score) <= math.ulp(expected_score)


@pytest.mark.parametrize("exponent", [64, 256, 1000, 1074])
def test_float_score_plateau_does_not_determine_optimum(exponent):
    # For a=1/2,L=2**-m, D_(m-2)>0 and D_(m-1)<0. The improvement can
    # be smaller than 1e-600: fixed Decimal80 score subtraction is insufficient.
    use, gamma, score = tk_slt_select_speculative(0.5, 2.0**-exponent, 0.0, 10**12)
    assert (use, gamma, score) == (True, exponent - 1, 2.0)


@pytest.mark.parametrize(
    "alpha,cost,cap,expected",
    [
        (0.9, 1e-300, 10**12, 6534),
        (math.nextafter(1.0, 0.0), 0.999999, 10**12, 134217),
        (math.nextafter(1.0, 0.0), math.ulp(0.0), 10**25, 6374424378296831398),
    ],
)
def test_large_optimum_matches_independent_800_digit_search(alpha, cost, cap, expected):
    # Reference: mpmath800 ternary search of S(g), then exhaustive final bracket.
    assert tk_slt_optimal_draft_length(alpha, cost, 0.0, cap)[0] == expected


def test_large_cap_is_not_enumerated():
    # Public-interface resource contract, not a private helper call-count mock.
    # subprocess.run terminates only its own child if an accidental O(cap) scan
    # is introduced, rather than leaving the test suite spinning indefinitely.
    subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            "from src.tk_slt_policy import tk_slt_optimal_draft_length as f; "
            "assert f(.5,.001,0.,10**100)[0] == 8",
        ],
        cwd=Path(__file__).resolve().parents[1],
        timeout=10,
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("alpha,cost", [(0.0, 0.0), (1.0, 1.0), (0.5, 0.5)])
def test_exact_ties_choose_smallest_gamma_and_standalone(alpha, cost):
    assert tk_slt_select_speculative(alpha, cost, 0.0, 64) == (False, 1, 1.0)


def test_saturated_acceptance_with_cost_above_one_does_not_choose_cap():
    use, gamma, score = tk_slt_select_speculative(1.0, 1.2, 0.0, 64)
    assert (use, gamma) == (False, 1)
    assert score == 2 / 2.2


def test_split_cost_uses_the_documented_rounded_sum():
    a, b, c = 0.3, 0.1, 0.2
    scores = _exhaustive_scores(a, b + c, 64)
    use, gamma, score = tk_slt_select_speculative(a, b, c)
    assert gamma == max(range(1, 65), key=lambda g: scores[g - 1])
    assert use == (scores[gamma - 1] > 1)
    assert score == float(scores[gamma - 1])


@pytest.mark.parametrize(
    "gamma,alpha,cost",
    [
        (0, 0.9, 1e308),
        (64, 0.5, 1e308),
        (64, math.nextafter(1.0, 0.0), 0.999999),
        (10**12, 1.0, 1e308),
    ],
)
def test_scalar_formula_avoids_cancellation_and_intermediate_overflow(
    gamma, alpha, cost
):
    a, cost_fraction = Fraction(alpha), Fraction(cost)
    total = Fraction(gamma + 1) if alpha == 1 else sum(a**j for j in range(gamma + 1))
    expected = float(total / (1 + gamma * cost_fraction))
    assert abs(tk_slt_speedup_ratio(gamma, alpha, cost) - expected) <= math.ulp(
        expected
    )


def test_extremely_large_integer_scalar_is_reported_without_float_conversion_overflow():
    gamma = 10**400
    assert math.isinf(tk_slt_speedup_ratio(gamma, 1.0, 0.0))
    assert tk_slt_speedup_ratio(gamma, 1.0, 1.0) == 1.0
    expected = float(Fraction(2) / (1 + gamma * Fraction(math.ulp(0.0))))
    assert tk_slt_speedup_ratio(gamma, 0.5, math.ulp(0.0)) == expected


@pytest.mark.parametrize(
    "alpha,b,c,cap",
    [
        (-0.1, 0.0, 0.0, 64),
        (1.1, 0.0, 0.0, 64),
        (math.nan, 0.0, 0.0, 64),
        (math.inf, 0.0, 0.0, 64),
        (0.5, -0.1, 0.2, 64),
        (0.5, 0.2, -0.1, 64),
        (0.5, math.nan, 0.0, 64),
        (0.5, 0.0, math.inf, 64),
        (0.5, 1e308, 1e308, 64),
        (0.5, 0.0, 0.0, 0),
        (0.5, 0.0, 0.0, -1),
        (0.5, 0.0, 0.0, 1.5),
        (0.5, 0.0, 0.0, True),
    ],
)
def test_invalid_policy_inputs_raise_instead_of_clamping(alpha, b, c, cap):
    with pytest.raises(ValueError):
        tk_slt_select_speculative(alpha, b, c, cap)


@pytest.mark.parametrize("z", [0.0, -0.0, -1 / math.e, -0.4, math.inf, math.nan])
def test_lambert_open_domain_is_explicit(z):
    with pytest.raises(ValueError):
        _lambert_w_minus_one(z)


@pytest.mark.parametrize(
    "gamma,alpha,cost",
    [
        (-1, 0.5, 0.1),
        (1.5, 0.5, 0.1),
        (True, 0.5, 0.1),
        (1, -1.0, 0.1),
        (1, 0.5, -0.1),
        (1, 0.5, math.inf),
    ],
)
def test_scalar_invalid_inputs_raise(gamma, alpha, cost):
    with pytest.raises(ValueError):
        tk_slt_speedup_ratio(gamma, alpha, cost)


def test_caller_decimal_context_does_not_change_policy_or_leak():
    with localcontext() as caller:
        caller.prec = 2
        caller.rounding = ROUND_DOWN
        caller.traps[Inexact] = True
        assert tk_slt_select_speculative(0.5, 0.001, 0.0) == (
            True,
            8,
            1.9802517361111112,
        )
        assert _lambert_w_minus_one(-0.2) == -2.5426413577735265
        assert tk_slt_optimal_draft_length(0.9, 1e-300, 0.0, 10**12)[0] == 6534
        assert caller.prec == 2 and caller.rounding == ROUND_DOWN
        assert caller.traps[Inexact]
