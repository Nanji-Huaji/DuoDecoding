"""Pure TK-SLT ODLD/AS² policy; compatibility names remain in helpers.

Optimize the bounded discrete objective, not a rounded Lambert stationary point.
Inputs are binary64 estimates; L is the ordinary rounded sum b+c. Comparisons
use exact rationals or directed Decimal enclosures, not rounded speedup scores.
The reported score is a floating approximation. Exact ties choose smaller gamma.
"""

from __future__ import annotations

import math
import operator
from decimal import (
    MAX_EMAX,
    MIN_EMIN,
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    Context,
    Decimal,
    DivisionByZero,
    InvalidOperation,
    Overflow,
    localcontext,
)
from fractions import Fraction
from typing import Tuple


def _context(precision, rounding=ROUND_HALF_EVEN):
    # Do not inherit a caller's Decimal precision, rounding mode or Inexact trap.
    return Context(
        prec=precision,
        rounding=rounding,
        Emin=MIN_EMIN,
        Emax=MAX_EMAX,
        traps=[InvalidOperation, DivisionByZero, Overflow],
    )


def _integer(value, name, minimum):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    try:
        result = operator.index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer >= {minimum}") from error
    if result < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return result


def _estimate(value, name, maximum=math.inf):
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= maximum:
        raise ValueError(f"{name} must be finite and in [0, {maximum}]")
    return result


def _cost(b, c):
    return _estimate(_estimate(b, "b") + _estimate(c, "c"), "b+c")


def _lambert_w_minus_one(z: float) -> float:
    """Binary64 W_-1 approximation for representable -1/e < z < 0.

    Solve t-ln(t)=-ln(-z), t>=1, instead of underflowing w*exp(w).
    Decimal guard digits also retain the distance from the branch point that
    a binary64 logarithm can round away. This compatibility helper is no longer
    used by ODLD; no 1e-28 absolute-accuracy claim is made for its float result.
    """
    z = float(z)
    if not (-1.0 / math.e < z < 0.0):
        raise ValueError(f"W_{{-1}} 的定义域是 (−1/e, 0)，收到 z={z!r}")
    with localcontext(_context(80)):
        target = -Decimal.from_float(-z).ln()
        lo, hi = Decimal(1), target + target.ln() + 2
        for _ in range(256):
            if float(lo) == float(hi):
                break
            mid = (lo + hi) / 2
            if mid - mid.ln() < target:
                lo = mid
            else:
                hi = mid
        return -float((lo + hi) / 2)


def tk_slt_speedup_ratio(gamma: int, alpha: float, L: float) -> float:
    """Report sum(alpha**j, j=0..gamma)/(1+gamma*L), including gamma=0.

    Decimal guard digits avoid cancellation and intermediate binary64 overflow.
    The result is a float, not an exact score/tie oracle; extreme scores may
    legitimately underflow to zero or overflow to infinity on conversion.
    """
    gamma = _integer(gamma, "gamma", 0)
    alpha, L = _estimate(alpha, "alpha", 1), _estimate(L, "L")
    with localcontext(_context(80)):
        a, cost = Decimal.from_float(alpha), Decimal.from_float(L)
        if alpha == 1:
            total = Decimal(gamma + 1)
        elif alpha == 0:
            total = Decimal(1)
        else:
            # For any binary64 alpha<1, alpha**(2**64)<exp(-2048).
            # Its omitted contribution is far below a float's representable
            # absolute resolution; this is only the reported score, not search.
            power = Decimal(0) if gamma >= 2**64 else a ** (gamma + 1)
            total = (1 - power) / (1 - a)
        return float(total / (1 + Decimal(gamma) * cost))


def _power_bound(base, exponent, context):
    """Positive integer power with directed rounding at *every* multiplication."""
    result = Decimal(1)
    while exponent:
        if exponent & 1:
            result = context.multiply(result, base)
        exponent >>= 1
        if exponent:
            base = context.multiply(base, base)
    return result


def _increasing(gamma, alpha, cost, decimal_alpha):
    """Whether S(g+1)>S(g), without cancellation of two rounded scores.

    Equivalent to alpha**(g+1) > L/[L+(1-alpha)*(1+gL)]. Small powers are
    compared as integers. Large ones use successively refined enclosures; no
    arbitrary epsilon makes the decision and no full-cap enumeration is used.
    """
    threshold = cost / (cost + (1 - alpha) * (1 + gamma * cost))
    n = gamma + 1
    binary_exponent = (alpha.denominator.bit_length() - 1) * n
    if n <= 4096:
        left = pow(alpha.numerator, n) * threshold.denominator
        right_bits = threshold.numerator.bit_length() + binary_exponent
        if left.bit_length() != right_bits:
            return left.bit_length() > right_bits
        return left > threshold.numerator << binary_exponent

    # A rational exact equality must not make interval refinement run forever.
    # alpha has a power-of-two denominator; equality can only occur if the
    # threshold has the same reduced denominator. Its numerator then bounds
    # the size of the one exact power needed to check equality.
    denominator = threshold.denominator
    if (
        denominator & (denominator - 1) == 0
        and denominator.bit_length() - 1 == binary_exponent
    ):
        numerator = alpha.numerator
        if (
            n * (numerator.bit_length() - 1) + 1 <= threshold.numerator.bit_length()
            and pow(numerator, n) == threshold.numerator
        ):
            return False

    precision = 64
    while True:
        lower, upper = (
            _context(precision, ROUND_FLOOR),
            _context(precision, ROUND_CEILING),
        )
        numerator, denominator = (
            Decimal(threshold.numerator),
            Decimal(threshold.denominator),
        )
        threshold_lo = lower.divide(numerator, denominator)
        threshold_hi = upper.divide(numerator, denominator)
        power_lo = _power_bound(decimal_alpha, n, lower)
        power_hi = _power_bound(decimal_alpha, n, upper)
        if power_lo > threshold_hi:
            return True
        if power_hi < threshold_lo:
            return False
        precision *= 2


def tk_slt_optimal_draft_length(
    alpha: float,
    b: float,
    c: float,
    gamma_cap: int = 64,
) -> Tuple[int, float]:
    """Maximize the discrete ODLD objective over integers 1..gamma_cap.

    D_g = alpha**(g+1)*(1+gL) - L*sum(alpha**j, j=0..g) has the sign of
    S(g+1)-S(g), and D_(g+1)-D_g = -(1-alpha)*alpha**(g+1)*(1+(g+1)L).
    Thus exponential bracketing and binary search find the first non-increase
    in O(log gamma_cap) comparisons. Precision cost depends on the inputs.
    Public estimates must be finite, 0<=alpha<=1, b,c>=0; cap is a positive
    integer. Invalid inputs raise instead of silently clamping the objective.
    """
    alpha, cost = _estimate(alpha, "alpha", 1), _cost(b, c)
    cap = _integer(gamma_cap, "gamma_cap", 1)
    if cap == 1 or alpha == 0 or cost >= alpha:
        gamma = 1
    elif cost == 0 or alpha == 1:
        gamma = cap
    else:
        a, cost_fraction = Fraction(alpha), Fraction(cost)
        da = Decimal.from_float(alpha)
        if not _increasing(1, a, cost_fraction, da):
            gamma = 1
        else:
            lo, hi = 1, min(2, cap)
            while hi < cap and _increasing(hi, a, cost_fraction, da):
                lo, hi = hi, min(2 * hi, cap)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if _increasing(mid, a, cost_fraction, da):
                    lo = mid
                else:
                    hi = mid
            gamma = hi
    return gamma, tk_slt_speedup_ratio(gamma, alpha, cost)


def tk_slt_select_speculative(
    alpha: float,
    b: float,
    c: float,
    gamma_cap: int = 64,
) -> Tuple[bool, int, float]:
    """AS² uses the mathematical S*>1 condition, not a rounded report.

    S(g)>1 iff the average of alpha**1..alpha**g exceeds L. This average
    cannot exceed alpha, and gamma=1 attains alpha. Thus S*>1 iff alpha>L.
    A tiny positive gain can legitimately return True alongside float score 1.
    """
    gamma, score = tk_slt_optimal_draft_length(alpha, b, c, gamma_cap)
    return float(alpha) > _cost(b, c), gamma, score
