"""TK-SLT ODLD/AS² policy math, independent of torch and communication.

This module owns the paper's draft-length decision (Theorem 2 / Algorithms 1
and 2).  It is a pure function family: no model, KV cache, RNG or wire state is
read here, so both the decode loop and focused CPU tests can call it directly.

The implementations were moved verbatim out of ``baselines_helpers``; the old
names remain re-exported there for compatibility.  Arithmetic, guard ordering
and return values are intentionally unchanged. This includes known legacy
numeric limitations: the fixed Lambert bracket and exponent rounding can fail
on valid parameters (e.g. alpha=.5, L=.0005). Extraction is not certification of
those boundaries; the separate numerical follow-up must address them.
"""

from __future__ import annotations

import math
from typing import Tuple


def _lambert_w_minus_one(z: float) -> float:
    """Lambert W 的 −1 分支：解 w·e^w = z，z ∈ (−1/e, 0)，返回 w ≤ −1。

    论文 Theorem 2 式 (21) 需要 W_{−1}。g(w)=w·e^w 在 (−∞,−1] 上从 0⁻
    单调降到 −1/e，故对给定 z ∈ (−1/e, 0) 解唯一。这里保留旧固定区间
    与 100 次二分，受 FP64 精度限制；并未保证覆盖整个定义域或任意精度。
    """
    if not (-1.0 / math.e < z < 0.0):
        raise ValueError(f"W_{{-1}} 的定义域是 (−1/e, 0)，收到 z={z!r}")
    lo, hi = -200.0, -1.0  # Historical bracket; not sufficient for every valid z.
    for _ in range(100):
        mid = (lo + hi) / 2.0
        if mid * math.exp(mid) > z:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def tk_slt_speedup_ratio(gamma: int, alpha: float, L: float) -> float:
    """TK-SLT 论文式 (11)：S_inf(γ) = (1−α^{γ+1}) / ((1−α)(1+γL))。

    α=1 时取极限 (γ+1)/(1+γL)（式 (12) 的期望 token 数除以式 (7) 的
    归一化时延）；α=0 时为 1/(1+γL)。
    """
    gamma = int(gamma)
    if alpha >= 1.0:
        return (gamma + 1) / (1.0 + gamma * L)
    if alpha <= 0.0:
        return 1.0 / (1.0 + gamma * L)
    return (1.0 - alpha ** (gamma + 1)) / ((1.0 - alpha) * (1.0 + gamma * L))


def tk_slt_optimal_draft_length(
    alpha: float,
    b: float,
    c: float,
    gamma_cap: int = 64,
) -> Tuple[int, float]:
    """论文 Algorithm 1（ODLD）：闭式最优草稿长度 γ*（Theorem 2 式 (21)）。

        γ0 = (1/ln α)·(W_{−1}(−(1/e)·α^{1/L−1}) + 1) − 1/L，L = b + c

    γ0 < 1 ⇒ γ* = 1；否则在 {⌊γ0⌋, ⌈γ0⌉} 里取 S_inf 更大者（平局取
    ⌈γ0⌉，与论文 Algorithm 1 的 ≤ 分支一致）。返回 (γ*, S_inf(γ*))。

    论文约束外的退化输入（估计噪声 early rounds 常见）按保守方向处理：
    · α ≥ 1 / L ≤ 0：接受率饱和或相对开销为零 ⇒ 加大 γ 只赚不亏，
      取上限 gamma_cap（防噪声下的病态值）；
    · α ≤ 0：全拒 ⇒ γ* = 1（AS² 随后会选择 standalone LLM）；
    · L ≥ 1：论文式 (19) 要求 L < 1，此时 z 越界无解；取 γ* = 1，
      S_inf(1) = (1+α)/(1+L) < 1 恒成立 ⇒ AS² 会退回 standalone LLM。
    """
    L = float(b) + float(c)
    gamma_cap = max(1, int(gamma_cap))
    if L <= 0.0 or alpha >= 1.0:
        return gamma_cap, tk_slt_speedup_ratio(gamma_cap, alpha, L)
    if alpha <= 0.0 or L >= 1.0:
        return 1, tk_slt_speedup_ratio(1, alpha, L)

    z = -(1.0 / math.e) * alpha ** (1.0 / L - 1.0)
    w = _lambert_w_minus_one(z)
    gamma0 = (w + 1.0) / math.log(alpha) - 1.0 / L
    if gamma0 < 1.0:
        return 1, tk_slt_speedup_ratio(1, alpha, L)

    floor_g, ceil_g = math.floor(gamma0), math.ceil(gamma0)
    s_floor = tk_slt_speedup_ratio(floor_g, alpha, L)
    s_ceil = tk_slt_speedup_ratio(ceil_g, alpha, L)
    if s_floor <= s_ceil:
        gamma_star = ceil_g
    else:
        gamma_star = floor_g
    gamma_star = max(1, min(gamma_star, gamma_cap))
    return gamma_star, tk_slt_speedup_ratio(gamma_star, alpha, L)


def tk_slt_select_speculative(
    alpha: float,
    b: float,
    c: float,
    gamma_cap: int = 64,
) -> Tuple[bool, int, float]:
    """论文 Algorithm 2（AS²）：S_inf(γ*) > 1 才用 DSD，否则 standalone LLM。

    返回 (use_dsd, γ*, S*)。与 ODLD 一样吃 (α, b, c)——在线使用时由
    运行估计喂入（接受率、T_V/T_LLM、T_SLM/T_LLM）。
    """
    gamma_star, s_star = tk_slt_optimal_draft_length(alpha, b, c, gamma_cap)
    return s_star > 1.0, gamma_star, s_star
