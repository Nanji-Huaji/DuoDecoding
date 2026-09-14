"""Reward design for the CEE-SD network adapters (DRA).

Why this module exists
----------------------
The v1 reward used in the paper submission is

    r = exp(min(N_acc / (T_wall + T_comm), 100) / 20) * (N_acc / gamma)^2      (v1)

with three hand-tuned ingredients and two structural problems:

  * `exp(./20)` + `min(., 100)` + `reward_scale=0.01`: magic constants that
    compensate each other's scale.  The exponential is *convex*, so under the
    (measured, +-17%) host-load jitter of the compute time it inflates the
    value of high-variance actions (Jensen), and the cap flattens the gradient
    exactly in the fast-link regime the controller is supposed to exploit.
  * `(N_acc/gamma)^2`: measured to span 0.001..1.0 across samples, i.e. reward
    variance is dominated by task difficulty rather than by the agent's action.
    It also rewards *small* gamma (a small draft raises N_acc/gamma), which is
    the opposite of what a fast link wants.  Over-drafting is already paid for
    by the verification compute time, so the term is redundant as well as
    harmful.
  * `T_wall`: the compute time is measured with the wall clock, so the same
    action earns a different reward on a busy host.  The learned policy is then
    optimal for "this machine right now", not for "this network condition".

The principled objective
------------------------
The system-level goal is tokens per unit time, i.e. max E[N] / E[T] (the
renewal-reward / semi-Markov average-reward criterion, which is what Eq. 1 of
the paper states).  Its Lagrangian relaxation is exact at the optimal shadow
price lambda* and gives a *linear* per-decision reward:

    r_t = N_acc,t - lambda * T_t,        T_t = T_comm,t + T_comp,t        (v2)

with `lambda` = shadow price of time (tokens/s).  No warp, no cap, no
acceptance multiplier: rejected drafts already cost verification compute, which
enters through T_comp.

Variants
--------
    legacy      v1, kept bit-exact so earlier results stay reproducible
    linear      N_acc / T                     (raw instantaneous goodput)
    lagrangian  N_acc - lambda * T            (v2, recommended default)
    slo         N_acc * 1{T <= D} - c * max(0, T - D)   (deadline / real-time)
    energy      N_acc - lambda * T - mu * E_comm        (green inference)

`T` uses either the legacy wall clock or a *host-independent* compute model
(`compute_mode="model"`), which is what makes the reward reproducible across
machines and removes the incentive to game the current machine load.

Any additional shaping term must be a potential difference
(F(s, s') = gamma * phi(s') - phi(s), Ng et al. 1999) to leave the optimal
policy unchanged; `potential_shaping()` is provided for that purpose.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class RewardConfig:
    mode: str = "legacy"           # legacy | linear | lagrangian | slo | energy
    # lagrangian / energy shadow price (tokens per second); 0 -> adaptive EMA
    lam: float = 0.0
    lam_ema: float = 0.99
    lam_init: float = 20.0
    # slo
    deadline_ms: float = 0.0
    deadline_penalty: float = 1.0
    # energy
    energy_weight: float = 0.0
    # byte price as an equivalent *time* penalty (seconds per megabyte) added to
    # the time term; 0 keeps the historical behaviour (bytes are then only priced
    # through the physical transmission time inside comm_s).  See F17.
    byte_price_s_per_mb: float = 0.0
    # compute-time source: "wall" (legacy, host dependent) or "model"
    compute_mode: str = "wall"
    # seconds per forward pass, per model; used when compute_mode == "model"
    compute_cost_s: dict[str, float] = field(default_factory=lambda: {
        "little": 0.0, "draft": 0.0, "target": 0.0,
    })
    # legacy-only knobs (kept so v1 can be reproduced exactly)
    legacy_warp: float = 20.0
    legacy_cap: float = 100.0
    legacy_alpha2: bool = True
    # v1's "opportunistic" first stage returned the raw N_acc/T (no warp, no
    # alpha^2); the CEE-SD tri-decoding path always ran the little adapter that
    # way, so this must be preserved for bit-compatible legacy behaviour.
    opportunistic: bool = False


class RewardShaper:
    """Turns one decoding-stage decision into a scalar reward + its components.

    The caller passes the *measured* quantities of the decision interval:

        accepted      : number of draft tokens accepted (tokens emitted)
        generated     : number of draft tokens proposed (gamma)
        comm_s        : simulated communication time of this interval (seconds)
        compute_s     : measured compute time of this interval (seconds)
        energy_j      : simulated communication energy of this interval (joules)
        forward_counts: {"little": n, "draft": n, "target": n} forward passes

    and gets back (reward, components).  `components` is meant to be logged --
    a reward that cannot be decomposed cannot be debugged.
    """

    def __init__(self, cfg: RewardConfig | None = None):
        self.cfg = cfg or RewardConfig()
        if self.cfg.mode not in {"legacy", "linear", "lagrangian", "slo", "energy"}:
            raise ValueError(f"unknown reward mode: {self.cfg.mode}")
        self._lam = float(self.cfg.lam) if self.cfg.lam > 0 else float(self.cfg.lam_init)
        self._lam_fixed = self.cfg.lam > 0

    # ---------------------------------------------------------------- helpers
    @property
    def lam(self) -> float:
        return self._lam

    def model_compute_s(self, forward_counts: dict[str, float] | None) -> float:
        if not forward_counts:
            return 0.0
        cost = self.cfg.compute_cost_s
        return float(sum(cost.get(k, 0.0) * float(v) for k, v in forward_counts.items()))

    def update_lambda(self, accepted: float, total_time_s: float) -> None:
        """Track the shadow price as the observed tokens/second of the policy."""
        if self._lam_fixed or total_time_s <= 0:
            return
        rate = accepted / total_time_s
        a = self.cfg.lam_ema
        self._lam = a * self._lam + (1.0 - a) * rate

    # ----------------------------------------------------------------- reward
    def compute(
        self,
        *,
        accepted: float,
        generated: float,
        comm_s: float,
        compute_s: float,
        energy_j: float = 0.0,
        forward_counts: dict[str, float] | None = None,
        opportunistic: bool | None = None,
        transferred_bytes: float = 0.0,
    ) -> tuple[float, dict]:
        cfg = self.cfg
        t_comp = (
            self.model_compute_s(forward_counts)
            if cfg.compute_mode == "model"
            else float(compute_s)
        )
        # 字节定价（F17）：通信时间里的发送分量只有 3–9%，所以策略会乐意用 3× 的
        # 字节换往返次数。把字节按"等效秒"计价后就把它变成显式的部署旋钮：
        #   byte_price_s_per_mb = 0    -> 只按物理发送时间计（默认，保持可复现）
        #   byte_price_s_per_mb = 16   -> 0.5 Mbps 的物理发送成本（1 MB / 0.0625 MB/s）
        #   byte_price_s_per_mb > 16   -> 模拟按流量计费/更贵链路
        byte_s = (float(transferred_bytes) / 1e6) * float(cfg.byte_price_s_per_mb)
        t_total = max(float(comm_s) + t_comp + byte_s, 1e-9)
        comp = {
            "mode": cfg.mode,
            "accepted": float(accepted),
            "generated": float(generated),
            "comm_s": float(comm_s),
            "compute_s": t_comp,
            "compute_source": cfg.compute_mode,
            "total_s": t_total,
            "energy_j": float(energy_j),
            "transferred_bytes": float(transferred_bytes),
            "byte_price_s": byte_s,
            "lam": self._lam,
        }

        if cfg.mode == "legacy":
            tps_part = accepted / t_total
            comp["tps_part"] = tps_part
            comp["cap_hit"] = float(tps_part >= cfg.legacy_cap)
            opp = cfg.opportunistic if opportunistic is None else bool(opportunistic)
            comp["opportunistic"] = float(opp)
            if opp:  # v1 first stage: raw instantaneous goodput
                return tps_part, comp
            reward = math.exp(min(tps_part, cfg.legacy_cap) / cfg.legacy_warp)
            if cfg.legacy_alpha2 and generated > 1:
                acc_rate = accepted / generated
                comp["alpha2"] = acc_rate ** 2
                reward *= acc_rate ** 2
            return reward, comp

        if cfg.mode == "linear":
            reward = accepted / t_total
        elif cfg.mode == "lagrangian":
            reward = accepted - self._lam * t_total
        elif cfg.mode == "slo":
            budget = cfg.deadline_ms / 1000.0
            over = max(0.0, t_total - budget)
            comp["over_budget_s"] = over
            reward = (accepted if t_total <= budget else 0.0) - cfg.deadline_penalty * over
        elif cfg.mode == "energy":
            reward = accepted - self._lam * t_total - cfg.energy_weight * float(energy_j)
        else:  # pragma: no cover - guarded in __init__
            raise ValueError(cfg.mode)

        self.update_lambda(accepted, t_total)
        comp["reward"] = reward
        return reward, comp


def potential_shaping(phi_next: float, phi_state: float, gamma: float = 0.99) -> float:
    """F(s, s') = gamma * phi(s') - phi(s).

    Adding this to any reward leaves the optimal policy unchanged
    (Ng, Harada & Russell, ICML 1999), which is the only admissible way to add
    auxiliary incentives such as "draft tokens that look likely to be accepted".
    """
    return gamma * float(phi_next) - float(phi_state)


class RewardLogger:
    """Windowed reward statistics (the v1 code logged a whole-run mean, which
    tracks curriculum difficulty rather than policy improvement)."""

    def __init__(self, window: int = 200):
        self.window: deque[float] = deque(maxlen=window)
        self.count = 0

    def add(self, reward: float) -> None:
        self.window.append(float(reward))
        self.count += 1

    def summary(self) -> dict:
        if not self.window:
            return {"n": self.count, "mean": 0.0, "min": 0.0, "max": 0.0}
        w = self.window
        return {
            "n": self.count,
            "mean": sum(w) / len(w),
            "min": min(w),
            "max": max(w),
        }

    def format(self, components: dict | None = None) -> str:
        s = self.summary()
        text = (f"Reward(win)={s['mean']:.4f} [{s['min']:.3f},{s['max']:.3f}] n={s['n']}")
        if components:
            text += (
                f" | acc={components.get('accepted', 0):.0f}"
                f" gamma={components.get('generated', 0):.0f}"
                f" comm={components.get('comm_s', 0):.4f}s"
                f" comp={components.get('compute_s', 0):.4f}s({components.get('compute_source', '?')})"
                f" lam={components.get('lam', 0):.2f}"
            )
            if "cap_hit" in components:
                text += f" cap_hit={components['cap_hit']:.0f}"
        return text
