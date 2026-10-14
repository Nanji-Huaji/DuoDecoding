"""CEE-SD 阈值学偏修复的回归测试。

背景（experiment_results/ceesd_gamma_threshold_probe.md）：
- 根因①：fluid+per_round 下 RL 奖励的通信项在 flush 前读差值恒为 0，
  训练奖励 ≡ accepted − λ·compute——NTT/排队/带宽三项成本全部不可见。
  修复 = --rl_comm_cost_mode nominal（NTT + 本轮 pending 字节/带宽）。
- 根因②：main agent 阈值候选格上限 0.4，深草稿不可达。修复 =
  --rl_threshold_candidates 扩格 + 候选格随 checkpoint 持久化、加载时
  自动还原（防训练/评测格错配）。
"""

from argparse import Namespace

import pytest

from src.baselines import _reward_comm_s
from src.communication import CommunicationSimulator
from src.rl_adapter import RLNetworkAdapter


def _sim() -> CommunicationSimulator:
    sim = CommunicationSimulator(
        46.0,
        563.0,
        46.0,
        dimension="Mbps",
        ntt_ms_edge_cloud=50.0,
        ntt_ms_edge_end=0.317,
        use_stochastic=False,
        bw_model="fluid",
    )
    sim.coalesce_rounds = True
    sim.set_round(0)
    return sim


class TestNominalComm:
    def test_ntt_only_when_no_pending_bytes(self):
        sim = _sim()
        assert sim.pending_bytes("edge_cloud") == 0.0
        assert sim.nominal_comm_s("edge_cloud") == pytest.approx(0.05)

    def test_ntt_plus_pending_drain(self):
        sim = _sim()
        # 575000 B @ 46 Mbps = 0.1 s 排水
        sim.simulate_transfer(575_000, "edge_cloud")
        assert sim.pending_bytes("edge_cloud") == 575_000.0
        assert sim.nominal_comm_s("edge_cloud") == pytest.approx(0.15)

    def test_edge_end_link_independent(self):
        sim = _sim()
        sim.simulate_transfer(70_375, "edge_end")  # 70375B @ 563Mbps = 1ms
        assert sim.nominal_comm_s("edge_end") == pytest.approx(0.001317)

    def test_flush_clears_pending_and_settles_bill(self):
        sim = _sim()
        sim.simulate_transfer(575_000, "edge_cloud")
        sim.flush_round()
        assert sim.pending_bytes("edge_cloud") == 0.0
        assert sim.edge_cloud_comm_time > 0.0


class TestRewardCommModes:
    """根因①的语义回归。nominal 轮内可读（修复后行为）；settled 已删除
    （Phase 1，2026-10-11）——其轮内差值恒 0 是坏观测本身，删除后传入
    必须显式失败而非静默降级。"""

    def test_settled_mode_is_rejected(self):
        """settled 已删除：传入即 ValueError，不静默落到任何口径。"""
        sim = _sim()
        with pytest.raises(ValueError, match="settled"):
            _reward_comm_s(
                Namespace(rl_comm_cost_mode="settled"), sim, "edge_cloud"
            )

    def test_nominal_mode_reads_ntt_plus_drain_mid_round(self):
        sim = _sim()
        sim.simulate_transfer(575_000, "edge_cloud")
        assert _reward_comm_s(
            Namespace(rl_comm_cost_mode="nominal"), sim, "edge_cloud"
        ) == pytest.approx(0.15)

    def test_default_mode_is_nominal(self):
        sim = _sim()
        sim.simulate_transfer(575_000, "edge_cloud")
        # Namespace 缺属性（历史调用方）→ getattr 默认 nominal
        assert _reward_comm_s(Namespace(), sim, "edge_cloud") == pytest.approx(
            0.15
        )

    def test_off_mode_is_zero(self):
        sim = _sim()
        sim.simulate_transfer(575_000, "edge_cloud")
        assert (
            _reward_comm_s(
                Namespace(rl_comm_cost_mode="off"), sim, "edge_cloud"
            )
            == 0.0
        )

    def test_edge_end_nominal(self):
        sim = _sim()
        sim.simulate_transfer(70_375, "edge_end")
        assert _reward_comm_s(
            Namespace(rl_comm_cost_mode="nominal"), sim, "edge_end"
        ) == pytest.approx(0.001317)


class TestActionGridPersistence:
    """根因②的防错配回归：候选格随 checkpoint 走。"""

    EXTENDED_GRID = [0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.99]
    REGISTRY_GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]

    def test_grid_round_trip(self, tmp_path):
        checkpoint = tmp_path / "latest.pth"
        first = RLNetworkAdapter(
            Namespace(),
            model_path=str(checkpoint),
            best_model_path=str(checkpoint),
            device="cpu",
            init_seed=73,
            init_strategy="fresh",
            threshold_candidates=self.EXTENDED_GRID,
        )
        first.save(1.0)

        # 构造侧换成 registry 默认格（同长度、不同标签）——同一 action_idx
        # 的语义会整体漂移，加载必须还原训练格
        resumed = RLNetworkAdapter(
            Namespace(),
            model_path=str(checkpoint),
            best_model_path=str(checkpoint),
            device="cpu",
            init_seed=73,
            init_strategy="resume",
            threshold_candidates=self.REGISTRY_GRID,
        )
        assert resumed.threshold_candidates == self.EXTENDED_GRID
        assert resumed.topk_candidates is not None

    def test_grid_saved_into_checkpoint_payload(self, tmp_path):
        import torch

        checkpoint = tmp_path / "latest.pth"
        first = RLNetworkAdapter(
            Namespace(),
            model_path=str(checkpoint),
            best_model_path=str(checkpoint),
            device="cpu",
            init_seed=73,
            init_strategy="fresh",
            threshold_candidates=self.EXTENDED_GRID,
        )
        first.save(1.0)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["threshold_candidates"] == self.EXTENDED_GRID
        assert "topk_candidates" in payload

    def test_force_indices_reresolved_after_grid_adoption(self, tmp_path):
        checkpoint = tmp_path / "latest.pth"
        first = RLNetworkAdapter(
            Namespace(),
            model_path=str(checkpoint),
            best_model_path=str(checkpoint),
            device="cpu",
            init_seed=73,
            init_strategy="fresh",
            threshold_candidates=self.EXTENDED_GRID,
        )
        first.save(1.0)

        resumed = RLNetworkAdapter(
            Namespace(rl_force_threshold=0.2),
            model_path=str(checkpoint),
            best_model_path=str(checkpoint),
            device="cpu",
            init_seed=73,
            init_strategy="resume",
            threshold_candidates=self.REGISTRY_GRID,
        )
        # 0.2 在两个格里都在档位 1；但解析必须发生在**还原后**的格上
        assert resumed.threshold_candidates == self.EXTENDED_GRID
        assert resumed._force_threshold_idx == 1
        assert resumed._force_threshold_value == 0.2
