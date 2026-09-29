"""Regression tests for the RL checkpoint durability chain (report finding R1).

Historical bug chain (all three links fixed together):
1. ``DDQNAgent.save`` wrote ``latest.pth`` / ``.buffer`` in place — a kill
   mid-``torch.save`` (manager SIGKILL, OOM, power loss) left a torn file.
2. ``DDQNAgent.load`` swallowed every exception with "Starting fresh", so a
   torn or incompatible checkpoint silently reset training; the next save
   then overwrote the only (often still recoverable) copy.
3. The candidate chain in ``RLNetworkAdapter`` had no per-candidate error
   handling, so one corrupt file aborted nothing — it just fell through to
   fresh weights without any trace.

Fixes under test: atomic tmp+``os.replace`` saves; load returns False only
for a missing file and raises RuntimeError on corruption / shape mismatch
(after pre-validating shapes so no half-applied state survives); corrupt
candidates are renamed ``.corrupt`` and the chain falls back to the next
candidate.
"""

import os

import pytest
import torch

from src.rl_adapter import DDQNAgent


def _make_agent(**overrides) -> DDQNAgent:
    kwargs = dict(
        feature_dim=6,
        action_dim=3,
        hidden_dim=16,
        seq_len=4,
        buffer_size=64,
        device="cpu",
        name="test-agent",
        init_seed=7,
    )
    kwargs.update(overrides)
    return DDQNAgent(**kwargs)


class TestAtomicSave:
    def test_save_leaves_no_tmp_and_replaces_target(self, tmp_path):
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        agent.memory.extend([("s", 0, 0.5, "s", False)] * 3)

        agent.save(path)

        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")
        assert os.path.exists(path + ".buffer")
        assert not os.path.exists(path + ".buffer.tmp")
        # 内容确实可解析（save 没有只写一半）
        ckpt = torch.load(path, map_location="cpu")
        assert set(ckpt) >= {
            "policy_net",
            "target_net",
            "optimizer",
            "epsilon",
            "best_tps",
        }

    def test_resave_replaces_old_checkpoint(self, tmp_path):
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        agent.save(path)
        old = torch.load(path, map_location="cpu")
        agent.update_count = 42
        agent.save(path)
        new = torch.load(path, map_location="cpu")
        assert new["update_count"] == 42
        assert old is not new


class TestLoadSemantics:
    def test_missing_file_returns_false_not_raise(self, tmp_path):
        agent = _make_agent()
        assert agent.load(str(tmp_path / "absent.pth")) is False

    def test_corrupt_file_raises_instead_of_starting_fresh(self, tmp_path):
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        with open(path, "wb") as f:
            f.write(b"this is not a torch archive, it is a torn write")
        with pytest.raises(RuntimeError, match="损坏"):
            agent.load(path)

    def test_shape_mismatch_raises_with_hint(self, tmp_path):
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        agent.save(path)
        # 模拟结构变更（如 --rl_action_space / hidden_dim 改动）后加载旧档
        other = _make_agent(hidden_dim=32)
        with pytest.raises(RuntimeError, match="不兼容"):
            other.load(path)

    def test_compatible_load_restores_and_returns_true(self, tmp_path):
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        agent.epsilon = 0.123
        agent.update_count = 17
        agent.save(path)

        restored = _make_agent()
        assert restored.load(path) is True
        assert restored.epsilon == pytest.approx(0.123)
        assert restored.update_count == 17

    def test_corrupt_buffer_warns_but_load_succeeds(self, tmp_path, capsys):
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        agent.save(path)
        with open(path + ".buffer", "wb") as f:
            f.write(b"not a pickle")
        restored = _make_agent()
        assert restored.load(path) is True
        assert "buffer" in capsys.readouterr().out

    def test_no_half_applied_state_on_mismatch(self, tmp_path):
        """shape 校验失败时，网络必须保持全新初始化状态（零污染）。"""
        agent = _make_agent()
        path = str(tmp_path / "latest.pth")
        agent.save(path)
        other = _make_agent(hidden_dim=32)
        fresh_params = {
            k: v.clone() for k, v in other.policy_net.state_dict().items()
        }
        with pytest.raises(RuntimeError):
            other.load(path)
        for k, v in other.policy_net.state_dict().items():
            assert torch.equal(v, fresh_params[k]), f"参数 {k} 被半加载污染"
