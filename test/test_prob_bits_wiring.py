"""B15：上行概率载荷位宽（--prob_payload_bits）的接线闭环测试。

- 原子 helper：量化与计费位宽必须成对返回（杜绝"声明 bits 不量化"）
- argparse 校验：不支持的模式显式拒绝 bits<16（此前静默忽略）
- 能力表：supports_prob_bits 精确覆盖 tri 系双投机阶段协议
"""

from argparse import Namespace
from unittest.mock import patch

import pytest
import torch

from src.baselines import _uplink_prob_payload
from src.mode_features import MODE_FEATURES


def _args(bits):
    return Namespace(prob_payload_bits=bits)


class TestUplinkProbPayloadAtomicity:
    def test_bits8_quantizes_and_bills(self):
        probs = torch.softmax(torch.randn(1, 4, 32), dim=-1)
        out, bits = _uplink_prob_payload(probs, _args(8))
        assert bits == 8
        assert out.shape == probs.shape
        assert out.dtype == probs.dtype
        # 值确实被量化：与原分布不同（默认 16 路径则完全相同）
        assert not torch.equal(out, probs)

    def test_bits16_passes_through_unchanged(self):
        probs = torch.softmax(torch.randn(1, 4, 32), dim=-1)
        out, bits = _uplink_prob_payload(probs, _args(16))
        assert bits is None
        assert out is probs  # 原对象，零开销

    def test_none_probs(self):
        assert _uplink_prob_payload(None, _args(8)) == (None, None)

    def test_none_bits_arg_falls_back_to_16(self):
        # getattr(..., 16) or 16 的容错：None/0 视为默认全宽
        probs = torch.softmax(torch.randn(1, 4, 32), dim=-1)
        out, bits = _uplink_prob_payload(probs, Namespace(prob_payload_bits=None))
        assert bits is None and out is probs

    def test_quantized_is_normalized(self):
        # 量化后仍为分布（和为 1）——接受判据的比值不失真
        probs = torch.softmax(torch.randn(1, 4, 32), dim=-1)
        out, _ = _uplink_prob_payload(probs, _args(4))
        assert torch.allclose(out.sum(dim=-1), torch.ones(1, 4), atol=1e-4)


class TestProbBitsModeGate:
    def test_unsupported_mode_rejects_low_bits(self, monkeypatch):
        import src.utils as utils

        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--eval_mode", "sd",
                "--draft_model", "llama-68m",
                "--target_model", "llama-2-13b",
                "--prob_payload_bits", "8",
            ],
        )
        with patch("src.utils.model_zoo"):
            with pytest.raises(SystemExit) as e:
                utils.parse_arguments()
            assert e.value.code == 2

    def test_supported_mode_accepts_low_bits(self, monkeypatch):
        import src.utils as utils

        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--eval_mode", "tridecoding",
                "--draft_model", "llama-68m",
                "--target_model", "llama-2-13b",
                "--prob_payload_bits", "8",
            ],
        )
        with patch("src.utils.model_zoo"):
            args = utils.parse_arguments()
        assert args.prob_payload_bits == 8

    def test_default_16_never_rejected(self, monkeypatch):
        import src.utils as utils

        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--eval_mode", "sd",
                "--draft_model", "llama-68m",
                "--target_model", "llama-2-13b",
            ],
        )
        with patch("src.utils.model_zoo"):
            args = utils.parse_arguments()
        assert args.prob_payload_bits == 16


def test_supports_prob_bits_covers_exactly_tri_protocols():
    # B15 能力位精确集：仅两个双投机阶段协议
    supported = {
        name for name, spec in MODE_FEATURES.items() if spec.supports_prob_bits
    }
    assert supported == {"tridecoding", "adaptive_tridecoding"}
