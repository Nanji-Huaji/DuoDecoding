"""Phase 1 地基：计费口径构造点。

本文件只测"读取是否正确、投影是否与协议一致"。计费数字的等价性由
各方法迁移时的 golden/fingerprint 负责——那时才会有生产路径 import
src.comm_accounting。
"""

from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from src.comm_accounting import DEFAULTS, FALLBACKS, PROFILES, CommConfig
from src.protocols import PROTOCOLS


def test_defaults_are_the_current_cli_defaults():
    assert DEFAULTS == {
        "comm_round_trip_mode": "per_round",
        "charge_residual_payload": True,
        "transfer_top_k_cap": 16,
        "force_full_vocab_transfer": False,
        "prob_payload_bits": 16,
    }


def test_from_values_falls_back_to_defaults():
    cfg = CommConfig.from_values({"charge_residual_payload": False})
    assert cfg.charge_residual_payload is False
    assert cfg.comm_round_trip_mode == "per_round"
    assert cfg.transfer_top_k_cap == 16
    assert cfg.force_full_vocab_transfer is False
    assert cfg.prob_payload_bits == 16


def test_from_args_reads_every_attribute():
    args = SimpleNamespace(
        comm_round_trip_mode="per_transfer",
        charge_residual_payload=False,
        transfer_top_k_cap=0,
        force_full_vocab_transfer=True,
        prob_payload_bits=8,
    )
    assert CommConfig.from_args(args).as_dict() == {
        "comm_round_trip_mode": "per_transfer",
        "charge_residual_payload": False,
        "transfer_top_k_cap": 0,
        "force_full_vocab_transfer": True,
        "prob_payload_bits": 8,
    }


def test_from_args_falls_back_to_pre_migration_values():
    """合成 Namespace 必须与迁移前逐字一致（FALLBACKS，见模块注释）。"""
    cfg = CommConfig.from_args(SimpleNamespace())
    assert cfg.as_dict() == FALLBACKS


def test_profiles_cover_every_protocol():
    assert set(PROFILES) == set(PROTOCOLS)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("legacy", {"charge_residual_payload": False, "comm_round_trip_mode": "per_transfer", "transfer_top_k_cap": 0}),
        ("honest", {"charge_residual_payload": True, "comm_round_trip_mode": "per_round", "transfer_top_k_cap": 0}),
        ("paper_table5", {"charge_residual_payload": True, "comm_round_trip_mode": "per_round", "transfer_top_k_cap": 0}),
    ],
)
def test_accounting_profiles_match_the_protocol_values(name, expected):
    cfg = PROFILES[name]
    for key, value in expected.items():
        assert getattr(cfg, key) == value


def test_profiles_do_not_invent_values():
    """协议没冻结的键必须回落 DEFAULTS，而不是某个 profile 的私有值。"""
    for name, spec in PROTOCOLS.items():
        cfg = PROFILES[name]
        for key, default in DEFAULTS.items():
            assert getattr(cfg, key) == spec.values.get(key, default), name


def test_config_is_frozen():
    cfg = PROFILES["legacy"]
    with pytest.raises(FrozenInstanceError):
        cfg.comm_round_trip_mode = "per_round"  # type: ignore[misc]


def test_profiles_are_derived_not_copied():
    """改 PROTOCOLS 必须同时改 PROFILES——同一份字典投影，不是两份常量。"""
    assert isinstance(PROFILES["paper_table5"], CommConfig)
    assert PROFILES["paper_table5"].charge_residual_payload is PROTOCOLS[
        "paper_table5"
    ].values["charge_residual_payload"]
