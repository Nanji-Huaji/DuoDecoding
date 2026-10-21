"""训练 staging 与显式晋升的回归测试。"""

import json
from pathlib import Path

import torch

from src.baselines import _rl_train_paths
from tools import promote_checkpoint as pc


def test_rl_train_paths_keep_production_by_default():
    m, b = _rl_train_paths(None, "main", "checkpoints/rl_agents/main/p/latest.pth",
                          "checkpoints/rl_agents/main/p/best.pth")
    assert m.endswith("rl_agents/main/p/latest.pth")
    assert b.endswith("rl_agents/main/p/best.pth")


def test_rl_train_paths_redirect_to_staging():
    m, b = _rl_train_paths(
        "checkpoints/staging/run1",
        "little",
        "checkpoints/rl_agents/little/p/latest.pth",
        "checkpoints/rl_agents/little/p/best.pth",
    )
    assert m == "checkpoints/staging/run1/little/p/latest.pth"
    assert b == "checkpoints/staging/run1/little/p/best.pth"


def _payload():
    return {
        "policy_net": {"w": torch.zeros(1)},
        "target_net": {"w": torch.zeros(1)},
        "optimizer": {},
    }


def test_promote_copies_and_updates_manifest(tmp_path):
    root = tmp_path
    staging = root / "checkpoints" / "staging" / "run1" / "main" / "pairA"
    staging.mkdir(parents=True)
    torch.save(_payload(), staging / "latest.pth")
    (staging / "latest.pth.buffer").write_bytes(b"buffer")

    promoted = pc.promote(
        staging=root / "checkpoints" / "staging" / "run1",
        roles=["main"],
        dry_run=False,
        evidence="unit-test",
        root=root,
    )
    assert len(promoted) == 2
    dst = root / "checkpoints" / "rl" / "main" / "pairA" / "latest.pth"
    assert dst.is_file()
    manifest = json.loads((root / "checkpoints" / "manifest.json").read_text())
    entry = manifest["entries"]["checkpoints/rl/main/pairA/latest.pth"]
    assert entry["role"] == "rl_main"
    assert entry["sha256"] == pc.sha256_file(dst)
    assert entry["provenance"]["evidence"] == "unit-test"


def test_promote_rejects_invalid_checkpoint(tmp_path):
    staging = tmp_path / "checkpoints" / "staging" / "run1" / "main" / "pairA"
    staging.mkdir(parents=True)
    (staging / "latest.pth").write_bytes(b"not a torch file")
    try:
        pc.promote(
            staging=tmp_path / "checkpoints" / "staging" / "run1",
            roles=["main"],
            dry_run=False,
            root=tmp_path,
        )
    except Exception as exc:
        assert "latest.pth" in str(exc) or "checkpoint" in str(exc).lower()
    else:  # pragma: no cover
        raise AssertionError("无效 checkpoint 应被拒绝")
