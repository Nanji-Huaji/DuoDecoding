"""运行时 checkpoint provenance：manifest 命中/篡改/未跟踪。"""

import json
from argparse import Namespace

import pytest

from src import checkpoint_provenance as cp


def _make(tmp_path, *, track=True, data=b"payload"):
    root = tmp_path
    ckpt = root / "checkpoints"
    path = ckpt / "rl" / "main" / "pairA" / "latest.pth"
    path.parent.mkdir(parents=True)
    path.write_bytes(data)
    rel = str(path.relative_to(root))
    entries = {}
    if track:
        entries[rel] = {
            "remote_path": rel,
            "sha256": cp.sha256_file(path),
            "size": path.stat().st_size,
            "role": "rl_main",
            "pair": "pairA",
            "status": "production",
        }
    (ckpt / "manifest.json").write_text(
        json.dumps({"entries": entries}), encoding="utf-8"
    )
    return root, path


def test_manifest_hit_and_record(tmp_path):
    root, path = _make(tmp_path)
    info = cp.describe_path(path, root=root)
    assert info["source"] == "manifest"
    assert info["role"] == "rl_main"
    args = Namespace()
    cp.record(args, "rl_main", info)
    assert args._checkpoint_provenance["rl_main"]["sha256"] == info["sha256"]


def test_untracked_path_is_allowed(tmp_path):
    root, path = _make(tmp_path, track=False)
    info = cp.describe_path(path, root=root)
    assert info["source"] == "untracked"
    assert info["sha256"]


def test_sha_mismatch_fails_closed(tmp_path):
    root, path = _make(tmp_path)
    path.write_bytes(b"tampered")  # 大小不同，绕开 sha 缓存
    with pytest.raises(cp.CheckpointIntegrityError):
        cp.describe_path(path, root=root)


def test_missing_path_reported(tmp_path):
    info = cp.describe_path(tmp_path / "nope.pth", root=tmp_path)
    assert info["source"] == "missing"


def test_directory_verification(tmp_path):
    root = tmp_path
    head = root / "src" / "SpecDec_pp" / "checkpoints" / "acc_head" / "p" / "run"
    head.mkdir(parents=True)
    (head / "config.json").write_text("{}", encoding="utf-8")
    ckpt = root / "checkpoints"
    ckpt.mkdir()
    rel = str((head / "config.json").relative_to(root))
    (ckpt / "manifest.json").write_text(
        json.dumps(
            {
                "entries": {
                    rel: {
                        "sha256": cp.sha256_file(head / "config.json"),
                        "size": 2,
                        "role": "acc_head",
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    info = cp.describe_dir(head, root=root)
    assert info["source"] == "manifest"
    (head / "config.json").write_text("{\"x\":1}", encoding="utf-8")
    with pytest.raises(cp.CheckpointIntegrityError):
        cp.describe_dir(head, root=root)
