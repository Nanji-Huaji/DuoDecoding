"""Golden 固件外部 checkpoint 的资产策略：缺失=未验收，漂移=永远红灯。

用小字节文件代替真实权重，覆盖匹配、漂移、默认缺失、严格缺失、混合缺失+漂移。
"""

from __future__ import annotations

import pytest
from golden_assets import assert_golden_checkpoints

MATCHING = b"golden-weights-v1"  # sha256_16 = 80fd62270fdce76c
DRIFTED = b"other-bytes"  # sha256_16 = fbf22ec45a215e0d


def _spec(path: str, digest: str) -> dict[str, str]:
    return {"path": path, "sha256_16": digest}


@pytest.mark.parametrize("require", [False, True])
def test_matching_asset_has_no_error(tmp_path, require):
    asset = tmp_path / "ckpt" / "latest.pth"
    asset.parent.mkdir()
    asset.write_bytes(MATCHING)

    assert_golden_checkpoints(
        {"main": _spec("ckpt/latest.pth", "80fd62270fdce76c")},
        require=require,
        repo_root=tmp_path,
    )


@pytest.mark.parametrize("require", [False, True])
@pytest.mark.parametrize("missing_first", [False, True])
def test_drift_fails_even_when_another_role_is_absent(tmp_path, require, missing_first):
    present = tmp_path / "ckpt" / "main.pth"
    present.parent.mkdir()
    present.write_bytes(DRIFTED)
    specs = {
        "little": _spec("ckpt/little.pth", "80fd62270fdce76c"),
        "main": _spec("ckpt/main.pth", "80fd62270fdce76c"),
    }
    if not missing_first:
        specs = dict(reversed(specs.items()))

    with pytest.raises(pytest.fail.Exception) as failure:
        assert_golden_checkpoints(specs, require=require, repo_root=tmp_path)

    message = str(failure.value)
    assert "ckpt/main.pth" in message
    assert "80fd62270fdce76c" in message and "fbf22ec45a215e0d" in message
    assert ("--require-golden-assets 要求" in message) is require


def test_default_missing_asset_skips_with_reason(tmp_path):
    with pytest.raises(pytest.skip.Exception) as skipped:
        assert_golden_checkpoints(
            {"main": _spec("ckpt/main.pth", "80fd62270fdce76c")},
            require=False,
            repo_root=tmp_path,
        )

    message = str(skipped.value)
    assert "ckpt/main.pth" in message
    assert "未验收" in message


def test_strict_mode_turns_missing_asset_into_failure(tmp_path):
    with pytest.raises(pytest.fail.Exception) as failure:
        assert_golden_checkpoints(
            {"little": _spec("ckpt/little.pth", "80fd62270fdce76c")},
            require=True,
            repo_root=tmp_path,
        )

    message = str(failure.value)
    assert "--require-golden-assets" in message
    assert "ckpt/little.pth" in message
