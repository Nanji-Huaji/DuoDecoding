"""pin_model_revisions 的离线测试（mock HfApi）。"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pin_model_revisions as pin  # noqa: E402


def test_pin_sets_revision_for_all_aliases(tmp_path, monkeypatch):
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps(
            {
                "models": {
                    "a": {"hf_repo": "org/model"},
                    "b": {"hf_repo": "org/model"},
                    "c": {"local_dir": "x"},
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeApi:
        def model_info(self, repo):
            return SimpleNamespace(sha="deadbeef" * 5)

    monkeypatch.setattr("huggingface_hub.HfApi", FakeApi)
    assert pin.pin(cfg) == 0
    data = json.loads(cfg.read_text())
    assert data["models"]["a"]["hf_revision"] == "deadbeef" * 5
    assert data["models"]["b"]["hf_revision"] == "deadbeef" * 5
