import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPO_ROOT / "scripts" / "run_temperature_validation.py"


def test_dry_run_writes_matrix_manifest_without_starting_inference(
    tmp_path: Path,
) -> None:
    output_root = tmp_path / "results"
    command = [
        sys.executable,
        str(DRIVER),
        "--dry-run",
        "--output-root",
        str(output_root),
        "--temperatures",
        "0.2",
        "0.8",
        "--seeds",
        "7",
        "--draft-model",
        "selected-draft-model",
    ]

    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    manifest_paths = list(output_root.glob("temperature_validation_*/manifest.json"))
    assert len(manifest_paths) == 1
    manifest = json.loads(manifest_paths[0].read_text(encoding="utf-8"))
    assert manifest["dry_run"] is True
    assert len(manifest["runs"]) == 4
    assert {run["mode"] for run in manifest["runs"]} == {"cee_sd", "large"}
    assert {run["temperature"] for run in manifest["runs"]} == {0.2, 0.8}
    assert all("--temp" in run["command"] for run in manifest["runs"])
    assert all(
        "--little_model" in run["command"]
        for run in manifest["runs"]
        if run["mode"] == "cee_sd"
    )
    assert all(
        "--little_model" not in run["command"]
        for run in manifest["runs"]
        if run["mode"] == "large"
    )
    assert all(
        run["command"][run["command"].index("--draft_model") + 1]
        == "selected-draft-model"
        for run in manifest["runs"]
        if run["mode"] == "large"
    )
    assert all(run["exit_code"] is None for run in manifest["runs"])
    assert all(run["status"] == "planned" for run in manifest["runs"])
