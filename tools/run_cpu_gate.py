"""Run the offline CPU gate with the invoking Python, never installing packages.

Usage: .venv/bin/python tools/run_cpu_gate.py \
           [--output-dir NEW_DIRECTORY] [--require-golden-assets]
The manifest records main-repository source files, not model weights or a GPU
acceptance claim. External subproject revisions/dirty state are recorded by Git.
Snapshots compare run endpoints, not every intermediate state. The caller must
stop other writers before using these results as frozen-source acceptance.
By default the 14 historical golden checkpoint checks skip when the external
weights are absent (skip is not asset acceptance); --require-golden-assets turns
that absence into failure and the policy is recorded in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "src", "test", "tools", "eval", "scripts", "configs",
    "pyproject.toml", "uv.lock", "requirements.txt", "requirements-dev.txt",
    "*.py", "pytest.ini", "tox.ini", "setup.cfg",
)
CPU_ENV = {
    "CUDA_VISIBLE_DEVICES": "",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1",
    "HF_HUB_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, check=True, capture_output=True, text=True,
    ).stdout


def source_snapshot(root: Path) -> dict:
    """Include untracked new source files; do not traverse models/virtualenvs."""
    names = _git(
        root, "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        "--", *SOURCE_PATHS,
    ).split("\0")
    files = {}
    for name in sorted(set(names) - {""}):
        path = root / name
        # Gitlinks name directories, not ordinary main-repository source files.
        if path.is_file():
            files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        elif not path.exists():
            files[name] = "deleted"
    return {
        "git_sha": _git(root, "rev-parse", "HEAD").strip(),
        "source_status": _git(root, "status", "--porcelain", "--", *SOURCE_PATHS),
        "submodules": _git(root, "submodule", "status", "--recursive"),
        "files": files,
    }


def environment_metadata() -> dict:
    packages = {}
    for name in ("torch", "transformers", "pytest", "pytest-subtests", "ruff"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": packages,
        "overrides": CPU_ENV,
    }


GOLDEN_ASSETS_FLAG = "--require-golden-assets"


def pytest_command(
    executable: str,
    mode: str,
    report: Path,
    *,
    require_golden_assets: bool = False,
) -> list[str]:
    if mode not in {"default", "importlib"}:
        raise ValueError(f"Unsupported collection mode: {mode}")
    command = [
        executable, "-B", "-m", "pytest", "-q", "-p", "no:cacheprovider",
        "-o", "addopts=", "-rs",
    ]
    if mode == "importlib":
        command.append("--import-mode=importlib")
    if require_golden_assets:
        command.append(GOLDEN_ASSETS_FLAG)
    return [*command, "test", f"--junitxml={report}"]


def run_gate(
    root: Path,
    output: Path,
    executable: str = sys.executable,
    *,
    require_golden_assets: bool = False,
) -> int:
    root, output = root.resolve(), output.resolve()
    # Refuse to merge with old evidence, even if an earlier run failed.
    output.mkdir(parents=True, exist_ok=False)
    before = source_snapshot(root)
    manifest = {
        "schema": "duodecoding.cpu-gate.v1",
        "scope": "CPU only; no real-model/CUDA execution acceptance",
        "golden_assets": {
            "required": bool(require_golden_assets),
            "flag": GOLDEN_ASSETS_FLAG,
        },
        "environment": environment_metadata(),
        "source_before": before,
        "results": [],
        "status": "running",
    }
    manifest_path = output / "manifest.json"

    def save() -> None:
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    save()
    try:
        env = dict(os.environ, **CPU_ENV)
        # Explicit env plugins bypass DISABLE_PLUGIN_AUTOLOAD. Configuration
        # addopts are separately overridden in pytest_command (e.g. collect-only).
        for name in ("PYTEST_ADDOPTS", "PYTEST_PLUGINS"):
            env.pop(name, None)
        for mode in ("default", "importlib"):
            command = pytest_command(
                executable, mode, output / f"{mode}.xml",
                require_golden_assets=require_golden_assets,
            )
            start = time.monotonic()
            with (output / f"{mode}.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(
                    command, cwd=root, env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=False,
                )
            manifest["results"].append({
                "mode": mode, "command": command, "returncode": result.returncode,
                "seconds": time.monotonic() - start, "log": f"{mode}.log",
                "junit": f"{mode}.xml",
            })
            save()
            print(f"{mode}: exit={result.returncode}; {output / (mode + '.log')}")
        after = source_snapshot(root)
        manifest["source_after"] = after
        manifest["source_unchanged"] = before == after
        passed = before == after and all(
            item["returncode"] == 0 for item in manifest["results"]
        )
        manifest["status"] = "passed" if passed else "failed"
        save()
        return 0 if passed else 1
    except BaseException as exc:
        manifest["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "error"
        )
        manifest["error"] = f"{type(exc).__name__}: {exc}"
        save()
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        GOLDEN_ASSETS_FLAG,
        action="store_true",
        help="缺少历史 golden checkpoint 时让 14 项资产检查失败，而不是跳过。",
    )
    args = parser.parse_args(argv)
    output = args.output_dir or ROOT / ".refactor" / "cpu-gates" / uuid.uuid4().hex
    return run_gate(
        ROOT, output, require_golden_assets=args.require_golden_assets,
    )


if __name__ == "__main__":
    raise SystemExit(main())
