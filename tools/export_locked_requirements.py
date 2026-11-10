"""Check (default) or regenerate (--write) the offline frozen dependency export.

The small manifest binds the COMPLETE export bytes to both production inputs.
CPU tests can detect package/hash/marker drift without requiring uv at test time.
Only this explicit offline generator should update the export and its manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPORT_ARGS = [
    "export", "--frozen", "--offline", "--no-dev", "--no-emit-project",
    "--format", "requirements-txt",
]
MANIFEST_PATH = Path("test/golden/dependency_export.json")


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def source_identity(root: Path) -> dict[str, str]:
    return {
        name: sha256((root / name).read_bytes())
        for name in ("pyproject.toml", "uv.lock")
    }


def validate_export_manifest(
    content: bytes, sources: dict[str, str], manifest: dict,
) -> None:
    if manifest.get("schema") != 1 or manifest.get("uv_export_args") != EXPORT_ARGS:
        raise ValueError("dependency export manifest schema/command mismatch")
    if manifest.get("source_sha256") != sources:
        raise ValueError("production dependency inputs differ from verified export")
    if manifest.get("requirements_sha256") != sha256(content):
        raise ValueError("complete dependency export differs from verified manifest")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    before = source_identity(ROOT)
    env = dict(
        os.environ,
        UV_PYTHON_DOWNLOADS="never",
        UV_CACHE_DIR=str(ROOT / ".refactor" / "dependency-export-cache"),
    )
    version = subprocess.check_output(["uv", "--version"], env=env, text=True).strip()
    exported = subprocess.check_output(["uv", *EXPORT_ARGS], cwd=ROOT, env=env)
    if source_identity(ROOT) != before:
        raise RuntimeError("production inputs changed during export; retry when frozen")
    manifest = {
        "schema": 1,
        "uv_version": version,
        "uv_export_args": EXPORT_ARGS,
        "source_sha256": before,
        "requirements_sha256": sha256(exported),
    }
    requirements = ROOT / "requirements.txt"
    manifest_path = ROOT / MANIFEST_PATH
    if args.write:
        # Explicit regeneration only. No installer or dependency resolver runs.
        requirements.write_bytes(exported)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8",
        )
    else:
        current = requirements.read_bytes()
        if current != exported:
            raise ValueError("requirements differs from current offline frozen export")
        validate_export_manifest(
            current, before, json.loads(manifest_path.read_text(encoding="utf-8")),
        )
    print(f"Verified frozen export with {version}; production inputs unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
