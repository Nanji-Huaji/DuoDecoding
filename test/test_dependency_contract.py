"""Dependency-source contracts for the CPU validation path.

These checks only read repository files: they never invoke uv or pip, install
packages, or access the network. The production source of truth is
``pyproject.toml`` plus the committed ``uv.lock``; ``requirements.txt`` must be
a frozen export of that lock, and ``requirements-dev.txt`` is a small list of
exact tool pins outside the production lock.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version

from tools.export_locked_requirements import (
    MANIFEST_PATH,
    source_identity,
    validate_export_manifest,
)

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
LOCK = ROOT / "uv.lock"
REQUIREMENTS = ROOT / "requirements.txt"
DEV_REQUIREMENTS = ROOT / "requirements-dev.txt"

KEY_PACKAGES = ("torch", "transformers", "fastchat")
DEV_PINS = {"pytest": "9.1.1", "ruff": "0.15.4"}

_CONTINUATION = re.compile(r"\\[ \t]*\n")
_HASH = re.compile(r"--hash=sha256:([0-9a-f]{64})\b")
_REQUIREMENT = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"(?:\[[^\]]*\])?"
    r"==(?P<version>[^\s;\\]+)"
    r"(?:\s*;\s*(?P<marker>[^#]+?))?\s*$"
)


@dataclass(frozen=True)
class ExportRecord:
    """One pinned requirement line from a uv requirements export."""

    name: str
    version: str
    marker: str | None
    hashes: tuple[str, ...]


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _toml_loads(text: str) -> dict:
    try:
        import tomllib
    except ModuleNotFoundError:  # Python 3.10; tomli is a required locked dependency.
        import tomli  # Missing prerequisites must fail the gate, never skip it.

        return tomli.loads(text)
    return tomllib.loads(text)


def parse_export(path: Path) -> list[ExportRecord]:
    """Parse pinned records from a uv requirements export.

    Handles logical lines continued with a trailing backslash and ``--hash``
    options, so the test does not hard-code the current package set or hashes.
    """

    text = path.read_text(encoding="utf-8")
    records: list[ExportRecord] = []
    for raw in _CONTINUATION.sub(" ", text).splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        hashes = tuple(_HASH.findall(line))
        line = _HASH.sub("", line).strip()
        match = _REQUIREMENT.match(line)
        assert match is not None, f"unparsed requirement in {path.name}: {raw!r}"
        records.append(
            ExportRecord(
                name=_normalize(match["name"]),
                version=match["version"],
                marker=(match["marker"] or "").strip() or None,
                hashes=hashes,
            )
        )
    return records


def _locked_versions() -> dict[str, set[str]]:
    data = _toml_loads(LOCK.read_text(encoding="utf-8"))
    locked: dict[str, set[str]] = {}
    for package in data["package"]:
        locked.setdefault(_normalize(package["name"]), set()).add(package["version"])
    return locked


def _direct_dependencies() -> dict[str, str]:
    project = _toml_loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    dependencies: dict[str, str] = {}
    for raw in project["dependencies"]:
        requirement = Requirement(raw)
        dependencies[_normalize(requirement.name)] = str(requirement.specifier)
    return dependencies


def _single(records: list[ExportRecord], name: str) -> ExportRecord:
    matches = [record for record in records if record.name == name]
    assert len(matches) == 1, f"expected one {name} record, found {len(matches)}"
    return matches[0]


def _satisfies(version: str, specifier: str) -> bool:
    return Version(version) in SpecifierSet(specifier)


def test_requirements_is_a_frozen_lock_export_with_hashes():
    header = "\n".join(REQUIREMENTS.read_text(encoding="utf-8").splitlines()[:2])
    assert "uv export" in header
    for flag in ("--frozen", "--offline", "--no-dev", "--no-emit-project"):
        assert flag in header
    records = parse_export(REQUIREMENTS)
    assert len(records) > 50
    assert len(records) == len({(record.name, record.marker) for record in records})
    assert all(record.name != "duodecoding" for record in records)
    for record in records:
        assert record.hashes, f"{record.name} has no artifact hash"


def _manifest() -> dict:
    return json.loads((ROOT / MANIFEST_PATH).read_text(encoding="utf-8"))


def test_complete_export_bound_to_production_inputs_and_verified_generator():
    # Binds all package records, markers and exact artifact hashes, not merely
    # a subset of versions. Regenerate ONLY through the offline frozen tool.
    validate_export_manifest(
        REQUIREMENTS.read_bytes(), source_identity(ROOT), _manifest(),
    )


@pytest.mark.parametrize("mutation", ["hashes", "marker", "missing_package"])
def test_complete_export_rejects_review_counterexamples(mutation):
    original = REQUIREMENTS.read_text(encoding="utf-8")
    if mutation == "hashes":
        changed = _HASH.sub("--hash=sha256:" + "0" * 64, original)
    elif mutation == "marker":
        assert "sys_platform == 'darwin'" in original
        changed = original.replace(
            "sys_platform == 'darwin'", "sys_platform == 'linux'",
        )
    else:
        # Remove the entire first transitive record including hashes/via comment.
        lines = original.splitlines(keepends=True)
        start = next(i for i, line in enumerate(lines) if line.startswith("absl-py=="))
        end = start + 1
        while end < len(lines) and (not lines[end].strip() or lines[end][0].isspace()):
            end += 1
        changed = "".join(lines[:start] + lines[end:])
        assert "absl-py==" not in changed
    assert changed != original
    with pytest.raises(ValueError, match="complete dependency export differs"):
        validate_export_manifest(changed.encode(), source_identity(ROOT), _manifest())


def test_changed_production_inputs_require_new_verified_export():
    sources = source_identity(ROOT)
    sources["uv.lock"] = "changed"
    with pytest.raises(ValueError, match="production dependency inputs differ"):
        validate_export_manifest(REQUIREMENTS.read_bytes(), sources, _manifest())


def test_missing_required_toml_parser_fails_instead_of_skipping(monkeypatch):
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    with pytest.raises(ModuleNotFoundError):
        _toml_loads("[project]\nname = 'test'\n")


@pytest.mark.parametrize(
    "version, constraint, expected",
    [("1.0rc1", ">=1.0", False), ("1.0", ">=1.0.0", True),
     ("2.0", ">=1.0,<2.0", False)],
)
def test_version_constraints_follow_pep440(version, constraint, expected):
    assert _satisfies(version, constraint) is expected


def test_every_exported_version_matches_uv_lock():
    locked = _locked_versions()
    records = parse_export(REQUIREMENTS)
    assert {record.name for record in records} <= set(locked)
    mismatches = [
        (record.name, record.version, sorted(locked[record.name]))
        for record in records
        if record.version not in locked[record.name]
    ]
    assert not mismatches


def test_key_versions_and_fastchat_name_follow_lock():
    records = parse_export(REQUIREMENTS)
    locked = _locked_versions()
    for name in KEY_PACKAGES:
        record = _single(records, name)
        assert record.marker is None
        assert record.version in locked[name]
    assert "fschat" not in locked
    assert all(record.name != "fschat" for record in records)


def test_export_satisfies_every_pyproject_direct_dependency():
    records = parse_export(REQUIREMENTS)
    for name, specifier in _direct_dependencies().items():
        record = _single(records, name)
        assert record.marker is None
        assert _satisfies(record.version, specifier), (
            name,
            record.version,
            specifier,
        )


def test_torch_is_an_exact_pin_in_pyproject():
    specifier = _direct_dependencies()["torch"]
    assert specifier.startswith("==")
    torch = _single(parse_export(REQUIREMENTS), "torch")
    assert specifier[2:] == torch.version


def test_dev_requirements_pin_tools_outside_production_lock():
    records = parse_export(DEV_REQUIREMENTS)
    assert {record.name for record in records} == set(DEV_PINS)
    for record in records:
        assert record.marker is None
        assert record.version == DEV_PINS[record.name]
    assert not set(DEV_PINS) & set(_locked_versions())
