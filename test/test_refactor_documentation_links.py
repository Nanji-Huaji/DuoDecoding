"""Resolve repository Markdown links relative to their containing document."""

import re
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = (
    "docs/adaptive_decoding_refactor_20261106.md",
    "docs/comm_call_sites.md",
    "docs/verification_gates.md",
)


@pytest.mark.parametrize("relative", DOCS)
def test_refactor_documentation_links_resolve_from_document(relative):
    document = ROOT / relative
    targets = re.findall(r"\]\(<([^>]+)>\)", document.read_text(encoding="utf-8"))
    assert targets, f"No documented link targets found in {relative}"
    for target in targets:
        parsed = urlsplit(target)
        if parsed.scheme or not parsed.path:
            continue
        resolved = (document.parent / unquote(parsed.path)).resolve()
        assert resolved.is_relative_to(ROOT), (relative, target)
        assert resolved.exists(), f"{relative}: broken relative link {target}"
