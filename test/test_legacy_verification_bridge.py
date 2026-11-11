"""Architecture contract: clipped legacy windows cannot spread to strict decoders."""

import ast
import subprocess
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import torch

from src import decoding_ops

ROOT = Path(__file__).resolve().parents[1]
LEGACY_PREPARE = "_prepare_legacy_verification_inputs"
FP16_ACCEPT = "compute_legacy_tk_slt_fp16_acceptance_result"
BRIDGES = {LEGACY_PREPARE, "prepare_verification_inputs", FP16_ACCEPT}


def _calls(path):
    tree = ast.parse(path.read_text())
    aliases = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for name in node.names:
                if name.name in BRIDGES:
                    aliases[name.asname or name.name] = name.name
    found = Counter()
    pending = [(tree, ())]
    while pending:
        node, scope = pending.pop()
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            scope = (*scope, node.name)
        if isinstance(node, ast.Call):
            name = (
                node.func.id
                if isinstance(node.func, ast.Name)
                else node.func.attr
                if isinstance(node.func, ast.Attribute)
                else None
            )
            name = aliases.get(name, name)
            if name in BRIDGES:
                found[(path.relative_to(ROOT).as_posix(), ".".join(scope), name)] += 1
        pending.extend((child, scope) for child in ast.iter_child_nodes(node))
    return found


def test_legacy_bridge_callers_are_explicit_and_bounded():
    actual = Counter()
    # Follow repository ownership, not nested third-party virtual environments.
    paths = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
            "--",
            "src",
        ],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split("\0")
    for relative in set(paths):
        if relative.endswith(".py") and (ROOT / relative).is_file():
            actual.update(_calls(ROOT / relative))
    assert actual == Counter(
        {
            ("src/baselines/core.py", "Baselines.cee_cuhlm", LEGACY_PREPARE): 2,
            ("src/decoding_ops.py", "verify_draft_sequence", LEGACY_PREPARE): 1,
            ("src/decoding_ops.py", "verify_draft_sequence_result", LEGACY_PREPARE): 1,
        }
    )


def test_active_tk_slt_is_not_a_legacy_bridge_consumer():
    """TK-SLT's active loop verifies the wire payload through the strict contract.

    The dedicated historical FP16 bridge and the clipped legacy adapter stay
    exported (other algorithms and the retired-codec tests still use them), but
    the migrated TK-SLT module must not call either one.
    """
    assert _calls(ROOT / "src/baselines/tk_slt.py") == Counter()


def test_old_public_name_is_only_compatibility_for_explicit_legacy_implementation():
    assert decoding_ops.prepare_verification_inputs.__name__ == LEGACY_PREPARE
    # Existing callers retain clipping; this is not an implicit strict fallback.
    draft = SimpleNamespace(
        device=torch.device("cpu"), prob_history=torch.full((1, 2, 4), 0.25)
    )
    target = SimpleNamespace(prob_history=torch.full((1, 4, 4), 0.25))
    result = decoding_ops.prepare_verification_inputs(
        draft,
        target,
        torch.tensor([[0, 1, 2, 3]]),
        prefix_len=2,
        gamma=2,
    )
    assert result.gamma == 2 and result.actual_gamma == 1
    assert result.selected_draft_p.tolist() == [[0.25]]


def test_strict_modules_do_not_import_legacy_validation_or_stage_wrappers():
    forbidden = BRIDGES | {
        "verify_draft_sequence",
        "verify_draft_sequence_result",
        "resolve_stage_verification",
        "finalize_verification",
        "compute_acceptance_result",
        "VerificationInputs",
    }
    for relative in (
        "src/dual_decoding.py",
        "src/baselines/dual.py",
        "src/verification.py",
        "src/probability_laws.py",
        "src/cuhlm_verification.py",
        "src/baselines/cuhlm.py",
    ):
        tree = ast.parse((ROOT / relative).read_text())
        imports = {
            name.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            for name in node.names
        }
        assert not imports & forbidden, (relative, imports & forbidden)
