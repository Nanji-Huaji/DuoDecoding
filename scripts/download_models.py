#!/usr/bin/env python3
"""Download experiment models and checkpoints for DuoDecoding.

Downloads three categories of artifacts:
  1. Base Models — 9 LLM weights across llama / qwen3 / qwen1.5 series
  2. SpecDec++ Acceptance Heads — prediction head checkpoints for adaptive decoding
  3. RL Agent Checkpoints — guidance for locally-generated RL adapter weights

Usage:
  python scripts/download_models.py                        # models only
  python scripts/download_models.py --checkpoints          # models + acceptance heads
  python scripts/download_models.py --checkpoints --dry-run  # preview everything
  python scripts/download_models.py --rl-guide             # print RL agent guidance
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import snapshot_download

REPO_ROOT = Path(__file__).resolve().parent.parent

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Base Models
# ═══════════════════════════════════════════════════════════════════════════════
# Each entry: (alias, hf_repo_id, local_subdir)

LLAMA_SERIES: list[tuple[str, str, str]] = [
    ("llama-68m", "JackFram/llama-68m", "llama/llama-68m"),
    (
        "tiny-llama-1.1b",
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "llama/tiny-llama-1.1b",
    ),
    ("llama-2-13b", "meta-llama/Llama-2-13b-hf", "llama/Llama-2-13b-hf"),
]

QWEN3_SERIES: list[tuple[str, str, str]] = [
    ("Qwen3-0.6B", "Qwen/Qwen3-0.6B", "Qwen/Qwen3-0.6B"),
    ("Qwen3-1.7B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-1.7B"),
    ("Qwen3-14B", "Qwen/Qwen3-14B", "Qwen/Qwen3-14B"),
]

QWEN15_SERIES: list[tuple[str, str, str]] = [
    ("Qwen1.5-0.5B-Chat", "Qwen/Qwen1.5-0.5B-Chat", "Qwen/Qwen1.5-0.5B-Chat"),
    ("Qwen1.5-1.8B-Chat", "Qwen/Qwen1.5-1.8B-Chat", "Qwen/Qwen1.5-1.8B-Chat"),
    ("Qwen1.5-7B-Chat", "Qwen/Qwen1.5-7B-Chat", "Qwen/Qwen1.5-7B-Chat"),
]

SERIES_MAP: dict[str, list[tuple[str, str, str]]] = {
    "llama": LLAMA_SERIES,
    "qwen": QWEN3_SERIES,
    "qwen15": QWEN15_SERIES,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. SpecDec++ Acceptance Heads  (hosted on HuggingFace)
# ═══════════════════════════════════════════════════════════════════════════════

SPECDEC_HF_REPO = "ArcticHuaji/specdecpp-acc-heads"
ACC_HEAD_LOCAL_ROOT = Path("src/SpecDec_pp/checkpoints/acc_head")

# Subpaths within the HF repo, grouped by model series.
ACC_HEAD_SUBPATHS: dict[str, list[str]] = {
    "llama": [
        "llama-68m--to--llama-2-13b/exp-weight6-layer3",
        "llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3",
        "tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3",
    ],
    "qwen": [
        "qwen3-0.6b--to--qwen3-1.7b/exp-weight6-layer3",
        "qwen3-1.7b--to--qwen3-14b/exp-weight6-layer3",
    ],
    "qwen15": [
        "qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat/exp-weight-layer3",
        "qwen1.5-1.8b-chat--to--qwen1.5-7b-chat/exp-weight6-layer3",
    ],
}

# ═══════════════════════════════════════════════════════════════════════════════
# 3. RL Agent Checkpoints  (generated locally – NOT downloadable)
# ═══════════════════════════════════════════════════════════════════════════════

RL_AGENT_ROOT = Path("checkpoints/rl_agents")

# Pairs expected for each series: main = draft→target, little = little→draft
RL_AGENT_PAIRS: dict[str, dict[str, str]] = {
    "llama": {
        "main": "tiny-llama-1.1b--to--llama-2-13b",
        "little": "llama-68m--to--tiny-llama-1.1b",
    },
    "qwen": {
        "main": "qwen3-1.7b--to--qwen3-14b",
        "little": "qwen3-0.6b--to--qwen3-1.7b",
    },
    "qwen15": {
        "main": "qwen1.5-1.8b-chat--to--qwen1.5-7b-chat",
        "little": "qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat",
    },
}

RL_TRAINING_COMMAND = (
    "LITTLE_MODEL={little} DRAFT_MODEL={draft} TARGET_MODEL={target} "
    "bash cmds/train_rl.sh"
)

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _model_exists(local_dir: Path) -> bool:
    """Check whether a model directory exists and contains config.json."""
    return (local_dir / "config.json").exists()


def _acc_head_exists(local_dir: Path) -> bool:
    """Check whether an acceptance head directory contains model weights."""
    return (
        (local_dir / "model.safetensors").exists()
        or (local_dir / "pytorch_model.bin").exists()
    )


def _rl_agent_exists(local_dir: Path) -> bool:
    """Check whether an RL agent directory contains a checkpoint."""
    return (local_dir / "best.pth").exists() or (local_dir / "latest.pth").exists()


def _download_one(
    label: str,
    repo_id: str,
    local_dir: Path,
    *,
    force: bool = False,
    allow_patterns: list[str] | None = None,
) -> bool:
    """Download from HuggingFace Hub.  Returns True on success."""
    exists_check = (
        _acc_head_exists if "acc_head" in str(local_dir) else _model_exists
    )
    if not force and exists_check(local_dir):
        rel = local_dir.relative_to(REPO_ROOT)
        print(f"  [SKIP] {label} — already at {rel}")
        return True

    rel = local_dir.relative_to(REPO_ROOT)
    print(f"  [DOWN] {label}  ({repo_id} → {rel})")
    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            allow_patterns=allow_patterns,
        )
        print(f"  [ OK ] {label}")
        return True
    except Exception as exc:
        print(f"  [FAIL] {label}: {exc}", file=sys.stderr)
        return False


def _resolve_series(
    requested: list[str],
) -> tuple[list[str], list[tuple[str, str, str]]]:
    """Build flat model list and canonical series names from --series arg."""
    names: list[str] = []
    models: list[tuple[str, str, str]] = []
    if "all" in requested or "llama" in requested:
        models.extend(LLAMA_SERIES)
        names.append("llama")
    if "all" in requested or "qwen" in requested:
        models.extend(QWEN3_SERIES)
        names.append("qwen3")
    if "all" in requested or "qwen15" in requested:
        models.extend(QWEN15_SERIES)
        names.append("qwen1.5")
    return names, models


def _resolve_acc_head_series(
    requested: list[str],
) -> tuple[list[str], list[tuple[str, str]]]:
    """Build flat acceptance-head list from --series arg."""
    names: list[str] = []
    pairs: list[tuple[str, str]] = []  # (subpath, local_dir)
    if "all" in requested or "llama" in requested:
        for sp in ACC_HEAD_SUBPATHS.get("llama", []):
            pairs.append((sp, str(ACC_HEAD_LOCAL_ROOT / sp)))
        names.append("llama")
    if "all" in requested or "qwen" in requested:
        for sp in ACC_HEAD_SUBPATHS.get("qwen", []):
            pairs.append((sp, str(ACC_HEAD_LOCAL_ROOT / sp)))
        names.append("qwen3")
    if "all" in requested or "qwen15" in requested:
        for sp in ACC_HEAD_SUBPATHS.get("qwen15", []):
            pairs.append((sp, str(ACC_HEAD_LOCAL_ROOT / sp)))
        names.append("qwen1.5")
    return names, pairs


# ═══════════════════════════════════════════════════════════════════════════════
# Section runners
# ═══════════════════════════════════════════════════════════════════════════════


def _run_models(
    series_names: list[str],
    models: list[tuple[str, str, str]],
    *,
    dry_run: bool,
    force: bool,
) -> tuple[int, int, int]:
    """Download base models.  Returns (ok, skipped, failed)."""
    print("── Base Models ──")
    if dry_run:
        for alias, repo_id, rel_path in models:
            local_dir = REPO_ROOT / rel_path
            status = "EXISTS" if _model_exists(local_dir) else "MISSING"
            print(f"  [{status}] {alias}  ({repo_id} → {rel_path})")
        print()
        return 0, 0, 0

    ok = skipped = failed = 0
    for alias, repo_id, rel_path in models:
        local_dir = REPO_ROOT / rel_path
        if not force and _model_exists(local_dir):
            skipped += 1
            print(f"  [SKIP] {alias} — already at {rel_path}")
            continue
        if _download_one(alias, repo_id, local_dir, force=force):
            ok += 1
        else:
            failed += 1
    print()
    return ok, skipped, failed


def _run_acc_heads(
    series_names: list[str],
    pairs: list[tuple[str, str]],
    *,
    dry_run: bool,
    force: bool,
) -> tuple[int, int, int]:
    """Download SpecDec++ acceptance heads.  Returns (ok, skipped, failed)."""
    print("── SpecDec++ Acceptance Heads ──")
    if dry_run:
        for subpath, local_rel in pairs:
            local_dir = REPO_ROOT / local_rel
            status = "EXISTS" if _acc_head_exists(local_dir) else "MISSING"
            print(f"  [{status}] {subpath}  ({SPECDEC_HF_REPO} → {local_rel})")
        print()
        return 0, 0, 0

    ok = skipped = failed = 0
    for subpath, local_rel in pairs:
        local_dir = REPO_ROOT / local_rel
        label = subpath
        if not force and _acc_head_exists(local_dir):
            skipped += 1
            print(f"  [SKIP] {label} — already at {local_rel}")
            continue
        # Download only files within this subpath from the shared HF repo.
        if _download_one(
            label,
            SPECDEC_HF_REPO,
            local_dir,
            force=force,
            allow_patterns=[f"{subpath}/*"],
        ):
            ok += 1
        else:
            failed += 1
    print()
    return ok, skipped, failed


def _run_rl_guide(
    series_names: list[str],
) -> None:
    """Print guidance for locally-generated RL agent checkpoints."""
    print("── RL Agent Checkpoints (generated locally) ──")
    print("These checkpoints are produced by RL training and are NOT downloadable.")
    print("Expected paths and training commands:")
    print()

    canonical_aliases = {
        "llama": ("llama-68m", "tiny-llama-1.1b", "llama-2-13b"),
        "qwen": ("qwen-3-0.6b", "qwen-3-1.7b", "qwen-3-14b"),
        "qwen15": ("qwen1.5-0.5b-chat", "qwen1.5-1.8b-chat", "qwen1.5-7b-chat"),
    }

    name_to_series = [("llama", "llama"), ("qwen3", "qwen"), ("qwen1.5", "qwen15")]
    for name_key, series_key in name_to_series:
        if name_key not in series_names and "all" not in series_names:
            continue
        pairs = RL_AGENT_PAIRS.get(series_key)
        if pairs is None:
            continue
        aliases = canonical_aliases.get(series_key, ("?", "?", "?"))
        little_alias, draft_alias, target_alias = aliases

        print(f"  [{name_key}]")
        main_path = RL_AGENT_ROOT / "main" / pairs["main"]
        little_path = RL_AGENT_ROOT / "little" / pairs["little"]
        exists_main = _rl_agent_exists(REPO_ROOT / main_path)
        exists_little = _rl_agent_exists(REPO_ROOT / little_path)

        status_main = "[EXISTS]" if exists_main else "[MISSING]"
        status_little = "[EXISTS]" if exists_little else "[MISSING]"
        print(f"    main:   {main_path}  {status_main}")
        print(f"    little: {little_path}  {status_little}")
        print(f"    train:  LITTLE_MODEL={little_alias} DRAFT_MODEL={draft_alias} "
              f"TARGET_MODEL={target_alias} bash cmds/train_rl.sh")
        print()

    print("Run 'bash cmds/train_rl.sh' or 'bash cmds/train_rl_mixed.sh'")
    print("to generate RL checkpoints. See docs/rl_agent_checkpoints.md for details.")
    print()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download DuoDecoding experiment models and checkpoints."
    )
    parser.add_argument(
        "--series",
        nargs="+",
        choices=["llama", "qwen", "qwen15", "all"],
        default=["all"],
        help="Which model series to operate on (default: all).",
    )
    parser.add_argument(
        "--checkpoints",
        action="store_true",
        help="Also download SpecDec++ acceptance head checkpoints from HuggingFace.",
    )
    parser.add_argument(
        "--rl-guide",
        action="store_true",
        help="Print guidance for locally-generated RL agent checkpoints and exit.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the artifact already exists locally.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading.",
    )
    args = parser.parse_args()

    series_names, models = _resolve_series(args.series)

    # ── RL guide only mode ──
    if args.rl_guide:
        _run_rl_guide(series_names)
        return

    # ── Dry-run or full download ──
    print(f"Series: {', '.join(series_names)}")
    total_ok = total_skipped = total_failed = 0

    # 1. Base models
    ok, skipped, failed = _run_models(
        series_names, models, dry_run=args.dry_run, force=args.force,
    )
    total_ok += ok
    total_skipped += skipped
    total_failed += failed

    # 2. Acceptance heads (only when --checkpoints is passed)
    if args.checkpoints:
        _, acc_pairs = _resolve_acc_head_series(args.series)
        ok, skipped, failed = _run_acc_heads(
            series_names, acc_pairs, dry_run=args.dry_run, force=args.force,
        )
        total_ok += ok
        total_skipped += skipped
        total_failed += failed

    # 3. RL agent guidance (always shown when --checkpoints, as a reminder)
    if args.checkpoints and not args.dry_run:
        _run_rl_guide(series_names)

    if args.dry_run:
        print("Dry-run complete. No artifacts were downloaded.")
        return

    print(
        f"Done — {total_ok} downloaded, {total_skipped} skipped, "
        f"{total_failed} failed"
    )
    if total_failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
