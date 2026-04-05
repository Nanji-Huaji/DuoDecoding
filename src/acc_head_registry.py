import json
import os
import re
import argparse
from pathlib import Path


_REGISTRY_PATH = (
    Path(__file__).resolve().parent
    / "SpecDec_pp"
    / "checkpoints"
    / "acc_head_registry.json"
)
_DEFAULT_LOCAL_ROOT = Path("src/SpecDec_pp/checkpoints/acc_head")


CANONICAL_MODEL_ALIASES = {
    "llama-68m": "llama-68m",
    "jackfram/llama-68m": "llama-68m",
    "tiny-llama-1.1b": "tiny-llama-1.1b",
    "tinyllama/tinyllama-1.1b-chat-v1.0": "tiny-llama-1.1b",
    "llama-2-7b-chat": "llama-2-7b-chat",
    "meta-llama/llama-2-7b-chat-hf": "llama-2-7b-chat",
    "llama-2-13b": "llama-2-13b",
    "meta-llama/llama-2-13b-hf": "llama-2-13b",
    "llama-2-chat-70b": "llama-2-chat-70b",
    "meta-llama/llama-2-70b-chat-hf": "llama-2-chat-70b",
    "vicuna-68m": "vicuna-68m",
    "double7/vicuna-68m": "vicuna-68m",
    "tiny-vicuna-1b": "tiny-vicuna-1b",
    "jiayi-pan/tiny-vicuna-1b": "tiny-vicuna-1b",
    "vicuna-13b-v1.5": "vicuna-13b-v1.5",
    "lmsys/vicuna-13b-v1.5": "vicuna-13b-v1.5",
    "qwen/qwen3-0.6b": "qwen3-0.6b",
    "qwen3-0.6b": "qwen3-0.6b",
    "qwen/qwen3-1.7b": "qwen3-1.7b",
    "qwen3-1.7b": "qwen3-1.7b",
    "qwen/qwen3-14b": "qwen3-14b",
    "qwen3-14b": "qwen3-14b",
    "qwen/qwen3-32b": "qwen3-32b",
    "qwen3-32b": "qwen3-32b",
    "qwen/qwen1.5-0.5b-chat": "qwen1.5-0.5b-chat",
    "qwen1.5-0.5b-chat": "qwen1.5-0.5b-chat",
    "qwen/qwen1.5-1.8b-chat": "qwen1.5-1.8b-chat",
    "qwen1.5-1.8b-chat": "qwen1.5-1.8b-chat",
    "qwen/qwen1.5-7b-chat": "qwen1.5-7b-chat",
    "qwen1.5-7b-chat": "qwen1.5-7b-chat",
}


def canonicalize_model_name(model_name: str) -> str:
    normalized = model_name.strip().rstrip("/")
    basename = os.path.basename(normalized)
    candidates = [
        normalized,
        basename,
        normalized.lower(),
        basename.lower(),
    ]
    for candidate in candidates:
        alias = CANONICAL_MODEL_ALIASES.get(candidate.lower())
        if alias is not None:
            return alias

    # Fallback for common Hugging Face model ids. We preserve the full model id
    # semantics but convert it into a filesystem-safe slug, for example:
    # `Qwen/Qwen2-3B` -> `qwen--qwen2-3b`.
    lowered = normalized.lower()
    if "/" in lowered and not lowered.startswith("/"):
        slug = lowered.replace("/", "--")
    else:
        slug = os.path.basename(lowered)

    slug = slug.replace("_", "-")
    slug = re.sub(r"[^a-z0-9.-]+", "-", slug)
    slug = re.sub(r"-{2,}", lambda m: "--" if len(m.group(0)) == 2 else "-", slug)
    slug = re.sub(r"\.-| -", "-", slug)
    slug = slug.strip("-.")
    return slug


def load_acc_head_registry() -> dict[tuple[str, str], dict[str, str]]:
    with _REGISTRY_PATH.open() as f:
        raw_entries = json.load(f)

    registry: dict[tuple[str, str], dict[str, str]] = {}
    for entry in raw_entries:
        key = (entry["source"], entry["target"])
        registry[key] = entry
    return registry


def default_run_name_for_pair(source_alias: str, target_alias: str) -> str:
    special_cases = {
        ("qwen1.5-0.5b-chat", "qwen1.5-1.8b-chat"): "exp-weight-layer3",
    }
    return special_cases.get((source_alias, target_alias), "exp-weight6-layer3")


def build_acc_head_pair_name(source_model: str, target_model: str) -> str:
    source_alias = canonicalize_model_name(source_model)
    target_alias = canonicalize_model_name(target_model)
    return f"{source_alias}--to--{target_alias}"


def build_default_acc_head_path(source_alias: str, target_alias: str) -> str:
    run_name = default_run_name_for_pair(source_alias, target_alias)
    return str(_DEFAULT_LOCAL_ROOT / f"{source_alias}--to--{target_alias}" / run_name)


def build_default_acc_head_path_for_models(source_model: str, target_model: str) -> str:
    source_alias = canonicalize_model_name(source_model)
    target_alias = canonicalize_model_name(target_model)
    return build_default_acc_head_path(source_alias, target_alias)


def resolve_acc_head_path(source_model: str, target_model: str) -> str:
    source_alias = canonicalize_model_name(source_model)
    target_alias = canonicalize_model_name(target_model)

    registry = load_acc_head_registry()
    entry = registry.get((source_alias, target_alias))
    if entry is not None:
        return entry["local_path"]

    return build_default_acc_head_path(source_alias, target_alias)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve SpecDec++ acceptance head pair names and paths."
    )
    parser.add_argument("source_model", help="Source model name or model id.")
    parser.add_argument("target_model", help="Target model name or model id.")
    parser.add_argument(
        "--format",
        choices=["pair", "default-path", "resolved-path"],
        default="pair",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.format == "pair":
        print(build_acc_head_pair_name(args.source_model, args.target_model))
    elif args.format == "default-path":
        print(
            build_default_acc_head_path_for_models(args.source_model, args.target_model)
        )
    else:
        print(resolve_acc_head_path(args.source_model, args.target_model))


if __name__ == "__main__":
    main()
