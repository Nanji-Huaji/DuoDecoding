import argparse
from dataclasses import dataclass
from pathlib import Path

from src.acc_head_registry import canonicalize_model_name


ROLE_MAIN = "main"
ROLE_LITTLE = "little"
VALID_ROLES = {ROLE_MAIN, ROLE_LITTLE}
DEFAULT_TOPK_CANDIDATES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
DEFAULT_RL_AGENT_ROOT = Path("checkpoints/rl_agents")


@dataclass(frozen=True)
class RLAgentSpec:
    role: str
    source_model: str
    target_model: str
    pair_name: str
    latest_path: str
    best_path: str
    agent_name: str
    threshold_candidates: list[float]
    topk_candidates: list[int]


def _validate_role(role: str) -> str:
    normalized = role.strip().lower()
    if normalized not in VALID_ROLES:
        raise ValueError(f"Unsupported RL agent role: {role}")
    return normalized


def build_rl_agent_pair_name(source_model: str, target_model: str) -> str:
    source_alias = canonicalize_model_name(source_model)
    target_alias = canonicalize_model_name(target_model)
    return f"{source_alias}--to--{target_alias}"


def default_threshold_candidates_for_role(role: str) -> list[float]:
    normalized = _validate_role(role)
    if normalized == ROLE_MAIN:
        return [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    return [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def default_agent_name_for_role(role: str) -> str:
    normalized = _validate_role(role)
    return f"rl_adapter_{normalized}"


def resolve_rl_agent_path(
    role: str,
    source_model: str,
    target_model: str,
    *,
    kind: str = "latest",
    checkpoint_root: str | Path = DEFAULT_RL_AGENT_ROOT,
) -> str:
    normalized_role = _validate_role(role)
    if kind not in {"latest", "best"}:
        raise ValueError(f"Unsupported RL checkpoint kind: {kind}")

    pair_name = build_rl_agent_pair_name(source_model, target_model)
    root = Path(checkpoint_root)
    return str(root / normalized_role / pair_name / f"{kind}.pth")


def resolve_rl_agent_paths(
    little_model: str,
    draft_model: str,
    target_model: str,
    *,
    checkpoint_root: str | Path = DEFAULT_RL_AGENT_ROOT,
) -> dict[str, str]:
    return {
        "main_latest": resolve_rl_agent_path(
            ROLE_MAIN,
            draft_model,
            target_model,
            kind="latest",
            checkpoint_root=checkpoint_root,
        ),
        "main_best": resolve_rl_agent_path(
            ROLE_MAIN,
            draft_model,
            target_model,
            kind="best",
            checkpoint_root=checkpoint_root,
        ),
        "little_latest": resolve_rl_agent_path(
            ROLE_LITTLE,
            little_model,
            draft_model,
            kind="latest",
            checkpoint_root=checkpoint_root,
        ),
        "little_best": resolve_rl_agent_path(
            ROLE_LITTLE,
            little_model,
            draft_model,
            kind="best",
            checkpoint_root=checkpoint_root,
        ),
    }


def build_legacy_rl_agent_paths(
    role: str,
    little_model: str | None,
    draft_model: str,
    target_model: str,
) -> list[str]:
    normalized_role = _validate_role(role)
    legacy_paths: list[str] = []

    legacy_filename = f"rl_adapter_{normalized_role}.pth"
    for model_name in {little_model, draft_model, target_model}:
        if not model_name:
            continue
        series = canonicalize_model_name(model_name)
        legacy_paths.append(str(Path("checkpoints") / series / legacy_filename))

    legacy_paths.append(str(Path("checkpoints") / legacy_filename))
    return legacy_paths


def resolve_legacy_rl_agent_load_path(
    role: str,
    little_model: str | None,
    draft_model: str,
    target_model: str,
) -> str | None:
    for candidate in build_legacy_rl_agent_paths(
        role, little_model, draft_model, target_model
    ):
        if Path(candidate).exists():
            return candidate
    return None


def get_rl_agent_spec(
    role: str,
    *,
    little_model: str | None,
    draft_model: str,
    target_model: str,
    checkpoint_root: str | Path = DEFAULT_RL_AGENT_ROOT,
) -> RLAgentSpec:
    normalized_role = _validate_role(role)
    if normalized_role == ROLE_MAIN:
        source_model = draft_model
        dest_model = target_model
    else:
        if little_model is None:
            raise ValueError("little_model is required for little RL agent")
        source_model = little_model
        dest_model = draft_model

    return RLAgentSpec(
        role=normalized_role,
        source_model=source_model,
        target_model=dest_model,
        pair_name=build_rl_agent_pair_name(source_model, dest_model),
        latest_path=resolve_rl_agent_path(
            normalized_role,
            source_model,
            dest_model,
            kind="latest",
            checkpoint_root=checkpoint_root,
        ),
        best_path=resolve_rl_agent_path(
            normalized_role,
            source_model,
            dest_model,
            kind="best",
            checkpoint_root=checkpoint_root,
        ),
        agent_name=default_agent_name_for_role(normalized_role),
        threshold_candidates=default_threshold_candidates_for_role(normalized_role),
        topk_candidates=list(DEFAULT_TOPK_CANDIDATES),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resolve pair-based RL agent names and checkpoint paths."
    )
    parser.add_argument("role", choices=sorted(VALID_ROLES), help="RL agent role.")
    parser.add_argument("source_model", help="Source model name or model id.")
    parser.add_argument("target_model", help="Target model name or model id.")
    parser.add_argument(
        "--kind",
        choices=["latest", "best"],
        default="latest",
        help="Checkpoint variant to resolve.",
    )
    parser.add_argument(
        "--format",
        choices=["pair", "path", "agent-name"],
        default="path",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.format == "pair":
        print(build_rl_agent_pair_name(args.source_model, args.target_model))
    elif args.format == "agent-name":
        print(default_agent_name_for_role(args.role))
    else:
        print(
            resolve_rl_agent_path(
                args.role,
                args.source_model,
                args.target_model,
                kind=args.kind,
            )
        )


if __name__ == "__main__":
    main()
