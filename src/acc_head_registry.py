import json
import warnings
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

#: 仓库根。由本文件位置推出而非 CWD，所以项目整体搬家、或从别的目录起
#: 评测脚本时都仍然指向正确位置（下面所有相对路径都锚定到这里）。
_REPO_ROOT = Path(__file__).resolve().parent.parent

#: `prepare_acc_head.py` 的早期输出布局根：head 直接落在
#: `checkpoints/<模型目录>/<run_name>/`，而不是规范的 `acc_head/<pair>/<run_name>/`。
_LEGACY_ACC_HEAD_ROOT = Path("src/SpecDec_pp/checkpoints")


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
    # B26：与 utils.model_zoo 的分歧别名统一归一。registry 在 zoo 映射之前
    # 拿到的是用户原始别名（parse_arguments 中 RL 解析先于 model_zoo），
    # 此前两种拼法（qwen-3-0.6b vs qwen3-0.6b）指向不同 series，
    # 合法别名静默错过已注册对、落到不存在的默认 checkpoint 路径
    "qwen-3-0.6b": "qwen3-0.6b",
    "qwen-3-1.7b": "qwen3-1.7b",
    "qwen-3-14b": "qwen3-14b",
    "llama-2-chat-7b": "llama-2-7b-chat",
}


def canonicalize_model_name(model_name: str) -> str:
    if not model_name:
        raise ValueError(
            "model name 为空：--draft_model/--target_model 是否缺失？"
            "（必填检查见 parse_arguments，此处为直接调用方的防御）"
        )
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
    # 注册表位于 src/SpecDec_pp 子模块内；fresh clone（未 submodule update）
    # 时不存在。此时降级为空表（resolve 走纯默认路径）而不是在 argparse
    # 默认值求值阶段裸崩——栈会指向 argparse 内部，极难定位。
    if not _REGISTRY_PATH.exists():
        warnings.warn(
            f"acc-head 注册表不存在（子模块未初始化？）：{_REGISTRY_PATH}，"
            "降级为默认 acc-head 路径"
        )
        return {}
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


def repo_root() -> Path:
    """仓库根目录（与 CWD 无关）。"""
    return _REPO_ROOT


def anchor_repo_path(path: str | os.PathLike[str]) -> Path:
    """把仓库相对路径锚定到仓库根，绝对路径原样返回。

    注册表、`assets.json` 与 CLI 里的路径都是仓库相对形式，直接交给
    ``Path(...)`` 会相对 CWD 解析——换个工作目录跑评测就会指错地方。
    加载侧统一用本函数解析，仓库整体搬家不会失效。
    """
    candidate = Path(path)
    return candidate if candidate.is_absolute() else _REPO_ROOT / candidate


def is_usable_acc_head_dir(path: str | os.PathLike[str]) -> bool:
    """目录里是否真有一个可加载的 acc head。

    判据是 ``config.json`` 存在（`from_pretrained` 与本地兜底加载都需要它），
    这能把「下载脚本建好但没填内容的空目录」与「已就位的 head」区分开。
    """
    resolved = anchor_repo_path(path)
    return resolved.is_dir() and (resolved / "config.json").is_file()


#: 目录名/alias 里常见的、对 head 身份无影响的尾部限定词。
#: 早期训练输出目录会省略它们（`qwen1.5-1.8b` vs alias `qwen1.5-1.8b-chat`）。
_MODEL_QUALIFIERS = ("instruct", "chat", "hf")


def _squash_slug(name: str) -> str:
    """归一化模型目录名，抹平命名差异。

    - 忽略大小写、连字符与点：`qwen-3-14b` == `qwen3-14b`
    - 去掉尾部限定词：`qwen1.5-1.8b-chat` == `qwen1.5-1.8b`

    仅用于早期布局的目录匹配，不参与注册表键或 canonical alias。
    """
    slug = re.sub(r"[^a-z0-9]", "", name.lower())
    changed = True
    while changed:
        changed = False
        for qualifier in _MODEL_QUALIFIERS:
            if slug.endswith(qualifier) and len(slug) > len(qualifier):
                slug = slug[: -len(qualifier)]
                changed = True
                break
    return slug


def find_legacy_acc_head_path(target_alias: str, run_name: str) -> str | None:
    """在早期训练输出布局里找 head：`checkpoints/<模型目录>/<run_name>/`。

    目录名与 canonical alias 不完全一致（`qwen-3-14b` vs `qwen3-14b`、
    `llama-13b` vs `llama-2-13b`），所以按归一化 slug 比对而非字符串相等。
    找到返回**仓库相对**路径，找不到返回 None。目录遍历顺序固定，
    同名候选按字典序取第一个，结果可复现。
    """
    root = anchor_repo_path(_LEGACY_ACC_HEAD_ROOT)
    if not root.is_dir():
        return None
    wanted = _squash_slug(target_alias)
    for model_dir in sorted(root.iterdir()):
        if not model_dir.is_dir() or model_dir.name == "acc_head":
            continue
        if _squash_slug(model_dir.name) != wanted:
            continue
        if is_usable_acc_head_dir(model_dir / run_name):
            return str(_LEGACY_ACC_HEAD_ROOT / model_dir.name / run_name)
    return None


def resolve_acc_head_path(source_model: str, target_model: str) -> str:
    """解析某一对模型的 acc head 路径（仓库相对形式）。

    按顺序取第一个**磁盘上确实可用**的候选，避免把无效路径交给
    ``from_pretrained``（那会被 huggingface_hub 当成 repo id，报出与真实
    原因无关的 HFValidationError）：

    1. 注册表登记的 ``local_path``（规范布局，也是下载脚本的目标位置）
    2. 默认命名规则推出的规范路径
    3. 早期训练输出布局 ``checkpoints/<模型目录>/<run_name>``

    判据是目录里有 ``config.json``，而不是「目录存在」：`download_assets.py`
    是先建目录再下载，失败时会留下空目录，只判存在会让空目录盖掉可用的回退
    布局，把本来能跑的实验变成加载失败。

    三者都没命中时，回落到「注册表优先、否则规范路径」的原语义，把
    「文件缺失」留给加载处报一条明确的错误。
    """
    source_alias = canonicalize_model_name(source_model)
    target_alias = canonicalize_model_name(target_model)

    registry = load_acc_head_registry()
    entry = registry.get((source_alias, target_alias))
    registered = entry["local_path"] if entry is not None else None
    canonical = build_default_acc_head_path(source_alias, target_alias)

    if registered and is_usable_acc_head_dir(registered):
        return registered
    if is_usable_acc_head_dir(canonical):
        return canonical

    run_name = (
        Path(registered).name
        if registered
        else default_run_name_for_pair(source_alias, target_alias)
    )
    legacy = find_legacy_acc_head_path(target_alias, run_name)
    if legacy is not None:
        return legacy

    return registered or canonical


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
