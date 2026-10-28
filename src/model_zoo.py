"""模型注册表（D3：从 utils 拆分，utils 再导出保兼容）。

别名→路径 zoo、未部署拒绝、vocab_size 查表（原始别名命中）。
"""

import json
import os
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_REPOS = ROOT / "configs" / "model_repos.json"


def get_vocab_size(model_name: str) -> int:
    try:
        with open(os.path.join(model_name, "config.json"), "r") as f:
            config = json.load(f)
            vocab_size = config.get("vocab_size", None)
            if vocab_size is not None:
                return vocab_size

            # Check text_config for multimodal models
            if "text_config" in config and isinstance(config["text_config"], dict):
                vocab_size = config["text_config"].get("vocab_size", None)
                if vocab_size is not None:
                    return vocab_size

    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # Fallback to AutoConfig if not found in JSON or file missing
    try:
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "vocab_size"):
            return config.vocab_size
        if hasattr(config, "text_config") and hasattr(config.text_config, "vocab_size"):
            return config.text_config.vocab_size
    except Exception:
        pass

    raise ValueError(f"Vocab size not found in config for model {model_name}.")


@lru_cache(maxsize=1)
def load_model_repos() -> dict:
    if not MODEL_REPOS.is_file():
        return {"models": {}}
    return json.loads(MODEL_REPOS.read_text(encoding="utf-8"))


def resolve_model(name: str, source: str, legacy_zoo: dict) -> tuple[str, str, dict]:
    """按 --model_source 解析别名；返回 (resolved, kind, entry)。

    kind: local（本地目录/文件）/ hub（HF repo id）/ legacy（旧硬编码映射或原样）。
    """
    if source == "legacy":
        return legacy_zoo.get(name, name), "legacy", {}
    entry = load_model_repos().get("models", {}).get(name)
    if entry is None:
        return legacy_zoo.get(name, name), "legacy", {}
    local_dir = entry.get("local_dir")
    local_path = ROOT / local_dir if local_dir else None
    has_local = bool(local_path and local_path.exists())
    hf_repo = entry.get("hf_repo")
    if source == "hub":
        if hf_repo:
            return hf_repo, "hub", entry
        return (str(local_path) if has_local else name), "local" if has_local else "legacy", entry
    if source == "local":
        if local_path:
            return str(local_path), "local", entry
        return name, "legacy", entry
    # auto：本地存在优先，否则用 HF repo id（首次运行由 transformers 下载）
    if has_local:
        return str(local_path), "local", entry
    if hf_repo:
        return hf_repo, "hub", entry
    return name, "legacy", entry


def model_zoo(args):
    # B24：显式必填。原默认 codellama-7b/codellama-70b 不可解析（zoo 无此
    # 键、本地无此目录、HF 无此仓库名），依赖默认只会在模型加载阶段以难懂
    # 的错误失败——提前到解析阶段给出明确报错。
    if not args.draft_model or not args.target_model:
        raise ValueError(
            "--draft_model 与 --target_model 必须显式指定"
            "（原默认 codellama-7b/codellama-70b 不可解析，已移除。"
            "例: --draft_model tiny-llama-1.1b --target_model llama-2-13b）"
        )

    # B23：未部署模型显式拒绝。此前 zoo 把它们映射成 "xxx还没部署" 占位
    # 路径静默传播，直到模型加载才以难懂的路径/HF 错误失败。
    undeployed = {
        "deepseek-1.3b",
        "deepseek-6.7b",
        "vicuna-7b-v1.5",
        "vicuna-7b-v1.3",
    }
    for role, model in (
        ("--draft_model", args.draft_model),
        ("--target_model", args.target_model),
        ("--little_model", getattr(args, "little_model", None)),
    ):
        if model in undeployed:
            raise ValueError(
                f"{role}={model} 对应模型未部署（zoo 占位符）。"
                f"请先下载到本地并更新 zoo 映射，或改用已部署的别名"
            )

    vocab_size = {
        "codellama-7b": 32000,
        "codellama-34b": 32000,
        "codellama-70b": 32000,
        "llama-2-7b": 32000,
        "llama-2-70b": 32000,
        "deepseek-1.3b": 32256,
        "deepseek-6.7b": 32256,
        "deepseek-33b": 32256,
        "llama-68m-q5-gguf": 32000,
        "llama-68m-q8-gguf": 32000,
        "llama-68m": 32000,
        "llama-68m-fp16": 32000,
        "llama-160m-q5-gguf": 32000,
        "llama-160m": 32000,
        "vicuna-68m-q5-gguf": 32000,
        "vicuna-68m": 32000,
        "vicuna-7b-v1.5": 32000,
        "vicuna-7b-v1.3": 32000,
        "llama-290m-q5-gguf": 32000,
        "llama-290m": 32000,
        "llama-543m": 32000,
        "llama-543m-q5-gguf": 32000,
        "llama-2-7b-chat": 32000,
        "llama-68m-chat-q5-gguf": 32000,
        "llama-3.2-1b": 32000,
        "llama-2-13b": 32000,
        "tiny-vicuna-1b": 32000,
        "vicuna-13b-v1.5": 32000,
        "tiny-llama-1.1b": 32000,
        "Llama-2-13b": 32000,
        "llama-3-70b": 32000,
        "qwen-3-0.6b": 151936,
        "qwen-3-1.7b": 151936,
        "qwen-3-14b": 151936,
    }
    # 注：原字典末尾有重复键 "llama-2-70b"（Python 静默取后者），已去重（B22）。

    zoo = {
        "llama-2-chat-7b": "meta-llama/Llama-2-7b-chat-hf",
        "llama-68m-q5-gguf": "llama/llama-68m-gguf-series/Llama-68M-Chat-v1-Q5_0.gguf",
        "llama-68m-q8-gguf": "llama/llama-68m-gguf-series/Llama-68M-Chat-v1-Q8_0.gguf",
        "llama-68m-fp16": "llama/llama-68m-gguf-series/llama-68m-chat-v1.fp16.gguf",
        "llama-68m": "llama/llama-68m",
        "llama-160m-q5-gguf": "llama/llama-160m-q5-gguf",
        "llama-160m": "llama/llama-160m",
        "vicuna-68m-q5-gguf": "vicuna/vicuna-68m.Q5_K_M-gguf/vicuna-68m.Q5_K_M.gguf",
        "vicuna-68m": "vicuna/vicuna-68m",
        "llama-2-7b-chat": "meta-llama/Llama-2-7b-chat-hf",
        "llama-68m-chat-q5-gguf": "llama/llama-68m-gguf-series/llama-68m-chat-v1.q5_k_m.gguf",
        "llama-3.2-1b": "llama/llama-3.2-1b",
        "llama-2-13b": "llama/Llama-2-13b-hf",
        "llama-2-70b": "llama/llama-2-70b",
        "llama-13b-hf": "llama/Llama-2-13b-hf",
        "tiny-vicuna-1b": "vicuna/tiny-vicuna-1b",
        "vicuna-13b-v1.5": "vicuna/vicuna-13b-v1.5",
        "tiny-llama-1.1b": "llama/tiny-llama-1.1b",
        "Llama-2-13b": "llama/Llama-2-13b-hf",
        "llama-3-70b": "llama/llama-70B",
        "qwen-3-0.6b": "Qwen/Qwen3-0.6B",
        "qwen-3-1.7b": "Qwen/Qwen3-1.7B",
        "qwen-3-14b": "Qwen/Qwen3-14B",
        "Qwen/Qwen3-32B-FP8": "Qwen/Qwen3-32B-FP8",
        "llama-2-chat-70b": "meta-llama/Llama-2-70b-chat-hf",  # mapping to HuggingFace model
    }
    # B22：vocab 查表必须在 zoo 映射前用原始别名命中。此前查表放在映射
    # 之后，键是别名而值已变成本地路径/HF id，最常用别名全部 miss，每次
    # parse 都落 get_vocab_size 读 config.json 或走网络回退。
    draft_alias = args.draft_model
    source = getattr(args, "model_source", "auto")
    _prov: dict[str, dict] = {}
    for role, attr in (
        ("draft", "draft_model"),
        ("target", "target_model"),
        ("little", "little_model"),
    ):
        if not hasattr(args, attr):
            continue
        alias = getattr(args, attr)
        resolved, kind, entry = resolve_model(alias, source, zoo)
        setattr(args, attr, resolved)
        _prov[role] = {
            "alias": alias,
            "resolved": resolved,
            "source": kind,
            "hf_repo": (entry or {}).get("hf_repo"),
            "hf_revision": (entry or {}).get("hf_revision"),
            "modelscope_id": (entry or {}).get("modelscope_id"),
        }
    if not hasattr(args, "little_model"):
        args.little_model = args.draft_model
    args._model_provenance = _prov
    # 注意不要写成 dict.get(a, dict.get(b, get_vocab_size(b)))：默认参数会
    # 被立即求值，hub id 在离线环境下会触发网络回退。
    if draft_alias in vocab_size:
        args.vocab_size = vocab_size[draft_alias]
    elif args.draft_model in vocab_size:
        args.vocab_size = vocab_size[args.draft_model]
    else:
        args.vocab_size = get_vocab_size(args.draft_model)
