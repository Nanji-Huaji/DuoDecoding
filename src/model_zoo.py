"""模型注册表（D3：从 utils 拆分，utils 再导出保兼容）。

别名→路径 zoo、未部署拒绝、vocab_size 查表（原始别名命中）。
"""

import json
import os


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
    args.draft_model = zoo.get(args.draft_model, args.draft_model)
    args.target_model = zoo.get(args.target_model, args.target_model)
    args.little_model = (
        zoo.get(args.little_model, args.little_model)
        if hasattr(args, "little_model")
        else args.draft_model
    )
    args.vocab_size = vocab_size.get(
        draft_alias,
        vocab_size.get(args.draft_model, get_vocab_size(args.draft_model)),
    )
