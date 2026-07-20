import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODELS = (
    "google/gemma-2-9b-it",
    "google/gemma-2-27b-it",
)

DEFAULT_PROMPT = "You are a helpful assistant.\nExplain speculative decoding in one paragraph."
DEFAULT_DATA = "data/mt_bench.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Check Gemma tokenizer/config/embedding alignment."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(DEFAULT_MODELS),
        help="Models to compare.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt used to compare tokenization and max token id.",
    )
    parser.add_argument(
        "--load-models",
        action="store_true",
        help="Load model weights and inspect embedding sizes.",
    )
    parser.add_argument(
        "--mt-bench-samples",
        type=int,
        default=0,
        help="Also inspect the first N MT-Bench first-turn prompts.",
    )
    parser.add_argument(
        "--data",
        default=DEFAULT_DATA,
        help="Dataset path for MT-Bench prompt inspection.",
    )
    return parser


def _config_vocab_size(config) -> int | None:
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    return None


def inspect_model(name: str, prompt: str, load_model: bool) -> dict[str, object]:
    tokenizer = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(name, trust_remote_code=True)
    encoded = tokenizer.encode(prompt, add_special_tokens=False)

    info: dict[str, object] = {
        "name": name,
        "tokenizer_class": tokenizer.__class__.__name__,
        "tokenizer_len": len(tokenizer),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "config_vocab_size": _config_vocab_size(config),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "special_tokens_map": tokenizer.special_tokens_map,
        "encoded": encoded,
        "encoded_len": len(encoded),
        "encoded_max": max(encoded) if encoded else None,
    }

    if load_model:
        model = AutoModelForCausalLM.from_pretrained(
            name,
            trust_remote_code=True,
            device_map="cpu",
            torch_dtype=torch.float16,
        )
        embeddings = model.get_input_embeddings()
        info["embedding_vocab_size"] = int(embeddings.weight.shape[0])
        lm_head = getattr(model, "lm_head", None)
        if lm_head is not None and hasattr(lm_head, "weight"):
            info["lm_head_vocab_size"] = int(lm_head.weight.shape[0])
        del model

    return info


def print_info(info: dict[str, object]) -> None:
    print(f"== {info['name']} ==")
    print(f"tokenizer_class: {info['tokenizer_class']}")
    print(f"len(tokenizer): {info['tokenizer_len']}")
    print(f"tokenizer.vocab_size: {info['tokenizer_vocab_size']}")
    print(f"config vocab_size: {info['config_vocab_size']}")
    if "embedding_vocab_size" in info:
        print(f"embedding vocab_size: {info['embedding_vocab_size']}")
    if "lm_head_vocab_size" in info:
        print(f"lm_head vocab_size: {info['lm_head_vocab_size']}")
    print(f"pad/eos/bos/unk: {info['pad_token_id']}/{info['eos_token_id']}/{info['bos_token_id']}/{info['unk_token_id']}")
    print(f"special_tokens_map: {info['special_tokens_map']}")
    print(
        f"prompt encoded_len={info['encoded_len']} max_token_id={info['encoded_max']}"
    )
    print()


def compare_infos(infos: list[dict[str, object]]) -> None:
    if len(infos) < 2:
        return

    first = infos[0]
    for current in infos[1:]:
        print(f"Comparing {first['name']} vs {current['name']}")
        print(
            "same tokenizer len:",
            first["tokenizer_len"] == current["tokenizer_len"],
        )
        print(
            "same tokenizer vocab_size:",
            first["tokenizer_vocab_size"] == current["tokenizer_vocab_size"],
        )
        print(
            "same config vocab_size:",
            first["config_vocab_size"] == current["config_vocab_size"],
        )
        print(
            "same special_tokens_map:",
            first["special_tokens_map"] == current["special_tokens_map"],
        )
        print("same encoded prompt:", first["encoded"] == current["encoded"])

        if "embedding_vocab_size" in first and "embedding_vocab_size" in current:
            print(
                "same embedding vocab_size:",
                first["embedding_vocab_size"] == current["embedding_vocab_size"],
            )
        print()


def inspect_mt_bench_prompts(model_name: str, data_path: str, sample_count: int) -> None:
    if sample_count <= 0:
        return

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    path = Path(data_path)
    print(f"Inspecting first {sample_count} MT-Bench prompts with {model_name}")
    with path.open() as handle:
        for idx, line in enumerate(handle):
            if idx >= sample_count:
                break
            datum = json.loads(line)
            content = datum["turns"][0]
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": content}],
                tokenize=False,
                add_generation_prompt=True,
            )
            encoded = tokenizer.encode(prompt, add_special_tokens=False)
            max_id = max(encoded) if encoded else None
            print(
                f"mt_bench[{idx}] question_id={datum.get('question_id', idx)} "
                f"encoded_len={len(encoded)} max_token_id={max_id}"
            )
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    infos = [inspect_model(name, args.prompt, args.load_models) for name in args.models]
    for info in infos:
        print_info(info)
    compare_infos(infos)
    inspect_mt_bench_prompts(args.models[0], args.data, args.mt_bench_samples)


if __name__ == "__main__":
    main()
