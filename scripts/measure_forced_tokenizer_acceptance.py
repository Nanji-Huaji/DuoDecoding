import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure acceptance using a forced shared tokenizer."
    )
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--tokenizer-model", required=True)
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument(
        "--disable-target-dense-attention",
        action="store_true",
        help="Set target config dense_attention_every_n_layers=0 before loading.",
    )
    parser.add_argument(
        "--disable-draft-dense-attention",
        action="store_true",
        help="Set draft config dense_attention_every_n_layers=0 before loading.",
    )
    parser.add_argument("--output", default="forced_tokenizer_acceptance.json")
    return parser


def resolve_torch_dtype(name: str):
    mapping = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def load_questions(data_path: str, limit: int) -> list[dict]:
    items: list[dict] = []
    with Path(data_path).open() as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            items.append(json.loads(line))
    return items


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n" + question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def summarize_topk_overlap(draft_probs: torch.Tensor, target_probs: torch.Tensor, top_k: int) -> int:
    draft_top = set(torch.topk(draft_probs, top_k).indices.tolist())
    target_top = set(torch.topk(target_probs, top_k).indices.tolist())
    return len(draft_top & target_top)


def get_config_vocab_size(model_name: str) -> int:
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    if hasattr(config, "vocab_size"):
        return int(config.vocab_size)
    text_config = getattr(config, "text_config", None)
    if text_config is not None and hasattr(text_config, "vocab_size"):
        return int(text_config.vocab_size)
    raise ValueError(f"Could not determine vocab size for {model_name}")


def load_model(model_name: str, device: str, dtype, disable_dense_attention: bool):
    config = None
    if disable_dense_attention:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        if hasattr(config, "dense_attention_every_n_layers"):
            config.dense_attention_every_n_layers = 0
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        config=config,
        device_map=device,
        dtype=dtype,
    ).eval()


def main() -> None:
    args = build_parser().parse_args()
    dtype = resolve_torch_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_model,
        trust_remote_code=True,
    )
    questions = load_questions(args.data, args.samples)
    prompts = [build_prompt(tokenizer, item["turns"][0]) for item in questions]
    encoded_rows = [
        tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        for prompt in prompts
    ]

    tokenizer_len = len(tokenizer)
    tokenizer_vocab_size = int(getattr(tokenizer, "vocab_size", tokenizer_len))
    draft_vocab_size = get_config_vocab_size(args.draft_model)
    target_vocab_size = get_config_vocab_size(args.target_model)

    max_token_id = max(int(row.max().item()) for row in encoded_rows if row.numel() > 0)
    compatibility = {
        "tokenizer_len": tokenizer_len,
        "tokenizer_vocab_size": tokenizer_vocab_size,
        "draft_vocab_size": draft_vocab_size,
        "target_vocab_size": target_vocab_size,
        "max_prompt_token_id": max_token_id,
        "draft_can_embed_prompt": max_token_id < draft_vocab_size,
        "target_can_embed_prompt": max_token_id < target_vocab_size,
    }

    if not compatibility["draft_can_embed_prompt"] or not compatibility["target_can_embed_prompt"]:
        result = {
            "draft_model": args.draft_model,
            "target_model": args.target_model,
            "tokenizer_model": args.tokenizer_model,
            "status": "incompatible",
            "compatibility": compatibility,
        }
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
        return

    draft_model = load_model(
        args.draft_model,
        args.device,
        dtype,
        args.disable_draft_dense_attention,
    )
    target_model = load_model(
        args.target_model,
        args.device,
        dtype,
        args.disable_target_dense_attention,
    )

    same_argmax = 0
    overlap_sum = 0
    acceptance_sum = 0.0
    greedy_accept = 0
    per_sample = []

    with torch.inference_mode():
        for prompt, input_ids in zip(prompts, encoded_rows):
            model_input = input_ids.unsqueeze(0).to(args.device)
            draft_logits = draft_model(input_ids=model_input, use_cache=False).logits[0, -1]
            target_logits = target_model(input_ids=model_input, use_cache=False).logits[0, -1]

            draft_probs = torch.softmax(draft_logits.detach().float().cpu(), dim=-1)
            target_probs = torch.softmax(target_logits.detach().float().cpu(), dim=-1)

            draft_argmax = int(draft_probs.argmax().item())
            target_argmax = int(target_probs.argmax().item())
            same_argmax += int(draft_argmax == target_argmax)
            overlap_sum += summarize_topk_overlap(draft_probs, target_probs, args.top_k)

            draft_mass = float(draft_probs[draft_argmax].item())
            target_mass = float(target_probs[draft_argmax].item())
            acceptance_ratio = target_mass / draft_mass if draft_mass > 0 else 0.0
            acceptance_sum += acceptance_ratio
            greedy_accept += int(acceptance_ratio >= 1.0)

            per_sample.append(
                {
                    "prompt": prompt,
                    "prompt_len": int(input_ids.shape[0]),
                    "draft_argmax": draft_argmax,
                    "target_argmax": target_argmax,
                    "acceptance_ratio": acceptance_ratio,
                    "greedy_accept": acceptance_ratio >= 1.0,
                }
            )

    total = len(per_sample)
    result = {
        "draft_model": args.draft_model,
        "target_model": args.target_model,
        "tokenizer_model": args.tokenizer_model,
        "status": "ok",
        "compatibility": compatibility,
        "same_argmax_rate": same_argmax / total,
        f"avg_top{args.top_k}_overlap": overlap_sum / total,
        "avg_acceptance_ratio": acceptance_sum / total,
        "greedy_accept_rate": greedy_accept / total,
        "samples": total,
        "per_sample": per_sample,
    }
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
