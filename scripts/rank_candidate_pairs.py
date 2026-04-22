import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure "
    "that your responses are socially unbiased and positive in nature.\n\n"
    "If a question does not make any sense, or is not factually coherent, explain why "
    "instead of answering something not correct. If you don't know the answer to a "
    "question, please don't share false information."
)


SERIES = {
    "gemma": [
        "google/gemma-2-2b-it",
        "google/gemma-2-9b-it",
        "google/gemma-2-27b-it",
    ],
    "phi3": [
        "microsoft/Phi-3-mini-4k-instruct",
        "microsoft/Phi-3-medium-4k-instruct",
    ],
    "qwen3": [
        "Qwen/Qwen3-0.6B",
        "Qwen/Qwen3-1.7B",
        "Qwen/Qwen3-14B",
    ],
    "qwen15": [
        "Qwen/Qwen1.5-0.5B-Chat",
        "Qwen/Qwen1.5-1.8B-Chat",
        "Qwen/Qwen1.5-7B-Chat",
    ],
    "llama": [
        "llama-68m",
        "tiny-llama-1.1b",
        "llama-2-13b",
    ],
    "vicuna": [
        "vicuna-68m",
        "tiny-vicuna-1b",
        "vicuna-13b-v1.5",
    ],
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rank candidate draft-target pairs.")
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument(
        "--series",
        nargs="*",
        default=["gemma", "phi3", "qwen3", "qwen15", "llama", "vicuna"],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--output", default="pair_ranking.json")
    return parser


def load_questions(data_path: str, limit: int) -> list[dict]:
    result = []
    with Path(data_path).open() as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            result.append(json.loads(line))
    return result


def build_prompt(tokenizer, question: str) -> str:
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n" + question}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def build_candidate_pairs(selected_series: list[str]) -> list[tuple[str, str, str]]:
    pairs: list[tuple[str, str, str]] = []
    for series_name in selected_series:
        models = SERIES[series_name]
        for i in range(len(models) - 1):
            pairs.append((series_name, models[i], models[i + 1]))
        if len(models) >= 3:
            pairs.append((series_name, models[0], models[-1]))
    return pairs


def summarize_topk_overlap(draft_probs: torch.Tensor, target_probs: torch.Tensor, top_k: int) -> int:
    draft_top = set(torch.topk(draft_probs, top_k).indices.tolist())
    target_top = set(torch.topk(target_probs, top_k).indices.tolist())
    return len(draft_top & target_top)


def collect_model_probs(model_name: str, prompts: list[str], device: str) -> list[torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map=device,
        dtype=torch.float16,
    ).eval()
    rows: list[torch.Tensor] = []
    with torch.inference_mode():
        for prompt in prompts:
            input_ids = tokenizer([prompt], add_special_tokens=False, return_tensors="pt").input_ids.to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].detach().float().cpu()
            rows.append(torch.softmax(logits, dim=-1))
    del model
    torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = build_parser().parse_args()
    questions = load_questions(args.data, args.samples)

    prompts_by_series: dict[str, list[str]] = {}
    for series_name in args.series:
        tokenizer = AutoTokenizer.from_pretrained(SERIES[series_name][-1], trust_remote_code=True)
        prompts_by_series[series_name] = [build_prompt(tokenizer, item["turns"][0]) for item in questions]

    all_pairs = build_candidate_pairs(args.series)
    results = []

    model_prob_cache: dict[str, list[torch.Tensor]] = {}

    for series_name, draft_model, target_model in all_pairs:
        prompts = prompts_by_series[series_name]
        if draft_model not in model_prob_cache:
            model_prob_cache[draft_model] = collect_model_probs(draft_model, prompts, args.device)
        if target_model not in model_prob_cache:
            model_prob_cache[target_model] = collect_model_probs(target_model, prompts, args.device)

        same_argmax = 0
        overlap_sum = 0
        acceptance_sum = 0.0
        greedy_accept = 0

        for draft_probs, target_probs in zip(model_prob_cache[draft_model], model_prob_cache[target_model]):
            draft_argmax = int(draft_probs.argmax().item())
            target_argmax = int(target_probs.argmax().item())
            same_argmax += int(draft_argmax == target_argmax)
            overlap_sum += summarize_topk_overlap(draft_probs, target_probs, args.top_k)
            draft_mass = float(draft_probs[draft_argmax].item())
            target_mass = float(target_probs[draft_argmax].item())
            acceptance_ratio = target_mass / draft_mass if draft_mass > 0 else 0.0
            acceptance_sum += acceptance_ratio
            greedy_accept += int(acceptance_ratio >= 1.0)

        total = len(prompts)
        result = {
            "series": series_name,
            "draft_model": draft_model,
            "target_model": target_model,
            "same_argmax_rate": same_argmax / total,
            f"avg_top{args.top_k}_overlap": overlap_sum / total,
            "avg_acceptance_ratio": acceptance_sum / total,
            "greedy_accept_rate": greedy_accept / total,
        }
        results.append(result)
        print(result)

    results.sort(key=lambda item: item["avg_acceptance_ratio"], reverse=True)
    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved ranking to {args.output}")


if __name__ == "__main__":
    main()
