import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


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
        description="Inspect Gemma target logits numerics on MT-Bench prompts."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--quantization", choices=["none", "4bit"], default="4bit")
    return parser


def load_questions(data_path: str, limit: int) -> list[dict]:
    questions = []
    with Path(data_path).open() as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            questions.append(json.loads(line))
    return questions


def build_prompt(tokenizer, question: str) -> str:
    content = SYSTEM_PROMPT + "\n" + question
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_model(model_name: str, device: str, quantization: str):
    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "device_map": device,
        "torch_dtype": torch.float16,
    }
    if quantization == "4bit":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs).eval()


def summarize_topk(tokenizer, logits: torch.Tensor, probs: torch.Tensor, top_k: int):
    k = min(top_k, logits.shape[-1])
    values, indices = torch.topk(probs, k)
    result = []
    for prob, index in zip(values.tolist(), indices.tolist()):
        result.append(
            {
                "id": int(index),
                "token": tokenizer.decode([index]).replace("\n", "\\n"),
                "prob": float(prob),
                "logit": float(logits[index].item()),
            }
        )
    return result


def analyze_sample(tokenizer, model, device: str, prompt: str, top_k: int) -> dict:
    encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(device)

    with torch.inference_mode():
        outputs = model(input_ids, use_cache=False)

    last_logits = outputs.logits[0, -1].detach().float().cpu()
    finite_mask = torch.isfinite(last_logits)
    nan_count = int(torch.isnan(last_logits).sum().item())
    posinf_count = int(torch.isposinf(last_logits).sum().item())
    neginf_count = int(torch.isneginf(last_logits).sum().item())
    finite_logits = last_logits[finite_mask]

    stats = {
        "prompt_len": int(input_ids.shape[1]),
        "nan_count": nan_count,
        "posinf_count": posinf_count,
        "neginf_count": neginf_count,
        "finite_count": int(finite_mask.sum().item()),
        "logit_min": float(finite_logits.min().item()) if finite_logits.numel() else None,
        "logit_max": float(finite_logits.max().item()) if finite_logits.numel() else None,
        "logit_mean": float(finite_logits.mean().item()) if finite_logits.numel() else None,
        "logit_std": float(finite_logits.std().item()) if finite_logits.numel() > 1 else 0.0,
    }

    probs = torch.softmax(last_logits, dim=-1)
    prob_finite = torch.isfinite(probs)
    stats.update(
        {
            "prob_nan_count": int(torch.isnan(probs).sum().item()),
            "prob_finite_count": int(prob_finite.sum().item()),
            "prob_sum": float(probs[prob_finite].sum().item()) if prob_finite.any() else None,
            "argmax_id": int(torch.argmax(last_logits).item()),
            "argmax_token": tokenizer.decode([int(torch.argmax(last_logits).item())]).replace(
                "\n", "\\n"
            ),
            "topk": summarize_topk(tokenizer, last_logits, probs, top_k),
        }
    )
    return stats


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.device, args.quantization)
    questions = load_questions(args.data, args.samples)

    nan_samples = 0
    prob_nan_samples = 0

    for idx, item in enumerate(questions):
        prompt = build_prompt(tokenizer, item["turns"][0])
        stats = analyze_sample(tokenizer, model, args.device, prompt, args.top_k)
        nan_samples += int(stats["nan_count"] > 0 or stats["posinf_count"] > 0 or stats["neginf_count"] > 0)
        prob_nan_samples += int(stats["prob_nan_count"] > 0)

        print(f"sample={idx} question_id={item['question_id']} category={item['category']}")
        print(
            f"prompt_len={stats['prompt_len']} nan={stats['nan_count']} posinf={stats['posinf_count']} "
            f"neginf={stats['neginf_count']} logit_min={stats['logit_min']} logit_max={stats['logit_max']} "
            f"logit_mean={stats['logit_mean']} logit_std={stats['logit_std']}"
        )
        print(
            f"prob_nan={stats['prob_nan_count']} prob_sum={stats['prob_sum']} "
            f"argmax_id={stats['argmax_id']} argmax_token={stats['argmax_token']!r}"
        )
        print(f"top{args.top_k}={stats['topk']}")
        print()

    total = max(len(questions), 1)
    print("Summary")
    print(f"sample_count={len(questions)}")
    print(f"logit_invalid_sample_rate={nan_samples / total:.3f}")
    print(f"prob_nan_sample_rate={prob_nan_samples / total:.3f}")


if __name__ == "__main__":
    main()
