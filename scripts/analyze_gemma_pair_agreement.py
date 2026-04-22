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
        description="Analyze next-token agreement between draft and target Gemma models."
    )
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--num-shots", type=int, default=0)
    parser.add_argument("--device-draft", default="cuda:1")
    parser.add_argument("--device-target", default="cuda:0")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--target-4bit", action="store_true")
    return parser


def build_gemma_prompt(tokenizer, question: str) -> str:
    content = SYSTEM_PROMPT + "\n" + question
    messages = [{"role": "user", "content": content}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def load_questions(data_path: str, limit: int) -> list[dict]:
    questions = []
    with Path(data_path).open() as handle:
        for idx, line in enumerate(handle):
            if idx >= limit:
                break
            questions.append(json.loads(line))
    return questions


def summarize_topk(tokenizer, probs: torch.Tensor, top_k: int) -> list[tuple[int, str, float]]:
    k = min(top_k, probs.shape[-1])
    values, indices = torch.topk(probs, k)
    result = []
    for value, index in zip(values.tolist(), indices.tolist()):
        token_text = tokenizer.decode([index]).replace("\n", "\\n")
        result.append((int(index), token_text, float(value)))
    return result


def analyze_prompt(
    prompt: str,
    tokenizer,
    draft_model,
    target_model,
    draft_device: str,
    target_device: str,
    top_k: int,
) -> dict:
    encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids

    with torch.inference_mode():
        draft_outputs = draft_model(input_ids.to(draft_device), use_cache=False)
        target_outputs = target_model(input_ids.to(target_device), use_cache=False)

    draft_probs = torch.softmax(draft_outputs.logits[0, -1].float().cpu(), dim=-1)
    target_probs = torch.softmax(target_outputs.logits[0, -1].float().cpu(), dim=-1)

    draft_argmax = int(draft_probs.argmax().item())
    target_argmax = int(target_probs.argmax().item())
    draft_rank_in_target = int((target_probs > target_probs[draft_argmax]).sum().item()) + 1
    target_rank_in_draft = int((draft_probs > draft_probs[target_argmax]).sum().item()) + 1

    overlap_topk = set(torch.topk(draft_probs, top_k).indices.tolist()) & set(
        torch.topk(target_probs, top_k).indices.tolist()
    )

    draft_token_under_target = float(target_probs[draft_argmax].item())
    draft_token_under_draft = float(draft_probs[draft_argmax].item())
    target_token_under_draft = float(draft_probs[target_argmax].item())
    acceptance_ratio = (
        draft_token_under_target / draft_token_under_draft
        if draft_token_under_draft > 0
        else 0.0
    )

    return {
        "prompt_len": int(input_ids.shape[1]),
        "draft_argmax": draft_argmax,
        "target_argmax": target_argmax,
        "same_argmax": draft_argmax == target_argmax,
        "draft_argmax_prob": float(draft_probs[draft_argmax].item()),
        "target_argmax_prob": float(target_probs[target_argmax].item()),
        "draft_rank_in_target": draft_rank_in_target,
        "target_rank_in_draft": target_rank_in_draft,
        "topk_overlap_count": len(overlap_topk),
        "draft_token_under_target": draft_token_under_target,
        "draft_token_under_draft": draft_token_under_draft,
        "target_token_under_draft": target_token_under_draft,
        "acceptance_ratio": acceptance_ratio,
        "accepts_if_greedy": acceptance_ratio >= 1.0,
        "draft_topk": summarize_topk(tokenizer, draft_probs, top_k),
        "target_topk": summarize_topk(tokenizer, target_probs, top_k),
    }


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
        device_map=args.device_draft,
        torch_dtype=torch.float16,
    ).eval()
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        trust_remote_code=True,
        device_map=args.device_target,
        torch_dtype=torch.float16,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            if args.target_4bit
            else None
        ),
    ).eval()

    questions = load_questions(args.data, args.samples)
    same_argmax_count = 0
    target_rank_sum = 0
    overlap_sum = 0
    acceptance_ratio_sum = 0.0
    greedy_accept_count = 0

    for idx, item in enumerate(questions):
        prompt = build_gemma_prompt(tokenizer, item["turns"][0])
        analysis = analyze_prompt(
            prompt,
            tokenizer,
            draft_model,
            target_model,
            args.device_draft,
            args.device_target,
            args.top_k,
        )
        same_argmax_count += int(analysis["same_argmax"])
        target_rank_sum += analysis["draft_rank_in_target"]
        overlap_sum += analysis["topk_overlap_count"]
        acceptance_ratio_sum += analysis["acceptance_ratio"]
        greedy_accept_count += int(analysis["accepts_if_greedy"])

        print(f"sample={idx} question_id={item['question_id']} category={item['category']}")
        print(
            f"prompt_len={analysis['prompt_len']} same_argmax={analysis['same_argmax']} "
            f"draft_rank_in_target={analysis['draft_rank_in_target']} "
            f"target_rank_in_draft={analysis['target_rank_in_draft']} "
            f"top{args.top_k}_overlap={analysis['topk_overlap_count']} "
            f"acceptance_ratio={analysis['acceptance_ratio']:.6f} "
            f"accepts_if_greedy={analysis['accepts_if_greedy']}"
        )
        print(f"draft_top{args.top_k}={analysis['draft_topk']}")
        print(f"target_top{args.top_k}={analysis['target_topk']}")
        print()

    total = max(len(questions), 1)
    print("Summary")
    print(f"same_argmax_rate={same_argmax_count / total:.3f}")
    print(f"avg_draft_rank_in_target={target_rank_sum / total:.2f}")
    print(f"avg_top{args.top_k}_overlap={overlap_sum / total:.2f}")
    print(f"avg_acceptance_ratio={acceptance_ratio_sum / total:.6f}")
    print(f"greedy_accept_rate={greedy_accept_count / total:.3f}")


if __name__ == "__main__":
    main()
