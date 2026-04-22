import argparse
import json
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze Gemma draft-target agreement with sharded non-4bit target."
    )
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--draft-device", default="cuda:2")
    parser.add_argument(
        "--target-max-memory",
        nargs="*",
        default=["0:36GiB", "1:36GiB", "2:36GiB"],
        help="Per-device target max memory entries like 0:36GiB 1:36GiB 2:36GiB",
    )
    parser.add_argument("--top-k", type=int, default=10)
    return parser


def parse_max_memory(entries: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for entry in entries:
        device, limit = entry.split(":", 1)
        result[int(device)] = limit
    return result


def load_questions(data_path: str, limit: int) -> list[dict]:
    items = []
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


def first_cuda_device(model) -> torch.device:
    if hasattr(model, "hf_device_map"):
        cuda_devices = sorted(
            {
                str(device)
                for device in model.hf_device_map.values()
                if str(device).startswith("cuda") or str(device).isdigit()
            }
        )
        if cuda_devices:
            first = cuda_devices[0]
            return torch.device(first if first.startswith("cuda") else f"cuda:{first}")
    return next(model.parameters()).device


def collect_last_token_probs(tokenizer, model, prompts: list[str], device: torch.device) -> list[torch.Tensor]:
    outputs: list[torch.Tensor] = []
    with torch.inference_mode():
        for prompt in prompts:
            encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
            input_ids = encoded.input_ids.to(device)
            logits = model(input_ids=input_ids, use_cache=False).logits[0, -1].detach().float().cpu()
            outputs.append(torch.softmax(logits, dim=-1))
    return outputs


def summarize_topk(tokenizer, probs: torch.Tensor, top_k: int) -> list[tuple[int, str, float]]:
    values, indices = torch.topk(probs, min(top_k, probs.shape[-1]))
    rows = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        rows.append(
            (int(token_id), tokenizer.decode([token_id]).replace("\n", "\\n"), float(prob))
        )
    return rows


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
    questions = load_questions(args.data, args.samples)
    prompts = [build_prompt(tokenizer, item["turns"][0]) for item in questions]

    print("Loading draft model...")
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        trust_remote_code=True,
        device_map=args.draft_device,
        dtype=torch.float16,
    ).eval()
    draft_probs = collect_last_token_probs(
        tokenizer, draft_model, prompts, torch.device(args.draft_device)
    )
    del draft_model
    torch.cuda.empty_cache()

    print("Loading sharded target model...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        trust_remote_code=True,
        device_map="auto",
        max_memory=parse_max_memory(args.target_max_memory),
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    target_device = first_cuda_device(target_model)
    print(f"target hf_device_map={getattr(target_model, 'hf_device_map', None)}")
    target_probs = collect_last_token_probs(tokenizer, target_model, prompts, target_device)

    same_argmax_count = 0
    overlap_sum = 0
    acceptance_ratio_sum = 0.0
    greedy_accept_count = 0

    for idx, (item, draft_p, target_p) in enumerate(zip(questions, draft_probs, target_probs)):
        draft_argmax = int(draft_p.argmax().item())
        target_argmax = int(target_p.argmax().item())
        same_argmax = draft_argmax == target_argmax
        same_argmax_count += int(same_argmax)
        draft_rank_in_target = int((target_p > target_p[draft_argmax]).sum().item()) + 1
        target_rank_in_draft = int((draft_p > draft_p[target_argmax]).sum().item()) + 1
        overlap_topk = set(torch.topk(draft_p, args.top_k).indices.tolist()) & set(
            torch.topk(target_p, args.top_k).indices.tolist()
        )
        overlap_sum += len(overlap_topk)
        acceptance_ratio = (
            float(target_p[draft_argmax].item()) / float(draft_p[draft_argmax].item())
            if float(draft_p[draft_argmax].item()) > 0
            else 0.0
        )
        acceptance_ratio_sum += acceptance_ratio
        greedy_accept = acceptance_ratio >= 1.0
        greedy_accept_count += int(greedy_accept)

        print(f"sample={idx} question_id={item['question_id']} category={item['category']}")
        print(
            f"same_argmax={same_argmax} draft_rank_in_target={draft_rank_in_target} "
            f"target_rank_in_draft={target_rank_in_draft} top{args.top_k}_overlap={len(overlap_topk)} "
            f"acceptance_ratio={acceptance_ratio:.6f} greedy_accept={greedy_accept}"
        )
        print(f"draft_top{args.top_k}={summarize_topk(tokenizer, draft_p, args.top_k)}")
        print(f"target_top{args.top_k}={summarize_topk(tokenizer, target_p, args.top_k)}")
        print()

    total = max(len(questions), 1)
    print("Summary")
    print(f"same_argmax_rate={same_argmax_count / total:.3f}")
    print(f"avg_top{args.top_k}_overlap={overlap_sum / total:.2f}")
    print(f"avg_acceptance_ratio={acceptance_ratio_sum / total:.6f}")
    print(f"greedy_accept_rate={greedy_accept_count / total:.3f}")


if __name__ == "__main__":
    main()
