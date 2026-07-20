import argparse
import json
import os
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
        description="Inspect Gemma 27B numerics with 3-GPU sharding."
    )
    parser.add_argument("--model", default="google/gemma-2-27b-it")
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument(
        "--max-memory",
        nargs="*",
        default=["0:40GiB", "1:40GiB", "2:40GiB"],
        help="Per-device max memory entries like 0:40GiB 1:40GiB 2:40GiB",
    )
    parser.add_argument("--generate-one", action="store_true")
    return parser


def parse_max_memory(entries: list[str]) -> dict[int, str]:
    result: dict[int, str] = {}
    for entry in entries:
        device, limit = entry.split(":", 1)
        result[int(device)] = limit
    return result


def load_question(data_path: str, sample_index: int) -> dict:
    with Path(data_path).open() as handle:
        for idx, line in enumerate(handle):
            if idx == sample_index:
                return json.loads(line)
    raise IndexError(f"sample_index {sample_index} out of range for {data_path}")


def build_prompt(tokenizer, user_prompt: str) -> str:
    messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n" + user_prompt}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def summarize_topk(tokenizer, logits: torch.Tensor, probs: torch.Tensor, top_k: int):
    values, indices = torch.topk(probs, min(top_k, probs.shape[-1]))
    rows = []
    for prob, token_id in zip(values.tolist(), indices.tolist()):
        rows.append(
            {
                "id": int(token_id),
                "token": tokenizer.decode([token_id]).replace("\n", "\\n"),
                "prob": float(prob),
                "logit": float(logits[token_id].item()),
            }
        )
    return rows


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


def main() -> None:
    args = build_parser().parse_args()
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"max_memory={args.max_memory}")
    max_memory = parse_max_memory(args.max_memory)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        device_map="auto",
        max_memory=max_memory,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()

    print(f"hf_device_map={getattr(model, 'hf_device_map', None)}")

    sample = load_question(args.data, args.sample_index)
    prompt = build_prompt(tokenizer, sample["turns"][0])
    encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_device = first_cuda_device(model)
    input_ids = encoded.input_ids.to(input_device)

    print(f"question_id={sample['question_id']} category={sample['category']}")
    print(f"prompt_len={input_ids.shape[1]} input_device={input_device}")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits[0, -1].detach().float().cpu()
    probs = torch.softmax(logits, dim=-1)
    finite_mask = torch.isfinite(logits)
    finite_logits = logits[finite_mask]

    print(
        f"logit_nan={int(torch.isnan(logits).sum().item())} "
        f"logit_posinf={int(torch.isposinf(logits).sum().item())} "
        f"logit_neginf={int(torch.isneginf(logits).sum().item())}"
    )
    if finite_logits.numel() > 0:
        print(
            f"logit_stats=min={float(finite_logits.min().item())} "
            f"max={float(finite_logits.max().item())} "
            f"mean={float(finite_logits.mean().item())}"
        )
    else:
        print("logit_stats=no_finite_values")
    print(f"prob_nan={int(torch.isnan(probs).sum().item())}")
    print(f"top{args.top_k}={summarize_topk(tokenizer, logits, probs, args.top_k)}")

    if args.generate_one:
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                max_new_tokens=1,
                do_sample=False,
                use_cache=True,
            )
        new_token = generated[0, input_ids.shape[1] :].detach().cpu().tolist()
        print(f"generated_one={new_token} decoded={tokenizer.decode(new_token)!r}")


if __name__ == "__main__":
    main()
