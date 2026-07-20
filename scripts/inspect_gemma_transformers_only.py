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
        description="Inspect Gemma numerics using raw transformers only."
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--quantization", choices=["none", "4bit"], default="none")
    parser.add_argument("--prompt")
    parser.add_argument("--data", default="data/mt_bench.jsonl")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--generate-one", action="store_true")
    return parser


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


def load_model(model_name: str, device: str, quantization: str):
    kwargs: dict[str, object] = {
        "trust_remote_code": True,
        "device_map": device,
        "dtype": torch.float16,
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


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = load_model(args.model, args.device, args.quantization)

    if args.prompt:
        user_prompt = args.prompt
        question_id = "manual"
    else:
        sample = load_question(args.data, args.sample_index)
        user_prompt = sample["turns"][0]
        question_id = sample["question_id"]

    prompt = build_prompt(tokenizer, user_prompt)
    encoded = tokenizer([prompt], add_special_tokens=False, return_tensors="pt")
    input_ids = encoded.input_ids.to(args.device)

    print(f"model={args.model}")
    print(f"quantization={args.quantization}")
    print(f"question_id={question_id}")
    print(f"prompt_len={input_ids.shape[1]}")

    with torch.inference_mode():
        outputs = model(input_ids=input_ids, use_cache=False)

    logits = outputs.logits[0, -1].detach().float().cpu()
    probs = torch.softmax(logits, dim=-1)

    nan_count = int(torch.isnan(logits).sum().item())
    posinf_count = int(torch.isposinf(logits).sum().item())
    neginf_count = int(torch.isneginf(logits).sum().item())
    finite_mask = torch.isfinite(logits)
    finite_logits = logits[finite_mask]

    print(f"logit_nan={nan_count} logit_posinf={posinf_count} logit_neginf={neginf_count}")
    if finite_logits.numel() > 0:
        print(
            "logit_stats="
            f"min={float(finite_logits.min().item())} "
            f"max={float(finite_logits.max().item())} "
            f"mean={float(finite_logits.mean().item())} "
            f"std={float(finite_logits.std().item()) if finite_logits.numel() > 1 else 0.0}"
        )
    else:
        print("logit_stats=no_finite_values")

    prob_nan_count = int(torch.isnan(probs).sum().item())
    prob_finite = probs[torch.isfinite(probs)]
    print(f"prob_nan={prob_nan_count}")
    if prob_finite.numel() > 0:
        print(f"prob_finite_sum={float(prob_finite.sum().item())}")
    else:
        print("prob_finite_sum=None")

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
