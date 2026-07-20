import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


DEFAULT_MODEL = "google/gemma-2-27b-it"
DEFAULT_DATA = "data/mt_bench.jsonl"
SYSTEM_PROMPT = (
    "You are a helpful, respectful and honest assistant. Always answer as helpfully "
    "as possible, while being safe. Your answers should not include any harmful, "
    "unethical, racist, sexist, toxic, dangerous, or illegal content."
)

EDGE_CASES = [
    "Hello world",
    "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
    "中文测试：请解释 speculative decoding。",
    "Emoji test: 😀🔥🚀",
    "Rare unicode: math aleph \u2135 and music \u266f",
    "Whitespace test:\n\n\tindented line\ntrailing spaces   ",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect Gemma prompt edge cases and token id ranges."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--samples", type=int, default=5)
    return parser


def summarize_encoding(tokenizer, label: str, text: str) -> None:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    max_id = max(encoded) if encoded else None
    min_id = min(encoded) if encoded else None
    over_32k = sum(token >= 32000 for token in encoded)
    over_128k = sum(token >= 128000 for token in encoded)
    print(f"[{label}] len={len(encoded)} min_id={min_id} max_id={max_id}")
    print(f"[{label}] tokens>=32000: {over_32k}, tokens>=128000: {over_128k}")
    print(f"[{label}] first_ids={encoded[:24]}")
    print()


def iter_mt_bench_prompts(tokenizer, data_path: Path, sample_count: int):
    with data_path.open() as handle:
        for idx, line in enumerate(handle):
            if idx >= sample_count:
                break
            datum = json.loads(line)
            user_prompt = datum["turns"][0]
            messages = [
                {
                    "role": "user",
                    "content": f"{SYSTEM_PROMPT}\n{user_prompt}",
                }
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            yield idx, datum.get("question_id", idx), prompt


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"model={args.model}")
    print(f"len(tokenizer)={len(tokenizer)} vocab_size={tokenizer.vocab_size}")
    print()

    for idx, text in enumerate(EDGE_CASES, start=1):
        summarize_encoding(tokenizer, f"edge-{idx}", text)

    data_path = Path(args.data)
    for idx, question_id, prompt in iter_mt_bench_prompts(
        tokenizer, data_path, args.samples
    ):
        summarize_encoding(tokenizer, f"mt-bench-{idx}-qid-{question_id}", prompt)


if __name__ == "__main__":
    main()
