import argparse

from transformers import AutoTokenizer


DEFAULT_MODEL = "google/gemma-2-27b-it"
DEFAULT_THRESHOLDS = (32000, 64000, 128000, 256000)
DEFAULT_PROMPTS = [
    "Hello world",
    "You are a helpful assistant. Explain speculative decoding.",
    "中文测试：请解释 speculative decoding 的作用。",
    "<bos><start_of_turn>user\nHi<end_of_turn>\n<start_of_turn>model\n",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Simulate vocab-size guard failures for Gemma prompts."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=int,
        default=list(DEFAULT_THRESHOLDS),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"model={args.model}")
    print(f"len(tokenizer)={len(tokenizer)} vocab_size={tokenizer.vocab_size}")
    print()

    for prompt in DEFAULT_PROMPTS:
        encoded = tokenizer.encode(prompt, add_special_tokens=False)
        max_id = max(encoded) if encoded else -1
        print(f"prompt={prompt!r}")
        print(f"encoded_len={len(encoded)} max_id={max_id}")
        for threshold in args.thresholds:
            status = "PASS" if max_id < threshold else "FAIL"
            print(f"  runtime_vocab={threshold}: {status}")
        print()


if __name__ == "__main__":
    main()
