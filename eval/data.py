"""Load the actual evaluation split, or fail before a run can report metrics."""

import json
from pathlib import Path

from eval.utils import select_eval_data

_HF_SPLITS = {
    "gsm8k": ("gsm8k", "main"),
    "humaneval": ("openai_humaneval",),
    "cnndm": ("cnn_dailymail", "3.0.0"),
    "xsum": ("xsum",),
}


def load_task_data(name, args):
    """Return selected rows. A mixed run uses the same loader for each task."""
    if name in ("mt_bench", "specbench"):
        stem = args.sub_domain if name == "specbench" else "mt_bench"
        source = Path(args.data_path) / f"{stem}.jsonl"
        with source.open(encoding="utf-8") as f:
            rows = []
            for number, line in enumerate(f, start=1):
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"{name}: invalid JSON in {source}:{number}"
                    ) from exc
        selected = select_eval_data(rows, args)
    else:
        from datasets import load_dataset

        source = _HF_SPLITS[name]
        try:
            dataset = load_dataset(*source, split="test")
            # Select indices first: a one-question run needn't materialize an
            # entire CNN/DailyMail split (nor change the old sampling order).
            indices = select_eval_data(range(len(dataset)), args)
            selected = [dict(dataset[i]) for i in indices]
        except Exception as exc:
            raise RuntimeError(
                f"{name}: could not load test data from {source}: {exc}"
            ) from exc

    if not selected:
        raise ValueError(f"{name}: no evaluation samples selected from {source}")
    return selected
