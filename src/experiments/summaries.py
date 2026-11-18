"""Experiment summaries: durable writes, semantic identity, resume and merge."""

import json
from enum import Enum
from itertools import chain
from pathlib import Path
from tempfile import NamedTemporaryFile

from .config import ExpConfig

# An experiment name contains a timestamp; the allocated GPU can also change on
# retry. Neither belongs in the historical resume/merge key. Older summaries
# lack the accounting fields and used these CLI defaults.
_ACCOUNTING_DEFAULTS = {
    "transfer_top_k": 300,
    "charge_residual_payload": True,
    "comm_round_trip_mode": "per_round",
    "transfer_top_k_cap": 0,
    "prob_payload_bits": 16,
}
RUN_IDENTITY_FIELDS = (
    "eval_dataset",
    "eval_mode",
    "little_model",
    "draft_model",
    "target_model",
    "edge_cloud_bandwidth",
    "min_bandwidth_mbps",
    "comm_bw_model",
    "edge_end_bandwidth",
    "cloud_end_bandwidth",
    "batch_delay",
    "gamma",
    "gamma1",
    "gamma2",
    "max_tokens",
    "num_shots",
    "eval_data_num",
    "num_samples_per_task",
    "sample_seed",
    "random_sample",
    *_ACCOUNTING_DEFAULTS,
)


def _identity_value(value):
    if isinstance(value, Enum):
        value = value.value
    return round(value, 9) if isinstance(value, float) else value


def run_identity(config: dict) -> tuple:
    """Historical result key, not the model compatibility key used for scheduling."""
    return tuple(
        (field, _identity_value(config.get(field, _ACCOUNTING_DEFAULTS.get(field))))
        for field in RUN_IDENTITY_FIELDS
        if field in config or field in _ACCOUNTING_DEFAULTS
    )


def load_summary(path: str | Path) -> list[dict]:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"找不到汇总文件: {source}")
    with source.open(encoding="utf-8") as stream:
        entries = json.load(stream)
    if not isinstance(entries, list):
        raise ValueError(f"汇总文件顶层不是 list: {source}")
    return entries


def save_summary(path: str | Path, entries: list[dict]) -> None:
    """Publish a complete JSON list; a failed write leaves the last snapshot intact."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    stream = NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=output.parent, suffix=".tmp", delete=False
    )
    temporary = Path(stream.name)
    try:
        with stream:
            json.dump(entries, stream, ensure_ascii=False, indent=2)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def _sort_key(entry: dict) -> tuple:
    config = entry.get("config") or {}
    return (
        str(config.get("eval_dataset", "")),
        str(config.get("eval_mode", "")),
        str(config.get("target_model", "")),
        str(entry.get("exp_name", "")),
    )


def merge_summaries(*summaries: list[dict]) -> list[dict]:
    """Last result wins, even if it failed; retain legacy entries without a config."""
    merged = {}
    for entry in chain.from_iterable(summaries):
        if config := entry.get("config"):
            key = run_identity(config)
        else:
            key = ("__exp_name__", str(entry.get("exp_name", "")))
        merged[key] = entry
    return sorted(merged.values(), key=_sort_key)


def successful_runs(entries: list[dict]) -> set[tuple]:
    return {
        run_identity(entry["config"])
        for entry in entries
        if entry.get("status") == "success" and entry.get("config")
    }


def filter_configs_for_resume(
    configs: list[ExpConfig], resume_from: str | Path | None = None
) -> list[ExpConfig]:
    """Skip recorded successes only; old logs are not a completion signal."""
    if not resume_from:
        return list(configs)
    done = successful_runs(load_summary(resume_from))
    return [config for config in configs if run_identity(config) not in done]
