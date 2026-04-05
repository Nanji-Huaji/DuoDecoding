from typing import Protocol, cast
import importlib.util
import os
from .metrics import DecodingMetrics, get_empty_metrics, INT_SIZE


class ArgsLike(Protocol):
    exp_name: str
    eval_dataset: str
    little_model: str
    draft_model: str
    target_model: str
    eval_mode: str
    gamma: int | None
    gamma1: int | None
    gamma2: int | None
    max_tokens: int
    use_early_stopping: bool
    dump_network_stats: bool


class MetricsDumpLike(Protocol):
    def get_filtered_dict(self, metrics: DecodingMetrics) -> dict: ...
    def dump_metrics(self, metrics: DecodingMetrics) -> str: ...
    def get_printable_metrics(
        self, metrics: DecodingMetrics
    ) -> str: ...  # Optional, for more flexible printing
    def get_save_dict(
        self, metrics: DecodingMetrics
    ) -> dict: ...  # Optional, for saving to file


class MetricsDumpFactoryLike(Protocol):
    def __call__(self, args: ArgsLike) -> MetricsDumpLike: ...


def _load_default_metrics_dumper_factory() -> MetricsDumpFactoryLike:
    eval_utils_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "eval", "utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        "duodecoding_eval_utils", eval_utils_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load metrics dumper from {eval_utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(MetricsDumpFactoryLike, module.ExpPrint)
