from argparse import Namespace

from eval.utils import ExpPrint
from src.metrics import get_empty_metrics


def test_exp_print_preserves_communication_components() -> None:
    # Given: a decoded result with separated communication components
    metrics = get_empty_metrics()
    metrics["communication_time"] = 1.25
    metrics["communication_serialization_time"] = 0.75
    metrics["communication_fixed_latency_time"] = 0.5
    printer = ExpPrint(Namespace(dump_network_stats=False))

    # When: the evaluator prepares metrics for persistence
    saved = printer.get_filtered_dict(metrics)

    # Then: the decomposition survives the serialization whitelist
    assert saved["communication_serialization_time"] == 0.75
    assert saved["communication_fixed_latency_time"] == 0.5
