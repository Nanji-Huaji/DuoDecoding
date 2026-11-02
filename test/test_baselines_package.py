"""S2 import compatibility, canonical globals and eager registration contracts."""

import importlib
import inspect
import json
import pickle
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import src.baselines as baselines
import src.baselines.simulators as simulators
from src.baselines import Baselines, _make_comm_simulator, _uplink_prob_payload, core
from src.register import Register


EXPORT_CONTRACT = json.loads(
    (Path(__file__).parent / "golden" / "baselines_exports.json").read_text(
        encoding="utf-8"
    )
)

# Frozen from the pre-S2 WIP, including engine registrations and historical aliases.
EXPECTED_REGISTRATIONS = {
    "adaptive_decoding": "Baselines.adaptive_decoding",
    "adaptive_tridecoding": "Baselines.adaptive_tridecoding",
    "cee_cuhlm": "Baselines.cee_cuhlm",
    "cee_dsd": "Baselines.cee_dsd",
    "cee_dssd": "Baselines.cee_dssd",
    "cee_sd": "Baselines.adaptive_tridecoding",
    "cee_sd_no_arp": "Baselines.cee_sd_no_arp",
    "cee_sd_opportunistic": "Baselines.cee_sd_opportunistic",
    "ceesd_w/o_arp": "Baselines.ceesd_without_arp",
    "ceesd_without_arp": "Baselines.ceesd_without_arp",
    "cuhlm": "Baselines.uncertainty_decoding",
    "dist_spec": "Baselines.dist_spec",
    "dist_split_spec": "Baselines.dist_split_spec",
    "dsd": "Baselines.dist_spec",
    "dssd": "Baselines.dist_split_spec",
    "large": "Decoding.autoregressive_sampling",
    "sd": "Decoding.speculative_decoding",
    "small": "Decoding.autoregressive_sampling",
    "speculative_decoding_with_bandwidth": (
        "Decoding.speculative_decoding_with_bandwidth"
    ),
    "target_only": "Baselines.target_only",
    "tk_slt": "Baselines.tk_slt",
    "tkslt": "Baselines.tk_slt",
    "tridecoding": "Baselines.tridecoding",
    "uncertainty_decoding": "Baselines.uncertainty_decoding",
}


def test_public_entry_and_core_share_class_and_historical_helpers():
    assert Baselines is core.Baselines
    assert _make_comm_simulator is core._make_comm_simulator
    assert _uplink_prob_payload is core._uplink_prob_payload
    for name in EXPORT_CONTRACT["exports"]:
        assert getattr(baselines, name) is getattr(core, name)
        assert name in dir(baselines)


def test_star_import_preserves_implementation_public_names():
    namespace = {}
    exec("from src.baselines import *", namespace)
    expected = set(EXPORT_CONTRACT["star_exports"])
    assert set(baselines.__all__) == expected
    assert set(namespace) - {"__builtins__"} == expected
    for name in expected:
        assert namespace[name] is getattr(core, name)


def test_package_metadata_and_missing_names_do_not_leak_core_metadata():
    assert Path(baselines.__file__).name == "__init__.py"
    assert Path(core.__file__).name == "core.py"
    assert Baselines.__module__ == "src.baselines.core"
    assert core._make_comm_simulator is simulators._make_comm_simulator
    assert simulators._make_comm_simulator.__globals__ is vars(simulators)
    assert core._make_comm_simulator.__module__ == "src.baselines.simulators"
    assert inspect.unwrap(Baselines.adaptive_tridecoding).__globals__ is vars(core)
    with pytest.raises(AttributeError, match="missing_s2_symbol"):
        getattr(baselines, "missing_s2_symbol")
    with pytest.raises(AttributeError, match="__missing_s2_metadata__"):
        getattr(baselines, "__missing_s2_metadata__")


def test_core_patch_reaches_function_imported_through_compatibility_entry():
    probs = torch.tensor([[0.25, 0.75]])
    quantized = torch.tensor([[0.5, 0.5]])
    with patch(
        "src.baselines.core._quantize_probs_logspace", return_value=quantized
    ) as quantize:
        payload, bits = _uplink_prob_payload(probs, 8)
        quantize.assert_called_once_with(probs, 8)
        assert baselines._quantize_probs_logspace is quantize
    assert payload is quantized
    assert bits == 8


def test_reloading_entry_does_not_repeat_decoding_registration():
    before = dict(Register._DECODING_REGISTRY)
    assert importlib.reload(baselines).Baselines is Baselines
    assert Register._DECODING_REGISTRY == before


def test_legacy_pickle_global_names_still_resolve():
    # Protocol-0 globals model existing references to the old module location.
    assert pickle.loads(b"csrc.baselines\nBaselines\n.") is Baselines
    assert pickle.loads(b"csrc.baselines\n_make_comm_simulator\n.") is _make_comm_simulator
    assert pickle.loads(pickle.dumps(Baselines)) is Baselines


@pytest.mark.parametrize(
    "order",
    [
        ["src.baselines", "src.baselines.core"],
        ["src.baselines.core", "src.baselines"],
        ["src.register", "src.engine", "src.baselines"],
        ["src.baselines.simulators", "src.baselines"],
        ["src.baselines.core", "src.baselines.simulators", "src.baselines"],
        ["src.register", "src.baselines.simulators", "src.baselines.core"],
    ],
)
def test_cold_import_orders_preserve_registry_and_alias_identity(order):
    code = """
import importlib, json, sys
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
from src.baselines import Baselines, core, simulators, _make_comm_simulator
from src.register import Register
assert Baselines is core.Baselines
assert _make_comm_simulator is core._make_comm_simulator
assert _make_comm_simulator is simulators._make_comm_simulator
registry = Register._DECODING_REGISTRY
for left, right in (
    ('cee_sd', 'adaptive_tridecoding'),
    ('dsd', 'dist_spec'),
    ('dssd', 'dist_split_spec'),
    ('tkslt', 'tk_slt'),
    ('ceesd_w/o_arp', 'ceesd_without_arp'),
):
    assert registry[left] is registry[right]
print(json.dumps({name: fn.__qualname__ for name, fn in registry.items()}))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, json.dumps(order)],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == EXPECTED_REGISTRATIONS
