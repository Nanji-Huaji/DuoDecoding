"""S2 import compatibility, canonical globals and eager registration contracts."""

import ast
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
import src.baselines.adaptive as adaptive
import src.baselines.cee as cee
import src.baselines.cuhlm as cuhlm
import src.baselines.dual as dual
import src.dual_decoding as local_dual
import src.baselines.operations as operations
import src.baselines.runtime as runtime
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
    "adaptive_decoding": "AdaptiveDecodingMixin.adaptive_decoding",
    "adaptive_tridecoding": "CeeSdMixin.adaptive_tridecoding",
    "cee_cuhlm": "Baselines.cee_cuhlm",
    "cee_dsd": "Baselines.cee_dsd",
    "cee_dssd": "Baselines.cee_dssd",
    "cee_sd": "CeeSdMixin.adaptive_tridecoding",
    "cee_sd_no_arp": "CeeSdMixin.cee_sd_no_arp",
    "cee_sd_opportunistic": "CeeSdMixin.cee_sd_opportunistic",
    "ceesd_w/o_arp": "Baselines.ceesd_without_arp",
    "ceesd_without_arp": "Baselines.ceesd_without_arp",
    "cuhlm": "CuhlmMixin.uncertainty_decoding",
    "dist_spec": "DualDecodingMixin.dist_spec",
    "dist_split_spec": "DualDecodingMixin.dist_split_spec",
    "dsd": "DualDecodingMixin.dist_spec",
    "dssd": "DualDecodingMixin.dist_split_spec",
    "large": "Decoding.autoregressive_sampling",
    "sd": "SpeculativeDecodingMixin.speculative_decoding",
    "small": "Decoding.autoregressive_sampling",
    "speculative_decoding_with_bandwidth": (
        "Decoding.speculative_decoding_with_bandwidth"
    ),
    "target_only": "Baselines.target_only",
    "tk_slt": "Baselines.tk_slt",
    "tkslt": "Baselines.tk_slt",
    "tridecoding": "CeeSdMixin.tridecoding",
    "uncertainty_decoding": "CuhlmMixin.uncertainty_decoding",
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
    assert inspect.unwrap(Baselines.adaptive_tridecoding).__globals__ is vars(cee)
    with pytest.raises(AttributeError, match="missing_s2_symbol"):
        getattr(baselines, "missing_s2_symbol")
    with pytest.raises(AttributeError, match="__missing_s2_metadata__"):
        getattr(baselines, "__missing_s2_metadata__")


def test_operations_patch_reaches_function_imported_through_compatibility_entry():
    probs = torch.tensor([[0.25, 0.75]])
    quantized = torch.tensor([[0.5, 0.5]])
    with patch(
        "src.baselines.operations._quantize_probs_logspace", return_value=quantized
    ) as quantize:
        payload, bits = _uplink_prob_payload(probs, 8)
        quantize.assert_called_once_with(probs, 8)
        assert operations._quantize_probs_logspace is quantize
        assert baselines._quantize_probs_logspace is core._quantize_probs_logspace
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
        ["src.baselines.cee", "src.baselines"],
        ["src.baselines.cuhlm", "src.baselines"],
        ["src.dual_decoding", "src.engine", "src.baselines.dual"],
        ["src.baselines.dual", "src.baselines", "src.engine"],
        ["src.baselines.runtime", "src.baselines"],
        ["src.baselines.adaptive", "src.baselines.operations", "src.baselines"],
    ],
)
def test_cold_import_orders_preserve_registry_and_alias_identity(order):
    code = """
import importlib, json, sys
for name in json.loads(sys.argv[1]):
    importlib.import_module(name)
from src.baselines import (
    Baselines,
    adaptive,
    cee,
    core,
    cuhlm,
    operations,
    runtime,
    simulators,
    _make_comm_simulator,
)
from src.register import Register
assert Baselines is core.Baselines
assert _make_comm_simulator is core._make_comm_simulator
assert _make_comm_simulator is simulators._make_comm_simulator
assert Baselines.adaptive_decoding is adaptive.AdaptiveDecodingMixin.adaptive_decoding
assert Baselines.adaptive_tridecoding is cee.CeeSdMixin.adaptive_tridecoding
assert Baselines.cee_sd_no_arp is cee.CeeSdMixin.cee_sd_no_arp
assert Baselines.tridecoding is cee.CeeSdMixin.tridecoding
assert Baselines.cee_sd_opportunistic is cee.CeeSdMixin.cee_sd_opportunistic
assert Baselines.uncertainty_decoding is cuhlm.CuhlmMixin.uncertainty_decoding
assert Baselines.__init__ is runtime.RuntimeResources.__init__
assert Baselines._stop_at_eos is operations.SharedOperations._stop_at_eos
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


def test_moved_families_use_their_real_module_globals():
    """Method bodies must resolve dependencies in the module that defines them."""
    cases = (
        (Baselines.speculative_decoding, local_dual, "SpeculativeDecodingMixin"),
        (Baselines.dist_spec, dual, "DualDecodingMixin"),
        (Baselines.dist_split_spec, dual, "DualDecodingMixin"),
        (Baselines.adaptive_decoding, adaptive, "AdaptiveDecodingMixin"),
        (Baselines._adaptive_decoding_verify, adaptive, "AdaptiveDecodingMixin"),
        (Baselines.uncertainty_decoding, cuhlm, "CuhlmMixin"),
        (Baselines.adaptive_tridecoding, cee, "CeeSdMixin"),
        (Baselines._cee_verify_proposal, cee, "CeeSdMixin"),
        (Baselines.cee_sd_opportunistic, cee, "CeeSdMixin"),
        (Baselines.__init__, runtime, "RuntimeResources"),
        (Baselines._acquire_three_layer_caches, runtime, "RuntimeResources"),
        (Baselines.load_acc_head, runtime, "RuntimeResources"),
        (Baselines._stop_at_eos, operations, "SharedOperations"),
        (
            Baselines._generate_with_optional_rebuilt_proposal,
            operations,
            "SharedOperations",
        ),
    )
    for method, module, owner in cases:
        function = inspect.unwrap(method)
        assert function.__globals__ is vars(module), (method, module.__name__)
        assert function.__qualname__.startswith(f"{owner}."), function.__qualname__

    assert runtime._build_cache.__globals__ is vars(runtime)
    assert operations._uplink_prob_payload.__globals__ is vars(operations)
    assert core._build_cache is runtime._build_cache
    assert core._uplink_prob_payload is operations._uplink_prob_payload


def test_real_modules_do_not_import_the_compatibility_package_or_core():
    """Extracted families depend on explicit modules, never back on core."""
    source_files = (
        Path(__file__).resolve().parents[1] / "src" / "baselines" / "runtime.py",
        Path(__file__).resolve().parents[1] / "src" / "baselines" / "adaptive.py",
        Path(__file__).resolve().parents[1] / "src" / "baselines" / "cee.py",
        Path(__file__).resolve().parents[1] / "src" / "baselines" / "cuhlm.py",
        Path(__file__).resolve().parents[1] / "src" / "baselines" / "operations.py",
        Path(__file__).resolve().parents[1] / "src" / "baselines" / "dual.py",
        Path(__file__).resolve().parents[1] / "src" / "dual_decoding.py",
    )
    forbidden = {
        "src.baselines",
        "src.baselines.core",
        ".core",
        "..core",
        "..baselines",
    }
    for path in source_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert module not in forbidden, (path.name, module)
                if module in ("", ".", ".."):
                    assert not (
                        {alias.name for alias in node.names} & {"core", "baselines"}
                    ), (path.name, module, [alias.name for alias in node.names])
            elif isinstance(node, ast.Import):
                assert all(
                    alias.name not in forbidden for alias in node.names
                ), (path.name, [alias.name for alias in node.names])
            elif isinstance(node, ast.Call):
                func = node.func
                dynamic = (
                    isinstance(func, ast.Name)
                    and func.id in ("__import__", "import_module")
                ) or (
                    isinstance(func, ast.Attribute)
                    and func.attr == "import_module"
                )
                if dynamic and node.args:
                    target = node.args[0]
                    if isinstance(target, ast.Constant) and isinstance(
                        target.value, str
                    ):
                        assert "src.baselines" not in target.value, (
                            path.name,
                            target.value,
                        )
                        assert not target.value.endswith(
                            (".core", "baselines.core")
                        ), (path.name, target.value)


def test_each_family_method_is_defined_in_exactly_one_real_module():
    """No copy of a moved body may remain in core or appear twice."""
    roots = [
        Path(__file__).resolve().parents[1] / "src" / "baselines" / name
        for name in (
            "core.py",
            "runtime.py",
            "adaptive.py",
            "cee.py",
            "cuhlm.py",
            "operations.py",
            "dual.py",
        )
    ] + [
        Path(__file__).resolve().parents[1] / "src" / "engine.py",
        Path(__file__).resolve().parents[1] / "src" / "dual_decoding.py",
    ]
    definitions: dict[str, list[str]] = {}
    for path in roots:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                definitions.setdefault(node.name, []).append(path.name)
    moved = (
        "speculative_decoding",
        "dist_spec",
        "dist_split_spec",
        "adaptive_decoding",
        "_adaptive_decoding_propose",
        "_adaptive_decoding_verify",
        "_adaptive_decoding_next_token",
        "_trace_adaptive_decoding_append",
        "_adaptive_decoding_finalize",
        "uncertainty_decoding",
        "adaptive_tridecoding",
        "_cee_propose",
        "_cee_first_stage",
        "_cee_target_proposal",
        "_cee_verify_proposal",
        "_cee_finalize",
        "_adaptive_rejection_prob_bytes",
        "_select_adaptive_little_config",
        "_select_adaptive_main_config",
        "_trace_adaptive_cache_preverify",
        "_trace_adaptive_calibration",
        "tridecoding",
        "cee_sd_no_arp",
        "cee_sd_opportunistic",
        "_acquire_three_layer_caches",
        "_acquire_draft_target_caches",
        "build_adaptive_tridecoding_caches",
        "load_acc_head",
        "_configure_ntt_trace",
        "_build_rl_adapters",
        "_generate_with_optional_rebuilt_proposal",
        "_residual_payload_bytes",
        "_eos_token_id",
        "_stop_at_eos",
        "_build_cache",
        "_uplink_prob_payload",
    )
    for name in moved:
        assert len(definitions.get(name, [])) == 1, (name, definitions.get(name))
