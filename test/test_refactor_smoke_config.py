"""``tools/run_refactor_smoke.py`` 的 CPU 侧安全测试（不 import torch、不加载模型）。

守的是"driver 自己写错"这一类失败：参数 dest 拼错、case 表被改、资产快照
没校验、NaN 混进 JSON、覆盖已有输出、把别的树的 src 抢进 import。
真实采样（GPU1 + 真实快照）不在本文件覆盖范围内。
"""

from __future__ import annotations

import ast
import builtins
import hashlib
import importlib.util
import json
import os
import sys
import types
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_driver():
    spec = importlib.util.spec_from_file_location(
        "refactor_smoke_driver_under_test", ROOT / "tools" / "run_refactor_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


driver = _load_driver()


# ---------------------------------------------------------------------------
# prompt / case / 参数
# ---------------------------------------------------------------------------


def test_prompt_table():
    assert len(driver.PROMPTS) == driver.MAX_SAMPLES == 12
    assert len(set(driver.PROMPTS)) == 12
    for prompt in driver.PROMPTS:
        assert prompt.strip() and len(prompt) <= 200
        assert "####" not in prompt and "gsm8k" not in prompt.lower()


def test_case_table_frozen():
    legacy = driver.resolve_case("legacy_static")
    honest = driver.resolve_case("honest_int8")
    assert (
        legacy["temperature"],
        legacy["transfer_top_k"],
        legacy["prob_payload_bits"],
    ) == (
        0.0,
        32,
        16,
    )
    assert legacy["comm_round_trip_mode"] == "per_transfer"
    assert legacy["charge_residual_payload"] is False
    assert (
        honest["temperature"],
        honest["transfer_top_k"],
        honest["prob_payload_bits"],
    ) == (
        0.7,
        8,
        8,
    )
    assert honest["comm_round_trip_mode"] == "per_round"
    assert honest["charge_residual_payload"] is True


def test_configuration_only_case_knobs_differ():
    first = driver.build_configuration(
        "legacy_static", samples=12, max_tokens=48, seed=1, eos_ids=[1]
    )
    second = driver.build_configuration(
        "honest_int8", samples=12, max_tokens=48, seed=1, eos_ids=[1]
    )
    assert {key for key in first if first[key] != second.get(key)} == {
        "temp",
        "transfer_top_k",
        "prob_payload_bits",
        "comm_round_trip_mode",
        "charge_residual_payload",
    }
    for key in ("output", "repo_root", "asset_manifest", "exp_name", "little_model"):
        assert key not in first
    assert first["eval_mode"] == "tridecoding"
    assert (
        first["use_cuda_graph"],
        first["disable_rl_update"],
        first["use_rl_adapter"],
    ) == (
        False,
        True,
        False,
    )
    assert first["little_quantization"] == "none"
    assert first["model_dtype"] == "bf16"
    assert (first["gamma1"], first["gamma2"]) == (3, 2)
    assert (first["edge_cloud_bandwidth"], first["edge_end_bandwidth"]) == (20.0, 100.0)
    assert (first["ntt_ms_edge_cloud"], first["ntt_ms_edge_end"]) == (10.0, 1.0)
    assert first["batch_delay"] == 0.0 and first["comm_bw_model"] == "instant"
    assert first["disable_eos_stop"] is False
    assert first["eos_token_ids"] == [1]
    assert (first["samples"], first["seed"]) == (12, 1)


def test_namespace_matches_case():
    namespace = driver.build_namespace(
        driver.resolve_case("honest_int8"),
        models={"small": "/abs/small", "target": "/abs/target"},
        max_tokens=48,
        seed=20261103,
    )
    assert namespace.little_model == namespace.draft_model == "/abs/small"
    assert namespace.target_model == "/abs/target"
    assert namespace.eval_mode == "tridecoding"
    assert namespace.probe_cache_max_length is None
    assert (namespace.transfer_top_k, namespace.prob_payload_bits) == (8, 8)
    assert namespace.comm_round_trip_mode == "per_round"
    assert namespace.charge_residual_payload is True
    assert (namespace.use_cuda_graph, namespace.use_rl_adapter) == (False, False)
    assert namespace.disable_rl_update is True
    assert namespace.use_stochastic_comm is False
    assert namespace.disable_eos_stop is False
    assert namespace.comm_bw_model == "instant"
    assert (namespace.little_quantization, namespace.target_quantization) == (
        "none",
        "none",
    )
    for forbidden in ("output", "repo_root", "asset_manifest"):
        assert not hasattr(namespace, forbidden)


def _dests_from_utils(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    dests: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            not isinstance(func, ast.Attribute)
            or func.attr != "add_argument"
            or not node.args
        ):
            continue
        first = node.args[0]
        if not isinstance(first, ast.Constant) or not isinstance(first.value, str):
            continue
        dest = first.value[2:].replace("-", "_")
        for keyword in node.keywords:
            if keyword.arg == "dest" and isinstance(keyword.value, ast.Constant):
                dest = str(keyword.value.value)
        dests.add(dest)
    return dests


def test_namespace_keys_are_real_cli_dests():
    """Namespace 键必须是真实 argparse dest；拼错等于静默用默认值。"""
    namespace = driver.build_namespace(
        driver.resolve_case("legacy_static"),
        models={"small": "/abs/small", "target": "/abs/target"},
        max_tokens=48,
        seed=1,
    )
    unknown = sorted(
        (set(vars(namespace)) - {"eval_dataset", "probe_cache_max_length"})
        - _dests_from_utils(ROOT / "src" / "utils.py")
    )
    assert not unknown, f"这些键在 src/utils.py 里没有对应开关: {unknown}"


def test_driver_imports_no_heavy_frameworks():
    for name in ("torch", "transformers", "numpy"):
        assert name not in vars(driver)


# ---------------------------------------------------------------------------
# 词表指纹（与主代理 manifest 的算法逐字节一致）
# ---------------------------------------------------------------------------


class _StubTokenizer:
    def __init__(self, mapping):
        self._mapping = dict(mapping)

    def get_vocab(self):
        return dict(self._mapping)


def _reference_digest(mapping: dict[str, int]) -> str:
    return hashlib.sha256(
        json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_vocab_digest_matches_manifest_algorithm():
    mapping = {"b": 1, "a": 0, "c": 2}
    expected = _reference_digest(mapping)
    assert driver.tokenizer_vocab_digest(_StubTokenizer(mapping)) == expected
    assert (
        driver.tokenizer_vocab_digest(_StubTokenizer({"c": 2, "a": 0, "b": 1}))
        == expected
    )
    # 别名差异必须改变指纹（不反转、不去重）
    assert (
        driver.tokenizer_vocab_digest(_StubTokenizer({"b": 1, "a": 0, "d": 2}))
        != expected
    )
    assert len(expected) == 64


def test_vocab_digest_rejects_missing_or_empty():
    with pytest.raises(driver.SmokeError):
        driver.tokenizer_vocab_digest(object())
    with pytest.raises(driver.SmokeError):
        driver.tokenizer_vocab_digest(_StubTokenizer({}))


# ---------------------------------------------------------------------------
# 资产清单
# ---------------------------------------------------------------------------


def _make_model(root: Path, name: str) -> dict:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    blobs = {
        "config.json": b'{"vocab_size": 16, "model_type": "qwen2"}',
        "tokenizer.json": b'{"model": {}}',
        "model.safetensors": b"weights",
    }
    entries = []
    for filename, blob in blobs.items():
        (directory / filename).write_bytes(blob)
        entries.append(
            {
                "name": filename,
                "size": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
            }
        )
    return {"path": str(directory), "files": entries}


def _manifest(tmp_path: Path) -> dict:
    return {
        "schema_version": 1,
        "metadata": {"vocab_size": 151936, "model_type": "qwen3"},
        "models": {
            "small": _make_model(tmp_path, "small"),
            "target": _make_model(tmp_path, "target"),
        },
        "tokenizer_vocab_sha256": "a" * 64,
        "submodule_commit": "d340008fb03fd9a8e766ba83e932f0e5600d9d0d",
    }


def test_assets_accepts_valid_and_keeps_fields(tmp_path):
    payload = _manifest(tmp_path)
    validated = driver.validate_assets(payload)
    assert validated["metadata"] == {"vocab_size": 151936, "model_type": "qwen3"}
    assert validated["submodule_commit"] == payload["submodule_commit"]


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda p: p.update(schema_version=2), "schema_version"),
        (lambda p: p.update(schema_version=True), "schema_version"),
        (lambda p: p.pop("models"), "models"),
        (lambda p: p["models"].pop("target"), "target"),
        (lambda p: p["models"]["small"].update(path="relative"), "绝对路径"),
        (lambda p: p["models"]["small"]["files"][0].update(sha256="b" * 64), "sha256"),
        (lambda p: p["models"]["small"]["files"][0].update(size=999), "大小"),
        (
            lambda p: p["models"]["small"]["files"][0].update(name="../x.json"),
            "纯文件名",
        ),
    ],
)
def test_assets_rejects_invalid(tmp_path, mutate, message):
    payload = json.loads(json.dumps(_manifest(tmp_path)))
    mutate(payload)
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.validate_assets(payload)
    assert message in str(excinfo.value)


def test_load_assets_rejects_missing_and_broken(tmp_path):
    with pytest.raises(driver.SmokeError):
        driver.load_assets(tmp_path / "nope.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{not json", encoding="utf-8")
    with pytest.raises(driver.SmokeError):
        driver.load_assets(broken)


# ---------------------------------------------------------------------------
# GPU pid / 参数 / JSON / 输出
# ---------------------------------------------------------------------------


def test_parse_and_foreign_pids():
    assert driver.parse_compute_apps_csv("") == []
    assert driver.parse_compute_apps_csv("123, 2048\n456\n[N/A]\n") == [123, 456]
    assert driver.foreign_pids([1, 2], baseline=[1], allowed=[2]) == []
    assert driver.foreign_pids([1, 2, 3], baseline=[1], allowed=[2]) == [3]


def test_cuda_visible_devices_bounds_and_revision():
    assert driver.validate_cuda_visible_devices("1") == "1"
    for bad in ("0", "2", "0,1", "", " 1", None):
        with pytest.raises(driver.SmokeError):
            driver.validate_cuda_visible_devices(bad)
    driver.validate_bounds(1, 1)
    driver.validate_bounds(12, 256)
    for samples in (0, 13):
        with pytest.raises(driver.SmokeError):
            driver.validate_bounds(samples, 48)
    for max_tokens in (0, 257):
        with pytest.raises(driver.SmokeError):
            driver.validate_bounds(12, max_tokens)
    assert driver.normalize_revision("A" * 40) == "a" * 40
    for bad in ("50c43ba", "a" * 39, "main", None):
        with pytest.raises(driver.SmokeError):
            driver.normalize_revision(bad)
    with pytest.raises(driver.SmokeError):
        driver.resolve_case("nope")


def test_strict_json_rejects_non_finite():
    """非有限浮点必须失败，不能落成 null 冒充有效数据。"""
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.strict_json({"nan": float("nan")})
    assert "非有限" in str(excinfo.value)
    with pytest.raises(driver.SmokeError):
        driver.strict_json({"nested": [float("inf")]})
    with pytest.raises(driver.SmokeError):
        driver.strict_json(float("-inf"))


def test_strict_json_converts_numeric_likes():
    class Scalar:
        def item(self):
            return 7

    class Vector:
        def tolist(self):
            return [1, 2, 3]

    assert driver.strict_json({"s": Scalar(), "v": Vector(), "ok": 1.5, "b": True}) == {
        "s": 7,
        "v": [1, 2, 3],
        "ok": 1.5,
        "b": True,
    }
    with pytest.raises(driver.SmokeError):
        driver.strict_json({"bad": object()})


def _samples(count: int) -> list[dict]:
    return [
        {
            "sample_id": index,
            "prompt": f"prompt {index}",
            "prompt_sha256": driver.prompt_sha256(f"prompt {index}"),
            "seed": 100 + index,
            "prefix_ids": [1, 2],
            "gen_ids": [3],
            "metrics": {"generated_tokens": 1, "target_forward_times": 1},
        }
        for index in range(count)
    ]


def _output_kwargs(**overrides):
    base = {
        "case": "legacy_static",
        "revision": "a" * 40,
        "code": {
            "definition_module": "src.baselines",
            "definition_file": "src/baselines.py",
            "source_sha256": "b" * 64,
        },
        "configuration": driver.build_configuration(
            "legacy_static", samples=2, max_tokens=8, seed=100, eos_ids=[1]
        ),
        "runtime": {
            "python": "3.10.18",
            "torch": "2.6.0+cu124",
            "cuda": "12.4",
            "transformers": "4.57.6",
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "gpu_uuid": "GPU-0000",
            "attention_backend": "sdpa",
        },
        "assets": {"schema_version": 1, "models": {}},
        "samples": _samples(2),
        "expected_samples": 2,
    }
    base.update(overrides)
    return base


def test_build_output_order_and_roundtrip():
    payload = driver.build_output(**_output_kwargs())
    assert tuple(payload) == driver.OUTPUT_FIELDS
    assert payload["schema_version"] == 1
    assert (
        json.loads(json.dumps(payload, allow_nan=False))["samples"][1]["sample_id"] == 1
    )


def test_build_output_rejects_bad_count_and_non_finite_metrics():
    with pytest.raises(driver.SmokeError):
        driver.build_output(**_output_kwargs(samples=_samples(1)))
    with pytest.raises(driver.SmokeError):
        driver.build_output(**_output_kwargs(samples=[], expected_samples=0))
    broken = _samples(1)
    broken[0]["metrics"] = {"throughput": float("nan")}
    with pytest.raises(driver.SmokeError):
        driver.build_output(**_output_kwargs(samples=broken, expected_samples=1))


def test_write_output_exclusive(tmp_path):
    target = tmp_path / "out.json"
    driver.write_output_exclusive(target, {"a": 1})
    original = target.read_text(encoding="utf-8")
    with pytest.raises(driver.SmokeError):
        driver.write_output_exclusive(target, {"a": 2})
    assert target.read_text(encoding="utf-8") == original


def test_eos_token_ids_normalization():
    class Tok:
        def __init__(self, value):
            self.eos_token_id = value

    assert driver.eos_token_ids(Tok(151645)) == [151645]
    assert driver.eos_token_ids(Tok([1, 2])) == [1, 2]
    assert driver.eos_token_ids(Tok(None)) == []
    assert driver.eos_token_ids(object()) == []


def test_generation_completeness_rule():
    """短于 max_tokens 必须自带 EOS；允许中间位置；自然 EOS 单 token 合法。"""
    driver.check_generation_complete([7, 8, 9], max_tokens=3, eos_ids=[9], label="s")
    driver.check_generation_complete([9], max_tokens=8, eos_ids=[9], label="s")
    driver.check_generation_complete([9, 7, 8], max_tokens=8, eos_ids=[9], label="s")
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.check_generation_complete([7, 8], max_tokens=8, eos_ids=[9], label="s")
    assert "无 EOS" in str(excinfo.value)
    with pytest.raises(driver.SmokeError):
        driver.check_generation_complete([7, 8], max_tokens=8, eos_ids=[], label="s")
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.check_generation_complete(
            [7, 8, 9], max_tokens=2, eos_ids=[9], label="s"
        )
    assert "max_tokens" in str(excinfo.value)


def _metrics(gen_len=1, bytes_=0.0, forward=0, trace=None):
    return {
        "generated_tokens": gen_len,
        "target_forward_times": forward,
        "edge_cloud_data_bytes": bytes_,
        "comm_trace_edge_cloud": [] if trace is None else trace,
    }


def test_sample_result_accepts_long_prefix_short_generation():
    """--max-tokens 是新增 token 预算：prefix 可以远长于它。"""
    driver.check_sample_result(
        prefix_ids=list(range(300)),
        gen_ids=[9],
        max_tokens=8,
        eos_ids=[9],
        metrics=_metrics(gen_len=1, bytes_=0.0, forward=0, trace=None),
        label="s",
    )


def test_sample_result_allows_first_layer_eos_without_trace():
    """第一层就被 EOS 收下：没有二层字节、没有 target forward、没有 trace 也合法。"""
    driver.check_sample_result(
        prefix_ids=[1, 2, 3],
        gen_ids=[9],
        max_tokens=8,
        eos_ids=[9],
        metrics=_metrics(gen_len=1, bytes_=0.0, forward=0, trace=None),
        label="s",
    )


def test_sample_result_requires_trace_only_when_bytes_positive():
    driver.check_sample_result(
        prefix_ids=[1],
        gen_ids=[9],
        max_tokens=8,
        eos_ids=[9],
        metrics=_metrics(gen_len=1, bytes_=120.0, forward=1, trace=[[120.0, 0]]),
        label="s",
    )
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.check_sample_result(
            prefix_ids=[1],
            gen_ids=[9],
            max_tokens=8,
            eos_ids=[9],
            metrics=_metrics(gen_len=1, bytes_=120.0, forward=1, trace=None),
            label="s",
        )
    assert "trace 为空" in str(excinfo.value)
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.check_sample_result(
            prefix_ids=[1],
            gen_ids=[9],
            max_tokens=8,
            eos_ids=[9],
            metrics=_metrics(gen_len=1, bytes_=120.0, forward=1, trace=[[100.0, 0]]),
            label="s",
        )
    assert "不一致" in str(excinfo.value)


def test_sample_result_rejects_token_account_mismatch():
    with pytest.raises(driver.SmokeError):
        driver.check_sample_result(
            prefix_ids=[1],
            gen_ids=[9, 10],
            max_tokens=8,
            eos_ids=[9],
            metrics=_metrics(gen_len=3, bytes_=0.0),
            label="s",
        )
    with pytest.raises(driver.SmokeError):
        driver.check_sample_result(
            prefix_ids=[1],
            gen_ids=[9, 8, 7],
            max_tokens=2,
            eos_ids=[9],
            metrics=_metrics(gen_len=3, bytes_=0.0),
            label="s",
        )
    with pytest.raises(driver.SmokeError):
        driver.check_sample_result(
            prefix_ids=[],
            gen_ids=[9],
            max_tokens=2,
            eos_ids=[9],
            metrics=_metrics(gen_len=1, bytes_=0.0),
            label="s",
        )


def test_run_aggregate_requires_one_real_second_layer():
    first_layer_eos = {
        "sample_id": 0,
        "metrics": _metrics(gen_len=1, bytes_=0.0, forward=0, trace=None),
    }
    full = {
        "sample_id": 1,
        "metrics": _metrics(gen_len=4, bytes_=300.0, forward=2, trace=[[300.0, 0]]),
    }
    driver.check_run_aggregate([first_layer_eos, full])
    for broken in (
        [first_layer_eos],
        [dict(first_layer_eos, metrics=_metrics(gen_len=1, bytes_=5.0))],
        [
            dict(
                full,
                metrics=_metrics(
                    gen_len=4, bytes_=300.0, forward=0, trace=[[300.0, 0]]
                ),
            )
        ],
    ):
        with pytest.raises(driver.SmokeError):
            driver.check_run_aggregate(broken)


def test_cache_max_length_covers_last_round_bonus():
    """末轮 T-1：两级各一个 bonus，最坏 T+g1+g2+1；实现必须留出余量。"""
    for prefix in (1, 50, 300):
        for max_tokens in (1, 8, 48):
            cap = driver.cache_max_length(prefix, max_tokens)
            worst = prefix + max_tokens + driver.GAMMA1 + driver.GAMMA2 + 1
            assert cap >= worst + 1, (prefix, max_tokens, cap, worst)


def test_insert_repo_on_sys_path_guards(tmp_path, monkeypatch):
    good = tmp_path / "good"
    (good / "src").mkdir(parents=True)
    (good / "src" / "__init__.py").write_text("", encoding="utf-8")

    monkeypatch.delitem(sys.modules, "src", raising=False)
    # 只看这条 guard：把 sys.path 收窄成"不含本仓库"的最小集合
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    driver.insert_repo_on_sys_path(good)
    assert sys.path[0] == str(good.resolve())

    sibling = tmp_path / "sibling"
    (sibling / "src" / "baselines").mkdir(parents=True)
    (sibling / "src" / "baselines" / "__init__.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(sys, "path", [str(sibling), str(tmp_path)])
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.insert_repo_on_sys_path(good)
    assert "别处" in str(excinfo.value)

    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.setitem(sys.modules, "src", types.ModuleType("src"))
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.insert_repo_on_sys_path(good)
    assert "sys.modules" in str(excinfo.value)


def test_definition_info_requires_repo_internal_source(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "src").mkdir(parents=True)
    inside = repo / "src" / "baselines.py"
    inside.write_text("class Baselines:\n    pass\n", encoding="utf-8")
    outside = tmp_path / "outside.py"
    outside.write_text("class Baselines:\n    pass\n", encoding="utf-8")

    def fake_module(name, path):
        module = types.ModuleType(name)
        module.__file__ = str(path)
        monkeypatch.setitem(sys.modules, name, module)
        cls = type("Baselines", (), {})
        cls.__module__ = name
        return cls

    info = driver.definition_info(fake_module("fake_inside", inside), repo)
    assert info["definition_module"] == "fake_inside"
    assert info["definition_file"] == "src/baselines.py"
    assert info["source_sha256"] == driver.sha256_file(inside)

    with pytest.raises(driver.SmokeError) as excinfo:
        driver.definition_info(fake_module("fake_outside", outside), repo)
    assert "不在 repo 内" in str(excinfo.value)


def test_snapshot_self_pids_fails_closed(monkeypatch):
    monkeypatch.setattr(driver, "query_pids", lambda: [os.getpid()])
    assert driver.snapshot_self_pids() == (os.getpid(),)

    # 只有一个 pid 但不是自己：不能猜，必须失败
    monkeypatch.setattr(driver, "query_pids", lambda: [os.getpid() + 1])
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.snapshot_self_pids()
    assert "pid=" in str(excinfo.value)

    # 自己 + 外来 pid：独占被破坏
    monkeypatch.setattr(driver, "query_pids", lambda: [os.getpid(), os.getpid() + 1])
    with pytest.raises(driver.GpuExclusivityError):
        driver.snapshot_self_pids()

    monkeypatch.setattr(driver, "query_pids", lambda: [])
    with pytest.raises(driver.SmokeError):
        driver.snapshot_self_pids()


def test_enforce_offline_env(monkeypatch):
    for key in (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "TOKENIZERS_PARALLELISM",
    ):
        monkeypatch.delenv(key, raising=False)
    driver.enforce_offline_env()
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"
    assert os.environ["TOKENIZERS_PARALLELISM"] == "false"


def test_assets_require_every_existing_file_declared(tmp_path):
    payload = _manifest(tmp_path / "assets")
    root = Path(payload["models"]["small"]["path"])
    readme = root / "README.md"
    readme.write_bytes(b"readme")

    with pytest.raises(driver.SmokeError) as excinfo:
        driver.validate_assets(payload)
    assert "README.md" in str(excinfo.value)
    assert "未声明" in str(excinfo.value)

    payload["models"]["small"]["files"].append(
        {
            "name": "README.md",
            "size": readme.stat().st_size,
            "sha256": hashlib.sha256(readme.read_bytes()).hexdigest(),
        }
    )
    driver.validate_assets(payload)

    # HF 风格：目录里是指向 blob 的符号链接，按内容 hash 后允许
    blob = tmp_path / "blob.bin"
    blob.write_bytes(b"blob-weights")
    shard = root / "model-00001-of-00002.safetensors"
    shard.symlink_to(blob)
    payload["models"]["small"]["files"].append(
        {
            "name": shard.name,
            "size": blob.stat().st_size,
            "sha256": hashlib.sha256(blob.read_bytes()).hexdigest(),
        }
    )
    driver.validate_assets(payload)

    # 未声明的符号链接同样拒绝（不能有"没 hash 但会被 loader 读到"的文件）
    (root / "extra.safetensors").symlink_to(blob)
    with pytest.raises(driver.SmokeError) as excinfo:
        driver.validate_assets(payload)
    assert "extra.safetensors" in str(excinfo.value)


GPU1_UUID = "GPU-1081b666-dc44-95ea-ead9-832937f52281"
GPU0_UUID = "GPU-00000000-0000-0000-0000-000000000000"
GPU_NAME = "NVIDIA GeForce RTX 4090"


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """Exercise run() without importing frameworks or accessing any GPU/model."""
    events = []
    props = types.SimpleNamespace(name=GPU_NAME, uuid=GPU1_UUID)
    torch = types.SimpleNamespace(
        __version__="2.6.0+cu124",
        version=types.SimpleNamespace(cuda="12.4"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda index: props,
        ),
    )

    class Runner:
        def __init__(self, namespace):
            events.append("Runner")
            self.tokenizer = types.SimpleNamespace(eos_token_id=9)

        def load_tokenizer(self):
            events.append("tokenizer")

        def load_model(self):
            events.append("model")

    modules = {
        "torch": torch,
        "src.baselines": types.SimpleNamespace(Baselines=object),
        "src.engine": types.SimpleNamespace(attn_impl="sdpa"),
        "src.fixed_prefix_probe_runner": types.SimpleNamespace(
            FixedPrefixProbeRunner=Runner
        ),
        "src.utils": types.SimpleNamespace(model_zoo=lambda ns: None),
        "transformers": types.SimpleNamespace(__version__="4.57.6"),
    }
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name in modules:
            events.append(("import", name, os.environ.get("CUDA_VISIBLE_DEVICES")))
            return modules[name]
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setattr(driver, "insert_repo_on_sys_path", lambda root: None)
    monkeypatch.setattr(driver, "query_pids", lambda: [])
    monkeypatch.setattr(driver, "query_identity", lambda: (GPU_NAME, GPU1_UUID))

    def snapshot_self_pids():
        events.append("snapshot")
        return (os.getpid(),)

    def collect_samples(**kwargs):
        events.append("collect")
        assert kwargs["baseline_pids"] == (os.getpid(),)
        return _samples(1)

    monkeypatch.setattr(driver, "snapshot_self_pids", snapshot_self_pids)
    monkeypatch.setattr(
        driver, "definition_info", lambda *args: _output_kwargs()["code"]
    )
    monkeypatch.setattr(driver, "assert_models", lambda *args: None)
    monkeypatch.setattr(driver, "collect_samples", collect_samples)
    manifest = tmp_path / "assets.json"
    manifest.write_text(json.dumps(_manifest(tmp_path / "assets")), encoding="utf-8")
    args = driver.parse_args(
        [
            "--repo-root",
            str(ROOT),
            "--revision",
            "a" * 40,
            "--case",
            "legacy_static",
            "--asset-manifest",
            str(manifest),
            "--samples",
            "1",
            "--output",
            str(tmp_path / "out.json"),
        ]
    )
    return types.SimpleNamespace(args=args, props=props, events=events, torch=torch)


def test_run_rejects_same_name_wrong_uuid_before_allocation(fake_run):
    fake_run.props.uuid = GPU0_UUID
    with pytest.raises(driver.SmokeError, match="UUID"):
        driver.run(fake_run.args)
    assert not {"Runner", "tokenizer", "model"}.intersection(fake_run.events)
    assert not Path(fake_run.args.output).exists()


@pytest.mark.parametrize(
    "missing_uuid", [None, "", "GPU-1081", "not-a-uuid", b"invalid", b"\xff" * 17]
)
def test_run_rejects_unavailable_torch_uuid_before_allocation(fake_run, missing_uuid):
    if missing_uuid is None:
        del fake_run.props.uuid
    else:
        fake_run.props.uuid = missing_uuid
    with pytest.raises(driver.SmokeError, match="UUID"):
        driver.run(fake_run.args)
    assert not {"Runner", "tokenizer", "model"}.intersection(fake_run.events)
    assert not Path(fake_run.args.output).exists()


@pytest.mark.parametrize("missing_uuid", [None, "", "GPU-1081"])
def test_run_rejects_unavailable_nvml_uuid_before_runtime(
    fake_run, monkeypatch, missing_uuid
):
    monkeypatch.setattr(driver, "query_identity", lambda: (GPU_NAME, missing_uuid))
    with pytest.raises(driver.SmokeError, match="UUID"):
        driver.run(fake_run.args)
    assert fake_run.events == []
    assert not Path(fake_run.args.output).exists()


def test_run_rejects_ineffective_uuid_visibility_before_runtime(fake_run, monkeypatch):
    class UnchangedVisibility(dict):
        def __setitem__(self, key, value):
            if key != "CUDA_VISIBLE_DEVICES":
                super().__setitem__(key, value)

    monkeypatch.setattr(os, "environ", UnchangedVisibility(os.environ))
    with pytest.raises(driver.SmokeError, match="CUDA_VISIBLE_DEVICES"):
        driver.run(fake_run.args)
    assert fake_run.events == []
    assert not Path(fake_run.args.output).exists()


def test_run_rejects_preimported_torch_before_runtime(fake_run, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_run.torch)
    with pytest.raises(driver.SmokeError, match="torch.*UUID"):
        driver.run(fake_run.args)
    assert fake_run.events == []
    assert not Path(fake_run.args.output).exists()


@pytest.mark.parametrize("bad_visibility", ["0", "2", "0,1", "", None])
def test_run_rejects_faulty_initial_visibility_before_runtime(
    fake_run, monkeypatch, bad_visibility
):
    if bad_visibility is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", bad_visibility)
    with pytest.raises(driver.SmokeError, match="CUDA_VISIBLE_DEVICES"):
        driver.run(fake_run.args)
    assert fake_run.events == []
    assert not Path(fake_run.args.output).exists()


def test_run_rejects_runtime_visibility_change_before_allocation(fake_run, monkeypatch):
    original_load = driver._load_runtime

    def load_runtime():
        runtime = original_load()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        return runtime

    monkeypatch.setattr(driver, "_load_runtime", load_runtime)
    with pytest.raises(driver.SmokeError, match="CUDA_VISIBLE_DEVICES"):
        driver.run(fake_run.args)
    assert not {"Runner", "tokenizer", "model"}.intersection(fake_run.events)
    assert not Path(fake_run.args.output).exists()


@pytest.mark.parametrize("guard", ["busy", "name", "count", "unavailable"])
def test_run_preserves_supplementary_gpu_guards(fake_run, monkeypatch, guard):
    if guard == "busy":
        monkeypatch.setattr(driver, "query_pids", lambda: [os.getpid() + 1])
    elif guard == "name":
        fake_run.props.name = "different GPU model"
    elif guard == "count":
        fake_run.torch.cuda.device_count = lambda: 2
    else:
        fake_run.torch.cuda.is_available = lambda: False
    with pytest.raises(driver.SmokeError):
        driver.run(fake_run.args)
    assert not {"Runner", "tokenizer", "model"}.intersection(fake_run.events)
    assert not Path(fake_run.args.output).exists()


@pytest.mark.parametrize(
    "torch_uuid",
    [
        GPU1_UUID,
        GPU1_UUID.removeprefix("GPU-"),
        uuid.UUID("1081b666-dc44-95ea-ead9-832937f52281"),
        bytes.fromhex("1081b666dc4495eaead9832937f52281"),
        GPU1_UUID.encode("ascii"),
    ],
)
def test_run_binds_uuid_before_runtime_and_preserves_loaded_snapshot(
    fake_run, monkeypatch, torch_uuid
):
    fake_run.props.uuid = torch_uuid
    original_load = driver._load_runtime

    def load_runtime():
        assert os.environ["CUDA_VISIBLE_DEVICES"] == GPU1_UUID
        assert os.environ["CUDA_DEVICE_ORDER"] == "FASTEST_FIRST"
        fake_run.events.append("runtime")
        return original_load()

    monkeypatch.setattr(driver, "_load_runtime", load_runtime)
    payload = driver.run(fake_run.args)
    assert fake_run.events[0] == "runtime"
    imported = [event for event in fake_run.events if isinstance(event, tuple)]
    assert imported and all(event[2] == GPU1_UUID for event in imported)
    assert [event for event in fake_run.events if isinstance(event, str)] == [
        "runtime",
        "Runner",
        "tokenizer",
        "model",
        "snapshot",
        "collect",
    ]
    assert payload["runtime"]["gpu_uuid"] == GPU1_UUID
    assert json.loads(Path(fake_run.args.output).read_text()) == payload


def test_cli_defaults():
    args = driver.parse_args(
        [
            "--repo-root",
            "/abs/tree",
            "--revision",
            "a" * 40,
            "--case",
            "legacy_static",
            "--asset-manifest",
            "/abs/assets.json",
            "--output",
            "/abs/out.json",
        ]
    )
    assert (args.samples, args.max_tokens, args.seed) == (12, 48, 20261103)
    with pytest.raises(SystemExit):
        driver.parse_args(
            [
                "--repo-root",
                "/abs/tree",
                "--revision",
                "a" * 40,
                "--case",
                "nope",
                "--asset-manifest",
                "/abs/assets.json",
                "--output",
                "/abs/out.json",
            ]
        )
