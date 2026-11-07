#!/usr/bin/env python3
"""重构冒烟回归 driver：真实模型 + 12 条固定 prompt 的整段生成（跨提交 A/B 用）。

每个 prompt 走注册表 ``tridecoding``（static 完整生成），逐样本 seed = --seed +
sample_id，记录 prefix_ids / gen_ids / 全部 metrics；``--output`` 用 mode "x"。

    CUDA_VISIBLE_DEVICES=1 python tools/run_refactor_smoke.py --repo-root /abs/tree \
      --revision <40hex> --case legacy_static|honest_int8 \
      --asset-manifest /abs/assets.json \
      --output /abs/out.json [--samples 12] [--max-tokens 48] [--seed 20261103]

顶层 keys: schema_version, case, revision, code, configuration, runtime, assets,
samples。严格 schema 归比较器；本文件只保证字段齐、可 JSON、**绝不出现**
NaN/Infinity（非有限数在采样阶段直接失败，不退化成 null）。

case 表（两臂同 case）：legacy_static temp0.0/k32/bits16/per_transfer/residualFalse；
honest_int8 temp0.7/k8/bits8/per_round/residualTrue。``transfer_top_k`` 是上行截断
宽度，**同时改变提案分布与计费**。其余固定：gamma1=3、gamma2=2、20/100/100 Mbps、
NTT 10/1 ms、batch 0、instant、不随机通信、graph off、disable_rl_update、
disable_eos_stop=False、无 RL/heads、不量化、bf16、单 GPU1。

assets：schema_version=1、models.small/target={path,files:[{name,size,sha256}]}、
tokenizer_vocab_sha256、submodule_commit（其余字段原样保留）。目录里实际存在的每个
文件（含 HF 指向 blob 的符号链接）都必须已声明并 hash，否则 fail closed。
``tokenizer_vocab_sha256`` = sha256(json.dumps(tokenizer.get_vocab(),
sort_keys=True, separators=(",", ":")).encode())，small 与 target 必须相等。

安全：import src 前 sys.path 只插 --repo-root 且 Baselines 定义须在 repo 内；离线 env
先置位；须使用尚未 import torch 的新进程。输入 CUDA_VISIBLE_DEVICES=1 仅为准入条件，
实际在 import torch 前改绑 NVML GPU1 的完整 UUID；构造 Runner 前须验证 torch UUID。
GPU1 开跑前零 compute、加载后只有自己、采样后出现新 pid 立即报错退出（不
kill）；不写 checkpoint。完整性：len(gen_ids)<=max_tokens，短于 max_tokens 时 gen_ids
必须含 EOS（只验结果，不加提前终止）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = 1
EVAL_MODE = "tridecoding"
GPU_INDEX = 1
MAX_SAMPLES, MAX_TOKENS_LIMIT = 12, 256
GAMMA1, GAMMA2 = 3, 2
BANDWIDTHS = (20.0, 100.0, 100.0)  # edge_cloud / edge_end / cloud_end, Mbps
MIN_BW_MBPS, NTT_MS, BATCH_DELAY_S = 5.0, (10.0, 1.0), 0.0
OUTPUT_FIELDS = (
    "schema_version",
    "case",
    "revision",
    "code",
    "configuration",
    "runtime",
    "assets",
    "samples",
)

#: case 的全部差异项（顺序 = CASE_FIELDS）。transfer_top_k 同时影响提案分布与计费。
CASE_FIELDS = (
    "temperature",
    "transfer_top_k",
    "prob_payload_bits",
    "comm_round_trip_mode",
    "charge_residual_payload",
)
CASES: dict[str, tuple] = {
    "legacy_static": (0.0, 32, 16, "per_transfer", False),
    "honest_int8": (0.7, 8, 8, "per_round", True),
}

#: 12 条自包含普通 prompt（非 GSM8K、不含 "####"）。
PROMPTS: tuple[str, ...] = (
    "Summarize the water cycle in three sentences.",
    "List four common kitchen appliances and one use for each.",
    "Rewrite formally: the meeting got moved because half the team was out.",
    "Explain the difference between a compiler and an interpreter to a beginner.",
    "Name three renewable energy sources and mention one limitation of each.",
    "Write a short product description for a stainless steel water bottle.",
    "Describe three practical ways to keep a shared kitchen clean.",
    "Translate the sentence 'The library closes at six' into Spanish and French.",
    "Outline a weekly plan for learning basic Python data structures.",
    "Compare the advantages of trains and airplanes for a 600 km trip.",
    "Explain why leaves change color in autumn, in plain language.",
    "Draft a polite email rescheduling a dentist appointment to next Tuesday.",
)


class SmokeError(RuntimeError):
    """可预期的配置/资产/一致性错误。"""


class GpuExclusivityError(SmokeError):
    """GPU1 上出现了不属于本次运行的 compute 进程。"""


def normalize_revision(text: str | None) -> str:
    value = (text or "").strip().lower()
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise SmokeError(f"--revision 必须是 40 位完整 git sha，收到 {text!r}")
    return value


def validate_cuda_visible_devices(value: str | None) -> str:
    if value != "1":
        raise SmokeError(
            f"CUDA_VISIBLE_DEVICES 准入值必须恰好是 '1'，当前 {value!r}；"
            "执行时将按 NVML 物理 GPU1 UUID 绑定"
        )
    return "1"


def validate_bounds(samples: int, max_tokens: int) -> None:
    if (
        isinstance(samples, bool)
        or not isinstance(samples, int)
        or not 1 <= samples <= MAX_SAMPLES
    ):
        raise SmokeError(f"--samples 必须在 1..{MAX_SAMPLES}，收到 {samples!r}")
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 1 <= max_tokens <= MAX_TOKENS_LIMIT
    ):
        raise SmokeError(
            f"--max-tokens 必须在 1..{MAX_TOKENS_LIMIT}，收到 {max_tokens!r}"
        )


def resolve_case(name: str) -> dict[str, Any]:
    values = CASES.get(name)
    if values is None:
        raise SmokeError(f"未知 --case {name!r}；可用: {sorted(CASES)}")
    return dict(zip(CASE_FIELDS, values, strict=True))


def prompt_sha256(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tokenizer_vocab_digest(tokenizer: Any) -> str:
    """sha256(json.dumps(get_vocab(), sort_keys=True, separators=(",", ":")))。"""
    getter = getattr(tokenizer, "get_vocab", None)
    if not callable(getter):
        raise SmokeError("tokenizer 没有 get_vocab()，无法算词表指纹")
    mapping = getter()
    if not isinstance(mapping, Mapping) or not mapping:
        raise SmokeError("tokenizer.get_vocab() 未返回非空映射")
    payload = json.dumps(mapping, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def strict_json(value: Any, label: str = "metrics") -> Any:
    """转成严格 JSON 值；**非有限浮点直接失败**（绝不落成 null 冒充有效数据）。"""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SmokeError(f"{label} 出现非有限浮点 {value!r}，拒绝写入（不转 null）")
        return float(value)
    if isinstance(value, Mapping):
        return {
            str(key): strict_json(item, f"{label}.{key}") for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [strict_json(item, f"{label}[{i}]") for i, item in enumerate(value)]
    for attribute in ("item", "tolist"):  # numpy / torch 标量或数组
        convert = getattr(value, attribute, None)
        if callable(convert):
            return strict_json(convert(), label)
    raise SmokeError(f"{label} 出现无法序列化的类型 {type(value)!r}")


def parse_compute_apps_csv(text: str) -> list[int]:
    heads = (line.split(",")[0].strip() for line in text.splitlines())
    return sorted({int(head) for head in heads if head.lstrip("-").isdigit()})


def foreign_pids(
    current: Sequence[int], *, baseline: Sequence[int], allowed: Sequence[int] = ()
) -> list[int]:
    return sorted(
        {int(p) for p in current}
        - {int(p) for p in baseline}
        - {int(p) for p in allowed}
    )


def _nvidia_smi(args: Sequence[str]) -> str:
    command = ["nvidia-smi", *args]
    try:
        done = subprocess.run(
            command, capture_output=True, text=True, timeout=30, check=False
        )
    except FileNotFoundError as exc:
        raise SmokeError("找不到 nvidia-smi：无法证明 GPU1 独占，拒绝继续") from exc
    if done.returncode != 0:
        detail = (done.stderr or done.stdout).strip()[:300]
        raise SmokeError(f"{' '.join(command)} 失败（rc={done.returncode}）: {detail}")
    return done.stdout


def query_pids() -> list[int]:
    return parse_compute_apps_csv(
        _nvidia_smi(
            [
                f"--id={GPU_INDEX}",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
            ]
        )
    )


def query_identity() -> tuple[str, str]:
    out = _nvidia_smi(
        [f"--id={GPU_INDEX}", "--query-gpu=name,uuid", "--format=csv,noheader"]
    ).strip()
    name, _, uuid = out.rpartition(", ")
    if not name.strip() or not uuid.strip():
        raise SmokeError(f"无法解析 GPU{GPU_INDEX} 身份信息: {out!r}")
    return name.strip(), uuid.strip()


def snapshot_self_pids() -> tuple[int, ...]:
    """加载后确认 GPU1 上只有本进程；认不出自己就 fail closed。

    不做 PID-namespace 猜测：如果列表里没有 os.getpid()，那"唯一那个 pid"可能
    是别人刚启动的进程，把它当成自己会让独占检查彻底失效。
    """
    current = query_pids()
    if not current:
        raise SmokeError("模型加载后 GPU1 上仍无 compute 进程：CUDA 上下文未建立")
    pid = os.getpid()
    if pid not in current:
        raise SmokeError(
            f"GPU1 的 compute 进程 {current} 里没有本进程 pid={pid}："
            "无法证明独占（不做 namespace 猜测），拒绝继续"
        )
    intruders = sorted(set(current) - {pid})
    if intruders:
        raise GpuExclusivityError(
            f"加载期间 GPU1 上出现别的进程 {intruders}（不 kill）"
        )
    return (pid,)


def load_assets(path: str | Path) -> dict:
    manifest = Path(path)
    if not manifest.is_file():
        raise SmokeError(f"--asset-manifest 不存在: {manifest}")
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise SmokeError(f"--asset-manifest 不是合法 JSON: {manifest}") from exc
    if not isinstance(payload, dict):
        raise SmokeError("--asset-manifest 顶层必须是对象")
    return payload


def validate_assets(payload: Mapping[str, Any], *, verify_files: bool = True) -> dict:
    """校验"能不能真跑"：绝对且存在的根 + 每个声明文件的 size/sha256。"""
    if payload.get("schema_version") != 1 or isinstance(
        payload.get("schema_version"), bool
    ):
        raise SmokeError(
            f"assets.schema_version 必须为 1，收到 {payload.get('schema_version')!r}"
        )
    models = payload.get("models")
    if not isinstance(models, Mapping):
        raise SmokeError("assets.models 必须是对象")
    for role in ("small", "target"):
        entry = models.get(role)
        if not isinstance(entry, Mapping):
            raise SmokeError(f"assets.models.{role} 缺失")
        raw_path = entry.get("path")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise SmokeError(
                f"assets.models.{role}.path 必须是绝对路径，收到 {raw_path!r}"
            )
        root = Path(raw_path)
        if verify_files and not root.is_dir():
            raise SmokeError(f"assets.models.{role}.path 不是目录: {root}")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise SmokeError(f"assets.models.{role}.files 必须是非空数组")
        declared: set[str] = set()
        for item in files:
            if not isinstance(item, Mapping):
                raise SmokeError(f"assets.models.{role}.files 项必须是对象")
            name = item.get("name")
            if (
                not isinstance(name, str)
                or not name
                or Path(name).name != name
                or name in {".", ".."}
            ):
                raise SmokeError(
                    f"assets.models.{role}.files[].name 必须是纯文件名: {name!r}"
                )
            if name in declared:
                raise SmokeError(f"assets.models.{role}.files 重复声明 {name!r}")
            declared.add(name)
            size = item.get("size")
            if isinstance(size, bool) or not isinstance(size, int) or size < 0:
                raise SmokeError(f"assets.models.{role}.{name}.size 非法: {size!r}")
            digest = str(item.get("sha256") or "").strip().lower()
            if not verify_files:
                continue
            target = root / name  # HF snapshot 内部是 blob 符号链接，按内容校验
            if not target.is_file():
                raise SmokeError(f"资产缺失: {target}")
            if target.stat().st_size != size:
                raise SmokeError(f"资产大小不符: {target}")
            actual = sha256_file(target)
            if actual != digest:
                raise SmokeError(
                    f"资产 sha256 不符: {target}（{actual[:16]}… != {digest[:16]}…）"
                )
        # fail closed：目录里**实际存在**的文件（含 HF 指向 blob 的符号链接）必须
        # 全部被 hash 过。否则 loader 可能读到没进 manifest 的 config/tokenizer/权重。
        for existing in root.iterdir():
            if not (existing.is_file() or existing.is_symlink()):
                continue
            if existing.name not in declared:
                raise SmokeError(
                    f"{root} 存在未声明的文件 {existing.name!r}："
                    "必须先 hash 进 assets.models.files"
                )
    if (
        Path(str(models["small"]["path"])).resolve()
        == Path(str(models["target"]["path"])).resolve()
    ):
        raise SmokeError("assets.models.small 与 target 指向同一目录")
    return dict(payload)


#: smoke_arguments 里不属于"行为参数"的键（不进 configuration）。
NON_BEHAVIOR_ARGS = frozenset(
    {
        "eval_dataset",
        "exp_name",
        "data_path",
        "probe_cache_max_length",
        "little_model",
        "draft_model",
        "target_model",
    }
)


def smoke_arguments(
    case: Mapping[str, Any], *, models: Mapping[str, str], max_tokens: int, seed: int
) -> dict[str, Any]:
    """decoder 需要的行为参数（键 = argparse dest）。"""
    return {
        "eval_mode": EVAL_MODE,
        "eval_dataset": "refactor_smoke",
        "little_model": models["small"],
        "draft_model": models["small"],
        "target_model": models["target"],
        "model_source": "local",
        "model_dtype": "bf16",
        "little_quantization": "none",
        "draft_quantization": "none",
        "target_quantization": "none",
        "temp": case["temperature"],
        "top_k": 0,
        "top_p": 0.0,
        "gamma": GAMMA1,
        "gamma1": GAMMA1,
        "gamma2": GAMMA2,
        "max_tokens": max_tokens,
        "seed": seed,
        "transfer_top_k": case["transfer_top_k"],
        "transfer_top_k_cap": 0,
        "prob_payload_bits": case["prob_payload_bits"],
        "force_full_vocab_transfer": False,
        "comm_round_trip_mode": case["comm_round_trip_mode"],
        "charge_residual_payload": case["charge_residual_payload"],
        "edge_cloud_bandwidth": BANDWIDTHS[0],
        "edge_end_bandwidth": BANDWIDTHS[1],
        "cloud_end_bandwidth": BANDWIDTHS[2],
        "min_bandwidth_mbps": MIN_BW_MBPS,
        "comm_bw_model": "instant",
        "ntt_ms_edge_cloud": NTT_MS[0],
        "ntt_ms_edge_end": NTT_MS[1],
        "batch_delay": BATCH_DELAY_S,
        "use_stochastic_comm": False,
        "use_early_stopping": False,
        "use_cuda_graph": False,
        "use_rl_adapter": False,
        "disable_rl_update": True,
        "disable_eos_stop": False,
        "dump_network_stats": False,
        "exp_name": "refactor_smoke",
        "data_path": "data/",
        "probe_cache_max_length": None,
    }


def build_namespace(case: Mapping[str, Any], **kwargs: Any) -> Any:
    from argparse import Namespace

    return Namespace(**smoke_arguments(case, **kwargs))


def eos_token_ids(tokenizer: Any) -> list[int]:
    """tokenizer.eos_token_id 归一化；None → []，单 int → [int]。"""
    value = getattr(tokenizer, "eos_token_id", None)
    if value is None:
        return []
    if isinstance(value, int) and not isinstance(value, bool):
        return [int(value)]
    return [int(item) for item in value]


def check_generation_complete(
    gen_ids: Sequence[int], *, max_tokens: int, eos_ids: Sequence[int], label: str
) -> None:
    """只验结果：短于 max_tokens 的生成必须自带 EOS，不能只靠"非空"冒充完整。"""
    if len(gen_ids) > max_tokens:
        raise SmokeError(f"{label} 生成 {len(gen_ids)} token > max_tokens={max_tokens}")
    if len(gen_ids) == max_tokens:
        return
    if not eos_ids:
        raise SmokeError(
            f"{label} 提前结束（{len(gen_ids)}<{max_tokens}）且无 eos_token_id，"
            "无法确认是自然 EOS"
        )
    if not set(gen_ids) & set(eos_ids):
        raise SmokeError(
            f"{label} 提前结束（{len(gen_ids)}<{max_tokens}）但 gen_ids 无 EOS "
            f"{sorted(eos_ids)}：疑似被截断"
        )


def build_configuration(
    case_name: str, *, samples: int, max_tokens: int, seed: int, eos_ids: Sequence[int]
) -> dict:
    """行为参数 = 实际传给 decoder 的那一份（单一真源，避免与 Namespace 漂移）。"""
    case = resolve_case(case_name)
    arguments = smoke_arguments(
        case,
        models={"small": "small", "target": "target"},
        max_tokens=max_tokens,
        seed=seed,
    )
    configuration = {k: v for k, v in arguments.items() if k not in NON_BEHAVIOR_ARGS}
    configuration.update(
        {"samples": samples, "eos_token_ids": [int(item) for item in eos_ids]}
    )
    return configuration


def build_output(
    *,
    case: str,
    revision: str,
    code: Mapping[str, Any],
    configuration: Mapping[str, Any],
    runtime: Mapping[str, Any],
    assets: Mapping[str, Any],
    samples: Sequence[Mapping[str, Any]],
    expected_samples: int,
) -> dict:
    """组装输出；只做本文件自己的完整性检查（严格 schema 归比较器）。"""
    resolve_case(case)
    if len(samples) != expected_samples or not 1 <= len(samples) <= MAX_SAMPLES:
        raise SmokeError(f"samples 数量 {len(samples)} != 期望 {expected_samples}")
    payload = strict_json(
        {
            "schema_version": SCHEMA_VERSION,
            "case": case,
            "revision": normalize_revision(revision),
            "code": dict(code),
            "configuration": dict(configuration),
            "runtime": dict(runtime),
            "assets": dict(assets),
            "samples": [dict(sample) for sample in samples],
        },
        "output",
    )
    if tuple(payload) != OUTPUT_FIELDS:  # pragma: no cover - 自检
        raise SmokeError("输出字段顺序被破坏")
    json.dumps(payload, ensure_ascii=False, allow_nan=False)
    return payload


def write_output_exclusive(path: str | Path, payload: Mapping[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    try:
        with open(target, "x", encoding="utf-8") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise SmokeError(f"--output 已存在，拒绝覆盖: {target}") from exc
    return target


def enforce_offline_env() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def insert_repo_on_sys_path(repo_root: Path) -> None:
    if "src" in sys.modules:
        raise SmokeError("import src 之前 sys.modules 已存在 'src'，拒绝复用脏状态")
    root = repo_root.resolve()
    for entry in sys.path:
        if not entry:
            continue
        other = Path(entry).resolve()
        if other != root and (
            (other / "src" / "baselines.py").exists()
            or (other / "src" / "baselines" / "__init__.py").exists()
        ):
            raise SmokeError(f"sys.path 上别处也能提供 src/baselines：{other}")
    sys.path.insert(0, str(root))


def definition_info(Baselines: Any, repo_root: Path) -> dict:
    import importlib
    import inspect

    module = sys.modules.get(Baselines.__module__) or importlib.import_module(
        Baselines.__module__
    )
    module_file = Path(getattr(module, "__file__", "")).resolve()
    try:
        relative = module_file.relative_to(repo_root).as_posix()
    except ValueError as exc:
        raise SmokeError(f"Baselines 定义不在 repo 内: {module_file}") from exc
    if inspect.getsourcefile(Baselines) is None or not relative.endswith(".py"):
        raise SmokeError(f"Baselines 源码归属异常: {relative}")
    return {
        "definition_module": Baselines.__module__,
        "definition_file": relative,
        "source_sha256": sha256_file(module_file),
    }


def assert_models(
    decoder: Any, torch: Any, assets: Mapping[str, Any], small_path: str
) -> None:
    """bf16/未量化 + 共享词表 + 两份 tokenizer 指纹一致。"""
    from transformers import AutoTokenizer

    sizes = {
        role: int(getattr(decoder, role).get_input_embeddings().weight.shape[0])
        for role in ("little_model", "draft_model", "target_model")
    }
    vocab = int(getattr(decoder, "vocab_size", -1))
    if vocab <= 0 or vocab != min(sizes.values()):
        raise SmokeError(f"共享词表未校验：vocab_size={vocab} sizes={sizes}")
    tokenizer = decoder.tokenizer
    if (
        max(len(tokenizer), int(getattr(tokenizer, "vocab_size", len(tokenizer))))
        > vocab
    ):
        raise SmokeError("tokenizer 可发出超出共享词表的 id")
    for role in ("little_model", "draft_model", "target_model"):
        model = getattr(decoder, role)
        if model.get_input_embeddings().weight.dtype != torch.bfloat16:
            raise SmokeError(f"{role} 不是 bfloat16（禁止量化）")
        if getattr(model, "is_loaded_in_4bit", False) or getattr(
            model, "is_quantized", False
        ):
            raise SmokeError(f"{role} 被量化（禁止量化）")
    digest = tokenizer_vocab_digest(tokenizer)
    expected = str(assets.get("tokenizer_vocab_sha256", "")).strip().lower()
    if digest != expected:
        raise SmokeError(f"tokenizer_vocab_sha256 不符：清单 {expected} 实测 {digest}")
    small_digest = tokenizer_vocab_digest(
        AutoTokenizer.from_pretrained(
            small_path, trust_remote_code=True, local_files_only=True
        )
    )
    if small_digest != digest:
        raise SmokeError(f"small/target 词表不一致：{small_digest} vs {digest}")


def cache_max_length(prefix_len: int, max_tokens: int) -> int:
    """KV/prob 缓存容量：prefix + 新增 token 预算 + 两级 bonus + 余量。

    末轮从 T-1 出发时，little 与 draft 两级各可能再各拿一个 bonus token，最坏
    T+g1+g2+1；这里给到 +2，避免最后一轮触发动态扩容。
    """
    return prefix_len + max_tokens + GAMMA1 + GAMMA2 + 2


def seed_all(seed: int) -> None:
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def check_sample_result(
    *,
    prefix_ids: Sequence[int],
    gen_ids: Sequence[int],
    max_tokens: int,
    eos_ids: Sequence[int],
    metrics: Mapping[str, Any],
    label: str,
) -> None:
    """单样本事后校验（纯函数）：EOS 完整性 + token 账 + 字节账一致性。

    只要求"本样本真的收了二层字节时必须给出 trace"：第一层就被 EOS 收下的样本
    可以没有 trace、没有 target forward —— 那是合法结果，不是空跑，整体由
    ``check_run_aggregate`` 保证至少发生过一次。
    """
    if not prefix_ids:
        raise SmokeError(f"{label} prefix_ids 为空")
    if not gen_ids:
        raise SmokeError(f"{label} gen_ids 为空")
    check_generation_complete(
        gen_ids, max_tokens=max_tokens, eos_ids=eos_ids, label=label
    )
    generated = int(metrics.get("generated_tokens", -1))
    if generated != len(gen_ids):
        raise SmokeError(f"{label} generated_tokens={generated} != {len(gen_ids)}")
    recorded = float(metrics.get("edge_cloud_data_bytes") or 0.0)
    if recorded <= 0:
        return
    trace = metrics.get("comm_trace_edge_cloud")
    if not isinstance(trace, list) or not trace:
        raise SmokeError(f"{label} edge_cloud_data_bytes={recorded} 但 trace 为空")
    traced = sum(float(entry[0]) for entry in trace)
    if not math.isclose(traced, recorded, rel_tol=0.0, abs_tol=1e-6):
        raise SmokeError(f"{label} 逐消息账与聚合字节不一致：{traced} != {recorded}")


def check_run_aggregate(samples: Sequence[Mapping[str, Any]]) -> None:
    """整轮聚合校验：二层计费 / 真实 target verify / trace 至少各发生一次。"""
    total_bytes = sum(
        float(sample["metrics"].get("edge_cloud_data_bytes") or 0.0)
        for sample in samples
    )
    total_forward = sum(
        int(sample["metrics"].get("target_forward_times") or 0) for sample in samples
    )
    if total_bytes <= 0:
        raise SmokeError("整轮 edge_cloud_data_bytes 合计为 0：二层计费从未发生")
    if total_forward <= 0:
        raise SmokeError("整轮 target_forward_times 合计为 0：没有真实 verify")
    if not any(sample["metrics"].get("comm_trace_edge_cloud") for sample in samples):
        raise SmokeError("整轮没有任何 comm_trace_edge_cloud：计费账为空")


def collect_samples(
    *,
    decoder: Any,
    torch: Any,
    case: Mapping[str, Any],
    samples: int,
    max_tokens: int,
    base_seed: int,
    baseline_pids: Sequence[int],
    eos_ids: Sequence[int],
) -> list[dict]:
    from src.baselines import Baselines

    method = decoder.get_decoding_method()
    if getattr(method, "__func__", None) is not getattr(Baselines, "tridecoding", None):
        raise SmokeError(f"注册表返回的不是 Baselines.tridecoding：{method!r}")
    results: list[dict] = []
    for index in range(samples):
        prompt = PROMPTS[index]
        sample_seed = base_seed + index
        seed_all(sample_seed)
        prefix_ids = [
            int(t)
            for t in decoder.tokenizer(prompt, add_special_tokens=False).input_ids
        ]
        input_ids = torch.tensor([prefix_ids], dtype=torch.long)
        prefix_len = len(prefix_ids)
        # --max-tokens 是**新增 token 预算**，不是 prefix 预算；cache 容量含 prefix。
        decoder.args.probe_cache_max_length = cache_max_length(prefix_len, max_tokens)
        with torch.no_grad():
            output, metrics = method(
                input_ids.clone(),
                transfer_top_k=case["transfer_top_k"],
                use_precise_comm_sim=False,
                use_stochastic_comm=False,
                ntt_ms_edge_cloud=NTT_MS[0],
                ntt_ms_edge_end=NTT_MS[1],
                use_early_stopping=False,
            )
        if (
            not torch.is_tensor(output)
            or output.dim() != 2
            or output.shape[1] <= prefix_len
        ):
            raise SmokeError(f"prompt[{index}] 没有产生新 token")
        if not torch.equal(output[0, :prefix_len].detach().cpu(), input_ids[0]):
            raise SmokeError(f"prompt[{index}] 返回张量前缀被改写（verify 异常）")
        gen_ids = [int(t) for t in output[0, prefix_len:].tolist()]
        safe = strict_json(metrics, f"prompt[{index}].metrics")
        check_sample_result(
            prefix_ids=prefix_ids,
            gen_ids=gen_ids,
            max_tokens=max_tokens,
            eos_ids=eos_ids,
            metrics=safe,
            label=f"prompt[{index}]",
        )
        intruders = foreign_pids(
            query_pids(), baseline=baseline_pids, allowed=(os.getpid(),)
        )
        if intruders:
            raise GpuExclusivityError(
                f"采样后 GPU1 出现新 compute 进程 {intruders}（不 kill）；"
                "本次数据不可信，换新的 --output 重跑"
            )
        results.append(
            {
                "sample_id": index,
                "prompt": prompt,
                "prompt_sha256": prompt_sha256(prompt),
                "seed": sample_seed,
                "prefix_ids": prefix_ids,
                "gen_ids": gen_ids,
                "metrics": safe,
            }
        )
        print(
            f"[refactor-smoke] {index + 1}/{samples} seed={sample_seed} "
            f"prefix={prefix_len} gen={len(gen_ids)} "
            f"fwd={safe.get('target_forward_times', 0)} "
            f"bytes={float(safe.get('edge_cloud_data_bytes') or 0.0):.0f}",
            file=sys.stderr,
            flush=True,
        )
    check_run_aggregate(results)
    return results


def normalize_gpu_uuid(value: Any) -> str:
    """Accept only a full GPU UUID, never a model name or ordinal."""
    if isinstance(value, uuid.UUID):
        return f"GPU-{value}"
    if isinstance(value, bytes):
        if len(value) == 16:
            return f"GPU-{uuid.UUID(bytes=value)}"
        try:
            value = value.decode("ascii")
        except UnicodeDecodeError as exc:
            raise SmokeError(f"无法解析完整 GPU UUID: {value!r}") from exc
    if not isinstance(value, str):
        raise SmokeError(f"无法取得完整 GPU UUID: {value!r}")
    text = value.removeprefix("GPU-")
    try:
        canonical = str(uuid.UUID(text))
    except ValueError as exc:
        raise SmokeError(f"无法解析完整 GPU UUID: {value!r}") from exc
    if text.lower() != canonical:
        raise SmokeError(f"要求完整 GPU UUID: {value!r}")
    return f"GPU-{canonical}"


def _load_runtime() -> tuple[Any, Any, Any, Any, Any]:
    """Import frameworks only after binding visibility to physical GPU1's UUID."""
    import torch

    from src.baselines import Baselines
    from src.engine import attn_impl
    from src.fixed_prefix_probe_runner import FixedPrefixProbeRunner
    from src.utils import model_zoo

    return torch, Baselines, attn_impl, FixedPrefixProbeRunner, model_zoo


def run(args: argparse.Namespace) -> dict:
    case = resolve_case(args.case)
    revision = normalize_revision(args.revision)
    validate_bounds(args.samples, args.max_tokens)
    validate_cuda_visible_devices(os.environ.get("CUDA_VISIBLE_DEVICES"))
    output_path = Path(args.output)
    if output_path.exists():
        raise SmokeError(f"--output 已存在，拒绝覆盖: {output_path}")

    assets = validate_assets(load_assets(args.asset_manifest))
    repo_root = Path(args.repo_root).resolve()
    if not (repo_root / "src").is_dir():
        raise SmokeError(f"--repo-root 下没有 src/: {repo_root}")

    enforce_offline_env()
    insert_repo_on_sys_path(repo_root)
    if query_pids():
        raise GpuExclusivityError(
            "开跑前 GPU1 上已有 compute 进程（不 kill），拒绝采样"
        )

    gpu_name, selected_uuid = query_identity()
    gpu_uuid = normalize_gpu_uuid(selected_uuid)
    if "torch" in sys.modules:
        raise SmokeError("torch 已 import：无法保证 GPU UUID 绑定先于 CUDA 初始化")
    # CUDA ordinal 1 need not mean NVML physical GPU1 under FASTEST_FIRST.
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise SmokeError("CUDA_VISIBLE_DEVICES 未绑定选定 GPU UUID，拒绝加载 runtime")
    torch, Baselines, attn_impl, FixedPrefixProbeRunner, model_zoo = _load_runtime()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu_uuid:
        raise SmokeError("runtime 改写了 CUDA_VISIBLE_DEVICES GPU UUID 绑定")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise SmokeError(
            f"要求恰好 1 张可见 GPU（CUDA_VISIBLE_DEVICES={gpu_uuid!r}），"
            f"实际 available={torch.cuda.is_available()} "
            f"count={torch.cuda.device_count()}"
        )
    props = torch.cuda.get_device_properties(0)
    actual_uuid = normalize_gpu_uuid(getattr(props, "uuid", None))
    if actual_uuid != gpu_uuid:
        raise SmokeError(
            f"torch GPU UUID 与物理 GPU1 不一致：{actual_uuid!r} vs {gpu_uuid!r}"
        )
    if str(getattr(props, "name", "")).strip() != gpu_name:
        raise SmokeError(
            f"torch 看到的 GPU 与物理 GPU1 不一致：{props.name!r} vs {gpu_name!r}"
        )
    code = definition_info(Baselines, repo_root)
    models = {
        "small": str(assets["models"]["small"]["path"]),
        "target": str(assets["models"]["target"]["path"]),
    }
    namespace = build_namespace(
        case, models=models, max_tokens=args.max_tokens, seed=args.seed
    )
    model_zoo(namespace)
    decoder = FixedPrefixProbeRunner(namespace)
    decoder.load_tokenizer()
    decoder.load_model()
    assert_models(decoder, torch, assets, models["small"])

    baseline_pids = snapshot_self_pids()
    import transformers

    runtime = {
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "cuda": str(torch.version.cuda),
        "transformers": str(transformers.__version__),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "attention_backend": str(attn_impl),
    }
    eos_ids = eos_token_ids(decoder.tokenizer)
    if not eos_ids:
        raise SmokeError(
            "tokenizer 没有 eos_token_id：无法验证生成完整性（配置要求非空）"
        )
    samples = collect_samples(
        decoder=decoder,
        torch=torch,
        case=case,
        samples=args.samples,
        max_tokens=args.max_tokens,
        base_seed=args.seed,
        baseline_pids=baseline_pids,
        eos_ids=eos_ids,
    )
    payload = build_output(
        case=args.case,
        revision=revision,
        code=code,
        configuration=build_configuration(
            args.case,
            samples=args.samples,
            max_tokens=args.max_tokens,
            seed=args.seed,
            eos_ids=eos_ids,
        ),
        runtime=runtime,
        assets=assets,
        samples=samples,
        expected_samples=args.samples,
    )
    write_output_exclusive(output_path, payload)
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="真实模型固定 prompt 的整段生成冒烟 driver"
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--revision", required=True, help="40 位完整 git sha")
    parser.add_argument("--case", required=True, choices=sorted(CASES))
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--samples", type=int, default=MAX_SAMPLES)
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--output", required=True, help="输出 JSON（mode x，不覆盖）")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    try:
        payload = run(parse_args(argv))
    except SmokeError as exc:
        print(f"[refactor-smoke] ERROR: {exc}", file=sys.stderr)
        return 3 if isinstance(exc, GpuExclusivityError) else 2
    except Exception:  # pragma: no cover - 未预期异常保留栈
        traceback.print_exc()
        return 1
    print(
        f"[refactor-smoke] OK case={payload['case']} "
        f"revision={payload['revision'][:12]} samples={len(payload['samples'])} "
        f"definition={payload['code']['definition_file']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
