#!/usr/bin/env python3
"""Stage-A real-model observation entry. Never invoked by CPU unit tests.

Explicit config + assets + --authorize-gpu-execution are required. Single UUID
only; no cross-device claim. Tools-local guards are not an arbitrary-code sandbox.
"""

from __future__ import annotations

import argparse
import builtins
import functools
import importlib.metadata
import inspect
import io
import os
import platform
import sys
import time
import weakref
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

try:
    from . import run_refactor_smoke as legacy
    from . import stage_a_smoke_schema as schema
    from .run_cpu_gate import source_snapshot
except ImportError:
    import run_refactor_smoke as legacy
    import stage_a_smoke_schema as schema
    from run_cpu_gate import source_snapshot

StageAError = schema.StageAError
need = schema.need
ROOT = Path(__file__).resolve().parents[1]


def namespace_for(config, assets, features):
    """One recorded argument source, no model_zoo/CLI implicit resource resolution."""
    behavior = config["behavior"]
    args = dict(behavior)
    args.update(
        eval_mode=config["mode"],
        eval_dataset="stage_a_smoke",
        exp_name="stage_a_smoke",
        data_path="data/",
        model_source="local",
        seed=config["seed"],
        max_tokens=config["max_tokens"],
        probe_cache_max_length=config["cache_capacity"],
        use_cuda_graph=config["backend"] == "graph",
        disable_rl_update=True,
        use_rl_adapter=features["uses_main_rl"] or features["uses_little_rl"],
        disable_eos_stop=False,
        use_stochastic_comm=False,
        use_early_stopping=False,
        dump_network_stats=False,
        rl_init_strategy="resume",
        rl_init_seed=config["seed"],
        rl_output_dir=None,
        rl_reward_mode="legacy",
        rl_comm_cost_mode="nominal",
        rl_team_reward=False,
        rl_track_best_tps=False,
        ntt_trace_file="",
    )
    for role in ("little", "draft", "target"):
        args[f"{role}_quantization"] = "none"
        args[f"{role}_model"] = assets["models"].get(role, {}).get("path")
    for role, entry in assets["heads"].items():
        key = (
            "acc_head_path"
            if features["acc_head"] == "draft_target"
            else f"{role}_acc_head_path"
        )
        args[key] = entry["path"]
    for role, entry in assets["rl"].items():
        path = str(Path(entry["path"]) / entry["entrypoint"])
        args[f"{role}_rl_path"] = path
        args[f"{role}_rl_best_path"] = path  # exactly one allowed candidate per role
    return argparse.Namespace(**args)


def call_kwargs(config):
    knobs = config["behavior"]
    return {
        "transfer_top_k": knobs["transfer_top_k"],
        "use_precise_comm_sim": False,
        "use_stochastic_comm": False,
        "ntt_ms_edge_cloud": knobs["ntt_ms_edge_cloud"],
        "ntt_ms_edge_end": knobs["ntt_ms_edge_end"],
        "use_early_stopping": False,
    }


def check_environment():
    need(
        "torch" not in sys.modules and "src" not in sys.modules,
        "fresh process required: torch/src already imported before guards",
    )
    forbidden = [
        key
        for key in os.environ
        if (
            key
            in {
                "RANK",
                "WORLD_SIZE",
                "LOCAL_RANK",
                "ARP_CALIB_MAP",
                "MODEL_SERIES_NAME",
                "COMM_TRACE",
                "APPEND_TRACE",
                "CACHE_CHECK_TRACE",
                "ARP_CALIB_TRACE",
                "TOPK_TRACE",
            }
            or key.startswith(("RL_", "DUODECODING_", "DUODEC_"))
        )
        and os.environ[key]
    ]
    need(
        not forbidden,
        f"unrecorded behavior/distributed environment forbidden: {forbidden}",
    )


def preflight(args):
    need(
        args.authorize_gpu_execution is True,
        "explicit --authorize-gpu-execution required after user authorization",
    )
    need(
        Path(args.repo_root).resolve() == ROOT,
        "runner and --repo-root must be the same source tree",
    )
    check_environment()
    config = schema.validate_config(schema.load_json(args.config))
    features = schema.requirements(config["mode"])
    need(
        config["behavior"]["prob_payload_bits"] >= 16 or features["supports_prob_bits"],
        "mode_features rejects lossy prob_payload_bits for this mode/alias",
    )
    assets = schema.validate_assets(schema.load_json(args.asset_manifest), features)
    output = Path(args.output).resolve()
    need(not output.exists(), "output exists; refusing overwrite")
    need(
        not any(
            output.is_relative_to(ROOT / folder)
            for folder in ("src", "test", "tools", "eval", "scripts", "configs")
        ),
        "evidence output must be outside source snapshot directories",
    )
    for group in ("models", "heads", "rl"):
        for entry in assets[group].values():
            need(
                not output.is_relative_to(Path(entry["path"]).resolve()),
                "output cannot be inside assets",
            )
    legacy.enforce_offline_env()
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    return config, features, assets


def query_identity(selected):
    text = legacy._nvidia_smi(
        [f"--id={selected}", "--query-gpu=name,uuid", "--format=csv,noheader"]
    ).strip()
    name, sep, value = text.rpartition(", ")
    need(
        sep and name and schema.gpu_uuid(value) == selected,
        "nvidia-smi UUID selection mismatch",
    )
    return name


def query_pids(selected):
    text = legacy._nvidia_smi(
        [
            f"--id={selected}",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ]
    )
    # Do not silently ignore unknown rows: they cannot establish exclusivity.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    need(all(line.isdigit() for line in lines), "unparseable compute process list")
    return sorted({int(line) for line in lines})


def check_exclusive(selected, *, loaded):
    pids = query_pids(selected)
    expected = [os.getpid()] if loaded else []
    need(
        pids == expected,
        f"GPU not exclusive: observed {pids}, expected {expected}; "
        "never kill processes",
    )


def verify_torch_device(torch, selected, name):
    need(
        os.environ.get("CUDA_VISIBLE_DEVICES") == selected,
        "runtime changed UUID visibility",
    )
    need(
        torch.cuda.is_available() and torch.cuda.device_count() == 1,
        "exactly one visible CUDA device required",
    )
    props = torch.cuda.get_device_properties(0)
    need(
        legacy.normalize_gpu_uuid(getattr(props, "uuid", None)) == selected,
        "torch actual UUID mismatch",
    )
    need(props.name == name, "torch actual GPU name mismatch")


@contextmanager
def readonly_assets(assets):
    """Reversible guards for ordinary Python file writes/renames to declared assets.

    Also protect resolved HF blob links. Not an OS sandbox against native code.
    Every attempt is remembered even if a loader catches the raised error.
    """
    roots = [
        Path(e["path"]).resolve()
        for group in ("models", "heads", "rl")
        for e in assets[group].values()
    ]
    files = [
        (Path(e["path"]) / f["name"]).resolve()
        for group in ("models", "heads", "rl")
        for e in assets[group].values()
        for f in e["files"]
    ]
    attempts = []

    def protect(path, *, mutation=False):
        if isinstance(path, (str, bytes, os.PathLike)):
            actual = Path(os.fsdecode(path)).resolve()
            protected = actual in files or any(
                actual.is_relative_to(root) for root in roots
            )
            # Moving/removing an ancestor also moves the protected entries.
            # Do not apply this broader check to ordinary file opens.
            if mutation:
                protected |= any(p.is_relative_to(actual) for p in (*roots, *files))
            if protected:
                attempts.append(str(actual))
                raise StageAError(f"asset write/rename forbidden: {actual}")

    def opening(original):
        @functools.wraps(original)
        def wrapped(file, mode="r", *args, **kwargs):
            if any(c in mode for c in "wax+"):
                protect(file)
            return original(file, mode, *args, **kwargs)

        return wrapped

    original_open = os.open

    def reject_write_dir_fd(kwargs):
        # A pathname resolved against cwd does not describe a dir_fd operation.
        # Conservatively reject that write form, without changing read behavior.
        for name in ("dir_fd", "src_dir_fd", "dst_dir_fd"):
            if kwargs.get(name) is not None:
                attempts.append(name)
                raise StageAError(f"asset guard cannot authorize write with {name}")

    def os_open(path, flags, *args, **kwargs):
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            reject_write_dir_fd(kwargs)
            protect(path)
        return original_open(path, flags, *args, **kwargs)

    def mutation(original, two=False):
        @functools.wraps(original)
        def wrapped(path, *args, **kwargs):
            reject_write_dir_fd(kwargs)
            protect(path, mutation=True)
            if two:
                protect(args[0] if args else kwargs["dst"], mutation=True)
            return original(path, *args, **kwargs)

        return wrapped

    with ExitStack() as stack:
        for owner, name in ((builtins, "open"), (io, "open"), (Path, "open")):
            # Python 3.10 pathlib caches io.open on its accessor; patch Path too.
            stack.enter_context(
                patch.object(owner, name, opening(getattr(owner, name)))
            )
        stack.enter_context(patch.object(os, "open", os_open))
        for owner, names in (
            (os, ("replace", "rename", "remove", "unlink", "rmdir")),
            (Path, ("replace", "rename", "unlink", "rmdir")),
        ):
            for name in names:
                stack.enter_context(
                    patch.object(
                        owner,
                        name,
                        mutation(getattr(owner, name), name in ("replace", "rename")),
                    )
                )
        yield attempts
        need(
            not attempts,
            "asset mutation was attempted, even though a loader suppressed the failure",
        )


@contextmanager
def frozen_rl(module, torch, assets):
    """Observe real load; fail before the production corrupt/legacy fallback."""
    allowed = {str(Path(e["path"]) / e["entrypoint"]) for e in assets["rl"].values()}
    loaded = weakref.WeakKeyDictionary()
    original_load = module.DDQNAgent.load

    def load(agent, path):
        need(str(path) in allowed, f"undeclared RL load: {path}")
        try:
            ok = original_load(agent, path)
        except Exception as exc:
            # Deliberately NOT RuntimeError: _try_load catches that and renames.
            raise StageAError(
                f"RL checkpoint load failed (no fallback): {path}: {exc}"
            ) from exc
        need(ok is True, f"RL checkpoint not loaded: {path}")
        loaded[agent] = str(path)
        return ok

    def forbidden(*args, **kwargs):
        raise StageAError("online RL update/save forbidden during acceptance")

    with ExitStack() as stack:
        stack.enter_context(patch.object(module.DDQNAgent, "load", load))
        for cls, method in (
            (module.DDQNAgent, "update"),
            (module.DDQNAgent, "save"),
            (module.RLNetworkAdapter, "step"),
            (module.RLNetworkAdapter, "save"),
        ):
            stack.enter_context(patch.object(cls, method, forbidden))
        stack.enter_context(patch.object(torch, "save", forbidden))
        yield loaded


class WeakIdentities:
    """Sequential identities tied to live object references, never recycled id()."""

    def __init__(self):
        self.objects = []

    def get(self, obj):
        for index, reference in enumerate(self.objects, 1):
            if reference() is obj:
                return index
        self.objects.append(weakref.ref(obj))
        return len(self.objects)


class GraphRecorder:
    def __init__(self, models):
        self.models = models
        self.identities = WeakIdentities()
        self.events = []

    def record(self, runner, kind):
        roles = [role for role, model in self.models.items() if runner.model is model]
        need(len(roles) == 1, "graph runner does not belong to one actual model role")
        self.events.append(
            {
                "object": self.identities.get(runner),
                "kind": kind,
                "role": roles[0],
                "capture_count": runner.capture_count,
                "cache": self.identities.get(runner.cache),
                "max_len": runner.max_len,
                "verify_sizes": list(runner.verify_sizes),
            }
        )

    @contextmanager
    def install(self, cls):
        with ExitStack() as stack:
            for method in ("step", "verify", "begin_new_sequence", "prefill"):
                original = getattr(cls, method)

                def wrapper(
                    runner, *args, _original=original, _method=method, **kwargs
                ):
                    result = _original(runner, *args, **kwargs)
                    kind = {"begin_new_sequence": "reset"}.get(_method, _method)
                    if _method == "verify" and result is None:
                        kind = "fallback"
                    self.record(runner, kind)
                    return result

                stack.enter_context(patch.object(cls, method, wrapper))
            yield self


@contextmanager
def resource_recorder(decoder, features):
    """Wrap actual bound adapter calls, preserving originals and restoring finally."""
    counters = {}
    with ExitStack() as stack:
        for role in schema.roles_for(features)["models"]:
            key = f"models.{role}"
            counters[key] = 0

            def forward_hook(module, inputs, output, _key=key):
                # Python forward hooks include prefill/warmup/capture, not replay.
                counters[_key] += 1

            handle = getattr(decoder, f"{role}_model").register_forward_hook(
                forward_hook
            )
            stack.callback(handle.remove)
        for group, roles in schema.roles_for(features).items():
            if group == "models":
                continue
            for role in roles:
                if group == "rl":
                    obj = getattr(
                        decoder, "rl_adapter" if role == "main" else "little_rl_adapter"
                    )
                    method = "select_config"
                else:
                    name = (
                        "adapter"
                        if features["acc_head"] == "draft_target"
                        else f"{role}_adapter"
                    )
                    obj, method = getattr(decoder, name), "predict"
                key = f"{group}.{role}"
                counters[key] = 0
                original = getattr(obj, method)

                def wrapper(*args, _key=key, _original=original, **kwargs):
                    if _key.startswith("rl."):
                        bound = inspect.signature(_original).bind(*args, **kwargs)
                        bound.apply_defaults()
                        need(
                            bound.arguments.get("training") is False,
                            "RL select_config must use frozen inference",
                        )
                    result = _original(*args, **kwargs)
                    counters[_key] += 1
                    return result

                stack.enter_context(patch.object(obj, method, wrapper))
        yield counters


def binding_info(decoder, root):
    method = decoder.get_decoding_method()
    function = getattr(method, "__func__", None)
    need(
        function is decoder._DECODING_REGISTRY.get(decoder.args.eval_mode),
        "registry returned wrong bound method",
    )
    path = Path(inspect.getsourcefile(inspect.unwrap(function))).resolve()
    need(path.is_relative_to(root), "bound algorithm is outside source tree")
    return method, {
        "module": function.__module__,
        "qualname": function.__qualname__,
        "file": path.relative_to(root).as_posix(),
        "aliases": sorted(
            k for k, v in decoder._DECODING_REGISTRY.items() if v is function
        ),
    }


def consumption_info(decoder, config, features):
    from src.protocols import (
        ACCOUNTING_SWITCHES,
        DEPTH_KEYS,
        _collect,
        live_mode_consumption,
    )

    mode = config["mode"]
    live = live_mode_consumption(mode)
    need(
        live is not None and live.source == "runtime",
        "runtime protocol consumption unavailable",
    )
    names = _collect(
        decoder._DECODING_REGISTRY[mode], decoder._DECODING_REGISTRY, set()
    )
    bits_read = "prob_payload_bits" in names
    bits = config["behavior"]["prob_payload_bits"]
    need(
        bits >= 16 or (bits_read and features["supports_prob_bits"]),
        "lossy bits not consumed/supported by this mode",
    )
    return {
        "source": "runtime_introspection_not_branch_coverage",
        "accounting": {
            k: "consumed" if k in live.accounting else "not_consumed"
            for k in ACCOUNTING_SWITCHES
        },
        "depth": {
            k: "consumed" if k in live.depth_keys else "not_consumed"
            for k in DEPTH_KEYS
        },
        "prob_payload_bits": {
            "implementation_reads": bits_read,
            "supports_lossy": features["supports_prob_bits"],
            "semantics": "not_consumed"
            if not bits_read
            else "lossy_configured"
            if bits < 16
            else "pass_through",
        },
    }


def effective_resources(decoder, config, assets, features, loaded, torch):
    rows = {"models": {}, "heads": {}, "rl": {}}
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[
        config["behavior"]["model_dtype"]
    ]
    for role in schema.roles_for(features)["models"]:
        model = getattr(decoder, f"{role}_model", None)
        need(model is not None, f"missing actual model: {role}")
        devices = sorted(
            {str(p.device) for p in model.parameters()}
            | {str(b.device) for b in model.buffers()}
        )
        need(
            devices == ["cuda:0"],
            f"unsupported actual role placement: {role}={devices}",
        )
        need(
            model.get_input_embeddings().weight.dtype == dtype
            and not getattr(model, "is_quantized", False)
            and not getattr(model, "is_loaded_in_4bit", False),
            "model dtype/quantization differs from profile",
        )
        rows["models"][role] = {
            "path": assets["models"][role]["path"],
            "devices": devices,
            "gpu_uuid": config["gpu_uuids"][0],
            "dtype": str(dtype),
            "hf_device_map": legacy.strict_json(getattr(model, "hf_device_map", {})),
        }
    provenance = getattr(decoder.args, "_checkpoint_provenance", {})
    for role, entry in assets["heads"].items():
        name = (
            "adapter" if features["acc_head"] == "draft_target" else f"{role}_adapter"
        )
        adapter = getattr(decoder, name, None)
        need(
            adapter is not None
            and adapter.stop_mode == config["behavior"]["arp_stop_mode"],
            "ARP missing or stop mode differs",
        )
        pkey = (
            "acc_head" if features["acc_head"] == "draft_target" else f"acc_head_{role}"
        )
        recorded = provenance.get(pkey, {})
        need(
            recorded.get("source") != "missing" and bool(recorded.get("files")),
            "head checkpoint provenance missing",
        )
        actual = recorded.get("files", {})
        expected = {
            f["name"]: f["sha256"]
            for f in entry["files"]
            if ".cache" not in Path(f["name"]).parts
        }
        need(
            {k: v.get("sha256") for k, v in actual.items()} == expected,
            "head checkpoint provenance digest mismatch",
        )
        need(
            Path(recorded["dir"]) == Path(entry["path"])
            or (ROOT / recorded["dir"]).resolve() == Path(entry["path"]).resolve(),
            "head provenance directory mismatch",
        )
        rows["heads"][role] = {
            "path": entry["path"],
            "stop_mode": adapter.stop_mode,
            "threshold": adapter.threshold,
            "device": str(adapter.device),
        }
    for role, entry in assets["rl"].items():
        adapter = getattr(
            decoder, "rl_adapter" if role == "main" else "little_rl_adapter", None
        )
        expected_path = str(Path(entry["path"]) / entry["entrypoint"])
        need(
            adapter is not None and loaded.get(adapter.agent) == expected_path,
            "required RL checkpoint did not load",
        )
        recorded = provenance.get(f"rl_{role}", {})
        expected_sha = next(
            f["sha256"] for f in entry["files"] if f["name"] == entry["entrypoint"]
        )
        need(
            recorded.get("sha256") == expected_sha,
            "RL checkpoint provenance digest mismatch",
        )
        rows["rl"][role] = {
            "path": entry["path"],
            "loaded_path": expected_path,
            "sha256": expected_sha,
            "device": str(adapter.device),
            "action_space": adapter.action_space,
            "gamma_candidates": list(adapter.gamma_candidates),
            "topk_candidates": list(adapter.topk_candidates),
            "threshold_candidates": list(adapter.threshold_candidates),
            "update_count": adapter.agent.update_count,
        }
    need(
        bool(decoder.args.use_rl_adapter) == bool(assets["rl"]), "RL silently disabled"
    )
    return rows


def load_runtime():
    import torch
    import transformers

    from src import rl_adapter
    from src.engine import attn_impl
    from src.fixed_prefix_probe_runner import FixedPrefixProbeRunner
    from src.graph_decode import GraphDecodeRunner
    from src.mode_features import get_mode_spec

    return (
        torch,
        transformers,
        rl_adapter,
        attn_impl,
        FixedPrefixProbeRunner,
        GraphDecodeRunner,
        get_mode_spec,
    )


def collect_samples(
    decoder, torch, config, method, recorder, counters, selected, deadline
):
    prompts = legacy.PROMPTS[: config["samples"]]
    prefixes = [
        [int(t) for t in decoder.tokenizer(p, add_special_tokens=False).input_ids]
        for p in prompts
    ]
    margin = (
        max(config["behavior"]["gamma"], config["behavior"]["gamma1"])
        + config["behavior"]["gamma2"]
        + 130
    )
    need(
        all(prefixes)
        and max(map(len, prefixes)) + config["max_tokens"] + margin
        <= config["cache_capacity"],
        "fixed cache_capacity too small for prompts/budget/verification ladder",
    )
    results = []
    kwargs = call_kwargs(config)
    for index, (prompt, prefix) in enumerate(zip(prompts, prefixes, strict=True)):
        need(time.monotonic() <= deadline, "wall-clock budget exhausted before sample")
        before_calls = dict(counters)
        start_event = len(recorder.events)
        seed = config["seed"] + index
        legacy.seed_all(seed)
        inputs = torch.tensor([prefix], dtype=torch.long)
        with torch.no_grad():
            output, metrics = method(inputs.clone(), **kwargs)
        torch.cuda.synchronize()
        need(
            time.monotonic() <= deadline,
            "wall-clock budget exceeded (checked at sample boundary)",
        )
        need(
            torch.is_tensor(output)
            and output.ndim == 2
            and output.shape[0] == 1
            and output.shape[1] > len(prefix),
            "empty/invalid generated output",
        )
        need(
            torch.equal(output[0, : len(prefix)].detach().cpu(), inputs[0]),
            "generated prefix changed",
        )
        check_exclusive(selected, loaded=True)
        results.append(
            {
                "sample_id": index,
                "prompt": prompt,
                "prompt_sha256": legacy.prompt_sha256(prompt),
                "seed": seed,
                "prefix_ids": prefix,
                "gen_ids": [int(t) for t in output[0, len(prefix) :].tolist()],
                "metrics": legacy.strict_json(metrics),
                "graph_events": recorder.events[start_event:],
                "resource_calls": {k: counters[k] - before_calls[k] for k in counters},
            }
        )
    return results


def run(args):
    config, features, assets = preflight(args)  # stdlib + local files; before torch
    before = source_snapshot(ROOT)
    selected = config["gpu_uuids"][0]
    check_exclusive(selected, loaded=False)
    name = query_identity(selected)
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    need(os.environ.get("CUDA_VISIBLE_DEVICES") == selected, "UUID binding failed")
    legacy.insert_repo_on_sys_path(ROOT)
    deadline = time.monotonic() + config["max_seconds"]
    with readonly_assets(assets):
        torch, transformers, rl_module, attention, Runner, GraphRunner, get_spec = (
            load_runtime()
        )
        verify_torch_device(torch, selected, name)
        need(vars(get_spec(config["mode"])) == features, "runtime mode_features drift")
        namespace = namespace_for(config, assets, features)
        requested_arguments = vars(namespace).copy()
        with frozen_rl(rl_module, torch, assets) as loaded:
            decoder = Runner(namespace)
            decoder.load_tokenizer()
            decoder.load_model()
            resources = effective_resources(
                decoder, config, assets, features, loaded, torch
            )
            check_exclusive(selected, loaded=True)
            for entry in assets["models"].values():
                tokenizer = transformers.AutoTokenizer.from_pretrained(
                    entry["path"], local_files_only=True, trust_remote_code=False
                )
                need(
                    legacy.tokenizer_vocab_digest(tokenizer)
                    == assets["tokenizer_vocab_sha256"],
                    "model tokenizer vocab digest mismatch",
                )
            need(
                legacy.tokenizer_vocab_digest(decoder.tokenizer)
                == assets["tokenizer_vocab_sha256"],
                "runtime tokenizer vocab digest mismatch",
            )
            eos_ids = legacy.eos_token_ids(decoder.tokenizer)
            need(eos_ids, "EOS ids missing")
            method, binding = binding_info(decoder, ROOT)
            consumption = consumption_info(decoder, config, features)
            recorder = GraphRecorder(
                {role: getattr(decoder, f"{role}_model") for role in assets["models"]}
            )
            with (
                recorder.install(GraphRunner),
                resource_recorder(decoder, features) as counters,
            ):
                samples = collect_samples(
                    decoder,
                    torch,
                    config,
                    method,
                    recorder,
                    counters,
                    selected,
                    deadline,
                )
            need(
                all(getattr(namespace, k) == v for k, v in requested_arguments.items()),
                "decoder mutated fixed behavior arguments",
            )
            final_resources = effective_resources(
                decoder, config, assets, features, loaded, torch
            )
            need(
                final_resources["rl"] == resources["rl"],
                "RL checkpoint state/config changed during frozen inference",
            )
        check_exclusive(selected, loaded=True)
        schema.validate_assets(assets, features)  # assets unchanged at endpoints
    after = source_snapshot(ROOT)
    packages = {
        d.metadata["Name"]: d.version
        for d in importlib.metadata.distributions()
        if d.metadata["Name"]
    }
    packages.update(
        torch=str(torch.__version__), transformers=str(transformers.__version__)
    )
    payload = {
        "schema_version": 2,
        "profile": "stage-a",
        "status": "observed",
        "configuration": config,
        "code": {
            "before": before,
            "after": after,
            "sha256": schema.digest(before),
            "endpoint_only": True,
        },
        "assets": assets,
        "effective": {
            "arguments": requested_arguments,
            "call_kwargs": call_kwargs(config),
            "binding": binding,
            "consumption": consumption,
            "mode_features": features,
            **resources,
            "checkpoint_provenance": legacy.strict_json(
                getattr(namespace, "_checkpoint_provenance", {})
            ),
            "eos_ids": eos_ids,
        },
        "runtime": {
            "python": platform.python_version(),
            "invocation": {"argv": list(sys.argv), "executable": sys.executable},
            "packages": packages,
            "cuda": str(torch.version.cuda),
            "attention_backend": attention,
            "gpu_uuid": selected,
            "gpu_name": name,
            "idle_before": True,
            "exclusive_after": True,
            "offline": {
                key: os.environ.get(key)
                for key in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                )
            },
        },
        "samples": samples,
        "graph": schema.graph_coverage(samples, config["backend"] == "graph"),
        "limitations": schema.LIMITATIONS,
    }
    schema.validate_payload(payload)
    legacy.write_output_exclusive(args.output, payload)
    return payload


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--config",
        required=True,
        help="explicit v2 mode/UUID/sample/token/capacity/time budget JSON",
    )
    parser.add_argument("--asset-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--authorize-gpu-execution",
        action="store_true",
        help=(
            "requires prior user approval of this exact config/assets/window; "
            "never defaults on"
        ),
    )
    return parser.parse_args(argv)


def main(argv=None):
    try:
        payload = run(parse_args(argv))
    except (StageAError, legacy.SmokeError) as exc:
        print(f"[stage-a-smoke] INVALID: {exc}", file=sys.stderr)
        return 2
    print(
        f"[stage-a-smoke] OBSERVED mode={payload['configuration']['mode']} "
        f"source={payload['code']['sha256']} (not full acceptance)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
