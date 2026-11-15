"""Private fixed worker, launched only by the authorized supervisor.

Real models/caches/timers/communication. Resource-loader bypass is intentional:
this validates explicit native CEE execution, not production asset loading, RPC,
model quality, speed, every graph branch, or a universal probability law.
"""

from __future__ import annotations

import argparse
import gc
import os
import signal
import sys
import uuid
import weakref
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools import native_validation_schema as schema
from tools.native_gpu_guard import NativeSystem


def smoke(torch, ids):
    from src.decoding_rng import DecodingRng

    schema.require(
        torch.cuda.is_available() and torch.cuda.device_count() == len(ids),
        "CUDA device count mismatch",
    )
    devices = []
    for index, identity in enumerate(ids):
        raw_uuid = torch.cuda.get_device_properties(index).uuid
        if isinstance(raw_uuid, bytes):
            raw_uuid = (
                uuid.UUID(bytes=raw_uuid)
                if len(raw_uuid) == 16
                else raw_uuid.decode("ascii")
            )
        actual = "GPU-" + str(uuid.UUID(str(raw_uuid).removeprefix("GPU-")))
        schema.require(actual == identity, "actual CUDA UUID/order mismatch")
        with torch.cuda.device(index):
            schema.require(torch.cuda.is_bf16_supported(), "BF16 unavailable")
            left = torch.ones((64, 64), device="cuda", dtype=torch.bfloat16)
            right = torch.full_like(left, 0.5)
            stream = torch.cuda.Stream(device=index)
            stream.wait_stream(torch.cuda.current_stream(index))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    product = left @ right
            torch.cuda.current_stream(index).wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                product = left @ right
            for value in (1.0, 2.0, 3.0):
                left.fill_(value)
                product.fill_(-1)
                graph.replay()
                torch.cuda.synchronize(index)
                schema.require(
                    torch.equal(product, torch.full_like(product, 32 * value)),
                    "native GEMM/replay failed",
                )
            devices.append(
                {
                    "uuid": identity,
                    "logical": index,
                    "name": torch.cuda.get_device_name(index),
                    "capability": list(torch.cuda.get_device_capability(index)),
                    "graph_replays": 3,
                }
            )
            del graph, stream, left, right, product
    torch.manual_seed(31857)
    cpu_before = torch.get_rng_state().clone()
    cuda_before = [row.clone() for row in torch.cuda.get_rng_state_all()]
    rng = DecodingRng()
    schema.require(
        torch.equal(cpu_before, torch.get_rng_state()),
        "RNG construction consumed entropy",
    )
    torch.empty((), dtype=torch.int64).random_()
    expected_cpu = torch.get_rng_state().clone()
    torch.set_rng_state(cpu_before)
    uniforms = []
    for index in range(len(ids)):
        device = torch.device("cuda", index)
        uniform = rng.rand((4096,), device=device)
        schema.require(
            bool(
                torch.isfinite(uniform).all()
                and (uniform >= 0).all()
                and (uniform < 1).all()
            )
            and abs(float(uniform.mean()) - 0.5) < 0.03,
            "native RNG uniform sanity failed",
        )
        uniforms.append(uniform.cpu())
        probabilities = torch.tensor([0.5, 0.25, 0.125, 0.125], device=device).expand(
            16384, 4
        )
        samples = rng.sample(probabilities)
        histogram = torch.bincount(samples.flatten(), minlength=4).double() / 16384
        error = float(
            (histogram - torch.tensor([0.5, 0.25, 0.125, 0.125], device=device))
            .abs()
            .max()
        )
        schema.require(error < 0.025, "native RNG histogram sanity failed")
        devices[index].update(rng_histogram=histogram.cpu().tolist(), rng_error=error)
    schema.require(
        not torch.equal(uniforms[0], uniforms[1]), "cloned device RNG streams"
    )
    schema.require(
        torch.equal(torch.get_rng_state(), expected_cpu),
        "private RNG CPU allocation mismatch",
    )
    schema.require(
        all(
            torch.equal(a, b)
            for a, b in zip(cuda_before, torch.cuda.get_rng_state_all(), strict=True)
        ),
        "default CUDA RNG changed",
    )
    source = torch.arange(128, dtype=torch.float64, device="cuda:0")
    schema.require(
        torch.equal(source, source.to("cuda:1").to("cuda:0")),
        "cross-device copy failed",
    )
    return {
        "checks": dict.fromkeys(
            ("bf16_gemm", "graph_replay", "private_rng", "cross_device_copy"), "passed"
        ),
        "devices": devices,
    }


class Observer:
    """Transparent original calls; weak ordinals cannot recycle like id(obj)."""

    def __init__(self, torch, graph_class, cache_class):
        self.torch, self.Graph, self.Cache = torch, graph_class, cache_class
        self.identities = weakref.WeakKeyDictionary()
        self.ordinal = 0
        self.owner = self.case = self.spec = None

    def identity(self, obj):
        if obj not in self.identities:
            self.ordinal += 1
            self.identities[obj] = self.ordinal
        return self.identities[obj]

    def compare(self, left, right, role, kind):
        torch = self.torch
        atol, rtol = schema.tolerances(self.spec)
        schema.require(left.shape == right.shape, "probability history shape mismatch")
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        close = torch.isclose(left, right, atol=atol, rtol=rtol)
        passed = finite and bool(close.all())
        delta = (left.float() - right.float()).abs()
        finite_delta = torch.nan_to_num(delta, nan=0.0, posinf=0.0, neginf=0.0)
        row = {
            "kind": kind,
            "role": role,
            "rows": int(left.shape[1]),
            "atol": atol,
            "rtol": rtol,
            "passed": passed,
            "nonfinite": not finite,
            "max_abs": float(finite_delta.max()) if finite_delta.numel() else 0.0,
            "mismatch_rows": (~close.all(-1))[0].nonzero().flatten().cpu().tolist(),
        }
        self.case["numerical_observations"].append(row)
        field = "forward_rows" if kind == "forward" else "cache_probabilities"
        if not passed:
            self.case["checks"][field] = "failed"

    def __enter__(self):
        self.stack = ExitStack()
        original_forward = self.Cache.forward_context
        observer = self

        def forward(cache, tokens):
            result = original_forward(cache, tokens)
            owner = observer.owner
            roles = (
                []
                if owner is None
                else [
                    role
                    for role, value in owner.observed_caches.items()
                    if value is cache
                ]
            )
            if roles:
                role = roles[0]
                fresh = observer.Cache(
                    getattr(owner, role + "_model"),
                    temperature=owner.args.temp,
                    top_k=0 if role == "target" else 4,
                    top_p=0,
                )
                fresh.vocab_size = owner.vocab_size
                original_forward(fresh, tokens.to(fresh.device))
                start = min(observer.prompt_length - 1, int(cache.current_length))
                observer.compare(
                    cache.prob_history[:, start:],
                    fresh.prob_history[:, start:],
                    role,
                    "forward",
                )
            return result

        self.stack.enter_context(patch.object(self.Cache, "forward_context", forward))
        for name in ("step", "verify", "begin_new_sequence"):
            original = getattr(self.Graph, name)

            def observed(runner, *args, _name=name, _original=original, **kwargs):
                returned = _original(runner, *args, **kwargs)
                if observer.case is not None:
                    roles = [
                        role
                        for role in ("little", "draft", "target")
                        if getattr(observer.owner, role + "_model") is runner.model
                    ]
                    schema.require(len(roles) == 1, "unattributed graph call")
                    operation = (
                        "reset"
                        if _name == "begin_new_sequence"
                        else "fallback"
                        if returned is None
                        else _name
                    )
                    observer.case["graph_events"].append(
                        {
                            "case_id": observer.case["id"],
                            "operation": operation,
                            "runner": observer.identity(runner),
                            "cache": observer.identity(runner.cache),
                            "role": roles[0],
                            "capture_count": runner.capture_count,
                            "max_len": runner.max_len,
                            "device": str(runner.device),
                        }
                    )
                return returned

            self.stack.enter_context(patch.object(self.Graph, name, observed))
        return self

    def __exit__(self, *args):
        self.owner = self.case = None
        return self.stack.__exit__(*args)


def native_run(request, report, save):
    # No torch/src import occurs until supervisor authorization and attestations.
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        LlamaConfig,
        LlamaForCausalLM,
    )

    from src.baselines import Baselines
    from src.graph_decode import GraphDecodeRunner
    from src.model_gpu import KVCacheModel

    config = request["configuration"]
    ids = config["gpu_uuids"]
    report["gpu_initialized"] = True
    report["gpu_uuids"] = ids
    report["cuda_runtime"] = torch.version.cuda
    report["smoke"] = smoke(torch, ids)
    save()
    inventory = request["binding"]["assets"]
    tokenizer = (
        None
        if inventory is None
        else AutoTokenizer.from_pretrained(
            inventory["models"]["target"]["path"],
            local_files_only=True,
            trust_remote_code=False,
        )
    )

    class Probe(Baselines):
        def load_data(self):
            return None

        def preprocess(self, text):
            return text

        def postprocess(self, text, output):
            return output

        def eval(self):
            return None

        def _acquire_three_layer_caches(self, *args, **kwargs):
            caches = super()._acquire_three_layer_caches(*args, **kwargs)
            self.observed_caches = caches
            return caches

    def prompt(alternate):
        if tokenizer is None:
            return torch.tensor(
                [[3, 1]] if alternate else [[1, 2, 3]],
                device="cuda:0",
                dtype=torch.long,
            )
        text = (
            "Name two planets in our solar system."
            if alternate
            else "Explain why the sky is blue in one short sentence."
        )
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda:0")

    def models_for(spec):
        result = {}
        topology = {"single": (0, 0, 0), "ld-t": (0, 0, 1), "l-dt": (0, 1, 1)}[
            spec["topology"]
        ]
        dtype = getattr(torch, spec["dtype"])
        model_record = {
            "dtype": spec["dtype"],
            "topology": spec["topology"],
            "roles": {},
        }
        report.setdefault("models", []).append(model_record)
        for role, device, seed in zip(
            ("little", "draft", "target"), topology, (13, 29, 47), strict=True
        ):
            if inventory is not None:
                model = AutoModelForCausalLM.from_pretrained(
                    inventory["models"][role]["path"],
                    torch_dtype=dtype,
                    device_map={"": device},
                    attn_implementation="sdpa",
                    output_hidden_states=True,
                    local_files_only=True,
                    trust_remote_code=False,
                    use_safetensors=True,
                ).eval()
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed)
                    model_config = LlamaConfig(
                        vocab_size=7,
                        hidden_size=32,
                        intermediate_size=64,
                        num_hidden_layers=2,
                        num_attention_heads=4,
                        num_key_value_heads=2,
                        max_position_embeddings=128,
                        output_hidden_states=True,
                        attention_dropout=0.0,
                        eos_token_id=None,
                    )
                    model_config._attn_implementation = "sdpa"
                    model = LlamaForCausalLM(model_config).eval()
                    model.lm_head.weight.mul_(8.0)
                model = model.to(device=torch.device("cuda", device), dtype=dtype)
            schema.require(
                model.device == torch.device("cuda", device)
                and model.dtype == dtype
                and not getattr(model, "is_quantized", False),
                "actual model placement/dtype differs",
            )
            result[role] = model
            model_record["roles"][role] = {
                "class": type(model).__module__ + "." + type(model).__qualname__,
                "dtype": str(model.dtype),
                "device": str(model.device),
                "parameters": sum(
                    parameter.numel() for parameter in model.parameters()
                ),
                "quantized": bool(getattr(model, "is_quantized", False)),
                "attention": model.config._attn_implementation,
                "asset_path": inventory["models"][role]["path"]
                if inventory is not None
                else None,
            }
        save()
        return result

    def owner_for(models, spec):
        owner = object.__new__(Probe)
        owner.args = argparse.Namespace(
            eval_mode=spec["mode"],
            edge_cloud_bandwidth=46,
            edge_end_bandwidth=563,
            cloud_end_bandwidth=46,
            max_tokens=12,
            temp=spec["temperature"],
            top_k=0,
            top_p=0.0,
            batch_delay=0.0,
            seed=0,
            exp_name="native-validation",
            eval_dataset="native-validation",
            gamma=2,
            gamma1=2,
            gamma2=2,
            use_early_stopping=False,
            dump_network_stats=False,
            disable_rl_update=True,
            disable_eos_stop=False,
            use_cuda_graph=spec["backend"] == "graph",
            graph_verify_sizes="2,4,8",
            transfer_top_k_cap=1,
            charge_residual_payload=True,
            comm_round_trip_mode="per_transfer",
        )
        owner.accelerator = SimpleNamespace(is_main_process=False)
        owner.tokenizer = SimpleNamespace(eos_token_id=None)
        owner.vocab_size = min(
            int(model.get_input_embeddings().weight.shape[0])
            for model in models.values()
        )
        if tokenizer is not None:
            schema.require(
                max(len(tokenizer), tokenizer.vocab_size) <= owner.vocab_size,
                "tokenizer/model vocabulary mismatch",
            )
        owner.num_acc_tokens = []
        owner.draft_forward_times = owner.target_forward_times = 0
        owner.observed_caches = {}
        for role, model in models.items():
            setattr(owner, role + "_model", model)
        return owner

    models = owner = None
    model_key = owner_key = None
    primary_tokens = None
    with torch.no_grad(), Observer(torch, GraphDecodeRunner, KVCacheModel) as observer:
        for spec in request["plan"]:
            next_model = (spec["dtype"], spec["topology"])
            next_owner = (
                *next_model,
                spec["mode"],
                spec["backend"],
                spec["temperature"],
            )
            if next_owner != owner_key:
                observer.owner = None
                owner = None
                gc.collect()
                for index in range(len(ids)):
                    with torch.cuda.device(index):
                        torch.cuda.empty_cache()
            if next_model != model_key:
                models = None
                gc.collect()
                models = models_for(spec)
                model_key = next_model
            if next_owner != owner_key:
                owner = owner_for(models, spec)
                owner_key = next_owner
            prefix = prompt(spec["scenario"] == "new-sample")
            schema.require(
                prefix.shape[1] + 12
                <= min(m.config.max_position_embeddings for m in models.values()),
                "prompt exceeds configured model capacity",
            )
            if spec["scenario"] == "first-token-eos":
                schema.require(primary_tokens is not None, "EOS seed scenario missing")
                owner.tokenizer.eos_token_id = primary_tokens[0][prefix.shape[1]]
            owner.args.max_tokens = spec["budget"]
            case = {
                "id": spec["id"],
                "checks": dict.fromkeys(schema.required_checks(spec), "not_run"),
                "numerical_observations": [],
                "graph_events": [],
            }
            report["cases"].append(case)
            observer.owner, observer.case, observer.spec = owner, case, spec
            observer.prompt_length = prefix.shape[1]
            save()

            def checked(name, ok, *, case=case, spec=spec):
                case["checks"][name] = "passed" if bool(ok) else "failed"
                schema.require(bool(ok), f"{spec['id']}: {name} failed")

            try:
                torch.manual_seed(spec["seed"])
                before = torch.get_rng_state().clone()
                before_cuda = [
                    state.clone() for state in torch.cuda.get_rng_state_all()
                ]
                if spec["budget"]:
                    torch.empty((), dtype=torch.int64).random_()
                expected = torch.get_rng_state().clone()
                torch.set_rng_state(before)
                generated, metrics = getattr(owner, spec["mode"])(
                    prefix, transfer_top_k=4
                )
                for index in range(len(ids)):
                    torch.cuda.synchronize(index)
                checked("cpu_rng", torch.equal(torch.get_rng_state(), expected))
                checked(
                    "cuda_rng",
                    all(
                        torch.equal(a, b)
                        for a, b in zip(
                            before_cuda, torch.cuda.get_rng_state_all(), strict=True
                        )
                    ),
                )
                checked(
                    "prefix",
                    torch.equal(generated[:, : prefix.shape[1]].cpu(), prefix.cpu()),
                )
                amount = generated.shape[1] - prefix.shape[1]
                checked(
                    "budget",
                    amount
                    == (1 if spec["scenario"] == "first-token-eos" else spec["budget"]),
                )
                checked(
                    "token_range",
                    ((generated >= 0) & (generated < owner.vocab_size)).all(),
                )
                case.update(
                    tokens=generated.cpu().tolist(),
                    metrics=metrics,
                    caches={},
                    graph_runners={},
                )
                if spec["budget"]:
                    checked(
                        "metrics",
                        metrics["generated_tokens"] == amount
                        and metrics.get("cee_probability_policy") == "wire-draft-v1"
                        and metrics.get("rng_policy") == "private-device-v1",
                    )
                    schema.require(
                        set(owner.observed_caches) == {"little", "draft", "target"},
                        "actual caches missing",
                    )
                    states_valid = True
                    for role, cache in owner.observed_caches.items():
                        state, length = cache.debug_state(), int(cache.current_length)
                        states_valid &= (
                            0 <= length <= generated.shape[1] - 1
                            and state["tracked_seq_len"]
                            == length
                            == state["prob_history_len"]
                            == state["logits_history_len"]
                        )
                        case["caches"][role] = state
                        runner = cache._graph_runner
                        case["graph_runners"][role] = (
                            None
                            if runner is None
                            else {
                                "runner": observer.identity(runner),
                                "cache": observer.identity(runner.cache),
                                "capture_count": runner.capture_count,
                                "max_len": runner.max_len,
                            }
                        )
                        if length:
                            fresh = KVCacheModel(
                                getattr(owner, role + "_model"),
                                temperature=owner.args.temp,
                                top_k=0 if role == "target" else 4,
                                top_p=0,
                            )
                            fresh.vocab_size = owner.vocab_size
                            fresh.forward_context(
                                generated[:, :length].to(fresh.device)
                            )
                            schema.require(
                                fresh.current_length == length,
                                "fresh reference endpoint mismatch",
                            )
                            observer.compare(
                                cache.prob_history, fresh.prob_history, role, "cache"
                            )
                    checked("cache_state", states_valid)
                    for field, kind in (
                        ("cache_probabilities", "cache"),
                        ("forward_rows", "forward"),
                    ):
                        schema.require(
                            any(
                                row["kind"] == kind and row["rows"] > 0
                                for row in case["numerical_observations"]
                            ),
                            f"missing {kind} numerical checks",
                        )
                        if case["checks"][field] != "failed":
                            case["checks"][field] = "passed"
                if "greedy_target" in case["checks"]:
                    reference = prefix.to(owner.target_model.device)
                    for _ in range(spec["budget"]):
                        logits = owner.target_model(reference, use_cache=False).logits[
                            :, -1, : owner.vocab_size
                        ]
                        reference = torch.cat(
                            (reference, logits.argmax(-1, keepdim=True)), dim=1
                        )
                    checked(
                        "greedy_target", torch.equal(generated.cpu(), reference.cpu())
                    )
                if spec["scenario"] == "budget12":
                    primary_tokens = case["tokens"]
                save()
                if (
                    any(not row["passed"] for row in case["numerical_observations"])
                    and not config["continue_on_numerical_mismatch"]
                ):
                    return
            except BaseException as exc:
                case["error"] = f"{type(exc).__name__}: {exc}"
                save()
                raise
            finally:
                observer.case = None
        report["execution"] = "completed"
        report["peak_allocated_bytes"] = [
            torch.cuda.max_memory_allocated(i) for i in range(len(ids))
        ]


def main(argv=None):
    # The supervisor masks these only until Popen's ownership handoff completes.
    # exec inherits the mask; unblock before reading requests or importing CUDA.
    signal.pthread_sigmask(
        signal.SIG_UNBLOCK, {signal.SIGTERM, signal.SIGINT, signal.SIGHUP}
    )
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args(argv)
    request = schema.read_json(args.request)
    output = args.request.parent / "worker-result.json"
    report = {
        "schema": "duodecoding.native-worker.v1",
        "request_sha256": schema.digest(request),
        "scope": __doc__,
        "execution": "incomplete",
        "gpu_initialized": False,
        "cases": [],
        "smoke": {"checks": {}, "devices": []},
    }
    schema.atomic_json(output, report)
    try:
        schema.require(
            os.environ.get("DUODECODING_NATIVE_AUTHORIZATION")
            == request.get("authorization")
            and isinstance(request.get("authorization"), str)
            and len(request["authorization"]) == 32,
            "supervisor authorization missing",
        )
        schema.require(
            dict(os.environ) == request["worker_environment"],
            "worker environment mismatch",
        )
        report["worker_environment"] = dict(os.environ)
        system = NativeSystem()
        report["source_before"] = system.snapshot()
        report["environment"] = system.environment()
        report["tools"] = system.tools()
        schema.require(
            report["source_before"] == request["binding"]["source"]
            and report["tools"] == request["binding"]["tools"]
            and report["environment"] == request["binding"]["environment"],
            "worker source/environment binding mismatch",
        )
        schema.require(
            schema.configuration(request["configuration"]) == request["configuration"]
            and schema.plan(request["configuration"]) == request["plan"],
            "worker plan mismatch",
        )
        native_run(request, report, lambda: schema.atomic_json(output, report))
    except BaseException as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["source_after"] = NativeSystem().snapshot()
        schema.atomic_json(output, report)
    try:
        outcome = schema.worker_outcome(report, request)
        return (
            0
            if (
                "error" not in report
                and outcome["execution"] == "completed"
                and outcome["functional"] == outcome["numerical"] == "passed"
            )
            else 2
        )
    except (ValueError, KeyError, TypeError):
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
