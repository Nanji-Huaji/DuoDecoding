"""CPU-only guards/schema/recording tests. No real CUDA or checkpoint loading."""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from tools import compare_refactor_smoke as compare
from tools import run_refactor_smoke as legacy
from tools import stage_a_smoke as runner
from tools import stage_a_smoke_schema as schema

UUID = "GPU-00000000-0000-0000-0000-000000000001"
ROOT = Path(__file__).resolve().parents[1]


def config(mode="sd", backend="eager"):
    return schema.validate_config(
        {
            "mode": mode,
            "backend": backend,
            "gpu_uuids": [UUID],
            "samples": 2,
            "max_tokens": 2,
            "seed": 7,
            "cache_capacity": 512,
            "max_seconds": 60,
            "behavior": {},
        }
    )


def assets_for(tmp_path, mode="sd"):
    assets = {
        "schema_version": 2,
        "models": {},
        "heads": {},
        "rl": {},
        "tokenizer_vocab_sha256": "a" * 64,
    }
    for group, roles in schema.roles_for(schema.requirements(mode)).items():
        for role in roles:
            root = tmp_path / group / role
            root.mkdir(parents=True, exist_ok=True)
            name = "best.pth" if group == "rl" else "config.json"
            data = (group + role).encode()
            (root / name).write_bytes(data)
            entry = {
                "path": str(root),
                "files": [
                    {
                        "name": name,
                        "size": len(data),
                        "sha256": hashlib.sha256(data).hexdigest(),
                    }
                ],
            }
            if group == "rl":
                entry["entrypoint"] = name
            assets[group][role] = entry
    return assets


def payload(tmp_path, mode="sd"):
    cfg = config(mode)
    assets = assets_for(tmp_path, mode)
    features = schema.requirements(mode)
    source = {
        "git_sha": "a" * 40,
        "source_status": "?? src/new.py\n",
        "submodules": "",
        "files": {"src/new.py": "b" * 64, "tools/stage_a_smoke.py": "c" * 64},
    }
    resources = {"models": {}, "heads": {}, "rl": {}}
    provenance = {}
    for group, entries in assets.items():
        if group not in resources:
            continue
        for role, entry in entries.items():
            row = {"path": entry["path"]}
            if group == "models":
                row.update(
                    devices=["cuda:0"],
                    gpu_uuid=UUID,
                    dtype="torch.bfloat16",
                    hf_device_map={"": 0},
                )
            elif group == "rl":
                row.update(
                    loaded_path=str(Path(entry["path"]) / entry["entrypoint"]),
                    sha256=entry["files"][0]["sha256"],
                    device="cuda",
                    action_space="topk_thr",
                    gamma_candidates=[2, 4, 8, 16],
                    topk_candidates=[1, 2, 4, 8],
                    threshold_candidates=[0.1, 0.2, 0.4],
                    update_count=123,
                )
                provenance[f"rl_{role}"] = {"sha256": entry["files"][0]["sha256"]}
            else:
                row.update(stop_mode="per_token", threshold=0.8, device="cuda:0")
                name = (
                    "acc_head"
                    if features["acc_head"] == "draft_target"
                    else f"acc_head_{role}"
                )
                provenance[name] = {
                    "source": "untracked",
                    "files": {
                        f["name"]: {"sha256": f["sha256"]} for f in entry["files"]
                    },
                }
            resources[group][role] = row
    rows = []
    for i in range(2):
        prompt = f"prompt {i}"
        amount = 0 if mode == "sd" else 8
        rows.append(
            {
                "sample_id": i,
                "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "seed": 7 + i,
                "prefix_ids": [1],
                "gen_ids": [3, 4],
                "metrics": {
                    "generated_tokens": 2,
                    "little_forward_times": 0,
                    "draft_forward_times": 1,
                    "target_forward_times": 1,
                    "edge_cloud_data_bytes": amount,
                    "edge_end_data_bytes": 0,
                    "cloud_end_data_bytes": 0,
                    "comm_trace_edge_cloud": [[8, 0]] if amount else [],
                    "wall_time": 1.0,
                    "throughput": 2.0,
                    "computation_time": 0.2,
                    "little_computation_time": 0.0,
                    "draft_computation_time": 0.1,
                    "target_computation_time": 0.1,
                    "arp_overhead_time": 0.0,
                    "dra_overhead_time": 0.0,
                    "communication_time": 0.1,
                    "edge_end_comm_time": 0.0,
                    "queuing_time": 0.0,
                    "little_generated_tokens": 0,
                    "draft_generated_tokens": 1,
                    "little_accepted_tokens": 0,
                    "draft_accepted_tokens": 1,
                    "loop_times": 1,
                    "each_loop_draft_tokens": 1.0,
                    "comm_energy": 0.0,
                    "accuracy": None,
                    "avg_top_k": 32.0,
                    "avg_draft_len": 1.0,
                    "connect_times": {"edge_cloud": 1} if amount else {},
                    "edge_cloud_bandwidth_history": [],
                    "edge_cloud_ntt_history": [],
                    "edge_cloud_topk_history": [],
                    "edge_cloud_draft_len_history": [],
                    "little_entropy_history": [],
                    "draft_entropy_history": [],
                    "little_accept_rate_history": [],
                    "draft_accept_rate_history": [],
                    "little_accepted_vocab_rank_history": [],
                    "draft_accepted_vocab_rank_history": [],
                    "little_accepted_in_transfer_topk_history": [],
                    "draft_accepted_in_transfer_topk_history": [],
                    "little_accepted_transfer_topk_rank_history": [],
                    "draft_accepted_transfer_topk_rank_history": [],
                },
                "graph_events": [],
                "resource_calls": {
                    f"{group}.{role}": 1
                    for group, entries in resources.items()
                    for role in entries
                },
            }
        )
    return {
        "schema_version": 2,
        "profile": "stage-a",
        "status": "observed",
        "configuration": cfg,
        "code": {
            "before": source,
            "after": copy.deepcopy(source),
            "sha256": schema.digest(source),
            "endpoint_only": True,
        },
        "assets": assets,
        "effective": {
            "arguments": vars(runner.namespace_for(cfg, assets, features)),
            "call_kwargs": runner.call_kwargs(cfg),
            "mode_features": features,
            **resources,
            "consumption": {
                "source": "runtime_introspection_not_branch_coverage",
                "accounting": {
                    k: "not_consumed"
                    for k in (
                        "comm_round_trip_mode",
                        "charge_residual_payload",
                        "transfer_top_k_cap",
                        "force_full_vocab_transfer",
                    )
                },
                "depth": {
                    k: "consumed" if k == "gamma" else "not_consumed"
                    for k in ("gamma", "gamma1", "gamma2")
                },
                "prob_payload_bits": {
                    "implementation_reads": False,
                    "supports_lossy": features["supports_prob_bits"],
                    "semantics": "not_consumed",
                },
            },
            "binding": {
                "module": "src.new",
                "qualname": "NewImplementation",
                "file": "src/new.py",
                "aliases": [mode],
            },
            "checkpoint_provenance": provenance,
            "eos_ids": [9],
        },
        "runtime": {
            "python": "fake",
            "cuda": "fake",
            "attention_backend": "fake",
            "gpu_name": "fake",
            "invocation": {"argv": ["fake"], "executable": "fake"},
            "offline": {
                key: "1"
                for key in (
                    "HF_HUB_OFFLINE",
                    "TRANSFORMERS_OFFLINE",
                    "HF_DATASETS_OFFLINE",
                )
            },
            "gpu_uuid": UUID,
            "idle_before": True,
            "exclusive_after": True,
            "packages": {"torch": "fake", "transformers": "fake"},
        },
        "samples": rows,
        "graph": {"replay_count": 0, "reused_runners": []},
        "limitations": schema.LIMITATIONS,
    }


def test_cold_import_uses_only_standard_library():
    code = (
        "import sys; from tools import stage_a_smoke; "
        "assert not {'torch','numpy','transformers','src'} & set(sys.modules); "
        "stage_a_smoke.schema.requirements('cee_sd'); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="", PYTHONDONTWRITEBYTECODE="1"),
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c.update(gpu_uuids=[]), "exactly one"),
        (lambda c: c.update(gpu_uuids=[UUID, UUID]), "cross-device"),
        (lambda c: c.update(gpu_uuids=["1"]), "UUID"),
        (lambda c: c.update(samples=0), "samples"),
        (lambda c: c.update(max_tokens=True), "max_tokens"),
        (lambda c: c.update(max_seconds=0), "max_seconds"),
        (lambda c: c.update(backend="graph", samples=1), "two samples"),
        (lambda c: c["behavior"].update(disable_rl_update=False), "unknown behavior"),
        (lambda c: c["behavior"].update(use_rl_adapter=False), "unknown behavior"),
        (lambda c: c["behavior"].update(temp=float("nan")), "non-finite"),
        (lambda c: c["behavior"].update(gamma=-1), "gamma"),
        (lambda c: c["behavior"].update(rl_gamma_candidates=""), "invalid sizes"),
    ],
)
def test_config_guards(mutate, match):
    cfg = config()
    mutate(cfg)
    with pytest.raises(schema.StageAError, match=match):
        schema.validate_config(cfg)


@pytest.mark.parametrize(
    "mode,models,heads,rl",
    [
        ("sd", 2, 0, 0),
        ("dsd", 2, 0, 0),
        ("dssd", 2, 0, 0),
        ("adaptive_decoding", 2, 1, 1),
        ("adaptive_tridecoding", 3, 2, 2),
        ("cee_sd", 3, 2, 2),
        ("tridecoding", 3, 0, 0),
        ("cee_sd_no_arp", 3, 0, 2),
        ("cee_sd_opportunistic", 3, 2, 2),
    ],
)
def test_real_mode_features_resources_required(tmp_path, mode, models, heads, rl):
    spec = schema.requirements(mode)
    roles = schema.roles_for(spec)
    assert [len(roles[k]) for k in ("models", "heads", "rl")] == [models, heads, rl]
    assets = assets_for(tmp_path, mode)
    schema.validate_assets(assets, spec)
    ns = runner.namespace_for(config(mode), assets, spec)
    assert ns.disable_rl_update and ns.use_rl_adapter == bool(rl)
    for role in roles["rl"]:
        assert getattr(ns, f"{role}_rl_path") == getattr(ns, f"{role}_rl_best_path")
    schema.validate_payload(payload(tmp_path, mode))
    for group in ("heads", "rl"):
        if roles[group]:
            broken = copy.deepcopy(assets)
            broken[group].clear()
            with pytest.raises(schema.StageAError, match="required roles"):
                schema.validate_assets(broken, spec)


def test_all_local_files_and_rl_sidecars_must_be_declared(tmp_path):
    assets = assets_for(tmp_path, "adaptive_decoding")
    root = Path(assets["rl"]["main"]["path"])
    (root / "best.pth.buffer").write_bytes(b"undeclared sidecar")
    with pytest.raises(schema.StageAError, match="undeclared"):
        schema.validate_assets(assets, schema.requirements("adaptive_decoding"))
    (root / "best.pth.buffer").unlink()
    (root / "best.pth").write_bytes(b"corrupt")
    with pytest.raises(schema.StageAError, match="size mismatch|sha256"):
        schema.validate_assets(assets, schema.requirements("adaptive_decoding"))


@pytest.mark.parametrize(
    "operation,kwargs",
    [
        ("open", {"dir_fd": 123}),
        ("rename", {"src_dir_fd": 123}),
        ("rename", {"dst_dir_fd": 123}),
        ("replace", {"src_dir_fd": 123, "dst_dir_fd": 456}),
        ("unlink", {"dir_fd": 123}),
    ],
)
def test_readonly_fd_relative_writes_fail_before_inert_syscall(
    tmp_path,
    monkeypatch,
    operation,
    kwargs,
):
    assets = assets_for(tmp_path)
    calls = []
    monkeypatch.setattr(os, operation, lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(schema.StageAError, match="dir_fd"):
        with runner.readonly_assets(assets):
            if operation == "open":
                os.open("config.json", os.O_WRONLY, **kwargs)
            elif operation in ("rename", "replace"):
                getattr(os, operation)("config.json", "config.corrupt", **kwargs)
            else:
                os.unlink("config.json", **kwargs)
    assert calls == []


def test_readonly_fd_relative_reads_still_reach_inert_syscall(tmp_path, monkeypatch):
    assets = assets_for(tmp_path)
    calls = []
    sentinel = object()
    monkeypatch.setattr(
        os,
        "open",
        lambda *a, **kw: calls.append((a, kw)) or sentinel,
    )
    with runner.readonly_assets(assets):
        assert os.open("config.json", os.O_RDONLY, dir_fd=123) is sentinel
    assert calls == [(("config.json", os.O_RDONLY), {"dir_fd": 123})]


@pytest.mark.parametrize("protected", ["model_parent", "blob_parent"])
@pytest.mark.parametrize("operand", ["source", "destination"])
def test_readonly_protects_mutation_ancestors_with_inert_syscall(
    tmp_path,
    monkeypatch,
    protected,
    operand,
):
    assets = assets_for(tmp_path)
    root = Path(assets["models"]["draft"]["path"])
    blob_dir = tmp_path / "hf" / "blobs"
    blob_dir.mkdir(parents=True)
    blob = blob_dir / "blob"
    blob.write_bytes(b"fake")
    (root / "weights.bin").symlink_to(blob)
    assets["models"]["draft"]["files"].append({"name": "weights.bin"})
    ancestor = root.parent if protected == "model_parent" else blob_dir
    unrelated = tmp_path / "unrelated"
    source, destination = (
        (ancestor, unrelated) if operand == "source" else (unrelated, ancestor)
    )
    calls = []
    monkeypatch.setattr(os, "rename", lambda *a, **kw: calls.append((a, kw)))
    with pytest.raises(schema.StageAError, match="write/rename"):
        with runner.readonly_assets(assets):
            os.rename(source, destination)
    assert calls == []
    assert blob.read_bytes() == b"fake" and root.is_dir()


def test_readonly_does_not_protect_unrelated_siblings(tmp_path, monkeypatch):
    assets = assets_for(tmp_path)
    parent = Path(assets["models"]["draft"]["path"]).parent
    calls = []
    monkeypatch.setattr(os, "open", lambda *a, **kw: calls.append((a, kw)) or 456)
    monkeypatch.setattr(os, "rename", lambda *a, **kw: calls.append((a, kw)))
    with runner.readonly_assets(assets):
        assert os.open(parent / "unrelated.bin", os.O_WRONLY) == 456
        os.rename(parent / "unrelated", parent / "unrelated-new")
    assert len(calls) == 2


def test_readonly_guards_symlink_and_restore(tmp_path):
    assets = assets_for(tmp_path)
    root = Path(assets["models"]["draft"]["path"])
    original = root / "config.json"
    blob = tmp_path / "blob"
    blob.write_bytes(original.read_bytes())
    original.unlink()
    original.symlink_to(blob)
    replace = os.replace
    with pytest.raises(schema.StageAError, match="mutation was attempted"):
        with runner.readonly_assets(assets):
            assert original.read_bytes() == b"modelsdraft"
            for path in (original, blob):
                with pytest.raises(schema.StageAError, match="write/rename"):
                    path.write_bytes(b"bad")
            with pytest.raises(schema.StageAError):
                os.replace(original, root / "renamed")
    assert os.replace is replace
    assert blob.read_bytes() == b"modelsdraft" and original.is_symlink()


def fake_rl():
    events = []

    class Agent:
        def load(self, path):
            events.append("load")
            raise RuntimeError("bad checkpoint")

        def update(self):
            events.append("update")

        def save(self):
            events.append("save")

    class Adapter:
        def step(self):
            events.append("step")

        def save(self):
            events.append("save-adapter")

    return types.SimpleNamespace(DDQNAgent=Agent, RLNetworkAdapter=Adapter), events


def test_bad_rl_checkpoint_never_reaches_corrupt_rename_save_or_fallback(tmp_path):
    assets = assets_for(tmp_path, "adaptive_decoding")
    module, events = fake_rl()
    torch = types.SimpleNamespace(save=lambda *a: events.append("torch-save"))
    original = module.DDQNAgent.load
    path = str(Path(assets["rl"]["main"]["path"]) / "best.pth")
    before = set(Path(path).parent.iterdir())
    with runner.frozen_rl(module, torch, assets):
        with pytest.raises(schema.StageAError, match="no fallback"):
            # Same catch type as real RLNetworkAdapter._try_load.
            try:
                module.DDQNAgent().load(path)
            except RuntimeError:
                events.append("rename")
                os.replace(path, path + ".corrupt")
                torch.save({}, path)
        for callback in (
            module.DDQNAgent().update,
            module.DDQNAgent().save,
            module.RLNetworkAdapter().step,
            module.RLNetworkAdapter().save,
            lambda: torch.save({}, path),
        ):
            with pytest.raises(schema.StageAError, match="forbidden"):
                callback()
    assert events == ["load"]
    assert module.DDQNAgent.load is original
    assert set(Path(path).parent.iterdir()) == before


def test_rl_real_load_called_and_identity_recorded(tmp_path):
    assets = assets_for(tmp_path, "adaptive_decoding")
    module, events = fake_rl()
    module.DDQNAgent.load = lambda self, path: events.append(path) or True
    path = str(Path(assets["rl"]["main"]["path"]) / "best.pth")
    torch = types.SimpleNamespace(save=lambda *a: None)
    agent = module.DDQNAgent()
    with runner.frozen_rl(module, torch, assets) as loaded:
        assert agent.load(path) is True
        assert loaded[agent] == path and events == [path]
        with pytest.raises(schema.StageAError, match="undeclared"):
            agent.load(path + ".other")


class FakeObject:
    pass


class FakeGraph:
    def __init__(self, model):
        self.model, self.cache = model, FakeObject()
        self.capture_count, self.max_len, self.verify_sizes = 2, 512, [8]
        self.result = object()
        self.calls = []

    def step(self, token):
        self.calls.append("step")
        if token == "error":
            raise RuntimeError("fake failure")
        return self.result

    def verify(self, token):
        self.calls.append("verify")
        return None if token == "fallback" else self.result

    def prefill(self, token):
        self.calls.append("prefill")
        return self.result

    def begin_new_sequence(self):
        self.calls.append("reset")


def test_graph_wrapper_records_only_success_replay_and_restores_on_exception():
    model = FakeObject()
    graph = FakeGraph(model)
    recorder = runner.GraphRecorder({"draft": model})
    original = FakeGraph.step
    with pytest.raises(RuntimeError):
        with recorder.install(FakeGraph):
            assert graph.step("ok") is graph.result
            assert graph.verify("fallback") is None
            assert graph.verify("ok") is graph.result
            graph.begin_new_sequence()
            graph.step("error")
    assert graph.calls == ["step", "verify", "verify", "reset", "step"]
    assert [e["kind"] for e in recorder.events] == [
        "step",
        "fallback",
        "verify",
        "reset",
    ]
    assert FakeGraph.step is original


def test_graph_coverage_requires_same_live_cache_reset_and_stable_capture():
    model, rows = FakeObject(), []
    graph = FakeGraph(model)
    recorder = runner.GraphRecorder({"draft": model})
    with recorder.install(FakeGraph):
        graph.step("ok")
        rows.append({"sample_id": 0, "graph_events": list(recorder.events)})
        recorder.events.clear()
        graph.begin_new_sequence()
        graph.verify("ok")
        rows.append({"sample_id": 1, "graph_events": list(recorder.events)})
    assert schema.graph_coverage(rows, True) == {
        "replay_count": 2,
        "reused_runners": [1],
    }
    for broken in (
        [],
        rows[:1],
        [rows[0], {"sample_id": 1, "graph_events": rows[1]["graph_events"][1:]}],
    ):
        with pytest.raises(schema.StageAError, match="reuse"):
            schema.graph_coverage(broken, True)
    for field in ("cache", "capture_count", "object"):
        broken = copy.deepcopy(rows)
        broken[1]["graph_events"][-1][field] += 100
        with pytest.raises(schema.StageAError, match="reuse"):
            schema.graph_coverage(broken, True)
    with pytest.raises(schema.StageAError, match="eager"):
        schema.graph_coverage(rows, False)


@pytest.mark.parametrize(
    "fault", ["late_reset", "late_then_replay", "cache", "capture"]
)
def test_graph_reuse_requires_matching_reset_before_first_replay(fault):
    replay = {
        "object": 1,
        "cache": 2,
        "role": "draft",
        "capture_count": 1,
        "max_len": 512,
        "verify_sizes": [8],
        "kind": "step",
    }
    reset = dict(replay, kind="reset")
    rows = [
        {"sample_id": 0, "graph_events": [replay]},
        {"sample_id": 1, "graph_events": [reset, dict(replay)]},
    ]
    assert schema.graph_coverage(rows, True)["reused_runners"] == [1]
    if fault.startswith("late"):
        rows[1]["graph_events"].reverse()
        if fault == "late_then_replay":
            rows[1]["graph_events"].append(dict(replay, kind="verify"))
    else:
        reset["cache" if fault == "cache" else "capture_count"] += 1
    with pytest.raises(schema.StageAError, match="reuse"):
        schema.graph_coverage(rows, True)


def test_weak_identity_not_recycled_after_death():
    ids = runner.WeakIdentities()
    obj = FakeObject()
    assert ids.get(obj) == ids.get(obj) == 1
    del obj
    gc.collect()
    other = FakeObject()
    assert ids.get(other) == 2


def test_resource_recording_calls_original_and_restores():
    class Model:
        def register_forward_hook(self, callback):
            self.hook = callback
            return types.SimpleNamespace(remove=lambda: setattr(self, "hook", None))

    class Adapter:
        def predict(self, state):
            return state + 1

    class RL:
        def select_config(self, state, training=True):
            assert not training
            return state + 2

    decoder = types.SimpleNamespace(
        draft_model=Model(), target_model=Model(), adapter=Adapter(), rl_adapter=RL()
    )
    with runner.resource_recorder(
        decoder, schema.requirements("adaptive_decoding")
    ) as counters:
        assert decoder.adapter.predict(1) == 2
        assert decoder.rl_adapter.select_config(2, training=False) == 4
        decoder.target_model.hook(None, None, None)
        with pytest.raises(schema.StageAError, match="frozen"):
            decoder.rl_adapter.select_config(2)
    assert counters == {
        "models.draft": 0,
        "models.target": 1,
        "heads.draft_target": 1,
        "rl.main": 1,
    }
    assert "predict" not in vars(decoder.adapter) and "select_config" not in vars(
        decoder.rl_adapter
    )
    assert decoder.target_model.hook is None


def test_registry_binding_uses_actual_function_not_old_qualname():
    class Migrated:
        args = types.SimpleNamespace(eval_mode="sd")

        def new_entry(self):
            pass

        def get_decoding_method(self):
            return self.new_entry

    Migrated._DECODING_REGISTRY = {
        "sd": Migrated.new_entry,
        "alias": Migrated.new_entry,
    }
    _, info = runner.binding_info(Migrated(), ROOT)
    assert info["aliases"] == ["alias", "sd"] and "new_entry" in info["qualname"]


def test_device_validation_does_not_allocate(monkeypatch):
    props = types.SimpleNamespace(uuid=UUID, name="fake")
    torch = types.SimpleNamespace(
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda i: props,
        )
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", UUID)
    runner.verify_torch_device(torch, UUID, "fake")
    props.uuid = "GPU-00000000-0000-0000-0000-000000000002"
    with pytest.raises(schema.StageAError, match="UUID mismatch"):
        runner.verify_torch_device(torch, UUID, "fake")


def test_gpu_process_guard_rejects_foreign_and_missing_self(monkeypatch):
    monkeypatch.setattr(runner, "query_pids", lambda _: [])
    runner.check_exclusive(UUID, loaded=False)
    with pytest.raises(schema.StageAError, match="exclusive"):
        runner.check_exclusive(UUID, loaded=True)
    for processes in ([os.getpid() + 1], [os.getpid(), os.getpid() + 1]):
        monkeypatch.setattr(runner, "query_pids", lambda _, p=processes: p)
        with pytest.raises(schema.StageAError, match="never kill"):
            runner.check_exclusive(UUID, loaded=True)
    monkeypatch.setattr(runner, "query_pids", lambda _: [os.getpid()])
    runner.check_exclusive(UUID, loaded=True)


def test_authorization_and_bad_files_precede_runtime(tmp_path, monkeypatch):
    args = types.SimpleNamespace(
        authorize_gpu_execution=False,
        repo_root=str(ROOT),
        config=str(tmp_path / "missing.json"),
        asset_manifest="missing",
        output=str(tmp_path / "output"),
    )
    monkeypatch.setattr(
        runner, "load_runtime", lambda: pytest.fail("runtime must not load")
    )
    monkeypatch.setattr(
        runner, "query_pids", lambda *_: pytest.fail("GPU must not be queried")
    )
    with pytest.raises(schema.StageAError, match="authorize"):
        runner.run(args)
    args.authorize_gpu_execution = True
    monkeypatch.setattr(runner, "check_environment", lambda: None)
    with pytest.raises(schema.StageAError, match="cannot read JSON"):
        runner.run(args)
    assert not Path(args.output).exists()


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda p: p["code"]["after"]["files"].update(
                {"src/untracked.py": "d" * 64}
            ),
            "endpoints changed",
        ),
        (lambda p: p["code"].update(sha256="a" * 40), "endpoint identity"),
        (lambda p: p["samples"].clear(), "sample count"),
        (lambda p: p["samples"][0].update(gen_ids=[]), "gen_ids"),
        (lambda p: p["samples"][0].update(gen_ids=[3]), "without EOS"),
        (lambda p: p["samples"][0].update(gen_ids=[3, 4, 5]), "exceeds budget"),
        (
            lambda p: p["samples"][0]["metrics"].update(edge_cloud_data_bytes=1),
            "trace sum",
        ),
        (
            lambda p: [
                s["metrics"].update(target_forward_times=0) for s in p["samples"]
            ],
            "target forward",
        ),
        (
            lambda p: [
                s["resource_calls"].update({"models.target": 0}) for s in p["samples"]
            ],
            "target model forward",
        ),
        (lambda p: p["configuration"].update(backend="graph"), "backend mismatch"),
        (
            lambda p: p["effective"]["arguments"].update(disable_rl_update=False),
            "unsafe",
        ),
        (lambda p: p["runtime"].update(exclusive_after=False), "exclusivity"),
    ],
)
def test_artifact_fail_closed(tmp_path, mutation, match):
    p = payload(tmp_path)
    mutation(p)
    with pytest.raises(schema.StageAError, match=match):
        schema.validate_payload(p)


def test_graph_flag_without_replay_cannot_pass(tmp_path):
    p = payload(tmp_path)
    p["configuration"]["backend"] = "graph"
    p["effective"]["arguments"]["use_cuda_graph"] = True
    with pytest.raises(schema.StageAError, match="graph requested"):
        schema.validate_payload(p)


def test_missing_effective_resource_usage_or_checkpoint_provenance_fails(tmp_path):
    p = payload(tmp_path, "adaptive_decoding")
    for s in p["samples"]:
        s["resource_calls"]["heads.draft_target"] = 0
    with pytest.raises(schema.StageAError, match="never executed"):
        schema.validate_payload(p)
    p = payload(tmp_path, "adaptive_decoding")
    p["effective"]["checkpoint_provenance"].clear()
    with pytest.raises(schema.StageAError, match="provenance"):
        schema.validate_payload(p)


@pytest.mark.parametrize(
    "group,role,field",
    [
        ("models", "draft", "dtype"),
        ("models", "draft", "hf_device_map"),
        ("heads", "draft_target", "stop_mode"),
        ("heads", "draft_target", "threshold"),
        ("heads", "draft_target", "device"),
        *[
            ("rl", "main", field)
            for field in (
                "sha256",
                "device",
                "action_space",
                "gamma_candidates",
                "topk_candidates",
                "threshold_candidates",
                "update_count",
            )
        ],
    ],
)
def test_effective_resource_fields_are_mandatory_even_for_self_compare(
    tmp_path,
    group,
    role,
    field,
):
    p = payload(tmp_path, "adaptive_decoding")
    schema.validate_payload(p)
    del p["effective"][group][role][field]
    source = p["code"]["sha256"]
    report = compare.compare(
        p,
        copy.deepcopy(p),
        expect_samples=2,
        base_source=source,
        candidate_source=source,
        comparison="exact-replay",
    )
    assert report["verdict"] == "INVALID"
    with pytest.raises(schema.StageAError):
        schema.validate_payload(p)


@pytest.mark.parametrize(
    "group,role,field,value",
    [
        ("models", "draft", "dtype", "torch.float16"),
        ("models", "draft", "hf_device_map", {"": "cpu"}),
        ("models", "draft", "hf_device_map", []),
        ("heads", "draft_target", "stop_mode", "cumulative"),
        ("heads", "draft_target", "threshold", 0.1),
        ("heads", "draft_target", "threshold", None),
        ("heads", "draft_target", "device", "cpu"),
        ("rl", "main", "sha256", "f" * 64),
        ("rl", "main", "device", "cuda:1"),
        ("rl", "main", "action_space", "topk_thr_gamma"),
        ("rl", "main", "gamma_candidates", [2, 4]),
        ("rl", "main", "topk_candidates", []),
        ("rl", "main", "topk_candidates", [True]),
        ("rl", "main", "threshold_candidates", [1.1]),
        ("rl", "main", "threshold_candidates", ["0.1"]),
        ("rl", "main", "update_count", -1),
        ("rl", "main", "update_count", True),
    ],
)
def test_effective_resource_values_and_cross_relationships(
    tmp_path,
    group,
    role,
    field,
    value,
):
    p = payload(tmp_path, "adaptive_decoding")
    p["effective"][group][role][field] = value
    with pytest.raises(schema.StageAError):
        schema.validate_payload(p)


@pytest.mark.parametrize(
    "field",
    [
        "wall_time",
        "throughput",
        "computation_time",
        "little_computation_time",
        "draft_computation_time",
        "target_computation_time",
        "arp_overhead_time",
        "dra_overhead_time",
        "little_wall_time",
        "draft_wall_time",
        "target_wall_time",
    ],
)
@pytest.mark.parametrize("value", ["not-a-time", -1.0, None, True])
def test_identical_invalid_timing_is_invalid_before_comparison(tmp_path, field, value):
    p = payload(tmp_path)
    p["samples"][0]["metrics"][field] = value
    source = p["code"]["sha256"]
    report = compare.compare(
        p,
        copy.deepcopy(p),
        expect_samples=2,
        base_source=source,
        candidate_source=source,
        comparison="exact-replay",
    )
    assert report["verdict"] == "INVALID"
    assert any(field in error for error in report["errors"])
    with pytest.raises(schema.StageAError, match=field):
        schema.validate_payload(p)


@pytest.mark.parametrize(
    "field",
    [
        "wall_time",
        "throughput",
        "computation_time",
        "little_computation_time",
        "draft_computation_time",
        "target_computation_time",
        "arp_overhead_time",
        "dra_overhead_time",
        "loop_times",
        "connect_times",
        "accuracy",
        "draft_generated_tokens",
        "edge_cloud_bandwidth_history",
    ],
)
def test_required_metrics_cannot_disappear_on_both_sides(tmp_path, field):
    p = payload(tmp_path)
    del p["samples"][0]["metrics"][field]
    source = p["code"]["sha256"]
    report = compare.compare(
        p,
        copy.deepcopy(p),
        expect_samples=2,
        base_source=source,
        candidate_source=source,
        comparison="exact-replay",
    )
    assert report["verdict"] == "INVALID"
    with pytest.raises(schema.StageAError, match=field):
        schema.validate_payload(p)


def test_v2_comparison_is_observation_not_rng_equality(tmp_path):
    base = payload(tmp_path)
    cand = copy.deepcopy(base)
    cand["samples"][0]["gen_ids"][0] = 42
    source = base["code"]["sha256"]
    report = compare.compare(
        base, cand, expect_samples=2, base_source=source, candidate_source=source
    )
    assert (
        report["verdict"] == "OBSERVED"
        and report["observations"]
        and not report["mismatches"]
    )
    strict = compare.compare(
        base,
        cand,
        expect_samples=2,
        base_source=source,
        candidate_source=source,
        comparison="exact-replay",
    )
    assert strict["verdict"] == "FAIL"
    missing = compare.compare(
        base,
        base,
        expect_samples=2,
        base_revision="a" * 40,
        candidate_revision="a" * 40,
    )
    assert missing["verdict"] == "INVALID" and "source digests" in missing["errors"][0]
    assert (
        compare.compare(
            base,
            {"schema_version": 1},
            expect_samples=2,
            base_source=source,
            candidate_source=source,
        )["verdict"]
        == "INVALID"
    )


def test_exact_replay_rejects_changed_untracked_source(tmp_path):
    base = payload(tmp_path)
    cand = copy.deepcopy(base)
    cand["code"]["before"]["files"]["src/new.py"] = "d" * 64
    cand["code"]["after"] = copy.deepcopy(cand["code"]["before"])
    cand["code"]["sha256"] = schema.digest(cand["code"]["before"])
    report = compare.compare(
        base,
        cand,
        expect_samples=2,
        base_source=base["code"]["sha256"],
        candidate_source=cand["code"]["sha256"],
        comparison="exact-replay",
    )
    assert report["verdict"] == "INVALID" and "identical source" in report["errors"][0]


def test_explicit_stage_dispatch_preserves_legacy_default(monkeypatch):
    monkeypatch.setattr(runner, "main", lambda argv: 37 if argv == ["--help"] else 99)
    assert legacy.main(["--stage", "stage-a", "--help"]) == 37
    seen = []
    monkeypatch.setattr(legacy, "parse_args", lambda argv: seen.append(argv) or None)
    monkeypatch.setattr(
        legacy, "run", lambda args: (_ for _ in ()).throw(legacy.SmokeError("legacy"))
    )
    assert legacy.main(["--case", "legacy_static"]) == 2
    assert seen == [["--case", "legacy_static"]]


@pytest.mark.parametrize("failure", [None, "uuid", "source", "intruder"])
def test_run_wiring_with_entirely_fake_runtime(tmp_path, monkeypatch, failure):
    fixture = payload(tmp_path)
    cfg = fixture["configuration"]
    config_path, asset_path = tmp_path / "config.json", tmp_path / "assets.json"
    config_path.write_text(json.dumps(cfg))
    asset_path.write_text(json.dumps(fixture["assets"]))
    args = types.SimpleNamespace(
        authorize_gpu_execution=True,
        repo_root=str(ROOT),
        config=str(config_path),
        asset_manifest=str(asset_path),
        output=str(tmp_path / "new.json"),
    )
    events = []
    state = {"loaded": False, "collected": False, "snapshots": 0}
    original_path = list(sys.path)
    monkeypatch.setattr(runner, "check_environment", lambda: events.append("guard"))
    monkeypatch.setattr(legacy, "insert_repo_on_sys_path", lambda root: None)
    monkeypatch.setattr(runner, "query_identity", lambda _: "fake GPU")

    def pids(_):
        if state["collected"] and failure == "intruder":
            return [os.getpid(), os.getpid() + 1]
        return [os.getpid()] if state["loaded"] else []

    monkeypatch.setattr(runner, "query_pids", pids)

    def snapshot(_):
        state["snapshots"] += 1
        value = copy.deepcopy(fixture["code"]["before"])
        if failure == "source" and state["snapshots"] == 2:
            value["files"]["src/new-untracked.py"] = "d" * 64
        return value

    monkeypatch.setattr(runner, "source_snapshot", snapshot)

    class Model:
        def register_forward_hook(self, callback):
            return types.SimpleNamespace(remove=lambda: events.append("unhook"))

    class Runner:
        def __init__(self, namespace):
            events.append("construct")
            self.args = namespace
            self.draft_model, self.target_model = Model(), Model()
            self.tokenizer = types.SimpleNamespace(eos_token_id=9)

        def load_tokenizer(self):
            events.append("tokenizer")

        def load_model(self):
            events.append("model")
            state["loaded"] = True

    module, _ = fake_rl()
    torch = types.SimpleNamespace(
        __version__="fake",
        version=types.SimpleNamespace(cuda="fake"),
        save=lambda *args: pytest.fail("must not save"),
        cuda=types.SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_properties=lambda i: types.SimpleNamespace(
                uuid=UUID
                if failure != "uuid"
                else "GPU-00000000-0000-0000-0000-000000000002",
                name="fake GPU",
            ),
        ),
    )
    transformers = types.SimpleNamespace(
        __version__="fake",
        AutoTokenizer=types.SimpleNamespace(
            from_pretrained=lambda *a, **kw: types.SimpleNamespace()
        ),
    )

    def runtime():
        assert events == ["guard"]
        assert os.environ["CUDA_VISIBLE_DEVICES"] == UUID
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        return (
            torch,
            transformers,
            module,
            "fake attention",
            Runner,
            FakeGraph,
            lambda mode: types.SimpleNamespace(**schema.requirements(mode)),
        )

    monkeypatch.setattr(runner, "load_runtime", runtime)
    monkeypatch.setattr(legacy, "tokenizer_vocab_digest", lambda _: "a" * 64)
    monkeypatch.setattr(
        runner, "binding_info", lambda *a: (None, fixture["effective"]["binding"])
    )
    monkeypatch.setattr(
        runner, "consumption_info", lambda *a: fixture["effective"]["consumption"]
    )
    monkeypatch.setattr(
        runner,
        "effective_resources",
        lambda *a: {k: fixture["effective"][k] for k in ("models", "heads", "rl")},
    )

    def collect(decoder, *args):
        assert decoder.args.probe_cache_max_length == 512
        assert decoder.args.disable_rl_update is True
        events.append("collect")
        state["collected"] = True
        return fixture["samples"]

    monkeypatch.setattr(runner, "collect_samples", collect)
    if failure:
        with pytest.raises(schema.StageAError):
            runner.run(args)
        assert not Path(args.output).exists()
        if failure == "uuid":
            assert "construct" not in events
    else:
        observed = runner.run(args)
        assert json.loads(Path(args.output).read_text()) == observed
        assert events == [
            "guard",
            "construct",
            "tokenizer",
            "model",
            "collect",
            "unhook",
            "unhook",
        ]
    assert sys.path == original_path


def test_v2_cli_and_duplicate_json(tmp_path):
    p = payload(tmp_path)
    source = p["code"]["sha256"]
    path = tmp_path / "artifact.json"
    path.write_text(json.dumps(p))
    argv = [
        "--base",
        str(path),
        "--candidate",
        str(path),
        "--expect-samples",
        "2",
        "--base-source",
        source,
        "--candidate-source",
        source,
    ]
    assert compare.main(argv) == 0
    path.write_text('{"mode": "sd", "mode": "dsd"}')
    with pytest.raises(schema.StageAError, match="duplicate"):
        schema.load_json(path)
