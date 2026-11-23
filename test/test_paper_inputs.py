"""Paper preflight: tiny files only. No worker/model/checkpoint deserialization."""

import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
from types import SimpleNamespace

import pytest

from eval.data import load_task_data
from eval.utils import select_eval_data
from src.experiments import paper_inputs as lock


def _put(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(data, encoding="utf-8")
    return path


def _plan(n, **other):
    return dict(
        eval_data_num=n, sample_seed=1234, random_sample=True, num_shots=3, **other
    )


def test_source_hashes_worktree_and_submodule_python(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    _put(tmp_path / "dirty.py", b"one")
    _put(tmp_path / "new.py", b"")
    _put(tmp_path / "sub/.git", b"gitlink")
    _put(tmp_path / "sub/arp.py", b"unchanged")

    def git(cwd, *args):
        if args[0] == "rev-parse":
            return "12ab\n" if cwd == tmp_path else "de34\n"
        if args[0] == "status":
            return " M dirty.py\n?? new.py\n"
        if "--stage" in args:
            return "160000 de34 0\tsub\0"
        assert "--others" in args and "--exclude-standard" in args
        return "dirty.py\0deleted.py\0new.py\0" if cwd == tmp_path else "arp.py\0"

    monkeypatch.setattr(lock, "_git", git)
    first = lock._source()
    assert first["dirty"] and first["head"] == "12ab"
    assert set(first["files"]) == {"dirty.py", "deleted.py", "new.py", "sub/arp.py"}
    assert first["files"]["deleted.py"] is None
    assert first["submodules"]["sub"] == {"gitlink": "de34", "head": "de34"}
    _put(tmp_path / "dirty.py", b"two")  # same size, same Git status
    assert lock._source()["python_sha256"] != first["python_sha256"]
    _put(tmp_path / "dirty.py", b"one")
    _put(tmp_path / "sub/arp.py", b"changed")
    assert lock._source()["python_sha256"] != first["python_sha256"]


def test_public_freeze_mt_preserves_evaluation_order_and_turns(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    monkeypatch.setattr(lock, "_source", lambda: {"head": "test", "dirty": True})
    monkeypatch.setattr(lock, "_dependencies", lambda tasks: {"python": sys.version})
    target = "local/test-target"
    _put(tmp_path / target / "config.json", '{"model_type":"llama","vocab_size":20}')
    _put(tmp_path / target / "model.safetensors", b"tiny weights fixture")
    _put(tmp_path / target / "tokenizer.json", "{}")
    rows = [
        {
            "question_id": i + 81,
            "category": "test",
            "turns": [f"{i} first", f"{i} follow-up"],
        }
        for i in range(80)
    ]
    _put(tmp_path / "data/mt_bench.jsonl", "".join(json.dumps(r) + "\n" for r in rows))
    c = dict(
        **_plan(80),
        run_full_dataset=False,
        eval_dataset="eval/eval_mt_bench_noeval.py",
        eval_mode="large",
        model_source="local",
        use_rl_adapter=False,
        draft_model="llama-68m",
        little_model="llama-68m",
        target_model=target,
    )
    rng = random.getstate()
    frozen = lock.freeze_inputs(
        [c, {**c, "CUDA_VISIBLE_DEVICES": "99", "exp_name": "not-an-input"}]
    )
    assert random.getstate() == rng
    assert json.loads(json.dumps(frozen)) == frozen
    assert frozen["worker_env"]["HF_DATASETS_OFFLINE"] == "1"
    assert list(frozen["models"]) == [target]
    assert (
        frozen["checkpoints"]["files"] == {} and frozen["network"]["bandwidth"] is None
    )
    mt = frozen["samples"]["mt_bench_noeval"]
    actual = load_task_data(
        "mt_bench", SimpleNamespace(**c, data_path=tmp_path / "data")
    )
    assert [(r["question_id"], r["turns"]) for r in mt["rows"]] == [
        (r["question_id"], r["turns"]) for r in actual
    ]
    assert mt["count"] == 80 and mt["total_turns"] == 160 and mt["effective_shots"] == 0
    assert mt == lock.freeze_inputs([c])["samples"]["mt_bench_noeval"]
    assert "CUDA_VISIBLE_DEVICES" not in c


@pytest.mark.parametrize(
    "task,field,count", [("humaneval", "task_id", 80), ("cnndm", "id", 500)]
)
def test_hf_task_ids_in_worker_order(task, field, count):
    rows = [
        {
            field: f"{task}-{i}",
            "article": "article",
            "highlights": "reference",
            "prompt": "prompt",
            "test": "test code",
            "entry_point": "fn",
        }
        for i in range(count + 3)
    ]
    plan = _plan(count)
    manifest = lock._sample(task, plan, rows, {})
    assert manifest["count"] == count and manifest["effective_shots"] == 3
    assert [r[field] for r in manifest["rows"]] == [
        r[field] for r in select_eval_data(rows, SimpleNamespace(**plan))
    ]
    assert manifest == lock._sample(task, plan, rows, {})


def test_no_demonstration_or_short_datasets():
    with pytest.raises(ValueError, match="empty"):
        lock.freeze_inputs([])
    c = {**_plan(80), "eval_dataset": "eval/eval_gsm8k.py", "run_full_dataset": False}
    with pytest.raises(ValueError, match="sample count"):
        lock.freeze_inputs([{**c, "eval_data_num": 2}])
    with pytest.raises(ValueError, match="selection"):
        lock.freeze_inputs([{**c, "random_sample": False}])
    with pytest.raises(ValueError, match="differs"):
        lock.freeze_inputs([c, {**c, "eval_data_num": 500}])
    with pytest.raises(ValueError, match="500 test samples, only 499"):
        lock._sample("cnndm", _plan(500), [{}] * 499, {})
    with pytest.raises(ValueError, match="duplicate"):
        lock._sample("gsm8k", _plan(2), [{"question": "Q", "answer": "A"}] * 2, {})
    with pytest.raises(ValueError, match="turns"):
        lock._sample(
            "mt_bench_noeval", _plan(1), [{"question_id": 1, "turns": ["one only"]}], {}
        )


def test_only_loaded_checkpoints_and_exact_pair_v3_hashes(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    c = dict(
        eval_mode="adaptive_tridecoding",
        use_rl_adapter=True,
        disable_rl_update=True,
        little_model="llama-68m",
        draft_model="tiny-llama-1.1b",
        target_model="llama-2-13b",
    )
    entries = {}
    for role, pair, attr in [
        ("main", "tiny-llama-1.1b--to--llama-2-13b", "draft_target_acc_head_path"),
        ("little", "llama-68m--to--tiny-llama-1.1b", "small_draft_acc_head_path"),
    ]:
        c[attr] = "heads/" + pair
        latest = f"agents/{role}/{pair}/latest.pth"
        c[role + "_rl_path"] = c[role + "_rl_best_path"] = latest
        for path in (
            latest,
            latest + ".buffer",
            c[attr] + "/config.json",
            c[attr] + "/model.safetensors",
        ):
            p = _put(tmp_path / path, path)
            policy = path.startswith("agents/")
            e = dict(
                status="production",
                role="rl_" + role if policy else "acc_head",
                pair=pair,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
            )
            entries["production/" + path] = e
            if policy:
                entries[
                    f"checkpoints/rl_fluid_retrain_v3/llama/{role}/{pair}/{p.name}"
                ] = dict(e, status="archive")
    manifest = _put(
        tmp_path / "checkpoints/manifest.json", json.dumps({"entries": entries})
    )
    all_files = lock._checkpoints([c, c])["files"]
    assert len(all_files) == 8
    assert all_files[c["main_rl_path"]]["v3_manifest_entry"].startswith(
        "checkpoints/rl_fluid_retrain_v3/"
    )
    extension = lock._checkpoints(
        [{**c, "eval_mode": "cee_cuhlm", "use_rl_adapter": False}]
    )
    assert len(extension["files"]) == 4 and {
        r["role"] for r in extension["files"].values()
    } == {"acc_head"}
    assert lock._checkpoints([{**c, "eval_mode": "tridecoding"}])["files"] == {}
    with pytest.raises(ValueError, match="best"):
        lock._checkpoints([{**c, "main_rl_best_path": "another/best.pth"}])
    with pytest.raises(ValueError, match="frozen"):
        lock._checkpoints([{**c, "disable_rl_update": False}])
    # Even a real production hash belonging to another pair/role must not pass.
    with pytest.raises(ValueError, match="production manifest"):
        lock._checkpoints(
            [{**c, "draft_target_acc_head_path": c["small_draft_acc_head_path"]}]
        )
    saved = manifest.read_bytes()
    ref = all_files[c["main_rl_path"]]["v3_manifest_entry"]
    entries[ref]["sha256"] = "ab" * 32
    manifest.write_text(json.dumps({"entries": entries}))
    with pytest.raises(ValueError, match="fluid-v3"):
        lock._checkpoints([c])
    manifest.write_bytes(saved)
    _put(tmp_path / (c["main_rl_path"] + ".buffer"), b"drift")
    with pytest.raises(ValueError, match=r"latest.pth.buffer.*sha256"):
        lock._checkpoints([c])


def test_strict_model_path_and_indexed_shards(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    alias = "Qwen/local-fixture"
    real = tmp_path / "models" / "Qwen" / "local-fixture"
    _put(real / "config.json", '{"model_type":"qwen","vocab_size":20}')
    _put(real / "tokenizer.json", "{}")
    _put(
        real / "model.safetensors.index.json",
        json.dumps({"weight_map": {"x": "s1", "y": "s2"}}),
    )
    _put(real / "s1", b"first")
    with pytest.raises(ValueError, match="not available locally"):
        lock._model(alias, "local", {"target"})  # similar models/ dir != worker path
    (tmp_path / "Qwen").symlink_to(real.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="missing model shard s2"):
        lock._model(alias, "local", {"target"})
    _put(real / "s2", b"second")
    _put(
        real / ".cache/huggingface/download/s1.metadata",
        "f" * 40 + "\n" + "a" * 64 + "\n1.0",
    )
    info = lock._model(alias, "local", {"target"})
    assert info["weights_metadata_only"]["s1"]["download_commit"] == "f" * 40
    assert info["weights_metadata_only"]["s1"]["download_etag"] == "a" * 64
    assert info["loader_argument"] == alias and info["resolved"] == str(real)
    assert not info["weights_hashed"] and sorted(info["weights_metadata_only"]) == [
        "s1",
        "s2",
    ]
    assert (
        info["metadata_sha256"]["tokenizer.json"] == hashlib.sha256(b"{}").hexdigest()
    )


def test_large_qwen_needs_draft_vocab_but_not_draft_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    for alias in ("Qwen/draft", "Qwen/target"):
        _put(tmp_path / alias / "config.json", '{"model_type":"qwen","vocab_size":30}')
    _put(tmp_path / "Qwen/target/model.safetensors", b"tiny")
    _put(tmp_path / "Qwen/target/tokenizer.json", "{}")
    models = lock._models(
        [
            dict(
                eval_mode="large",
                draft_model="Qwen/draft",
                target_model="Qwen/target",
                model_source="local",
            )
        ]
    )
    assert models["Qwen/draft"]["roles"] == ["draft_vocab"]
    assert models["Qwen/draft"]["weights_metadata_only"] == {}
    assert "config.json" in models["Qwen/draft"]["metadata_sha256"]


def test_dependencies_fail_on_missing_quality_metric(monkeypatch):
    def dist(name):
        return SimpleNamespace(metadata={"Name": name}, version="1.0")

    base = [
        dist(n)
        for n in (
            "torch",
            "transformers",
            "accelerate",
            "numpy",
            "safetensors",
            "tokenizers",
            "tqdm",
            "shortuuid",
            "fschat",
            "datasets",
            "pyarrow",
        )
    ]
    monkeypatch.setattr(lock.metadata, "distributions", lambda: base)
    assert lock._dependencies(["gsm8k"])["packages"]["datasets"] == "1.0"
    with pytest.raises(ValueError, match="rouge-score"):
        lock._dependencies(["gsm8k", "cnndm"])


def test_network_uses_actual_runs_and_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setattr(lock, "ROOT", tmp_path)
    text = "Run 1\n 6,8\n###############################\nRun 3\n 42,50\n"
    p = _put(tmp_path / lock._STATIC_TRACE, text)
    _put(tmp_path / "rtt", "Run 3\n 51,49\n")
    c = dict(
        eval_mode="dist_spec",
        use_precise=False,
        use_stochastic_comm=True,
        edge_cloud_bandwidth=46,
        ntt_trace_file="rtt",
        ntt_trace_scale=0.0,
    )
    n = lock._network([c, {**c, "edge_cloud_bandwidth": 5}])
    runs = n["bandwidth"]["runs_by_target_mbps"]
    assert (runs["46"]["run_id"], runs["5"]["run_id"]) == (3, 1)
    assert (n["rtt_replays"][0]["run_id"], n["rtt_replays"][0]["scale"]) == (3, 0.0)
    p.write_text("Run 1\n 0,0\n")
    with pytest.raises(ValueError, match="No usable bandwidth run"):
        lock._network([c])


@pytest.mark.skipif(
    importlib.util.find_spec("datasets") is None, reason="needs local datasets package"
)
def test_hf_cache_reader_matches_offline_evaluator_without_cache_writes(
    tmp_path, monkeypatch
):
    # Two eligible local snapshots. Actual HF offline resolution uses newest
    # directory mtime, not lexicographically highest revision. Compare against
    # the evaluator's real data loader on this disposable cache (no evaluator).
    pa = pytest.importorskip("pyarrow")
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "cache"))
    plan = _plan(80)
    for key, stamp in [("f" * 40, 1), ("0" * 40, 2)]:
        folder = tmp_path / "cache/gsm8k/main/0.0.0" / key
        info = {
            "config_name": "main",
            "dataset_name": "gsm8k",
            "version": "0.0.0",
            "splits": {
                "test": {"name": "test", "num_examples": 83, "dataset_name": "gsm8k"}
            },
        }
        _put(folder / "dataset_info.json", json.dumps(info))
        table = pa.table(
            {"question": [f"{key}-{i}" for i in range(83)], "answer": ["#### 0"] * 83}
        )
        with (
            pa.OSFile(str(folder / "gsm8k-test.arrow"), "wb") as f,
            pa.ipc.new_stream(f, table.schema) as writer,
        ):
            writer.write_table(table)
        os.utime(folder, (stamp, stamp))
    old_files = set(tmp_path.rglob("*"))
    frozen = lock._samples({"gsm8k": plan})["gsm8k"]
    assert set(tmp_path.rglob("*")) == old_files  # no .lock, new arrow, cache, etc.
    assert frozen["source"]["cache_key"] == "0" * 40
    env = {
        **os.environ,
        **lock._OFFLINE,
        "USE_TORCH": "0",
        "USE_TF": "0",
        "CUDA_VISIBLE_DEVICES": "",
    }
    code = """
import json, sys
from types import SimpleNamespace
from eval.data import load_task_data
rows = load_task_data('gsm8k', SimpleNamespace(**json.load(sys.stdin)))
print(json.dumps([row['question'] for row in rows]))
"""
    actual = subprocess.run(
        [sys.executable, "-B", "-c", code],
        env=env,
        cwd=lock.ROOT,
        input=json.dumps(plan),
        text=True,
        check=True,
        capture_output=True,
    )
    expected = [
        hashlib.sha256(q.encode()).hexdigest() for q in json.loads(actual.stdout)
    ]
    assert [r["question_sha256"] for r in frozen["rows"]] == expected
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "absent"))
    with pytest.raises(ValueError, match="no download or substitution"):
        lock._samples({"gsm8k": plan})
    assert not (tmp_path / "absent").exists()
