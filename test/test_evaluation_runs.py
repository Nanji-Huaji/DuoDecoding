"""Behavior at the evaluation seam; no downloads, GPUs or model checkpoints."""

import json
import weakref
from argparse import Namespace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from eval.data import load_task_data
from eval.decoder import ModelStore, model_key
from eval.eval_gsm8k import EvalGSM8K
from src import engine
from src.register import Register


@pytest.fixture
def local_models(monkeypatch):
    class Tokenizer:
        vocab_size = 8
        pad_token_id = 0
        bos_token = eos_token = None

        def __len__(self):
            return self.vocab_size

        def encode(self, text, **kwargs):
            return [1, 2]

        def decode(self, ids, **kwargs):
            return str(int(ids[-1]))

    monkeypatch.setattr(
        engine.AutoTokenizer, "from_pretrained", lambda *a, **kw: Tokenizer()
    )
    monkeypatch.setattr(
        engine,
        "Accelerator",
        lambda: SimpleNamespace(
            device=torch.device("cpu"),
            is_main_process=True,
            wait_for_everyone=lambda: None,
        ),
    )
    model = SimpleNamespace(
        device=torch.device("cpu"),
        get_input_embeddings=lambda: SimpleNamespace(weight=torch.empty(8, 1)),
    )
    load = Mock(return_value=model)
    monkeypatch.setattr(engine, "load_causal_lm", load)
    store = ModelStore()
    try:
        yield store, load, model
    finally:
        store.close()


@pytest.fixture
def args(tmp_path):
    return Namespace(
        eval_mode="large",
        target_model="target",
        draft_model="draft",
        little_model=None,
        exp_name=str(tmp_path),
        seed=11,
        num_shots=0,
        num_samples_per_task=1,
        gamma=1,
        gamma1=1,
        gamma2=1,
        use_precise=False,
        use_stochastic_comm=False,
        use_early_stopping=False,
        ntt_ms_edge_end=1,
        ntt_ms_edge_cloud=2,
        transfer_top_k=4,
        dump_network_stats=False,
    )


def test_reused_weights_do_not_reuse_run_state(
    args, local_models, tmp_path, monkeypatch
):
    import datasets

    store, load, model = local_models
    rows = [{"question": "2+2?", "answer": "#### 4"}]
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: rows)

    # Exercise the real task, runner and model lifetime; replace only execution
    # that needs CUDA. A stale decoder would carry this per-run marker forward.
    def decode(decoder, prefix, **options):
        assert not hasattr(decoder, "was_used")
        decoder.was_used = True
        assert decoder.task == "gsm8k" and decoder.target_model is model
        token = options["transfer_top_k"]
        return torch.cat((prefix, torch.tensor([[token]])), dim=1), {
            "generated_tokens": 1,
            "wall_time": 0.5,
        }

    monkeypatch.setitem(Register._DECODING_REGISTRY, "large", decode)
    evaluator = EvalGSM8K(args, models=store)
    first = evaluator.eval()
    args.transfer_top_k = 3
    second = evaluator.eval()
    assert first["accuracy"] == 1 and second["accuracy"] == 0
    assert first["generated_tokens"] == second["generated_tokens"] == 1
    assert first["throughput"] == second["throughput"] == 2.0
    assert len((tmp_path / "large_gsm8k.jsonl").read_text().splitlines()) == 2
    assert load.call_count == 1
    # Changing how weights are actually loaded must miss the cache.
    args.model_dtype = "fp32"
    load.side_effect = OSError("missing weights")
    with pytest.raises(OSError, match="missing weights"):
        evaluator.eval()
    assert len((tmp_path / "large_gsm8k.jsonl").read_text().splitlines()) == 2
    load.side_effect = None
    evaluator.eval()
    assert load.call_count == 3


def test_specbench_counts_all_repetitions_but_not_warmup(
    args, local_models, tmp_path, monkeypatch
):
    from eval.eval_specbench import EvalSpecbench

    args.num_samples_per_task = 2
    args.sub_domain = "translation"
    args.data_path = str(tmp_path)
    (tmp_path / "translation.jsonl").write_text(
        '{"question_id": 1, "turns": ["Hallo"]}\n'
    )
    calls = []

    def decode(decoder, prefix, **options):
        # One warmup and one scored call for each repetition (scored steps 2,4).
        step = len(calls) + 1
        calls.append(decoder.task)
        return torch.cat((prefix, torch.ones(1, step, dtype=torch.long)), dim=1), {
            "generated_tokens": step,
            "wall_time": 0.5,
        }

    monkeypatch.setitem(Register._DECODING_REGISTRY, "large", decode)
    result = EvalSpecbench(args, models=local_models[0]).eval()
    assert calls == ["specbench"] * 4
    assert result["num_samples"] == 2
    assert result["generated_tokens"] == 6
    assert result["mean_generated_tokens"] == 3
    assert (
        len((tmp_path / "large_specbench_translation.jsonl").read_text().splitlines())
        == 2
    )


def test_load_failures_cannot_become_successful_empty_runs(tmp_path, monkeypatch):
    import datasets

    def broken(*args, **kwargs):
        raise OSError("offline")

    monkeypatch.setattr(datasets, "load_dataset", broken)
    args = Namespace(data_path=tmp_path)
    with pytest.raises(RuntimeError, match="gsm8k.*offline") as error:
        load_task_data("gsm8k", args)
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(FileNotFoundError):
        load_task_data("mt_bench", args)
    (tmp_path / "mt_bench.jsonl").touch()
    with pytest.raises(ValueError, match="no evaluation samples"):
        load_task_data("mt_bench", args)


def test_model_switch_releases_old_weights_before_loading(
    args, local_models, monkeypatch
):
    store, load, _ = local_models
    refs = []

    class Model(torch.nn.Embedding):
        def get_input_embeddings(self):
            return self

    def allocate(*_args, **_kwargs):
        assert not refs or refs[-1]() is None, "loading two weight sets simultaneously"
        model = Model(8, 1)
        refs.append(weakref.ref(model))
        return model

    # The distributed process group belongs to the caller, not to either run.
    monkeypatch.setenv("RANK", "0")
    monkeypatch.setattr(engine.dist, "is_initialized", lambda: True)
    for function in ["init_process_group", "destroy_process_group"]:
        monkeypatch.setattr(
            engine.dist,
            function,
            lambda *a, **kw: pytest.fail("unexpected process group change"),
        )
    load.side_effect = allocate
    store.new_decoder(args, "gsm8k")
    assert refs[-1]() is not None
    args.target_model = "other"
    store.new_decoder(args, "humaneval")
    assert load.call_count == 2


def test_grouping_depends_on_used_weights_not_the_benchmark():
    common = dict(eval_mode="dist_spec", draft_model="draft", target_model="target")
    assert model_key(common) == model_key(
        dict(
            common,
            eval_mode="dist_split_spec",
            little_model="unused",
            gamma=17,
            eval_dataset="eval/eval_mt_bench_noeval.py",
        )
    )
    assert model_key(common) != model_key(dict(common, target_model="other"))
    assert model_key(common) != model_key(dict(common, target_quantization="4bit"))
    assert model_key(common) != model_key(dict(common, eval_mode="adaptive_decoding"))


def test_matrix_groups_compatible_jobs_and_recovers_from_worker_death(
    tmp_path, monkeypatch
):
    import exp
    from src.experiments import scheduler, workers

    launches = []

    class Worker:
        alive = True

        def __init__(self, devices):
            assert devices == "0"
            self.jobs = []
            launches.append(self.jobs)

        def run(self, config, log_dir):
            self.jobs.append(config["exp_name"])
            if config["exp_name"] == "crash":
                self.alive = False
            status = "failed" if config["exp_name"] in {"crash", "bad"} else "success"
            return {
                "status": status,
                "exp_name": config["exp_name"],
                "result": {},
                "log_file": "",
            }

        def close(self):
            pass

    monkeypatch.setattr(scheduler, "get_available_gpus", lambda: [0])
    monkeypatch.setattr(workers, "_WorkerHandle", Worker)
    base = dict(
        eval_mode="large",
        little_model="unused-70b",
        target_model="one",
        draft_model="tiny",
    )
    configs = [
        dict(base, exp_name="a"),
        dict(base, exp_name="big", target_model="70b"),
        dict(base, exp_name="b", target_model="two"),
        dict(base, exp_name="bad", eval_mode="unknown"),
        dict(base, exp_name="c", gamma=3),
        dict(base, exp_name="crash"),
        dict(base, exp_name="d"),
    ]
    summary = tmp_path / "matrix.json"
    results = exp.run_experiments_parallel(
        configs, max_workers=1, summary_file=str(summary)
    )
    assert launches == [["a", "c", "crash"], ["d"], ["b"], ["bad"]]
    assert {r["exp_name"] for r in results if r["status"] == "failed"} == {
        "crash",
        "bad",
    }
    assert len(results) == len(configs)
    skipped = next(r for r in results if r["status"] == "skipped_capacity_mismatch")
    # Skips must still be identifiable for a retry/merge.
    assert skipped["config"]["target_model"] == "70b"
    assert all("CUDA_VISIBLE_DEVICES" not in c for c in configs)
    assert json.loads(summary.read_text()) == results


def test_standalone_runs_keep_only_one_worker_per_gpu(monkeypatch):
    from src.experiments import workers

    spawn = Mock(side_effect=lambda devices: Mock(alive=True))
    monkeypatch.setattr(workers, "_WorkerHandle", spawn)
    pool = workers._WorkerPool()
    with pool.acquire({"CUDA_VISIBLE_DEVICES": "0,1", "target_model": "A"}) as first:
        pass
    with pool.acquire({"CUDA_VISIBLE_DEVICES": "0,1", "target_model": "B"}) as second:
        assert (
            second is first
        )  # The worker swaps weights; do not spawn on the same cards.
    with pool.acquire({"CUDA_VISIBLE_DEVICES": "1"}) as other:
        first.close.assert_called_once_with()
        assert other is not first
    pool.shutdown()
    other.close.assert_called_once_with()
    assert spawn.call_count == 2


def test_worker_reports_this_runs_failure_not_an_old_summary(
    args, local_models, tmp_path, monkeypatch
):
    import datasets

    from eval.worker import _run_job
    from src import utils

    monkeypatch.setattr(utils, "parse_arguments", lambda argv: args)
    config = {
        "eval_mode": "large",
        "exp_name": "retry",
        "eval_dataset": "eval/eval_gsm8k.py",
    }
    stale = tmp_path / "large_gsm8k_metrics.json"
    stale.write_text('{"accuracy":1}')
    monkeypatch.setattr(datasets, "load_dataset", lambda *a, **kw: [])
    result = _run_job({"config": config}, local_models[0], tmp_path)
    assert result["status"] == "failed" and result["exp_name"] == "retry"
    assert "no evaluation samples" in result["result"]["error"]
    assert json.loads(stale.read_text()) == {"accuracy": 1}
    assert not (tmp_path / "large_gsm8k.jsonl").exists()
