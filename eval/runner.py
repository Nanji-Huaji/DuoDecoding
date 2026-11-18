"""Run one task with fresh state. Only ModelStore outlives an evaluation."""

import json
import random
import time
from contextlib import ExitStack
from copy import deepcopy
from pathlib import Path

import torch

from eval.decoder import ModelStore
from src.metrics import accumulate_metrics, get_empty_metrics
from src.utils import seed_everything


class EvaluationRun:
    """The measured decode and result-writing operations available to a task."""

    def __init__(self, decoder, record_path, files, *, collect_metrics=True):
        self.decoder = decoder
        self.args = decoder.args
        self.tokenizer = decoder.tokenizer
        self.device = decoder.accelerator.device
        self.is_main_process = decoder.accelerator.is_main_process
        self.metrics = get_empty_metrics()
        self._collect_metrics = collect_metrics
        self._decode = decoder.get_decoding_method()
        self._seeds = set()
        self._record_path = record_path
        self._files = files
        self._stream = None
        self.measured_decodes = 0

    def next_seed(self):
        seed = self.args.seed
        while seed in self._seeds:
            seed = random.randint(0, 1_000_000)
        self._seeds.add(seed)
        seed_everything(seed)
        return seed

    @torch.no_grad()
    def generate(self, input_ids, *, record=True, **overrides):
        # Mixed curricula change these values between examples, not between models.
        args = self.args
        options = dict(
            transfer_top_k=args.transfer_top_k,
            use_precise_comm_sim=args.use_precise,
            use_stochastic_comm=args.use_stochastic_comm,
            ntt_ms_edge_cloud=args.ntt_ms_edge_cloud,
            ntt_ms_edge_end=args.ntt_ms_edge_end,
            use_early_stopping=args.use_early_stopping,
        )
        options.update(overrides)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        start = time.perf_counter()
        result = self._decode(input_ids, **options)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - start
        ids, metrics = result[:2] if isinstance(result, tuple) else (result, {})
        if record:
            if self._collect_metrics:
                accumulate_metrics(self.metrics, metrics)
            self.measured_decodes += 1
        return ids, metrics, elapsed

    def write(self, row):
        if not self.is_main_process:
            return
        if self._stream is None:
            self._record_path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = self._files.enter_context(
                self._record_path.open("a", encoding="utf-8")
            )
        self._stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._stream.flush()


class Evaluation:
    """CLI-compatible entry point; reuse a ModelStore across config objects.

    A task supplies ``name``, ``load_data()`` and ``run(run, data)``. The task,
    its args, the decoder and all counters are new on every call to ``eval``;
    neither an earlier result file nor a previous decoder seeds this run.
    """

    def __init__(self, args, models=None):
        self.args = args
        self.models = models if models is not None else ModelStore()

    def eval(self, total=None):
        args = deepcopy(self.args)
        decoder = self.models.new_decoder(args, self.task_type.name)
        task = self.task_type(args, decoder.tokenizer)
        data = task.load_data()
        if total is not None and isinstance(data, list):
            data = data[:total]
        if not data:
            raise ValueError(f"{task.name}: no samples to evaluate")

        # A cold model load may consume RNG that a cache hit doesn't. Start
        # sampling from the run's seed in either case, after all initialization.
        seed_everything(args.seed)
        directory = Path(args.exp_name)
        stem = getattr(task, "record_name", task.name)
        save_metrics = getattr(task, "save_metrics", True)
        with ExitStack() as files:
            run = EvaluationRun(
                decoder,
                directory / f"{args.eval_mode}_{stem}.jsonl",
                files,
                collect_metrics=save_metrics,
            )
            extra = task.run(run, data) or {}
        decoder.accelerator.wait_for_everyone()
        if not run.measured_decodes:
            raise RuntimeError(
                f"{task.name}: no sample decoded successfully; no metrics written"
            )
        if not run.is_main_process or not save_metrics:
            return extra

        metrics = run.metrics
        if metrics["wall_time"]:
            metrics["throughput"] = metrics["generated_tokens"] / metrics["wall_time"]
        print(decoder.metrics_dumper.get_printable_metrics(metrics))
        summary = decoder.metrics_dumper.get_save_dict(metrics)
        summary.update(extra)
        path = directory / f"{args.eval_mode}_{task.name}_metrics.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=4)
        print(f"Decoding metrics saved to {path}")
        return summary
