"""One GPU worker: read one config per JSON line, return its own run summary.

The parent sets CUDA_VISIBLE_DEVICES before launch. Output from each evaluation
(including native libraries) goes to its log, never to the response pipe.
"""

import argparse
import contextlib
import importlib
import json
import os
import shlex
import sys
import traceback
from datetime import datetime
from pathlib import Path

# Import only the chosen task (e.g. no judge dependency for a noeval run).
_EVALUATORS = {
    "eval_gsm8k": "EvalGSM8K",
    "eval_humaneval": "EvalHumaneval",
    "eval_cnndm": "EvalCNNDM",
    "eval_xsum": "EvalXSum",
    "eval_mt_bench": "EvalMTBench",
    "eval_mt_bench_noeval": "EvalMTBench",
    "eval_specbench": "EvalSpecbench",
    "eval_mixed": "EvalMixed",
}


def build_argv(config: dict) -> list[str]:
    """Use exactly the CLI parser's protocol, defaults, and alias resolution."""
    argv = [
        "--protocol",
        config.get("protocol", "paper_table5"),
        "--comm_accounting",
        "honest",
        "--temp",
        "0.0",
    ]
    # The remaining ExpConfig fields are CLI options. Placement is the worker's
    # environment; the dataset chooses the entry point, not a parser option.
    for key, value in config.items():
        if key in {"protocol", "eval_dataset", "CUDA_VISIBLE_DEVICES"}:
            continue
        if key == "eval_data_num" and value is None:
            if not config.get("run_full_dataset"):
                argv.append("--run_full_dataset")
        elif isinstance(value, bool):
            if value:
                argv.append(f"--{key}")
        elif value not in (None, ""):
            argv.extend([f"--{key}", str(getattr(value, "value", value))])
    return argv


@contextlib.contextmanager
def _job_output(path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as log:
        sys.stdout.flush()
        sys.stderr.flush()
        saved_out, saved_err = os.dup(1), os.dup(2)
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(saved_out, 1)
            os.dup2(saved_err, 2)
            os.close(saved_out)
            os.close(saved_err)


def _run_job(job, models, log_dir):
    config = job["config"]
    name = config["exp_name"]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    log_file = job.get("log_file") or str(
        Path(log_dir) / f"{name.replace('/', '_')}_{stamp}.log"
    )
    with _job_output(log_file):
        # gamma_sweep recovers configs from this literal log header.
        print(f"实验配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
        try:
            script = config["eval_dataset"]
            argv = build_argv(config)
            print(f"执行命令: {shlex.join([sys.executable, script, *argv])}")
            print(
                f"[eval.worker] pid={os.getpid()} "
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            module = Path(script).stem
            evaluator = _EVALUATORS[module]
            from src.utils import parse_arguments

            args = parse_arguments(argv)
            cls = getattr(importlib.import_module(f"eval.{module}"), evaluator)
            result = cls(args, models=models).eval()
            status = "success"
        except (Exception, SystemExit) as exc:
            traceback.print_exc()
            status, result = "failed", {"error": f"{type(exc).__name__}: {exc}"}
    return {"exp_name": name, "result": result, "log_file": log_file, "status": status}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", default="exp_logs")
    opts = parser.parse_args(argv)
    with os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8") as responses:
        # Outside a job there may also be import/destructor/atexit prints.
        os.dup2(2, 1)
        from eval.decoder import ModelStore

        models = ModelStore()
        try:
            for line in sys.stdin:
                if not line.strip():
                    continue
                job = {}
                try:
                    job = json.loads(line)
                    result = _run_job(job, models, opts.log_dir)
                except Exception as exc:
                    cfg = job.get("config", {}) if isinstance(job, dict) else {}
                    result = {
                        "exp_name": cfg.get("exp_name", "<invalid-job>")
                        if isinstance(cfg, dict)
                        else "<invalid-job>",
                        "log_file": "",
                        "status": "failed",
                        "result": {"error": str(exc)},
                    }
                print(
                    json.dumps(result, ensure_ascii=False), file=responses, flush=True
                )
        finally:
            models.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
