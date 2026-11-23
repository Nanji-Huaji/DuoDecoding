"""Preview a paper stage, or run it on explicitly assigned physical GPU groups.

Use reserved GPUs: the idle check is a launch-time snapshot, not a cluster lock.
A run-id is single use (also for pilots). Before combining different runs,
compare manifest inputs: source, runtime, weights, network and shared samples.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from itertools import cycle
from pathlib import Path

from . import matrix
from .summaries import save_summary
from .workers import _WorkerPool

ROOT = Path(__file__).resolve().parents[2]
_FAMILIES = {
    matrix.llama_series[-1]: "llama",
    matrix.qwen_1_5_series[-1]: "qwen15",
    matrix.qwen_series[-1]: "qwen3",
}
_GPU_UUID = re.compile(
    r"GPU-[\da-fA-F]{8}(?:-[\da-fA-F]{4}){3}-[\da-fA-F]{12}", re.ASCII
)


def _csv(values):
    return [s.strip() for value in values or () for s in value.split(",")]


def _gpu_groups(values):
    """Each --gpus is one worker's ordered physical device list."""
    groups = []
    seen = set()
    for value in values or ():
        ids = _csv(value)
        if not ids or not all(re.fullmatch("[0-9]+", x) for x in ids):
            raise ValueError(
                "--gpus requires physical nvidia-smi indices, e.g. 0 or 0,2"
            )
        group = tuple(map(int, ids))
        if len(set(group)) != len(group) or not seen.isdisjoint(group):
            raise ValueError("--gpus indices must be distinct within and across groups")
        groups.append(group)
        seen.update(group)
    return groups


def _task(config):
    return Path(config["eval_dataset"]).stem.removeprefix("eval_")


def _cell_id(config):
    # exp_name contains a fresh timestamp; placement is not a scientific axis.
    semantic = {
        k: v for k, v in config.items() if k not in ("exp_name", "CUDA_VISIBLE_DEVICES")
    }
    data = json.dumps(semantic, sort_keys=True, separators=(",", ":"), allow_nan=False)
    tag = hashlib.sha256(data.encode()).hexdigest()[:16]
    return (
        f"{_FAMILIES[config['target_model']]}.{_task(config)}.{config['eval_mode']}."
        f"bw{config['edge_cloud_bandwidth']:g}."
        f"g{config['gamma']}-{config['gamma1']}-{config['gamma2']}.{tag}"
    )


def _select(configs, include, exclude):
    cells = [(_cell_id(c), c) for c in configs]
    # The short digest is also the artifact directory name; never silently alias it.
    if len({key.rsplit(".", 1)[1] for key, _ in cells}) != len(cells):
        raise ValueError("duplicate cell configuration or hash collision in stage")
    requested = include or exclude
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate ID in cell selection")
    unknown = set(requested).difference(key for key, _ in cells)
    if unknown:
        raise ValueError(f"unknown cell IDs for this stage: {sorted(unknown)}")
    selected = [
        (k, c) for k, c in cells if (not include or k in include) and k not in exclude
    ]
    if not selected:
        raise ValueError("empty cell selection")
    return selected


def _visible_cuda_uuids():
    """Resolve an inherited ordinal mask in CUDA's order, without a CUDA context."""
    import ctypes as ct
    from uuid import UUID

    driver = ct.CDLL("libcuda.so.1")

    def check(symbol, *args):
        func = getattr(driver, symbol, None)
        if func is None or func(*args) != 0:
            raise RuntimeError(f"cannot verify inherited CUDA mask via {symbol}")

    check("cuInit", 0)
    count = ct.c_int()
    check("cuDeviceGetCount", ct.byref(count))
    visible = set()
    for ordinal in range(count.value):
        device, raw = ct.c_int(), (ct.c_ubyte * 16)()
        check("cuDeviceGet", ct.byref(device), ordinal)
        check("cuDeviceGetUuid", ct.byref(raw), device)
        visible.add(f"GPU-{UUID(bytes=bytes(raw))}")
    return visible


def _check_masks(indices):
    # These are physical indices, never indices into a remapped device list.
    # Unknown mask syntax (including MIG, -1, empty or abbreviated UUIDs) is not
    # a reason to widen a reservation. Full UUID masks are checked after lookup.
    uuid_masks = []
    numeric_cuda = False
    for var in ("CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        value = os.environ.get(var)
        if value is None or (var == "NVIDIA_VISIBLE_DEVICES" and value == "all"):
            continue
        mask = _csv([value])
        if all(re.fullmatch("[0-9]+", item) for item in mask):
            numbers = set(map(int, mask))
            if len(numbers) == len(mask) and set(indices) <= numbers:
                numeric_cuda |= var == "CUDA_VISIBLE_DEVICES"
                continue
        elif all(_GPU_UUID.fullmatch(item) for item in mask):
            if len({item.lower() for item in mask}) == len(mask):
                uuid_masks.append((var, set(mask)))
                continue
        raise ValueError(f"--gpus not permitted by {var}={value!r}")
    if numeric_cuda:
        uuid_masks.append(
            ("CUDA_VISIBLE_DEVICES (driver order)", _visible_cuda_uuids())
        )
    return uuid_masks


def _smi(index, query):
    try:
        return subprocess.run(
            ["nvidia-smi", "-i", str(index), query, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=15,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"cannot verify physical GPU {index} via nvidia-smi: {exc}"
        ) from exc


def _hardware(indices, uuid_masks):
    cards = []
    seen_uuids = set()
    for index in indices:
        row = _smi(
            index,
            "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu,driver_version",
        )
        fields = _csv([row])
        if len(fields) != 7 or fields[0] != str(index):
            raise ValueError(f"unverified GPU {index}: {row!r}")
        _, name, uuid, total, mem, util, driver = fields
        if not name or not _GPU_UUID.fullmatch(uuid):
            raise ValueError(
                f"GPU {index}: expected a model name and full UUID, got {row!r}"
            )
        if uuid.lower() in seen_uuids:
            raise ValueError(f"multiple physical indices resolved to GPU {uuid}")
        seen_uuids.add(uuid.lower())
        for var, allowed in uuid_masks:
            if uuid not in allowed:
                raise ValueError(f"GPU {index} ({uuid}) not permitted by {var}")
        if not (0 <= int(mem) < 1024 and 0 <= int(util) < 5):
            raise ValueError(
                f"GPU {index} is not idle: {mem} MiB used, {util}% utilization"
            )
        if _smi(index, "--query-compute-apps=pid"):
            raise ValueError(
                f"GPU {index} is not idle or compute process state is unknown"
            )
        cards.append(
            dict(
                index=index,
                name=name,
                uuid=uuid,
                # Capacity is provenance, not an admission or model-placement policy.
                memory_total_mib=int(total) if total.isdecimal() else None,
                memory_used_mib=int(mem),
                utilization_percent=int(util),
                driver=driver,
            )
        )
    return cards


def _freeze_inputs(configs):
    # A normal preview must not import dataset/weight loaders or probe hardware.
    from .paper_inputs import freeze_inputs

    return freeze_inputs(configs)


@contextmanager
def _worker_env(settings):
    # _WorkerHandle copies this at Popen. Use a fresh pool per invocation; never
    # borrow a worker started with someone else's GPU mask or hub/cache settings.
    previous = {k: os.environ.get(k) for k in settings}
    os.environ.update(settings)
    try:
        yield
    finally:
        for k, v in previous.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _verify_samples(config, samples):
    task = _task(config)
    expected = samples[task]["rows"]
    name = f"{config['eval_mode']}_{task.removesuffix('_noeval')}.jsonl"
    with (Path(config["exp_name"]) / name).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    if len(rows) != config["eval_data_num"] or len(rows) != len(expected):
        raise ValueError(
            f"{name}: expected {len(expected)} frozen rows, found {len(rows)}"
        )
    # All executable paper stages have one repetition. For MT, answers need not
    # equal the input turns; check question identity and number of decoded turns.
    for n, (row, ref) in enumerate(zip(rows, expected, strict=True), 1):
        if task == "mt_bench_noeval":
            turns = row["choices"][0]["turns"]
            ok = (
                row["question_id"] == ref["question_id"]
                and row["category"] == ref["category"]
                and len(row["choices"]) == 1
                and isinstance(turns, list)
                and len(turns) == len(ref["turns"])
            )
        elif task == "gsm8k":
            ok = (
                hashlib.sha256(row["question"].encode()).hexdigest()
                == ref["question_sha256"]
            )
        elif task == "humaneval":
            ok = row["task_id"] == ref["task_id"]
        else:
            raise ValueError(f"no sample identity check for {task}")
        if not ok:
            raise ValueError(
                f"{name}: sample {n} differs from frozen selection/turn count"
            )
    return len(rows)


def _run_cell(job, pool, logs, samples):
    config = job["config"]
    out = Path(config["exp_name"])
    start = time.perf_counter()
    try:
        out.mkdir()  # Refuse even an empty existing directory: eval.runner appends.
        with pool.acquire(config) as worker:
            result = worker.run(config, str(logs))
    except Exception as exc:
        result = dict(
            exp_name=str(out), status="failed", result={"error": str(exc)}, log_file=""
        )
    result.update(job, elapsed_seconds=time.perf_counter() - start)
    if result["status"] == "success":
        try:
            result["sample_rows"] = _verify_samples(config, samples)
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            IndexError,
        ) as exc:
            result.update(status="failed", error=f"output validation: {exc}")
    return result


def _run_lanes(jobs, dest, samples):
    pool, lock, results = _WorkerPool(), threading.Lock(), []
    queues = {}
    for job in jobs:
        queues.setdefault(job["config"]["CUDA_VISIBLE_DEVICES"], []).append(job)

    def lane(queue):
        # Each lane owns one ordered, disjoint group. Reuse its worker and models.
        for job in sorted(
            queue, key=lambda j: (j["config"]["target_model"], j["config"]["eval_mode"])
        ):
            if pool.stopped.is_set():
                break
            entry = _run_cell(job, pool, dest / "logs", samples)
            with lock:
                results.append(entry)
                save_summary(dest / "summary.json", results)
                print(
                    f"{job['cell_id']}: {entry['status']} "
                    f"({entry['elapsed_seconds']:.1f}s)",
                    flush=True,
                )

    try:
        with ThreadPoolExecutor(max_workers=len(queues)) as executor:
            try:
                for future in as_completed(
                    [executor.submit(lane, q) for q in queues.values()]
                ):
                    future.result()
            except BaseException:
                pool.interrupt()  # unblock a worker pipe before joining the lanes
                raise
    finally:
        pool.shutdown()
    return len(results) == len(jobs) and all(r["status"] == "success" for r in results)


def _run(args):
    if args.execute and args.stage == "quality":
        raise ValueError(
            "quality is proposed only: preview and --preflight, never --execute"
        )
    if args.execute and not (args.gpus and args.run_id):
        raise ValueError("--execute requires explicit --gpus and a fresh --run-id")
    if args.run_id is not None and not re.fullmatch(
        "[A-Za-z0-9][A-Za-z0-9_.-]{0,47}", args.run_id
    ):
        raise ValueError("--run-id must be a single safe ASCII name (1–48 characters)")
    groups = _gpu_groups(args.gpus)
    indices = [index for group in groups for index in group]

    configs = matrix.build_rerun_matrix(args.stage)
    include, exclude = _csv(args.cells), _csv(args.exclude_cells)
    cells = _select(configs, include, exclude)
    placements = list(zip(cells, cycle(groups or [()])))
    mode = "execute" if args.execute else "preflight" if args.preflight else "dry-run"
    print(
        f"{mode}: stage={args.stage} cells={len(cells)}/{len(configs)}"
        + (" (proposed; preview/preflight only)" if args.stage == "quality" else "")
    )
    for (key, c), group in placements:
        placement = "GPU " + ",".join(map(str, group)) if group else "unassigned"
        print(
            f"{key}  {placement}  N={c['eval_data_num']} "
            f"tokens={c['max_tokens']} k={c['transfer_top_k']}"
        )
    if not (args.preflight or args.execute):
        return 0

    if args.execute:
        if Path.cwd() != ROOT:
            raise ValueError(
                f"run --execute from repo root {ROOT}: worker data paths are relative"
            )
        dest = ROOT / "experiments" / "paper_reruns" / args.run_id
        if dest.resolve() != dest or os.path.lexists(dest):
            raise ValueError(
                f"not a fresh run directory: {dest}; no resume or overwrite"
            )
        uuid_masks = _check_masks(indices)
        if any(c["use_rl_adapter"] and not c["disable_rl_update"] for _, c in cells):
            raise ValueError(
                "refusing an unfrozen RL adapter: disable_rl_update required"
            )
        # Worker logs flatten the *absolute* exp_name, then add a timestamp.
        # Keep that component short even though CLI cell IDs are descriptive.
        longest = f"{dest}/cells/{'0' * 16}_20000101_000000_000000.log"
        if len(os.fsencode(longest)) > os.pathconf(ROOT, "PC_NAME_MAX"):
            raise ValueError("checkout/run-id too long for worker log filenames")

    inputs = _freeze_inputs([c for _, c in cells])
    if args.preflight:
        # File- and row-level hashes go in the complete execution manifest.
        compact = dict(inputs)
        compact["source"] = {k: v for k, v in inputs["source"].items() if k != "files"}
        compact["samples"] = {
            t: {k: v for k, v in s.items() if k != "rows"}
            for t, s in inputs["samples"].items()
        }
        print(json.dumps(compact, ensure_ascii=False, indent=2))
        return 0

    cards = _hardware(indices, uuid_masks)  # check idle immediately before launch
    uuids = {card["index"]: card["uuid"] for card in cards}
    jobs = []
    for (key, config), group in placements:
        out = dest / "cells" / key.rsplit(".", 1)[1]
        mask = ",".join(uuids[index] for index in group)
        cfg = dict(config, CUDA_VISIBLE_DEVICES=mask, exp_name=str(out))
        job = dict(cell_id=key, gpu_indices=list(group), config=cfg)
        if len(group) == 1:
            job["gpu_index"] = group[0]  # compatibility with single-GPU output
        jobs.append(job)
    dest.mkdir(
        parents=True
    )  # atomic single-use reservation, including concurrent invocations
    (dest / "cells").mkdir()
    manifest = dict(
        stage=args.stage,
        run_id=args.run_id,
        started_utc=datetime.now(timezone.utc).isoformat(),
        stage_size=len(configs),
        selection={"include": include, "exclude": exclude},
        inputs=inputs,
        gpus=cards,
        gpu_groups=groups,
        cells=jobs,
    )
    with (dest / "manifest.json").open("x", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with _worker_env(inputs["worker_env"]):
        return 0 if _run_lanes(jobs, dest, inputs["samples"]) else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--stage", required=True, choices=matrix.RERUN_STAGES)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run", action="store_true", help="list cells and placement only (default)"
    )
    mode.add_argument(
        "--preflight",
        action="store_true",
        help="check local inputs without GPU or run artifacts",
    )
    mode.add_argument(
        "--execute",
        action="store_true",
        help="launch only with a fresh run-id and explicitly allocated GPUs",
    )
    parser.add_argument(
        "--gpus",
        metavar="INDEX[,INDEX...]",
        action="append",
        nargs="+",
        help="ordered physical nvidia-smi indices visible to one worker; "
        "repeat --gpus with disjoint groups to run cells in parallel",
    )
    parser.add_argument(
        "--run-id",
        help="fresh directory under experiments/paper_reruns (never resumes)",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--cells", nargs="+", help="only these preview IDs (space- or comma-separated)"
    )
    selection.add_argument(
        "--exclude-cells", nargs="+", help="all except these exact preview IDs"
    )
    args = parser.parse_args(argv)
    try:
        return _run(args)
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"paper_rerun: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
