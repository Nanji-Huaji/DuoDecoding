"""Read-only input lock for one selected paper-rerun stage, not a downloader.

Call ``freeze_inputs(configs)`` at launch, save the returned JSON, then start
workers from the repository root, using this Python and ``worker_env``. No
model, tokenizer, checkpoint deserializer, evaluator or GPU is needed here.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from types import SimpleNamespace

from eval.data import _HF_SPLITS, load_task_data
from eval.few_shot_examples import get_few_shot_prompt
from eval.utils import select_eval_data
from src.acc_head_registry import build_acc_head_pair_name
from src.mode_features import get_mode_spec
from src.model_zoo import resolve_model
from src.trace_io import read_trace_file, return_closest_mean_index

ROOT = Path(__file__).resolve().parents[2]
_OFFLINE = {
    k: "1" for k in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE")
}
_COUNTS = {
    "mt_bench_noeval": (80,),
    "humaneval": (80,),
    "gsm8k": (80, 500),
    "cnndm": (500,),
}
_STATIC_TRACE = Path(
    "data/sigcomm-5gmemu-5g-mmWave-uplink-data/throughput/static/5g/away_p1.list"
)


def _sha(path: Path) -> str:
    """Always read the launch-time bytes (no process-global mtime/hash cache)."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_sha(value) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _file(path: Path) -> dict:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"Missing or empty input: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha(path),
    }


def _git(root: Path, *args: str) -> str:
    p = subprocess.run(
        ["git", "--no-optional-locks", *args], cwd=root, capture_output=True
    )
    if p.returncode:
        raise RuntimeError(f"git {args[0]} in {root}: {os.fsdecode(p.stderr).strip()}")
    return os.fsdecode(p.stdout)


def _source() -> dict:
    # ls-files omits gitlinks' contents. Inspect initialized submodules too,
    # notably SpecDec++ (ARP source); never recurse into ignored venvs/caches.
    files, submodules = {}, {}
    status = _git(ROOT, "status", "--porcelain=v1").splitlines()
    roots = [ROOT]
    for entry in _git(ROOT, "ls-files", "--stage", "-z").split("\0"):
        if entry.startswith("160000 "):
            meta, rel = entry.split("\t", 1)
            path = ROOT / rel
            head = (
                _git(path, "rev-parse", "HEAD").strip()
                if (path / ".git").exists()
                else None
            )
            submodules[rel] = {"gitlink": meta.split()[1], "head": head}
            if rel == "src/SpecDec_pp" and (
                not head or not (path / "specdec_pp/wrap_model.py").is_file()
            ):
                raise ValueError(
                    f"Required evaluator source submodule is missing: {rel}"
                )
            if head:
                roots.append(path)
    for root in roots:
        names = _git(
            root,
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
        )
        for name in filter(None, names.split("\0")):
            p = root / name
            files[str(p.relative_to(ROOT))] = (
                _sha(p) if p.is_file() else None
            )  # tracked deletion
    return {
        "root": str(ROOT),
        "head": _git(ROOT, "rev-parse", "HEAD").strip(),
        "dirty": bool(status),
        "status": status,
        "submodules": submodules,
        "files": files,
        "python_sha256": _json_sha(files),
    }


def _dependencies(tasks) -> dict:
    # Distribution metadata, not imports (torch/flash-attn can touch CUDA).
    packages = {
        d.metadata["Name"].lower().replace("_", "-"): d.version
        for d in metadata.distributions()
        if d.metadata.get("Name")
    }
    required = {
        "torch",
        "transformers",
        "accelerate",
        "numpy",
        "safetensors",
        "tokenizers",
        "tqdm",
        "shortuuid",
    }
    if set(tasks) - {"gsm8k"} and not {"fastchat", "fschat"} & packages.keys():
        required.add("fastchat")
    if set(tasks) & _HF_SPLITS.keys():
        required.update(("datasets", "pyarrow"))
    if "cnndm" in tasks:
        required.add("rouge-score")
    if missing := sorted(required - packages.keys()):
        raise ValueError(f"Missing evaluation packages in {sys.executable}: {missing}")
    return {
        "python": sys.version,
        "executable": sys.executable,
        "packages": packages,
        "locks": {
            n: _sha(ROOT / n)
            for n in ("uv.lock", "requirements.txt")
            if (ROOT / n).is_file()
        },
    }


def _weight_inventory(path: Path) -> dict:
    # Transformers' default order: safetensors (single, then indexed), then bin.
    for name in (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    ):
        chosen = path / name
        if chosen.is_file():
            break
    else:
        raise ValueError(f"Model weights or shard index missing in {path}")
    index = json.loads(chosen.read_text()) if name.endswith(".json") else None
    shards = set(index["weight_map"].values()) if index is not None else {name}
    if not shards:
        raise ValueError(f"Empty weight_map in {chosen}")
    weights = {}
    for shard in sorted(shards):
        p = path / shard
        if Path(shard).name != shard or not p.is_file() or not p.stat().st_size:
            raise ValueError(f"Invalid/missing model shard {shard} in {path}")
        st = p.stat()
        if st.st_size < 1024 and p.read_bytes().startswith(
            b"version https://git-lfs.github.com/spec/"
        ):
            raise ValueError(f"Model shard is an unhydrated Git LFS pointer: {p}")
        weights[shard] = {
            "bytes": st.st_size,
            "mtime_ns": st.st_mtime_ns,
            "resolved": str(p.resolve()),
        }
        # A local_dir snapshot normally retains the separate HF download receipt.
        # Its etag is the upstream assertion, NOT a rehash of the current file.
        receipt = path / ".cache/huggingface/download" / (shard + ".metadata")
        if receipt.is_file():
            lines = receipt.read_text().splitlines()
            if len(lines) >= 2 and re.fullmatch("[0-9a-f]{40}", lines[0]):
                weights[shard].update(download_commit=lines[0], download_etag=lines[1])
    if index and sum(w["bytes"] for w in weights.values()) < index.get(
        "metadata", {}
    ).get("total_size", 0):
        raise ValueError(f"Truncated weight shards according to {chosen}")
    return weights


def _model(alias: str, policy: str, roles: set[str]) -> dict:
    if policy != "local":
        raise ValueError(
            f"{alias}: a paper worker needs model_source=local, not {policy}"
        )
    # Registered aliases use the worker's resolver. Literal Qwen/Qwen… names
    # are also real local paths in this checkout; never substitute an HF revision.
    resolved, kind, entry = resolve_model(alias, policy, {})
    path = (ROOT / resolved).resolve()
    if (kind == "legacy" and "/" not in alias) or not path.is_dir():
        raise ValueError(
            f"{alias}: resolved model directory not available locally: {path}"
        )
    config = json.loads((path / "config.json").read_text())
    if not config.get("model_type") or not config.get("vocab_size"):
        raise ValueError(
            f"{alias}: model_type/vocab_size missing in {path}/config.json"
        )
    # In a Qwen target-only cell the CLI still reads the draft config for its
    # vocabulary. Do not require or claim to load draft weights in that case.
    vocab_only = roles == {"draft_vocab"}
    metadata_files = {
        p.name: _sha(p)
        for p in path.iterdir()
        if p.is_file()
        and (
            p.name == "config.json"
            or not vocab_only
            and (
                p.suffix in {".json", ".model", ".py", ".jinja"}
                or p.name == "merges.txt"
            )
        )
    }
    if "target" in roles:

        def present(name):
            return name in metadata_files and (path / name).stat().st_size > 0

        if not (
            present("tokenizer.json")
            or present("tokenizer.model")
            or (present("vocab.json") and present("merges.txt"))
        ):
            raise ValueError(f"{alias}: target tokenizer data missing in {path}")
    revision = (
        path.name
        if path.parent.name == "snapshots" and re.fullmatch(r"[a-f0-9]{40}", path.name)
        else None
    )
    return {
        "alias": alias,
        "roles": sorted(roles),
        "resolved": str(path),
        "loader_argument": resolved,
        "registry_hf_repo": entry.get("hf_repo"),
        "registry_hf_revision": entry.get("hf_revision"),
        "snapshot_revision": revision,
        "metadata_sha256": metadata_files,
        "weights_metadata_only": {} if vocab_only else _weight_inventory(path),
        "weights_hashed": False,
    }


def _models(configs) -> dict:
    roles = {}
    for c in configs:
        topology = get_mode_spec(c["eval_mode"]).models
        used = {
            "large": ["target"],
            "dual": ["draft", "target"],
            "tri": ["little", "draft", "target"],
        }[topology]
        if topology == "large" and "/" in c["draft_model"]:
            used = [*used, "draft_vocab"]
        for role in used:
            alias = c["draft_model"] if role == "draft_vocab" else c[f"{role}_model"]
            policy, seen = roles.setdefault(alias, (c["model_source"], set()))
            if policy != c["model_source"]:
                raise ValueError(f"{alias}: conflicting model sources across cells")
            seen.add(role)
    return {
        alias: _model(alias, policy, seen) for alias, (policy, seen) in roles.items()
    }


def _checkpoints(configs) -> dict:
    # Compare the actual configured bytes (incl. symlinks or explicit copies)
    # to the production pair/role -- a valid hash for the wrong pair is not OK.
    needed = set()
    for c in configs:
        spec = get_mode_spec(c["eval_mode"])
        for role, src, dst in (
            ("main", "draft", "target"),
            ("little", "little", "draft"),
        ):
            pair = build_acc_head_pair_name(c[f"{src}_model"], c[f"{dst}_model"])
            if c["use_rl_adapter"] and getattr(spec, f"uses_{role}_rl"):
                latest = c[f"{role}_rl_path"]
                best = c[f"{role}_rl_best_path"]
                if (
                    not c["disable_rl_update"]
                    or c.get("rl_output_dir")
                    or c.get("rl_init_strategy", "resume") != "resume"
                ):
                    raise ValueError("Paper runs require frozen, deployed RL agents")
                if (
                    not latest
                    or Path(latest).name != "latest.pth"
                    or (ROOT / best).resolve() != (ROOT / latest).resolve()
                ):
                    raise ValueError(
                        f"{role}/{pair}: pin both paths (incl. best) to latest.pth"
                    )
                for suffix in ("", ".buffer"):
                    needed.add((f"rl_{role}", pair, latest + suffix))
            if spec.acc_head == "both" or (
                role == "main" and spec.acc_head == "draft_target"
            ):
                key = (
                    "acc_head_path"
                    if spec.acc_head == "draft_target"
                    else {
                        "main": "draft_target_acc_head_path",
                        "little": "small_draft_acc_head_path",
                    }[role]
                )
                if not c[key]:
                    raise ValueError(f"{role}/{pair}: missing {key}")
                folder = Path(c[key])
                weight = (
                    "model.safetensors"
                    if (ROOT / folder / "model.safetensors").is_file()
                    else "pytorch_model.bin"
                )
                for name in ("config.json", weight):
                    needed.add(("acc_head", pair, str(folder / name)))
    if not needed:
        return {"files": {}}
    manifest = ROOT / "checkpoints/manifest.json"
    entries = json.loads(manifest.read_text())["entries"]
    files = {}
    for role, pair, value in sorted(needed):
        name = Path(value).name
        candidates = [
            k
            for k, v in entries.items()
            if v.get("status") == "production"
            and (v.get("role"), v.get("pair"), Path(k).name) == (role, pair, name)
        ]
        info = _file(ROOT / value)
        if len(candidates) != 1 or info["sha256"] != entries[candidates[0]]["sha256"]:
            raise ValueError(
                f"{value}: {pair}/{role} sha256 mismatch with production manifest"
            )
        info.update(role=role, pair=pair, manifest_entry=candidates[0])
        if role.startswith("rl_"):
            tail = f"/{role[3:]}/{pair}/{name}"
            refs = [
                k
                for k, v in entries.items()
                if k.startswith("checkpoints/rl_fluid_retrain_v3/")
                and k.endswith(tail)
                and v["sha256"] == info["sha256"]
            ]
            if len(refs) != 1:
                raise ValueError(
                    f"{value}: hash does not match the fluid-v3 latest deployment"
                )
            info["v3_manifest_entry"] = refs[0]
        files[value] = info
    return {"manifest": _file(manifest), "files": files}


def _network(configs) -> dict:
    # Paper cells use only the static 5G campaign (other modes don't all
    # consume --comm_trace_mode). Don't certify an unsupported override.
    bws, rtts = set(), set()
    for c in configs:
        if c.get("ntt_trace_file"):
            rtts.add((c["ntt_trace_file"], c.get("ntt_trace_scale", 1.0)))
        if c["eval_mode"] == "large":  # no link simulator
            continue
        if (
            not c["use_stochastic_comm"]
            or c["use_precise"]
            or c.get("comm_trace_mode", "static") != "static"
        ):
            raise ValueError(
                "Paper links must use the static stochastic bandwidth trace"
            )
        bws.add(float(c["edge_cloud_bandwidth"]))
    out = {"bandwidth": None, "rtt_replays": []}
    if bws:
        path = ROOT / _STATIC_TRACE
        runs = {}
        for bw in sorted(bws):
            run = return_closest_mean_index(str(path), max(0.1, bw))
            if run == -1:
                raise ValueError(f"No usable bandwidth run in {path}")
            values = read_trace_file(str(path), run)
            runs[f"{bw:g}"] = {
                "run_id": run,
                "count": len(values),
                "cleaned_mbps_sha256": _json_sha(values),
            }
        out["bandwidth"] = {**_file(path), "runs_by_target_mbps": runs}
    for raw, scale in sorted(rtts):
        path = ROOT / raw
        for run in (1, 2, 3):  # baselines.runtime._configure_ntt_trace
            try:
                values = read_trace_file(str(path), run)
            except ValueError:
                continue
            if values:
                break
        else:
            raise ValueError(f"No usable RTT run in {path}")
        out["rtt_replays"].append(
            {**_file(path), "run_id": run, "scale": scale, "count": len(values)}
        )
    return out


def _plans(configs) -> dict:
    plans = {}
    for c in configs:
        task = Path(c["eval_dataset"]).stem.removeprefix("eval_")
        if task not in _COUNTS or c["eval_data_num"] not in _COUNTS[task]:
            raise ValueError(f"Not a paper sample count: {task} n={c['eval_data_num']}")
        if (
            c["run_full_dataset"]
            or c["random_sample"] is not True
            or c["sample_seed"] != 1234
            or c["num_shots"] != 3
        ):
            raise ValueError(
                f"{task}: expected random selection with seed=1234 and num_shots=3"
            )
        plan = {
            k: c[k]
            for k in ("eval_data_num", "random_sample", "sample_seed", "num_shots")
        }
        if task == "mt_bench_noeval":
            plan["data_path"] = str((ROOT / c.get("data_path", "data")).resolve())
        if plans.setdefault(task, plan) != plan:
            raise ValueError(f"Sampling differs between {task} cells in this stage")
    return plans


def _sample(task, plan, rows, source) -> dict:
    args = SimpleNamespace(**plan)
    indices = select_eval_data(range(len(rows)), args)
    if len(indices) != args.eval_data_num:
        raise ValueError(
            f"{task}: need {args.eval_data_num} test samples, only {len(indices)}"
        )
    records, identities = [], set()
    for index in indices:
        row = dict(rows[index])
        record = {"index": index, "row_sha256": _json_sha(row)}
        if task == "mt_bench_noeval":
            identity = row["question_id"]
            turns = row["turns"]
            if (
                not isinstance(turns, list)
                or len(turns) != 2
                or not all(isinstance(t, str) and t.strip() for t in turns)
            ):
                raise ValueError(
                    f"MT-Bench question {identity}: expected two nonempty turns"
                )
            record.update(question_id=identity, category=row["category"], turns=turns)
        else:
            required = {
                "gsm8k": ("question", "answer"),
                "humaneval": ("task_id", "prompt", "entry_point", "test"),
                "cnndm": ("id", "article", "highlights"),
            }[task]
            if not all(
                isinstance(row.get(k), str) and row[k].strip() for k in required
            ):
                raise ValueError(
                    f"{task} test row {index}: missing evaluation fields {required}"
                )
            if task == "gsm8k":
                identity = record["question_sha256"] = hashlib.sha256(
                    row["question"].encode()
                ).hexdigest()
            else:
                key = "task_id" if task == "humaneval" else "id"
                identity = record[key] = row[key]
        if identity in identities:
            raise ValueError(f"{task}: duplicate sample identity {identity}")
        identities.add(identity)
        records.append(record)
    # MT/noeval evaluates raw turns; even its warmup has no supported examples.
    shots = 0 if task == "mt_bench_noeval" else args.num_shots
    prompt = get_few_shot_prompt(task, shots)
    if shots and not prompt:
        raise ValueError(f"{task}: missing {shots}-shot prompt")
    out = {
        "source": source,
        "count": len(records),
        "population": len(rows),
        "sample_seed": args.sample_seed,
        "num_shots": args.num_shots,
        "effective_shots": shots,
        "few_shot_prefix_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "rows": records,
        "ordered_sha256": _json_sha(records),
    }
    if task == "mt_bench_noeval":
        out["total_turns"] = sum(len(r["turns"]) for r in records)
    return out


def _cached_split(task):
    """Use datasets' actual offline resolver + test Arrow reader, without its
    builder constructor (writes locks) or download_and_prepare (writes data).
    The pinned datasets version exposes these helpers; absence is a hard error.
    """
    from datasets import Dataset, DatasetInfo, config
    from datasets.arrow_reader import ArrowReader
    from datasets.packaged_modules.cache.cache import _find_hash_in_cache

    repo, *names = _HF_SPLITS[task]  # these three are unnamespaced, lowercase IDs
    if Path(repo).exists() or Path(repo + ".py").exists():
        raise ValueError(f"Local {repo} shadows the official HF input of eval.data")
    cache = Path(config.HF_DATASETS_CACHE).expanduser()
    cfg, version, key = _find_hash_in_cache(
        repo, names[0] if names else None, str(cache), {}, None
    )
    path = cache / repo / cfg / version / key
    info = DatasetInfo.from_directory(str(path))
    kwargs = ArrowReader(str(path), info).read(
        name=repo,
        instructions="test",
        split_infos=info.splits.values(),
        in_memory=False,
    )
    ds = Dataset(**kwargs)
    files = [_file(Path(f["filename"])) for f in ds.cache_files]
    if not files:
        raise ValueError(f"{repo}: no verifiable test split files in {path}")
    return ds, {
        "kind": "hf_offline_arrow",
        "repo": repo,
        "config": cfg,
        "split": "test",
        "version": version,
        "cache_key": key,
        "info": _file(path / "dataset_info.json"),
        "files": files,
    }


def _samples(plans) -> dict:
    samples = {}
    if (plan := plans.get("mt_bench_noeval")) is not None:
        args = SimpleNamespace(data_path=plan["data_path"], run_full_dataset=True)
        rows = load_task_data("mt_bench", args)
        samples["mt_bench_noeval"] = _sample(
            "mt_bench_noeval",
            plan,
            rows,
            {"kind": "jsonl", **_file(Path(args.data_path) / "mt_bench.jsonl")},
        )
    hf = {k: v for k, v in plans.items() if k in _HF_SPLITS}
    if hf:
        # Offline switches are read at import time. Do not mutate a parent
        # process's environment, dataset globals or RNG to obtain a preflight.
        env = {
            **os.environ,
            **_OFFLINE,
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "USE_TORCH": "0",
            "USE_TF": "0",
            "CUDA_VISIBLE_DEVICES": "",
        }
        p = subprocess.run(
            [
                sys.executable,
                "-B",
                "-m",
                "src.experiments.paper_inputs",
                "--cached-samples",
            ],
            cwd=ROOT,
            env=env,
            text=True,
            input=json.dumps(hf),
            capture_output=True,
        )
        if p.returncode:
            raise ValueError(
                f"HF cache failed (no download or substitution): {p.stderr.strip()}"
            )
        samples.update(json.loads(p.stdout))
    return samples


def freeze_inputs(configs) -> dict:
    """Fingerprint exactly the selected stage; raise rather than shrink or demo."""
    configs = list(configs)
    if not configs:
        raise ValueError("Cannot freeze an empty paper stage")
    plans = _plans(configs)
    return {
        "schema_version": 1,
        "source": _source(),
        "dependencies": _dependencies(plans),
        "models": _models(configs),
        "checkpoints": _checkpoints(configs),
        "network": _network(configs),
        "samples": _samples(plans),
        "worker_env": dict(_OFFLINE),
    }


if __name__ == "__main__" and sys.argv[1:] == ["--cached-samples"]:
    result = {}
    for task, plan in json.load(sys.stdin).items():
        try:
            result[task] = _sample(task, plan, *_cached_split(task))
        except Exception as exc:
            raise SystemExit(f"{task} ({_HF_SPLITS[task]}, test): {exc}") from exc
    print(json.dumps(result))
