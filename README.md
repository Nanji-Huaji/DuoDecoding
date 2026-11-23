# DuoDecoding

DuoDecoding runs speculative decoding on real language models and simulates the cost of exchanging proposals and verification results between device, edge, and cloud. Use it to compare two-model baselines with three-model CEE-SD under the same bandwidth and latency assumptions. Reported communication time is simulated; a single-GPU run is not a network benchmark.

[Run one question](#quickstart-run-one-mt-bench-question) · [Rerun paper experiments](#paper-experiments) · [Train an RL agent](#optional-train-an-rl-agent)

## Setup

Use CPython 3.10 and run the commands below from the repository root. Model evaluation requires CUDA; inspecting CLI options and previewing an experiment matrix do not require a GPU or model weights.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
  https://github.com/Nanji-Huaji/DuoDecoding.git
cd DuoDecoding
uv sync --frozen
source .venv/bin/activate
```

The clone above skips archived Git LFS weights; the examples do not need them. For an existing clone, run `git submodule update --init --recursive` after checking out the desired revision. Both SpecDec++ and the bandwidth traces are submodules. Even methods that use no acceptance head import **SpecDec++ source code**. Trace replay also needs the trace files.

If you prefer pip, install the same locked dependencies in a fresh Python 3.10 environment using the hash-checked command below **instead of** `uv sync`:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --require-hashes -r requirements.txt
```

## Quickstart: run one MT-Bench question

This example loads a **TinyLlama 1.1B draft** and a **Llama 2 13B target**. Both can run on one GPU if they and their caches fit; the default does not quantize this pair to 4-bit.

If the weights are not already in the configured local directories, download the two snapshots below. The Llama 2 repository is gated: obtain access for your Hugging Face account, then sign in with `hf auth login` or supply `HF_TOKEN` in your environment.

```bash
hf download TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 \
  --local-dir models/llama/tiny-llama-1.1b
hf download meta-llama/Llama-2-13b-hf \
  --revision 5c31dfb671ce7cfe2d7bb7c04375e44c55e815b1 \
  --local-dir models/llama/Llama-2-13b-hf
```

Generate answers for one question from the included MT-Bench data. Keep the GPU mask assigned by your scheduler. On a workstation, expose only the GPU(s) you intend to use. The model loader decides placement across the visible devices. The `noeval` entry point runs warm-up and generation, but does not grade answers or add few-shot examples to measured prompts.

```bash
python eval/eval_mt_bench_noeval.py \
  --eval_mode dist_spec --model_source local \
  --draft_model tiny-llama-1.1b --target_model llama-2-13b \
  --eval_data_num 1 --max_tokens 32 --temp 0.0 --exp_name mt_smoke
```

On success, look in `exp/mt_smoke/` for `dist_spec_mt_bench.jsonl` and `dist_spec_mt_bench_metrics.json`. Use a new `--exp_name` for an independent run: the sample JSONL is appended to, while the metrics JSON describes the latest run.

### GSM8K with a bandwidth trace

Use the same two models. GSM8K reads the Hugging Face `gsm8k` dataset, configuration `main`, split `test`: it must be cached locally or accessible for download. Unlike MT-Bench, it does not read `--data_path`. The task loader reports an error if the split is missing.

```bash
python eval/eval_gsm8k.py \
  --eval_mode dist_split_spec --model_source local \
  --draft_model tiny-llama-1.1b --target_model llama-2-13b \
  --eval_data_num 1 --num_shots 8 --max_tokens 128 \
  --edge_cloud_bandwidth 23.6 --use_stochastic_comm --transfer_top_k 1024 \
  --exp_name gsm8k_trace
```

DSSD models only the edge–cloud link. Here 23.6 Mbps is the **target mean** of the replayed trace; omit `--use_stochastic_comm` to use constant bandwidth. The 8-shot prompt and `k=1024` are demo settings. In DSSD a positive `--transfer_top_k` also sets the draft sampling top-k; the ordinary uplink sends a token and one probability per candidate, not 1,024 logits. The paper DSD/DSSD baselines use `--transfer_top_k 0`.

## Paper experiments

These stages rerun the experiments with the current code; they are not the published scores. Each configuration is one method, model family, task, and operating point.

| Stage | Configurations | What to run |
| --- | ---: | --- |
| `main46` | 45 | Table V at 46 Mbps: 3 model families × 3 tasks × 5 methods. |
| `core` | 63 | Includes `main46`, 9 CEE+ comparisons and 9 matched-window Static controls. |
| `lowbw` | 90 | Adds 10 and 5 Mbps. Run after `core` if you want all performance results. |
| `performance` | 153 | Union of `core` and `lowbw`; **do not** run it again after those two. |
| `quality` | 24 proposed | Preview or preflight only. Execution is disabled. |

Start with a CPU-only preview. It lists configurations without opening models, probing GPUs, or launching workers. The separate `core` preflight runs offline. It requires all nine local models across Llama, Qwen1.5, and Qwen3 (the six Qwen directories must be reachable as `Qwen/Qwen…` from the repository root), the six acceptance heads, and the deployed fluid-v3 RL checkpoints and replay buffers for both roles. It also requires the bandwidth trace, local MT-Bench questions, and prepared `datasets` test caches for `gsm8k` (`main`) and `openai_humaneval`.

```bash
python -m src.experiments.paper_rerun --stage core --dry-run
# Requires local assets; still does not start GPU jobs:
python -m src.experiments.paper_rerun --stage core --preflight
```

Reserve the GPUs first and pass their physical `nvidia-smi` indices. One `--gpus` flag defines the ordered devices visible to one worker: `--gpus 0` runs on one GPU; `--gpus 0,1` lets the model loader place a configuration across both. To run different configurations in parallel, repeat the flag with disjoint groups, e.g. `--gpus 0 --gpus 1` for the original one-worker-per-card layout. The runner does not require a particular GPU model or minimum VRAM. The 48 GB cards used for the original schedule are a sizing reference, not a guarantee that a given placement fits.

**Comparing performance:** GPU model, device order, interconnect, and cross-GPU transfers can change measured throughput and latency even when the simulated link settings are identical. Compare methods using the same hardware and placement, or report layouts separately. The manifest records the GPUs visible to each configuration; check the worker log for loading decisions. Visibility does not guarantee every GPU is used.

After preflight, set `GPUS` to your allocated group (e.g. `0,1`) and use a fresh run ID:

```bash
python -m src.experiments.paper_rerun --stage core --execute \
  --gpus "${GPUS:?set allocated physical GPU indices}" \
  --run-id "core-$(date -u +%Y%m%dT%H%M%SZ)"
```

Results go to `experiments/paper_reruns/<run-id>/`: `manifest.json` records the inputs and placement, `cells/` holds per-configuration results, `logs/` holds worker logs, and `summary.json` is updated as cells finish. The runner **does not resume or overwrite** a run ID. For a pilot, pass `--cells` with IDs from the preview. Use `--exclude-cells` in a new run to skip successful pilot cells only if the code, environment, hardware, weights, traces, and selected samples match.

### Legacy parameter sweep

The legacy launcher is for environment-tunable sweeps. Its default is **135** configurations across 46, 10, and 5 Mbps, not the 45 configurations of the main table. **Without `--dry-run` it starts jobs.**

```bash
python exp.py --dry-run
python exp.py --only-modes dsd,dssd --dry-run
```

The CLI writes logs to `exp_logs/` and a summary to `experiment_results/experiment_summary_<timestamp>.json` unless you set `--summary-file`. Here `--resume-from` skips successful configurations; add `--merge-from` to carry earlier results into a merged summary. For a custom Python sweep, import `run_experiments_parallel` from `exp` and pass `log_dir` and `summary_file` explicitly.

## Models, acceptance heads, and RL checkpoints

A base model generates tokens. An **acceptance head** (ARP) predicts acceptance for a directed model pair. An **RL agent** selects actions such as a top-k and a threshold, and in some modes a draft length.

| Asset | How to obtain or locate it |
| --- | --- |
| Base models | For a registered alias, `--model_source auto` uses the configured local path if it exists, even if the download is incomplete. Otherwise it uses the configured Hub ID if available. Use `--model_source local` with a registered alias to select its configured path. An unregistered Hub ID is not forced offline by that flag. |
| Acceptance heads | `python tools/fetch_checkpoints.py --groups acc_head` fetches the six pairs used by the paper from `ArcticHuaji/specdecpp-acc-heads`. The default download layout is `src/SpecDec_pp/checkpoints/acc_head/<source>--to--<target>/<run>/`. The resolver looks for a config in registered, standard, then legacy locations. |
| Production RL agents | `python tools/fetch_checkpoints.py --groups rl` uses the mirror in the manifest (`ArcticHuaji/DuoDecoding-checkpoints`). Files live in `checkpoints/rl/{main,little}/<pair>/`; `checkpoints/rl_agents/` supplies compatible paths. `git lfs pull` retrieves historical LFS objects, **not** these runtime weights. |

The older downloader (`python scripts/download_models.py --help`) lists nine base models under `--series llama`, `qwen`, or `qwen15`. It writes to `./llama/` and `./Qwen/`, not the catalog's `models/` directories. If you already have those downloads, point the catalog or the model arguments at them rather than downloading again. Use the manifest fetcher above for heads. For Qwen1.5 training use the full `Qwen/Qwen1.5-{size}-Chat` IDs or model paths, not the bare checkpoint pair names printed by `--rl-guide`.

### Which modes need which weights?

The following groups cover the commonly used `--eval_mode` values. The mode registry specifies which models, heads, and RL roles the loader constructs.

| Mode or aliases | Weights needed at startup |
| --- | --- |
| `small`; `large` | Draft only; target only, respectively. The CLI still requires both model arguments. |
| `sd`; `dsd` / `dist_spec`; `dssd` / `dist_split_spec`; `cuhlm` / `uncertainty_decoding`; `tk_slt` / `tkslt` | Draft and target; no acceptance heads or RL agents. |
| `adaptive_decoding` | Draft and target, plus one draft→target head. |
| `adaptive_tridecoding` / `cee_sd`; `cee_sd_opportunistic`; `cee_dsd`, `cee_dssd`, `cee_cuhlm` | Little, draft, and target; both little→draft and draft→target heads. |
| `tridecoding`; `cee_sd_no_arp` | Three models, no heads. The former is the fixed-window Static control without RL; the latter is the current no-ARP ablation, not Static. |
| `target_only` | Decodes with the target but currently loads all three models. Use `large` for the target-only loading baseline. |

Use `--acc_head_path PATH` for a single head; use `--small_draft_acc_head_path PATH` and `--draft_target_acc_head_path PATH` to override the two stage paths. For CEE-SD, `--gamma2` sets the little model’s proposal length and `--gamma1` sets how many additional tokens the draft proposes. The target also verifies tokens carried forward from the first stage. An RL policy with draft-length actions can change the next round’s lengths. Do not assume `--gamma` sets the number of candidates for every dual method: standalone CUHLM proposes one token at a time. For all standard options, run `python eval/eval_gsm8k.py --help` (use `--help-advanced` for RL and training options).

## Optional: train an RL agent

For inference with a trained policy, pass `--use_rl_adapter --disable_rl_update` in a supported mode and **check which file was loaded**. The main agent is keyed by draft→target; the little agent by little→draft. To print an expected path without loading or creating a checkpoint:

```bash
python -m src.rl_agent_registry main tiny-llama-1.1b llama-2-13b --kind latest --format path
```

In normal `resume` mode the loader tries `best.pth`, then `latest.pth`, then a compatible legacy file. If none loads it keeps the initialized policy; that alone does not start training. With only an explicit `--main_rl_path` or `--little_rl_path`, the corresponding best candidate is that same file. The opportunistic mode skips legacy files, freezes its main agent, and prefers latest for its little agent while learning.

For a **new** Llama training run, add `llama-68m` to the two demo models and fetch the two Llama acceptance heads using the head group above. The mixed task also needs MT-Bench plus the GSM8K, CNN/DailyMail, XSum, and HumanEval test splits. Use a new staging directory rather than training over production files. `fresh` skips old weights and refuses to overwrite existing latest/best checkpoints in that destination:

```bash
hf download JackFram/llama-68m --revision 9de84537b6aa98de634ad0fbb1608e9d6a019355 \
  --local-dir models/llama/llama-68m

run="llama-mixed-$(date -u +%Y%m%dT%H%M%SZ)"
stage="$(pwd -P)/checkpoints/staging/$run"
python eval/eval_mixed.py --eval_mode adaptive_tridecoding \
  --model_source local --little_model llama-68m \
  --draft_model tiny-llama-1.1b --target_model llama-2-13b \
  --use_rl_adapter --rl_init_strategy fresh --rl_output_dir "$stage" \
  --rl_init_seed 42 --eval_data_num 32 --max_tokens 64 --exp_name "$run"
```

Training writes `latest.pth` and its replay buffer under `$stage/<role>/<pair>/`. A new throughput high can write `best_tps.pth` only with `--rl_track_best_tps`; that diagnostic is **not** a production best. Validate the policy, then preview promotion with `python tools/promote_checkpoint.py --staging "$stage" --as-best --dry-run`. Use the **absolute, repository-local** staging path shown above. Only rerun with `--as-best` and without `--dry-run` if you intend to designate the staged `latest.pth` as production best. The older `bash cmds/train_rl_mixed.sh` command defaults to 68M/7B-chat/70B-chat and does not forward the staging flags in this example.

**Training new acceptance heads** is a separate workflow. SpecDec++ uses one environment for head training and another for vLLM data generation. For data generation, `bash scripts/setup_vllm_env.sh` from the submodule directory recreates its dedicated environment; do not install either stack in the root evaluation environment.

## Validation and debugging

In a development environment with the project dependencies and pytest installed, run `python -m pytest -q test/test_dependency_contract.py` to check dependencies. For the optional GSM8K vLLM benchmark, run `python test/eval_gsm8k_vllm.py --help` in a separate vLLM environment.

For a failing sample, prefix **one** single-evaluation command with `DUODEC_DEBUG_NUMERICS=1` for probability checks. On measured MT-Bench **noeval** turns, `DUODEC_DEBUG_TOKEN_CHECKS=1` also replaces out-of-range output IDs with zero.
