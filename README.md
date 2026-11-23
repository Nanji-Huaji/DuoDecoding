# DuoDecoding

Experiment framework for speculative decoding with communication simulation,
RL-based threshold selection, and adaptive candidate lengths (SpecDec++).

## Setup

### Prerequisites

- **Python** &ge; 3.10
- **CUDA**-capable GPU(s) with sufficient VRAM for the models you intend to run
- **HuggingFace Hub** access (for downloading models; a [HF token](https://huggingface.co/settings/tokens) may be required for gated models such as Llama 2)

### Environment

`pyproject.toml` plus the committed `uv.lock` are the only sources of truth for
production dependencies. Use `uv sync --frozen` for the primary install: it
installs exactly the locked versions and never re-resolves or rewrites
`uv.lock`.

```bash
# Clone and enter the repository
git clone https://github.com/Nanji-Huaji/DuoDecoding.git && cd DuoDecoding

# Option A: uv (recommended — installs the committed lock as-is)
uv sync --frozen
```

`requirements.txt` is a generated export of that same `uv.lock`, not a separate
dependency list; it is the supported pip entry point. The export records a
sha256 hash for every artifact, so pip (which enables hash-checking
automatically) verifies each downloaded file against the lock:

```bash
# Option B: pip, installing the same locked artifacts
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Do not edit `requirements.txt` by hand. Run `python tools/export_locked_requirements.py --write` to regenerate the hash-checked export from the frozen lock.

Development tools are deliberately separate: `requirements-dev.txt` pins
`pytest==9.1.1` and `ruff==0.15.4`, but the production `uv.lock` does not contain
those tools, so it is a direct pin list rather than a resolved dev lock.

### Git LFS

RL agent checkpoints (`.pth` / `.pth.buffer` files under `checkpoints/`) are
stored via [Git LFS](https://git-lfs.com). Install `git-lfs` before cloning, or
pull LFS objects after the fact:

```bash
# Install git-lfs (once per machine)
# Linux (Debian/Ubuntu):   sudo apt install git-lfs
# macOS (Homebrew):         brew install git-lfs
# Or download from:         https://git-lfs.com
git lfs install

# If you cloned without git-lfs, pull LFS objects now
git lfs pull
```

### Download Models & Checkpoints

Use `scripts/download_models.py` to fetch all artifacts needed for experiments.

```bash
# Download all 9 base models (llama + qwen3 + qwen1.5 series)
python scripts/download_models.py

# Also download SpecDec++ acceptance prediction heads
python scripts/download_models.py --checkpoints

# Preview what would be downloaded
python scripts/download_models.py --checkpoints --dry-run
```

| Flag | Effect |
|------|--------|
| `--series llama` | Only the llama series (llama-68m, tiny-llama-1.1b, llama-2-13b) |
| `--series qwen` | Only Qwen3 series (0.6B, 1.7B, 14B) |
| `--series qwen15` | Only Qwen1.5 series (0.5B-Chat, 1.8B-Chat, 7B-Chat) |
| `--checkpoints` | Also download SpecDec++ acceptance heads from HuggingFace |
| `--force` | Re-download even if files already exist |
| `--dry-run` | Show what would happen without downloading |
| `--rl-guide` | Print RL agent checkpoint status and training commands |

**What the script downloads:**

| Category | Models | Source | Local path |
|----------|--------|--------|------------|
| Llama series | llama-68m, tiny-llama-1.1b, llama-2-13b | HuggingFace | `./llama/` |
| Qwen3 series | Qwen3-0.6B, Qwen3-1.7B, Qwen3-14B | HuggingFace | `./Qwen/` |
| Qwen1.5 series | Qwen1.5-0.5B-Chat, Qwen1.5-1.8B-Chat, Qwen1.5-7B-Chat | HuggingFace | `./Qwen/` |
| Acceptance heads | 7 model pairs (speculative decoding) | `ArcticHuaji/specdecpp-acc-heads` | `src/SpecDec_pp/checkpoints/acc_head/` |

**RL agent checkpoints** come in two forms:

- **Legacy checkpoints** (`.pth` / `.pth.buffer` under `checkpoints/`) are
  stored via Git LFS and included with the repository. Make sure you have
  [Git LFS](#git-lfs) installed, then run `git lfs pull` after cloning.

- **Pair-based checkpoints** (`checkpoints/rl_agents/`) are produced by local RL
  training and are **not** committed to the repo. Run
  `python scripts/download_models.py --rl-guide` to see the expected paths,
  current status, and training commands for each model series.

```bash
# Example: train RL agents for the llama series
LITTLE_MODEL=llama-68m DRAFT_MODEL=tiny-llama-1.1b TARGET_MODEL=llama-2-13b \
  bash cmds/train_rl_mixed.sh
```

### Model Resolution

The repository supports three ways to specify models:

- **local aliases** defined in `src/utils.py::model_zoo` (e.g. `llama-68m`, `tiny-llama-1.1b`)
- **direct local paths** (e.g. `./llama/llama-68m`)
- **Hugging Face model IDs** (e.g. `Qwen/Qwen3-0.6B`)

For local aliases, models must be placed under paths expected by `model_zoo`
(e.g. `./llama/llama-68m`). The download script handles this automatically.

If paths don't match your environment, edit the `zoo` dict in `src/utils.py`.
  
### SpecDec++ Environment (optional)

The `src/SpecDec_pp/` subproject has its own dual-environment setup for training
acceptance prediction heads:

```bash
cd src/SpecDec_pp
uv venv --clear .venv --python 3.10
uv pip install --python .venv/bin/python -r requirements.txt
./scripts/setup_vllm_env.sh   # creates .venv-vllm for data generation
```

This is only needed if you plan to train new acceptance heads. Pre-trained heads
are downloaded by `scripts/download_models.py --checkpoints`.

## Validation

In a development environment, run `python -m pytest -q test/test_dependency_contract.py` to check the dependency lock. CPU checks do not test GPU compatibility or real-model throughput.

## Usage

There are two main ways to use this repository:

1. Run a single evaluation script in `eval/` with `python`.
2. Run a batch of predefined experiments through `exp.py`.

Model names can be passed as:

- aliases defined in `src/utils.py::model_zoo`, such as `llama-68m`, `tiny-llama-1.1b`, `llama-2-13b`, `tiny-vicuna-1b`, `vicuna-13b-v1.5`, `qwen-3-1.7b`, `qwen-3-14b`
- a local model path
- a Hugging Face model ID

### Run a Single Evaluation

Run a single task script directly with `python`. Keep the GPU mask assigned by your scheduler. On a workstation, expose only the GPU(s) you intend to use. The model loader decides placement across the visible devices.

Example: MT-Bench without judge:

```bash
python eval/eval_mt_bench_noeval.py \
    --eval_mode dist_spec \
    --draft_model tiny-llama-1.1b \
    --target_model llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --num_shots 5 \
    --temp 0.0 \
    --exp_name demo_mt_bench
```

Example: GSM8K with communication simulation:

```bash
python eval/eval_gsm8k.py \
    --eval_mode dist_split_spec \
    --draft_model tiny-llama-1.1b \
    --target_model llama-2-13b \
    --little_model llama-68m \
    --max_tokens 128 \
    --num_shots 8 \
    --edge_cloud_bandwidth 23.6 \
    --edge_end_bandwidth 563 \
    --cloud_end_bandwidth 23.6 \
    --transfer_top_k 1024 \
    --use_stochastic_comm \
    --exp_name demo_gsm8k
```

The output directory is created as:

```text
exp/<exp_name>/
```

### Supported `eval_mode`

The currently registered decoding modes are:

- `small`
- `large`
- `dist_spec` / `dsd`
- `dist_split_spec` / `dssd`
- `specedge` — [SpecEdge, single-request chain variant](docs/baselines.md#specedge): token-only verification with proactive edge drafting.
- `uncertainty_decoding` / `cuhlm`
- `tridecoding`（Table VIII Static 消融：`adaptive_tridecoding(_ablation="static")` 的 wrapper，固定 γ/top-k、无 RL/ARP）
- `adaptive_decoding`（独立方法：2 模型 + ARP）
- `adaptive_tridecoding` / `cee_sd`（CEE-SD，本方法）
- `cee_cuhlm` / `cee_dsd` / `cee_dssd`（Table VI CEE+ 组合）
- `ceesd_without_arp` / `ceesd_w/o_arp`（legacy：已归档，仅用于复现历史 run）

### Important Arguments

The common arguments are defined in `src/utils.py::parse_arguments()`.

| Argument | Meaning |
| --- | --- |
| `--eval_mode` | Decoding mode to run. |
| `--draft_model` | Draft model used by speculative methods. |
| `--target_model` | Target model used for verification or autoregressive baseline. |
| `--little_model` | The smallest model used by tri-decoding style methods. |
| `--gamma` | Draft length for two-model speculative decoding. |
| `--specedge_lookahead` | SpecEdge only: how many next-round candidates to draft ahead of the reply, after predicting the bonus token (default: `gamma`; `0` disables proactive drafting). |
| `--gamma1`, `--gamma2` | Draft lengths for the two stages in tri-decoding. |
| `--max_tokens` | Maximum number of generated tokens. |
| `--num_shots` | Few-shot examples for supported tasks. |
| `--eval_data_num` | Number of evaluation samples to run. |
| `--temp`, `--top_k`, `--top_p` | Sampling parameters. |
| `--edge_cloud_bandwidth`, `--edge_end_bandwidth`, `--cloud_end_bandwidth` | Link bandwidths for communication simulation. |
| `--ntt_ms_edge_cloud`, `--ntt_ms_edge_end` | Extra link latency in milliseconds. |
| `--transfer_top_k` | Top-k compression size for transmitted logits / probabilities. |
| `--use_precise` | Use the physics-level communication simulator. |
| `--use_stochastic_comm` | Use stochastic communication simulation. |
| `--use_early_stopping` | Enable early stopping inside supported decoding loops. |
| `--acc_head_path` | Acceptance head path for `adaptive_decoding`. |
| `--small_draft_acc_head_path`, `--draft_target_acc_head_path` | Acceptance head paths for `adaptive_tridecoding` and `cee_cuhlm`. |
| `--use_rl_adapter` | Enable RL-based threshold selection. |
| `--main_rl_path`, `--little_rl_path` | RL agent latest checkpoints. |
| `--main_rl_best_path`, `--little_rl_best_path` | RL agent best checkpoints. |
| `--disable_rl_update` | Freeze RL adapter updates during evaluation / inference. |

>[!NOTE]
> `adaptive_decoding` and `adaptive_tridecoding` depend on acceptance prediction heads. The repository now resolves acceptance heads through `src.acc_head_registry` and RL checkpoints through `src.rl_agent_registry`, so most common model pairs no longer need hardcoded local paths.

### RL Agent Checkpoints

RL checkpoints now use a pair-based layout rather than fixed filenames such as
`checkpoints/rl_adapter_main.pth`.

Default layout:

```text
checkpoints/
  rl_agents/
    main/
      <draft_alias>--to--<target_alias>/
        latest.pth
        best.pth
    little/
      <little_alias>--to--<draft_alias>/
        latest.pth
        best.pth
```

Examples:

```text
checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth
```

Loading behavior:

1. load `best.pth` if it exists
2. otherwise load `latest.pth`
3. otherwise try legacy hardcoded checkpoint locations
4. otherwise start RL training from scratch

During online RL training:

- `latest.pth` is updated continuously
- `best.pth` is updated when a higher TPS is observed

You can resolve checkpoint paths from the command line:

```bash
python -m src.rl_agent_registry main "tiny-llama-1.1b" "Llama-2-13b" --kind latest --format path
python -m src.rl_agent_registry little "llama-68m" "tiny-llama-1.1b" --kind best --format path

# Or use the download script to check RL agent status
python scripts/download_models.py --rl-guide
```

The training scripts already use this resolver automatically:

```bash
bash cmds/train_rl_mixed.sh
```

You can also override the models and keep automatic RL path resolution:

```bash
LITTLE_MODEL="llama-68m" \
DRAFT_MODEL="tiny-llama-1.1b" \
TARGET_MODEL="Llama-2-13b" \
bash cmds/train_rl_mixed.sh
```

### Paper Reruns and Batch Experiments

The staged paper runner separates the 45 configurations in the 46 Mbps main table from the full bandwidth sweep. From the repository root, preview the 63 core configurations without GPUs, model weights, or dataset caches:

```bash
python -m src.experiments.paper_rerun --stage core --dry-run
```

Before executing, repeat the command with `--preflight` instead of `--dry-run` to check the local models, checkpoints, network trace, and datasets without starting GPU jobs. Once it passes, run with `--execute`, a fresh `--run-id`, and the physical `nvidia-smi` indices of the idle GPUs you have reserved. `--gpus 0` uses one card. `--gpus 0,1` exposes both cards to one worker and leaves placement to the model loader. To run two configurations in parallel on one card each, use `--gpus 0 --gpus 1` instead. GPU groups cannot overlap. No particular GPU model or minimum VRAM is required; 48 GB is a sizing reference, not a fit check.

**Comparing performance:** GPU model, device order, interconnect, and cross-GPU transfers can change measured throughput and latency even when the simulated link settings are identical. Compare methods using the same hardware and placement, or report layouts separately. The manifest records the GPUs visible to each configuration; check the worker log for loading decisions. Visibility does not guarantee every GPU is used.

For the **legacy, environment-tunable sweep**, preview the default Table V matrix (135 configurations) before allocating GPUs:

```bash
python exp.py --dry-run
python exp.py --extra-modes=tridecoding --only-modes tridecoding --dry-run
python exp.py --only-modes dsd,dssd --max-workers 2 --summary-file experiment_results/current.json
```

The scheduler groups jobs by the weights they actually load. Each group runs serially in a persistent `eval.worker` process on its assigned GPUs; independent groups can run in parallel. Each completed job gets its own log under `exp_logs/` and is checkpointed to the batch summary. To retry only unfinished configurations and merge the results:

```bash
python exp.py --only-modes dsd,dssd --resume-from experiment_results/current.json \
    --merge-from experiment_results/current.json --merged-output experiment_results/merged.json
```

For a custom sweep, `from exp import create_config, run_experiments_parallel` and pass a list of configurations. Importing the module does not parse arguments or materialize the matrix. `build_matrix()` returns a fresh list; the old `config_to_run` name remains available on request. Edit [matrix.py](src/experiments/matrix.py) to change Table V, not the CLI.

### Debug Checks

The repository provides two optional debug checks that are disabled by default:

- `DUODEC_DEBUG_NUMERICS=1`: enable probability / acceptance-ratio validity checks during generation
- `DUODEC_DEBUG_TOKEN_CHECKS=1`: enable output token range checks in MT-Bench evaluation

Example:

```bash
DUODEC_DEBUG_NUMERICS=1 DUODEC_DEBUG_TOKEN_CHECKS=1 python exp.py
```

Leave both variables unset for normal benchmarking.

### Reading Results

- Single-run outputs are written under `exp/<exp_name>/`.
- Batch logs are written under `exp_logs/`.
- By default, batch summaries are written to `experiment_results/experiment_summary_<timestamp>.json`.
- `notebooks/table_generator_ver2.ipynb` can be pointed to a summary JSON for result aggregation.

### Bash Scripts and vLLM Test Scripts

- `cmds/` contains project-specific shell scripts such as `cmds/test.sh` and `cmds/train_rl.sh`.
- For a separate GSM8K benchmark, run `python test/eval_gsm8k_vllm.py --help` in a vLLM environment.
