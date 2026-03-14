# DuoDecoding

This is an experiment framework based on Duodecoding.

## Setup


Firstly, install the requirements with:
```bash
pip install -r requirements.txt
```

Then, prepare the models you want to evaluate. The repository currently supports:

- local aliases defined in `src/utils.py::model_zoo`
- direct local paths
- Hugging Face model IDs

The following are commonly used local alias targets:

Llama Series:

- [Llama-68M](https://huggingface.co/JackFram/llama-68m)
- [TinyLlama-1.1B](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0)
- [Llama-2-13B](https://huggingface.co/meta-llama/Llama-2-13b)

Vicuna Series:

- [Vicuna-68M](https://huggingface.co/double7/vicuna-68m)
- [TinyVicuna-1B](https://huggingface.co/Jiayi-Pan/Tiny-Vicuna-1B)
- [Vicuna-13B-v1.5](https://huggingface.co/lmsys/vicuna-13b-v1.5)

For the local alias-based workflow, place them under paths expected by `model_zoo`, for example:
```
./llama/<your-model-dir>
./vicuna/<your-model-dir>
```

Some newer models, such as Qwen variants, are already mapped to Hugging Face IDs in `model_zoo`, so they do not need to follow the `./llama/...` or `./vicuna/...` layout.

If a path does not match your environment, modify `model_zoo` in `src/utils.py`.

Model paths are defined on the `zoo` dict on the `model_zoo` function. And their vocab sizes are defined on the `vocab_size` dict on the same function.

## Usage

There are two main ways to use this repository now:

1. Run a single evaluation script in `eval/` with `accelerate launch`.
2. Run a batch of predefined experiments through `exp.py`.

Model names can be passed as:

- aliases defined in `src/utils.py::model_zoo`, such as `llama-68m`, `tiny-llama-1.1b`, `llama-2-13b`, `tiny-vicuna-1b`, `vicuna-13b-v1.5`, `qwen-3-1.7b`, `qwen-3-14b`
- a local model path
- a Hugging Face model ID

### Run a Single Evaluation

The most direct workflow is to launch one task script under `eval/`.

Example: MT-Bench without judge:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --main_process_port 29051 \
    eval/eval_mt_bench_noeval.py \
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
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --num_processes 1 \
    --main_process_port 29052 \
    eval/eval_gsm8k.py \
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
- `uncertainty_decoding` / `cuhlm`
- `tridecoding`
- `adaptive_decoding`
- `adaptive_tridecoding` / `cee_sd`
- `cee_cuhlm`
- `cee_dsd`
- `cee_dssd`
- `ceesd_without_arp` / `ceesd_w/o_arp`

### Important Arguments

The common arguments are defined in `src/utils.py::parse_arguments()`.

| Argument | Meaning |
| --- | --- |
| `--eval_mode` | Decoding mode to run. |
| `--draft_model` | Draft model used by speculative methods. |
| `--target_model` | Target model used for verification or autoregressive baseline. |
| `--little_model` | The smallest model used by tri-decoding style methods. |
| `--gamma` | Draft length for two-model speculative decoding. |
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
| `--main_rl_path`, `--little_rl_path` | RL adapter checkpoints. |
| `--disable_rl_update` | Freeze RL adapter updates during evaluation / inference. |

>[!NOTE]
> `adaptive_decoding` and `adaptive_tridecoding` depend on acceptance prediction heads. Some checkpoint paths are already wired in `exp.py` through `model_acc_head_map`, but if your local paths differ you still need to pass the correct head checkpoints explicitly.

### Batch Experiments via `exp.py`

`exp.py` is a batch runner, not a generic CLI wrapper. It does the following:

- builds `config_to_run` in Python
- detects idle GPUs with NVML
- launches experiments in parallel
- writes per-run logs to `exp_logs/`
- writes a summary JSON to the repository root as `experiment_summary_<timestamp>.json`

Run it with:

```bash
python exp.py
```

Before doing that, edit the `create_config(...)` calls near the bottom of `exp.py` to match the experiments you actually want to run.

Current caveat: `exp.py` launches `accelerate` through the hard-coded path `/home/tiantianyi/code/DuoDecoding/.venv/bin/accelerate`. If your environment is different, update `cmd_temp` in `exp.py` first.

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
- Batch summaries are written as `experiment_summary_<timestamp>.json`.
- `table_generator_ver2.ipynb` can be pointed to a summary JSON for result aggregation.

### Bash Scripts and vLLM Test Scripts

- `cmds/` contains project-specific shell scripts such as `cmds/test.sh` and `cmds/train_rl.sh`.
- `test/` contains separate vLLM-based evaluation utilities. See `test/README.md` and `test/QUICKSTART.md` if you want a lightweight benchmarking path outside the main `eval/` pipeline.
