# RL Agent Checkpoints

This project now resolves RL agent checkpoints through `src.rl_agent_registry`
instead of hardcoded checkpoint filenames.

## Roles

There are two RL agent roles in adaptive tri-decoding style workflows:

- `main`: controls the `draft_model -> target_model` stage
- `little`: controls the `little_model -> draft_model` stage

Each role is tied to a directed model pair.

## Default Layout

RL checkpoints are stored under a pair-based directory layout:

```text
checkpoints/
  rl_agents/
    main/
      <draft_alias>--to--<target_alias>/
        latest.pth
        latest.pth.buffer
        best.pth
        best.pth.buffer
    little/
      <little_alias>--to--<draft_alias>/
        latest.pth
        latest.pth.buffer
        best.pth
        best.pth.buffer
```

Examples:

```text
checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth
```

This avoids collisions between different model combinations and keeps RL agent
storage consistent with the pair-based acceptance head layout.

## Training And Evaluation Behavior

`RLNetworkAdapter` now follows this loading order:

1. load `best.pth` if it exists
2. otherwise load `latest.pth`
3. otherwise try legacy checkpoint paths
4. otherwise start from scratch

During online RL training:

- `latest.pth` is updated continuously
- `best.pth` is updated only when a better TPS is observed

This means training and later evaluation can share the same pair-based location
without extra path wiring.

## Legacy Compatibility

Older checkpoints may still live under legacy paths such as:

```text
checkpoints/<series>/rl_adapter_main.pth
checkpoints/<series>/rl_adapter_little.pth
checkpoints/rl_adapter_main.pth
checkpoints/rl_adapter_little.pth
```

The runtime keeps a legacy fallback for loading. If a legacy checkpoint is
found and the new pair-based path is missing, the checkpoint is loaded and then
saved to the new `latest.pth` location.

New saves always go to the pair-based layout.

## Command Line Usage

Resolve a pair name:

```bash
python -m src.rl_agent_registry main "tiny-llama-1.1b" "Llama-2-13b" --format pair
```

Resolve the default latest checkpoint path:

```bash
python -m src.rl_agent_registry main "tiny-llama-1.1b" "Llama-2-13b" --kind latest --format path
```

Resolve the default best checkpoint path:

```bash
python -m src.rl_agent_registry little "llama-68m" "tiny-llama-1.1b" --kind best --format path
```

Resolve the default agent log name:

```bash
python -m src.rl_agent_registry main "tiny-llama-1.1b" "Llama-2-13b" --format agent-name
```

## Python Usage

Use the resolver directly from Python:

```python
from src.rl_agent_registry import ROLE_LITTLE, ROLE_MAIN, get_rl_agent_spec

main_spec = get_rl_agent_spec(
    ROLE_MAIN,
    little_model="llama-68m",
    draft_model="tiny-llama-1.1b",
    target_model="Llama-2-13b",
)

little_spec = get_rl_agent_spec(
    ROLE_LITTLE,
    little_model="llama-68m",
    draft_model="tiny-llama-1.1b",
    target_model="Llama-2-13b",
)

print(main_spec.latest_path)
print(main_spec.best_path)
print(little_spec.latest_path)
print(little_spec.best_path)
```

Typical integration points:

- `src/baselines.py` uses `get_rl_agent_spec(...)` to instantiate RL adapters
- `src/utils.py` auto-fills `--main_rl_path` and `--little_rl_path`
- `cmds/train_rl.sh` and `cmds/train_rl_mixed.sh` resolve defaults through the CLI
- `auto_train_manager.py` passes pair-based paths into online training runs

## Runtime Arguments

The main runtime arguments are:

- `--main_rl_path`: path to the main agent latest checkpoint
- `--main_rl_best_path`: path to the main agent best checkpoint
- `--little_rl_path`: path to the little agent latest checkpoint
- `--little_rl_best_path`: path to the little agent best checkpoint
- `--use_rl_adapter`: enable RL-based threshold selection
- `--disable_rl_update`: load and use the RL agents without updating them

If these paths are omitted, the project computes them automatically from the
current `little_model`, `draft_model`, and `target_model`.

## Recommended Workflows

Online RL training with automatic defaults:

```bash
bash cmds/train_rl_mixed.sh
```

Online RL training for a specific trio:

```bash
LITTLE_MODEL="llama-68m" \
DRAFT_MODEL="tiny-llama-1.1b" \
TARGET_MODEL="Llama-2-13b" \
bash cmds/train_rl_mixed.sh
```

Evaluation with RL enabled and automatic pair-based checkpoint resolution:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --main_process_port 29051 \
  eval/eval_mixed.py \
  --eval_mode adaptive_tridecoding \
  --draft_model tiny-llama-1.1b \
  --target_model Llama-2-13b \
  --little_model llama-68m \
  --use_rl_adapter
```

Evaluation with explicit frozen checkpoints:

```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
  --num_processes 1 \
  --main_process_port 29051 \
  eval/eval_mixed.py \
  --eval_mode adaptive_tridecoding \
  --draft_model tiny-llama-1.1b \
  --target_model Llama-2-13b \
  --little_model llama-68m \
  --use_rl_adapter \
  --disable_rl_update \
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth \
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth \
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth \
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth
```
