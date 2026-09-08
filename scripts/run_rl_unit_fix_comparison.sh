#!/bin/bash
# Post-fix RL comparison experiments, sequential on one GPU.
#
# A: new agent @ 23.6 Mbps (in-distribution)
# B: old legacy agent @ 23.6 Mbps
# C: new agent @ 5 Mbps (constrained link)
# D: old legacy agent @ 5 Mbps
# E: no RL @ 23.6 Mbps (reference)
#
# Usage: bash scripts/run_rl_unit_fix_comparison.sh [GPU_ID]

set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID=${1:-3}
PORT_BASE=29630
ACC_ROOT=src/SpecDec_pp/checkpoints/acc_head
OLD_MAIN=checkpoints_llama/rl_adapter_main.pth
OLD_LITTLE=checkpoints_llama/rl_adapter_little.pth

COMMON=(
  --eval_mode adaptive_tridecoding
  --draft_model tiny-llama-1.1b
  --target_model llama-2-13b
  --little_model llama-68m
  --max_tokens 128
  --temp 0.0
  --use_stochastic_comm
  --ntt_ms_edge_cloud 10
  --edge_end_bandwidth 563
  --target_quantization 4bit
  --eval_data_num 60
  --small_draft_acc_head_path "$ACC_ROOT/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC_ROOT/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --acc_head_path "$ACC_ROOT/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
)

run_eval() {
  local exp_name=$1
  local port=$2
  shift 2
  echo "=================== $exp_name ==================="
  CUDA_VISIBLE_DEVICES=$GPU_ID .venv/bin/accelerate launch \
    --num_processes 1 --main_process_port $port \
    eval/eval_mixed.py -e "$exp_name" "$@" 2>&1 | tee "exp_logs/$exp_name.log"
}

PORT=$PORT_BASE

# A. New agent @ 23.6 Mbps (pair paths resolved automatically).
run_eval eval_rl_new_23mbps $PORT "${COMMON[@]}" \
  --edge_cloud_bandwidth 23.6 --use_rl_adapter --disable_rl_update
PORT=$((PORT + 1))

# B. Old legacy agent @ 23.6 Mbps.
run_eval eval_rl_old_23mbps $PORT "${COMMON[@]}" \
  --edge_cloud_bandwidth 23.6 --use_rl_adapter --disable_rl_update \
  --main_rl_path "$OLD_MAIN" --main_rl_best_path "$OLD_MAIN" \
  --little_rl_path "$OLD_LITTLE" --little_rl_best_path "$OLD_LITTLE"
PORT=$((PORT + 1))

# C. New agent @ 5 Mbps.
run_eval eval_rl_new_5mbps $PORT "${COMMON[@]}" \
  --edge_cloud_bandwidth 5 --use_rl_adapter --disable_rl_update
PORT=$((PORT + 1))

# D. Old legacy agent @ 5 Mbps.
run_eval eval_rl_old_5mbps $PORT "${COMMON[@]}" \
  --edge_cloud_bandwidth 5 --use_rl_adapter --disable_rl_update \
  --main_rl_path "$OLD_MAIN" --main_rl_best_path "$OLD_MAIN" \
  --little_rl_path "$OLD_LITTLE" --little_rl_best_path "$OLD_LITTLE"
PORT=$((PORT + 1))

# E. No-RL reference @ 23.6 Mbps.
run_eval eval_no_rl_23mbps $PORT "${COMMON[@]}" \
  --edge_cloud_bandwidth 23.6

echo "All comparison runs finished."
for log in exp_logs/eval_rl_new_23mbps.log exp_logs/eval_rl_old_23mbps.log \
           exp_logs/eval_rl_new_5mbps.log exp_logs/eval_rl_old_5mbps.log \
           exp_logs/eval_no_rl_23mbps.log; do
  echo "--- $log ---"
  .venv/bin/python scripts/summarize_rl_training_log.py "$log" --tail 60
done
