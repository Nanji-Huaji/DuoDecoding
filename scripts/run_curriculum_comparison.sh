#!/bin/bash
# Post-curriculum A/B evaluation with DETERMINISTIC network conditions.
#
# IMPORTANT: eval/eval_mixed.py's eval() loop overrides --edge_cloud_bandwidth
# per sample via the curriculum sampler. To pin an exact condition we collapse
# the curriculum ranges to a single point (uniform(x, x) == x).
#
# A: curriculum @ 23.6 Mbps   B: old(fixed-range) @ 23.6 Mbps
# C: curriculum @ 5 Mbps      D: old @ 5 Mbps
# E: curriculum @ 1 Mbps      F: old @ 1 Mbps
# G: no-RL @ 5 Mbps
#
# Usage: bash scripts/run_curriculum_comparison.sh [GPU_ID]

set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID=${1:-0}
PORT_BASE=29650
ACC_ROOT=src/SpecDec_pp/checkpoints/acc_head
OLD_MAIN=checkpoints_backup_pre_curriculum/main/tiny-llama-1.1b--to--llama-2-13b
OLD_LITTLE=checkpoints_backup_pre_curriculum/little/llama-68m--to--tiny-llama-1.1b
# Curriculum agent: use the FINAL trained policy (latest.pth, end of curriculum).
# The old agent: its best.pth (same convention as the previous A/B).
CURR_MAIN=checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
CURR_LITTLE=checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth

COMMON=(
  --eval_mode adaptive_tridecoding
  --draft_model tiny-llama-1.1b
  --target_model llama-2-13b
  --little_model llama-68m
  --max_tokens 128
  --temp 0.0
  --min_bandwidth_mbps 0.1
  --edge_end_bandwidth 563
  --target_quantization 4bit
  --eval_data_num 40
  --small_draft_acc_head_path "$ACC_ROOT/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC_ROOT/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --acc_head_path "$ACC_ROOT/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
)

# Deterministic condition helper: collapses the curriculum ranges to a point.
# bw/ntt are the exact per-sample values the loop will draw.
fixed_cond() {
  local bw=$1 ntt=$2
  echo --curriculum_bw_start "$bw,$bw" --curriculum_bw_end "$bw,$bw" \
       --curriculum_ntt_start "$ntt,$ntt" --curriculum_ntt_end "$ntt,$ntt"
}

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

# A / B: 23.6 Mbps, 20 ms
run_eval eval_cur_23mbps $PORT "${COMMON[@]}" $(fixed_cond 23.6 20) \
  --use_rl_adapter --disable_rl_update \
  --main_rl_path "$CURR_MAIN" --main_rl_best_path "$CURR_MAIN" \
  --little_rl_path "$CURR_LITTLE" --little_rl_best_path "$CURR_LITTLE"
PORT=$((PORT + 1))
run_eval eval_old_23mbps $PORT "${COMMON[@]}" $(fixed_cond 23.6 20) \
  --use_rl_adapter --disable_rl_update \
  --main_rl_path "$OLD_MAIN/best.pth" --main_rl_best_path "$OLD_MAIN/best.pth" \
  --little_rl_path "$OLD_LITTLE/best.pth" --little_rl_best_path "$OLD_LITTLE/best.pth"
PORT=$((PORT + 1))

# C / D / G: 5 Mbps, 20 ms
run_eval eval_cur_5mbps $PORT "${COMMON[@]}" $(fixed_cond 5 20) \
  --use_rl_adapter --disable_rl_update \
  --main_rl_path "$CURR_MAIN" --main_rl_best_path "$CURR_MAIN" \
  --little_rl_path "$CURR_LITTLE" --little_rl_best_path "$CURR_LITTLE"
PORT=$((PORT + 1))
run_eval eval_old_5mbps $PORT "${COMMON[@]}" $(fixed_cond 5 20) \
  --use_rl_adapter --disable_rl_update \
  --main_rl_path "$OLD_MAIN/best.pth" --main_rl_best_path "$OLD_MAIN/best.pth" \
  --little_rl_path "$OLD_LITTLE/best.pth" --little_rl_best_path "$OLD_LITTLE/best.pth"
PORT=$((PORT + 1))
run_eval eval_norl_5mbps $PORT "${COMMON[@]}" $(fixed_cond 5 20)
PORT=$((PORT + 1))

# E / F: 1 Mbps, 20 ms
run_eval eval_cur_1mbps $PORT "${COMMON[@]}" $(fixed_cond 1 20) \
  --use_rl_adapter --disable_rl_update \
  --main_rl_path "$CURR_MAIN" --main_rl_best_path "$CURR_MAIN" \
  --little_rl_path "$CURR_LITTLE" --little_rl_best_path "$CURR_LITTLE"
PORT=$((PORT + 1))
run_eval eval_old_1mbps $PORT "${COMMON[@]}" $(fixed_cond 1 20) \
  --use_rl_adapter --disable_rl_update \
  --main_rl_path "$OLD_MAIN/best.pth" --main_rl_best_path "$OLD_MAIN/best.pth" \
  --little_rl_path "$OLD_LITTLE/best.pth" --little_rl_best_path "$OLD_LITTLE/best.pth"

echo "All comparison runs finished."
for log in exp_logs/eval_cur_23mbps.log exp_logs/eval_old_23mbps.log \
           exp_logs/eval_cur_5mbps.log exp_logs/eval_old_5mbps.log \
           exp_logs/eval_norl_5mbps.log \
           exp_logs/eval_cur_1mbps.log exp_logs/eval_old_1mbps.log; do
  echo "--- $log ---"
  .venv/bin/python scripts/summarize_rl_training_log.py "$log" --tail 40
done
