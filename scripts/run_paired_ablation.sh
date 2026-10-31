#!/bin/bash
# 成对交替 A/B 跑消融：等 GPU 空闲 → A/B 或 B/A 交替 → 每轮落 metrics/jsonl。
#
# 需要环境变量（见下），示例：
#   GPU=1 REPS=3 OUT_ROOT=/data/user2/paired_ab \
#   A_NAME=cee_sd A_MODE=adaptive_tridecoding A_EXTRA="$ACC" \
#   B_NAME=cee_sd_no_arp B_MODE=cee_sd_no_arp B_EXTRA="" \
#   COMMON="--protocol paper_table5 ..." \
#   bash scripts/run_paired_ablation.sh
#
# 设计目的：共享 GPU 上单次吞吐不可比；A/B 交替 + 多轮取 median，结构指标
# （forwards/token、bytes/token）用于硬件无关对照。
set -u

: "${GPU:=1}"
: "${REPS:=3}"
: "${OUT_ROOT:=/data/user2/paired_ab}"
: "${EVAL_SCRIPT:=eval/eval_gsm8k.py}"
: "${A_NAME:=A}"
: "${A_MODE:=adaptive_tridecoding}"
: "${A_EXTRA:=}"
: "${B_NAME:=B}"
: "${B_MODE:=cee_sd_no_arp}"
: "${B_EXTRA:=}"
: "${COMMON:=}"
: "${MEM_FREE_MIN_MB:=46000}"
: "${UTIL_MAX_PCT:=5}"
: "${IDLE_SAMPLES:=3}"
: "${POLL_SECONDS:=20}"
: "${WAIT_TIMEOUT_SECONDS:=21600}"
: "${PYTHON:=.venv/bin/python}"
: "${ACCELERATE:=.venv/bin/accelerate}"

if [ -z "$COMMON" ]; then
  echo "COMMON 不能为空；请传入完整的评测参数（不含 --eval_mode/--exp_name）" >&2
  exit 2
fi

STAMP=$(date +%Y%m%d_%H%M%S)
RUN_ROOT="$OUT_ROOT/paired_$STAMP"
mkdir -p "$RUN_ROOT" "$RUN_ROOT/logs"
export HF_HOME="${HF_HOME:-$HOME/hf}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export CUDA_VISIBLE_DEVICES="$GPU"

wait_idle() {
  local start now free util streak
  start=$(date +%s)
  streak=0
  while :; do
    now=$(date +%s)
    if [ $((now - start)) -ge "$WAIT_TIMEOUT_SECONDS" ]; then
      echo "[wait] timeout after ${WAIT_TIMEOUT_SECONDS}s" >&2
      return 1
    fi
    read -r free util < <(
      nvidia-smi -i "$GPU" --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits | tr -d ' '
    )
    if [ "${free:-0}" -ge "$MEM_FREE_MIN_MB" ] && [ "${util:-100}" -le "$UTIL_MAX_PCT" ]; then
      streak=$((streak + 1))
      echo "[wait] gpu=$GPU free=${free}MiB util=${util}% streak=$streak/$IDLE_SAMPLES"
    else
      streak=0
      echo "[wait] gpu=$GPU free=${free:-?}MiB util=${util:-?}% (need >=${MEM_FREE_MIN_MB}MiB, <=${UTIL_MAX_PCT}%)"
    fi
    [ "$streak" -ge "$IDLE_SAMPLES" ] && return 0
    sleep "$POLL_SECONDS"
  done
}

run_one() {
  local rep=$1 slot=$2 name=$3 mode=$4 extra=$5
  local dir="$RUN_ROOT/rep${rep}/$slot"
  mkdir -p "$dir"
  local exp_dir="$dir/exp"
  echo "[run] rep=$rep slot=$slot name=$name mode=$mode $(date -Is)"
  "$ACCELERATE" launch --num_processes 1 --main_process_port $((29900 + RANDOM % 90)) \
    "$EVAL_SCRIPT" --eval_mode "$mode" $COMMON $extra \
    --exp_name "$exp_dir" > "$dir/run.log" 2>&1
  local rc=$?
  echo "[run] rc=$rc"
  local metrics jsonl
  metrics=$(find "$exp_dir" -maxdepth 1 -name '*_metrics.json' | sort | tail -1)
  jsonl=$(find "$exp_dir" -maxdepth 1 -name '*.jsonl' | sort | tail -1)
  if [ -n "${metrics:-}" ]; then cp "$metrics" "$dir/metrics.json"; fi
  if [ -n "${jsonl:-}" ]; then cp "$jsonl" "$dir/run.jsonl"; fi
  echo "$name" > "$dir/mode_name.txt"
  return $rc
}

echo "[start] root=$RUN_ROOT gpu=$GPU reps=$REPS A=$A_NAME B=$B_NAME"
for rep in $(seq 1 "$REPS"); do
  if [ $((rep % 2)) -eq 1 ]; then
    wait_idle || exit 1
    run_one "$rep" A "$A_NAME" "$A_MODE" "$A_EXTRA" || exit 1
    wait_idle || exit 1
    run_one "$rep" B "$B_NAME" "$B_MODE" "$B_EXTRA" || exit 1
  else
    wait_idle || exit 1
    run_one "$rep" B "$B_NAME" "$B_MODE" "$B_EXTRA" || exit 1
    wait_idle || exit 1
    run_one "$rep" A "$A_NAME" "$A_MODE" "$A_EXTRA" || exit 1
  fi
done
echo "[all done] $RUN_ROOT"
