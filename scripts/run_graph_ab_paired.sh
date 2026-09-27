#!/bin/bash
# CUDA Graph 的**配对**端到端 A/B：同一时段内交替跑 eager / graph 两组，
# 两组参数逐字相同，唯一差别是 --use_cuda_graph。
#
# 为什么要配对：本机长期有第三方任务，不同时段的吞吐能差 ±44%。只有把两个
# 变体放在同一时段、交替执行，它们的差异才能归因给开关本身，而不是机器状态。
#
# 顺序交替（奇偶轮对调）进一步抵消"总是先跑的那一侧占便宜"。
#
# 用法: GPU=0 N=40 ROUNDS=2 bash scripts/run_graph_ab_paired.sh
#   TAG_PREFIX=vg  结果目录前缀（默认 ab；换前缀避免覆盖历史配对）
#   GATE_SKIP=1   跳过内部守门（外层脚本已守门时用）
set -uo pipefail
cd /home/tiantianyi/code/DuoDecoding

GPU=${GPU:-0}
N=${N:-40}
ROUNDS=${ROUNDS:-2}
MAXTOK=${MAXTOK:-128}
PORT_BASE=${PORT_BASE:-29600}
TAG_PREFIX=${TAG_PREFIX:-ab}
GATE_SKIP=${GATE_SKIP:-0}

ACC=src/SpecDec_pp/checkpoints/acc_head
CK=checkpoints/rl_fixedg16_nobp

COMMON=(
  --data_path data
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --temp 0.0 --seed 20260911 --target_quantization 4bit --model_dtype bf16
  --num_shots 3 --eval_data_num "$N" --max_tokens "$MAXTOK"
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 0.5 --min_bandwidth_mbps 0.1
  --ntt_ms_edge_cloud 20
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
)

METHOD=(
  --eval_mode adaptive_tridecoding
  # 通信计费钉住遗留口径：与 exp/vg_*、exp/vg2_* 及 exp/ab_* 历史序列保持同口径
  # （仓库默认自 2026-09-25 起已改为诚实口径，见 src/utils.py；改口径会换基线）
  --no-charge_residual_payload --comm_round_trip_mode per_transfer --transfer_top_k_cap 0
  --use_rl_adapter --disable_rl_update --rl_action_space topk_thr
  --rl_force_threshold 0.4 --gamma1 16 --gamma2 16
  --main_rl_path "$CK/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth"
  --main_rl_best_path "$CK/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth"
  --little_rl_path "$CK/little/llama-68m--to--tiny-llama-1.1b/latest.pth"
  --little_rl_best_path "$CK/little/llama-68m--to--tiny-llama-1.1b/latest.pth"
)

if [ "$GATE_SKIP" != "1" ]; then
  echo "### 守门：等 GPU $GPU 空闲 ($(date +%H:%M:%S))"
  if ! .venv/bin/python scripts/wait_gpu_idle.py --gpu "$GPU" \
        --timeout "${GATE_TIMEOUT:-7200}"; then
    echo "### 守门未放行，结束"; exit 1
  fi
fi

run () {  # run <tag> <port> [extra...]
  local tag="$1"; local port="$2"; shift 2
  echo "### $tag  ($(date +%H:%M:%S))"
  CUDA_VISIBLE_DEVICES=$GPU \
  HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
  .venv/bin/accelerate launch --num_processes 1 --main_process_port "$port" \
    eval/eval_gsm8k.py -e "$tag" "${COMMON[@]}" "${METHOD[@]}" "$@" \
    2>&1 | tee "exp_logs/$tag.log" | grep -E "GSM8K Accuracy|error:"
  echo "### $tag 结束 ($(date +%H:%M:%S))"
}

port=$PORT_BASE
for r in $(seq 1 "$ROUNDS"); do
  if [ $((r % 2)) -eq 1 ]; then ORDER="eager cuda"; else ORDER="cuda eager"; fi
  echo "### 第 $r 轮，顺序: $ORDER"
  for MODE in $ORDER; do
    EXTRA=()
    [ "$MODE" = cuda ] && EXTRA=(--use_cuda_graph)
    run "${TAG_PREFIX}_${MODE}_r${r}" "$port" "${EXTRA[@]}"
    port=$((port + 1))
  done
done

echo
echo "### 汇总（配对比较）"
.venv/bin/python scripts/summarize_graph_ab.py "exp/${TAG_PREFIX}_*_r*"
