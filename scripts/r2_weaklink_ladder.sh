#!/bin/bash
# R2 弱链路阶梯（代表性场景：MEC 低RTT波动 + 带宽敏感）
#   NTT 10ms + stochastic_ntt(拥塞动态) + top-k cap 1024 + bw {0.5,1,2,5}
#   静态 pin: k {64,256,1024} × thr 0.4（R1 最优阈值），γ16
#   之后自动接: R2 域 DRA 重训(curriculum bw 0.5-5 loguniform) → DRA 探针 → 汇总
# R1(Table V 协议)结果原样保留;RTT 敏感性实验后补(sigcomm ping/ 真实数据可回放)。
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29810}
ACC=src/SpecDec_pp/checkpoints/acc_head

NTT_MS=${NTT_MS:-10}
TAGPFX=${TAGPFX:-r2}
NTT_TRACE_FILE=${NTT_TRACE_FILE:-}
NTT_TRACE_FLAGS=()
[ -n "$NTT_TRACE_FILE" ] && NTT_TRACE_FLAGS=(--ntt_trace_file "$NTT_TRACE_FILE" --ntt_trace_scale "${NTT_TRACE_SCALE:-1.0}")
SUMMARY=${SUMMARY:-exp_logs/${TAGPFX}_ladder_summary.txt}
COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 80
  --temp 0.0 --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud ${NTT_MS} --ntt_ms_edge_end 0.317 --batch_delay 0.05 ${NTT_TRACE_FLAGS[@]}
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm --stochastic_ntt
  --use_rl_adapter --disable_rl_update --use_cuda_graph --task_name gsm8k
  --gamma1 16 --gamma2 16 --gamma 5
  --rl_force_threshold 0.4
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth)

run() { local tag=$1 bw=$2 k=$3
  if ls exp/${tag}/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### $tag bw=$bw k=$k ($(date +%H:%M:%S))"
  mkdir -p "exp/${tag}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "${tag}" \
    --edge_cloud_bandwidth "$bw" --transfer_top_k "$k" --rl_force_topk "$k" "${COMMON[@]}" \
    > "exp_logs/${tag}.log" 2>&1
  echo "  exit=$? ($(date +%H:%M:%S))"
}

# ---- 1) 静态阶梯: 4 bw × 3 k = 12 跑 (~2h) ----
for bw in 0.5 1 2 5; do
  for k in 64 256 1024; do
    run "${TAGPFX}_bw${bw}_k${k}_thr04" "$bw" "$k"
  done
done

# ---- 2) R2 域 DRA 重训 (TRAIN=0 跳过; curriculum bw 0.5-5 loguniform) ----
if [ "${TRAIN:-1}" = "1" ]; then
echo "=== R2 DRA 重训 $(date +%H:%M:%S) ==="
OUT=checkpoints/rl_r2_weaklink EXP=rl_r2_weaklink PORT=29985 \
CURR_BW="0.5,5" CURR_NTT="10,10" SAMPLING=loguniform \
LAMBDA=0 RL_EXTRA="--rl_charge_queue" \
  bash scripts/run_rl_inregime_training.sh "$GPU" 500 \
  > exp_logs/nohup_train_r2.log 2>&1
echo "  训练 exit=$? ($(date +%H:%M:%S))"
else echo "  TRAIN=0 跳过重训"; fi

# ---- 3) DRA 探针 @ 4 个 bw点 (k 由策略自适应, cap 1024) ----
NEWCKPT=checkpoints/rl_r2_weaklink
for bw in 0.5 1 2 5; do
  tag="${TAGPFX}_bw${bw}_dra"
  if ls exp/${tag}/*_metrics.json >/dev/null 2>&1; then continue; fi
  mkdir -p "exp/${tag}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "${tag}" \
    --edge_cloud_bandwidth "$bw" --transfer_top_k 1024 \
    --main_rl_path "$NEWCKPT/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth" \
    --little_rl_path "$NEWCKPT/little/llama-68m--to--tiny-llama-1.1b/latest.pth" \
    --main_rl_best_path "$NEWCKPT/main/tiny-llama-1.1b--to--llama-2-13b/best.pth" \
    --little_rl_best_path "$NEWCKPT/little/llama-68m--to--tiny-llama-1.1b/best.pth" \
    "${COMMON[@]}" > "exp_logs/${tag}.log" 2>&1
  echo "  $tag exit=$? ($(date +%H:%M:%S))"
done

# ---- 4) 汇总: 每 bw 的静态最优 vs DRA ----
.venv/bin/python scripts/summarize_r2_ladder.py "$TAGPFX" "$SUMMARY"
cat "$SUMMARY"
echo "=== R2 链全部完成 $(date +%F\ %H:%M:%S) ==="
