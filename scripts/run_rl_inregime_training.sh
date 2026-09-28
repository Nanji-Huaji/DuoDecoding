#!/bin/bash
# P1：DRA 域内重训（46 Mbps / NTT-dominant + 动态 NTT）。
#
# 与 run_rl_3shot_training.sh 的区别（为什么要重训，本轮 L1 诊断）：
#   旧训练的课程域是 0.5-20 Mbps 字节主导（curriculum_bw_end 0.5），而论文
#   Table V 协议是 46 Mbps + NTT 50ms 的 NTT 主导域。已部署策略
#   （checkpoints/rl_agents/*/best.pth，t5a 实测 3.88 tok/fwd）在论文域内
#   保守过头，劣于钉死阈值 0.4（4.43 tok/fwd）—— 域错配。
#   本脚本在**论文域内**重训：固定 46 Mbps（trace 均值重标定，static 5G），
#   NTT 基值 50ms + L1 拥塞动态（--stochastic_ntt），legacy 口径与 t5a 一致。
#   策略的 (bw, ntt) 状态输入首次同时具备两个维度的真实变化。
#
# 用法: bash scripts/run_rl_inregime_training.sh [GPU] [SAMPLES]
# v3 (reward 修复版): LAMBDA=0(自适应EMA,不再冻12) + RL_EXTRA="--rl_charge_queue" + graph 标定
#   COST_JSON/configs/compute_cost_model_graph.json; 判据不变: tok/fwd >= 4.43
set -euo pipefail
cd "$(dirname "$0")/.."

GPU_ID=${1:-1}
SAMPLES=${2:-500}
GAMMA=${GAMMA:-16}
PORT=${PORT:-29951}

COST_JSON=${COST_JSON:-configs/compute_cost_model_graph.json}
OUT=${OUT:-checkpoints/rl_inregime_46mbps_sn}
EXP=${EXP:-rl_inregime_46mbps_sn}
SEED=checkpoints/rl_agents

[ -f "$COST_JSON" ] || { echo "missing $COST_JSON" >&2; exit 1; }

# 从论文部署的 DRA（t5a 所用 best.pth）warm-start，不覆盖原目录
for part in main/tiny-llama-1.1b--to--llama-2-13b little/llama-68m--to--tiny-llama-1.1b; do
  mkdir -p "$OUT/$part"
  for pth in latest best; do
    src_pth="$SEED/$part/${pth}.pth"
    if [ -f "$src_pth" ] && [ ! -f "$OUT/$part/${pth}.pth" ]; then
      cp "$src_pth" "$OUT/$part/${pth}.pth"
      echo "seeded $OUT/$part/${pth}.pth  <-  $src_pth"
    fi
  done
done

echo "=== 论文域重训: $EXP (46Mbps mean trace + stochastic NTT, gamma=$GAMMA) ==="
echo "    GPU=$GPU_ID steps=$SAMPLES  $(date '+%F %T')"
RL_ACTION_TRACE="exp_logs/trace_train_${EXP}.jsonl" \
RL_ACC_PROB_TRACE="exp_logs/accprob_train_${EXP}.jsonl" \
RL_REWARD_TRACE="exp_logs/reward_trace_train_${EXP}.jsonl" \
CUDA_VISIBLE_DEVICES=$GPU_ID \
HF_HUB_OFFLINE=1 \
.venv/bin/accelerate launch --num_processes 1 --main_process_port "$PORT" \
  eval/eval_mixed.py -e "$EXP" \
    --eval_mode adaptive_tridecoding \
    --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m \
    --max_tokens 128 --temp 0.0 --seed 7 \
    --num_shots 3 \
    --target_quantization 4bit \
    --eval_data_num "$SAMPLES" \
    --curriculum_bw_start "${CURR_BW:-46,46}" --curriculum_bw_end "${CURR_BW:-46,46}" \
    --curriculum_ntt_start "${CURR_NTT:-50,50}" --curriculum_ntt_end "${CURR_NTT:-50,50}" \
    --curriculum_sampling ${SAMPLING:-uniform} \
    --edge_end_bandwidth 563 \
    --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46 \
    --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 \
    --batch_delay 0.05 \
    --comm_accounting legacy \
    --use_stochastic_comm --stochastic_ntt \
    --min_bandwidth_mbps 0.1 \
    --gamma1 "$GAMMA" --gamma2 "$GAMMA" \
    --transfer_top_k 300 \
    --small_draft_threshold 0.6 --draft_target_threshold 0.7 \
    --small_draft_acc_head_path "src/SpecDec_pp/checkpoints/acc_head/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3" \
    --draft_target_acc_head_path "src/SpecDec_pp/checkpoints/acc_head/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3" \
    --state_bw_scaling log --state_latency_scaling centi \
    --rl_epsilon_decay 0.9997 \
    --rl_action_space topk_thr \
    --rl_buffer_size 20000 \
    --rl_reward_mode lagrangian --rl_reward_lambda ${LAMBDA:-0} \
    --rl_compute_time_mode model --rl_compute_cost_json "$COST_JSON" \
    --rl_byte_price 0 \
    --use_rl_adapter \
    --main_rl_path "$OUT/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth" \
    --main_rl_best_path "$OUT/main/tiny-llama-1.1b--to--llama-2-13b/best.pth" \
    --little_rl_path "$OUT/little/llama-68m--to--tiny-llama-1.1b/latest.pth" \
    --little_rl_best_path "$OUT/little/llama-68m--to--tiny-llama-1.1b/best.pth" \
    2>&1 | tee -a "exp_logs/train_${EXP}.log"

echo "=== done $(date '+%F %T') ==="
