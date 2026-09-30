#!/bin/bash
# 低延迟域主对比（近边 5G 工况）: t5a 完整口径 + 仅改 NTT 50→15 + 真实 ping trace 回放
#   N=40 先看地形; 基线 DSD/DSSD/CUHLM 首次进入低延迟域
#   与 t5a 同计费(per_round+无残差+cap0)、同 k300、同 RL checkpoints → 差值纯attributable于工况
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29850}
ACC=src/SpecDec_pp/checkpoints/acc_head
PING=data/sigcomm-5gmemu-5g-mmWave-uplink-data/ping/static/5g/away_p1.list

COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 40
  --temp 0.0 --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 15 --ntt_trace_file "$PING" --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --transfer_top_k 300 --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm
  --use_rl_adapter --disable_rl_update --use_cuda_graph
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth)

run() { local tag=$1 mode=$2 g1=$3 g2=$4
  if ls exp/lowlat40_${tag}/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### lowlat40_$tag mode=$mode γ1=$g1 γ2=$g2 ($(TZ=Asia/Shanghai date '+%H:%M'))"
  mkdir -p "exp/lowlat40_${tag}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode "$mode" -e "lowlat40_${tag}" \
    --gamma1 "$g1" --gamma2 "$g2" --gamma 5 "${COMMON[@]}" \
    > "exp_logs/lowlat40_${tag}.log" 2>&1
  echo "  exit=$? ($(TZ=Asia/Shanghai date '+%H:%M'))"
}

run dsd       dist_spec             3 3
run dssd      dist_split_spec       3 3
run cuhlm     uncertainty_decoding  3 3
run ceesd_g5  adaptive_tridecoding  5 5
run ceesd_g16 adaptive_tridecoding 16 16

.venv/bin/python scripts/summarize_lowlat.py
echo "=== 低延迟主对比完成 ($(TZ=Asia/Shanghai date '+%F %H:%M')) ==="
