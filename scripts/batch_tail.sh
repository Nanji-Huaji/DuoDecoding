#!/bin/bash
# 批次尾: (1) target锚(去通信旗标) (2) θ0.99@temp0 定峰位
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29865}
ACC=src/SpecDec_pp/checkpoints/acc_head
snap() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader --id=$GPU 2>/dev/null | sed "s/^/[$(date '+%H:%M:%S')] /"; }

BASE=(--data_path data --num_shots 3 --max_tokens 128
  --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --use_cuda_graph --task_name gsm8k)

# 1) target 锚: 无通信/无ARP/无RL
if ! ls exp/t072_target/*_metrics.json >/dev/null 2>&1; then
  echo "### t072_target ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p exp/t072_target
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode target_only -e t072_target \
    --temp 0.7 --eval_data_num 80 "${BASE[@]}" \
    > exp_logs/t072_target.log 2>&1
  echo "  exit=$?"
fi

# 2) θ0.99@temp0 (per_token, γ16)
if ! ls exp/t0thr_099/*_metrics.json >/dev/null 2>&1; then
  echo "### t0thr_099 ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p exp/t0thr_099
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e t0thr_099 \
    --temp 0.0 --eval_data_num 40 --gamma1 16 --gamma2 16 --gamma 5 \
    --rl_force_threshold 0.99 --arp_stop_mode per_token "${BASE[@]}" \
    --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46 \
    --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05 \
    --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0 \
    --transfer_top_k 300 --small_draft_threshold 0.6 --draft_target_threshold 0.7 \
    --uncertainty_threshold 0.8 --use_stochastic_comm \
    --use_rl_adapter --disable_rl_update \
    --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3" \
    --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3" \
    --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth \
    --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth \
    --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth \
    --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth \
    > exp_logs/t0thr_099.log 2>&1
  echo "  exit=$?"
fi

.venv/bin/python - <<'PYEOF'
import json, glob
for tag in ("t072_target","t0thr_099"):
    fs = glob.glob(f"exp/{tag}/*_metrics.json")
    if not fs: print(f"{tag}: 未产出"); continue
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"]) if m.get("draft_generated_tokens") else None
    acc = m.get("accuracy")
    print(f"{tag}: Thr {m['throughput']:.2f}, Fwd {tf}, acc {acc}"
          + (f", R_acce {r:.1f}, tf {m['generated_tokens']/tf:.2f}, draft {m['draft_generated_tokens']/tf:.2f}" if r is not None else ""))
print("temp0 峰位判定: θ0.8 tf 5.32 < θ0.95 tf 5.79 < θ0.99 tf ?")
PYEOF
echo "=== 尾批完成 ($(date '+%F %H:%M')) ==="
