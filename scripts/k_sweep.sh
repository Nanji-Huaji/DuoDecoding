#!/bin/bash
# top-k 扫描: k∈{1..1024}(候选网格) × adaptive_tridecoding temp0.7 θ0.8 γ16
# k 进验证路径(草稿分布→top-k上行→重建→draft_probs_override), R_acce 会真实移动
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29869}
ACC=src/SpecDec_pp/checkpoints/acc_head
snap() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader --id=$GPU 2>/dev/null | sed "s/^/[$(date '+%H:%M:%S')] /"; }

COMMON=(--data_path data --num_shots 3 --max_tokens 128
  --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --temp 0.7 --eval_data_num 40 --gamma1 16 --gamma2 16 --gamma 5
  --rl_force_threshold 0.8 --arp_stop_mode per_token
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05
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

for K in 1 2 4 8 16 32 64 128 256 512 1024; do
  TAG="ksweep_$(printf %04d $K)"
  if ls exp/$TAG/*_metrics.json >/dev/null 2>&1; then echo "跳过 $TAG"; continue; fi
  echo "### $TAG ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p "exp/$TAG"
  # k=256 跑挂 TOPK_TRACE（分布形状与 k 无关, 一份就够）
  if [ "$K" = "256" ]; then
    rm -f exp/topk_trace.jsonl
    export TOPK_TRACE="$PWD/exp/topk_trace.jsonl"
  else
    unset TOPK_TRACE
  fi
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "$TAG" \
    --rl_force_topk "$K" "${COMMON[@]}" \
    > "exp_logs/${TAG}.log" 2>&1
  echo "  exit=$?"
done
unset TOPK_TRACE

.venv/bin/python - <<'PYEOF'
import json, glob
print(f"{'k':>5} | {'R_acce':>6} {'tf':>5} {'Thr':>6} {'B/轮':>6} {'draft':>6} {'acc':>6}")
for K in [1,2,4,8,16,32,64,128,256,512,1024]:
    fs = glob.glob(f"exp/ksweep_{K:04d}/*_metrics.json")
    if not fs: print(f"{K:>5} | 未产出"); continue
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
    b = m.get("edge_cloud_data_bytes",0)/max(1,tf)
    print(f"{K:>5} | {r:>6.1f} {m['generated_tokens']/tf:>5.2f} {m['throughput']:>6.2f} {b:>6.0f} {m['draft_generated_tokens']/max(1,tf):>6.2f} {m.get('accuracy'):>6.3f}")
PYEOF
echo "=== k扫描完成 ($(date '+%F %H:%M')) ==="
