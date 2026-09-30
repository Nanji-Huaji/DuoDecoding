#!/bin/bash
# temp0.7 ARP阈值阶梯: γ16, θ∈{0.1,0.2,0.4,0.6,0.8} (rl_force_threshold 钉死)
# 回答: (1)回收空间总量 (2)静态θ能否兑现 (3)θ*相对temp0的0.4移向哪里
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29861}
ACC=src/SpecDec_pp/checkpoints/acc_head

COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 40
  --temp 0.7 --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --transfer_top_k 300 --arp_stop_mode per_token --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm
  --use_rl_adapter --disable_rl_update --use_cuda_graph
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth)

for THR in 0.2 0.4 0.6 0.8 0.95; do
  TAG="t07thr_${THR/./}"
  if ls exp/$TAG/*_metrics.json >/dev/null 2>&1; then echo "跳过 $TAG"; continue; fi
  echo "### $TAG θ=$THR ($(TZ=Asia/Shanghai date '+%H:%M'))"
  mkdir -p "exp/$TAG"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "$TAG" \
    --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold "$THR" "${COMMON[@]}" \
    > "exp_logs/${TAG}.log" 2>&1
  echo "  exit=$? ($(TZ=Asia/Shanghai date '+%H:%M'))"
done

.venv/bin/python - <<'PYEOF'
import glob, json
print(f"{'θ':>4} | {'Thr':>6} {'tokfwd':>6} {'R_acce':>6} {'草稿长/轮':>8} {'Fwd':>5} {'acc':>6}")
print("-"*56)
for thr in ["0.2","0.4","0.6","0.8","0.95"]:
    fs = glob.glob(f"exp/t07thr_{thr.replace('.','')}/*_metrics.json")
    if not fs: print(f"{thr:>4} | (未完成)"); continue
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
    dl = m["draft_generated_tokens"]/max(1,tf)
    acc = f"{m['accuracy']:.3f}" if m.get("accuracy") is not None else "—"
    print(f"{thr:>4} | {m['throughput']:>6.2f} {m['generated_tokens']/tf:>6.2f} {r:>6.1f} {dl:>8.2f} {tf:>5d} {acc:>6}")
print("参照: t07策略自选 θ≈? Thr 12.37 tf 3.12 R 61.4 | temp0 pin θ=0.4 tf 4.43")
PYEOF
echo "=== thr 阶梯完成 ($(TZ=Asia/Shanghai date '+%F %H:%M')) ==="
