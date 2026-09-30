#!/bin/bash
# ARP 校准诊断: 10样本 × temp{0.7, 0.0}, γ16 + per_token θ0.6, ARP_CALIB_TRACE 开
# 产出: exp_logs/arpcalib_{t07,t0}.jsonl → 可靠性曲线分析
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29862}
ACC=src/SpecDec_pp/checkpoints/acc_head

COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 10
  --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --transfer_top_k 300 --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm
  --use_rl_adapter --disable_rl_update --use_cuda_graph
  --arp_stop_mode per_token --rl_force_threshold 0.6
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth)

run_calib() { local tag=$1 temp=$2
  rm -f "exp_logs/arpcalib_${tag}.jsonl"
  mkdir -p "exp/arpcalib_$tag"
  echo "### arpcalib_$tag temp=$temp ($(TZ=Asia/Shanghai date '+%H:%M'))"
  ARP_CALIB_TRACE="exp_logs/arpcalib_${tag}.jsonl" \
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "arpcalib_$tag" \
    --gamma1 16 --gamma2 16 --gamma 5 --temp "$temp" "${COMMON[@]}" \
    > "exp_logs/arpcalib_${tag}.log" 2>&1
  echo "  exit=$? rows=$(wc -l < exp_logs/arpcalib_${tag}.jsonl 2>/dev/null || echo 0)"
}

run_calib t07 0.7
run_calib t0  0.0

.venv/bin/python - <<'PYEOF'
import json

def load(p):
    rows = []
    for line in open(p):
        try: rows.append(json.loads(line))
        except Exception: pass
    return rows

def reliability(rows, key):
    bins = {}
    for r in rows:
        b = min(int(r["arp"] * 10), 9)
        bins.setdefault(b, []).append(r[key])
    return bins

for tag, key in (("t0", "gmatch"), ("t07", "ratio")):
    p = f"exp_logs/arpcalib_{tag}.jsonl"
    rows = load(p)
    if not rows: print(f"{tag}: 无数据"); continue
    bins = reliability(rows, key)
    print(f"\n=== {tag} (真值={key}, n={len(rows)}) ===")
    print(f"{'ARP预测区间':>12} | {'n':>5} {'真值均值':>8} {'预测均值':>8} {'偏差':>7}")
    for b in sorted(bins):
        vs = bins[b]
        pred = sum(r["arp"] for r in rows if min(int(r["arp"]*10),9)==b)/len(vs)
        truth = sum(vs)/len(vs)
        print(f"[{b/10:.1f},{(b+1)/10:.1f}) | {len(vs):>5} {truth:>8.3f} {pred:>8.3f} {truth-pred:>+7.3f}")
    # 全局
    all_truth = sum(r[key] for r in rows)/len(rows)
    all_pred = sum(r["arp"] for r in rows)/len(rows)
    print(f"{'全局':>12} | {len(rows):>5} {all_truth:>8.3f} {all_pred:>8.3f} {all_truth-all_pred:>+7.3f}")
    # Brier
    bs = sum((r["arp"]-r[key])**2 for r in rows)/len(rows)
    print(f"Brier({tag}) = {bs:.4f}")
PYEOF
echo "=== ARP校准诊断完成 ($(TZ=Asia/Shanghai date '+%F %H:%M')) ==="
