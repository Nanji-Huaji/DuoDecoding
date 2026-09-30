#!/bin/bash
# temp>0 预实验: t5a 完整口径, 仅改 --temp 0.0 → 0.7, N=40
#   回答: (1)接受率掉多少(1-TVD vs 贪心匹配) (2)tok/fwd 各方法掉多少
#         (3)现有 ARP(贪心口径训练)在采样域的门槛行为是否退化 (4)精度变化
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29860}
ACC=src/SpecDec_pp/checkpoints/acc_head

COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 40
  --temp 0.7 --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
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

run() { local tag=$1 mode=$2 g1=$3 g2=$4
  if ls exp/t07_${tag}/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### t07_$tag mode=$mode γ1=$g1 γ2=$g2 ($(TZ=Asia/Shanghai date '+%H:%M'))"
  mkdir -p "exp/t07_${tag}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode "$mode" -e "t07_${tag}" \
    --gamma1 "$g1" --gamma2 "$g2" --gamma 5 "${COMMON[@]}" \
    > "exp_logs/t07_${tag}.log" 2>&1
  echo "  exit=$? ($(TZ=Asia/Shanghai date '+%H:%M'))"
}

run dsd       dist_spec             3 3
run dssd      dist_split_spec       3 3
run cuhlm     uncertainty_decoding  3 3
run ceesd_g5  adaptive_tridecoding  5 5
run ceesd_g16 adaptive_tridecoding 16 16

.venv/bin/python - <<'PYEOF'
import glob, json
W = open("exp_logs/t07_summary.txt", "w").write
W("temp=0.7 预实验 (N=40, 其余=t5a口径) vs t5a (temp=0, N=80)\n")
W(f"{'方法':<16} | {'Thr':>6} {'tokfwd':>6} {'Fwd':>5} {'R_acce':>6} {'acc':>6} | {'t5a Thr':>7} {'t5a tf':>6}\n")
W("-"*76 + "\n")
for name, tag in [("DSD","dsd"),("DSSD","dssd"),("CUHLM","cuhlm"),
                  ("CEE-SD γ5","ceesd_g5"),("CEE-SD γ16","ceesd_g16")]:
    fs = glob.glob(f"exp/t07_{tag}/*_metrics.json")
    if not fs: W(f"{name:<16} | (未完成)\n"); continue
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
    acc = f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "—"
    hi = glob.glob(f"exp/t5a_{tag}/*_metrics.json")
    if hi:
        h = json.load(open(hi[0]))
        hcell = f"{h['throughput']:>7.2f} {h['generated_tokens']/h['target_forward_times']:>6.2f}"
    else: hcell = f"{'—':>7} {'—':>6}"
    W(f"{name:<16} | {m['throughput']:>6.2f} {m['generated_tokens']/tf:>6.2f} {tf:>5d} {r:>6.1f} {acc:>6} | {hcell}\n")
PYEOF
cat exp_logs/t07_summary.txt
echo "=== temp0.7 预实验完成 ($(TZ=Asia/Shanghai date '+%F %H:%M')) ==="
