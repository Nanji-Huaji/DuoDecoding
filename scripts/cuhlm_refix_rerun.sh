#!/bin/bash
# CUHLM 修复后重测: temp0(N=80,t5a口径) + temp0.7(N=40) 干净窗口 + GPU快照审计
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29863}
ACC=src/SpecDec_pp/checkpoints/acc_head
snap() { nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader \
  --id=$GPU | sed "s/^/[$(TZ=Asia/Shanghai +%H:%M:%S)] /"; }

COMMON=(--data_path data --num_shots 3 --max_tokens 128
  --sample_seed 1234 --random_sample --num_samples_per_task 1
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

echo "=== GPU快照(跑前) ==="; snap
run() { local tag=$1 temp=$2 n=$3
  if ls exp/$tag/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### $tag temp=$temp N=$n ($(TZ=Asia/Shanghai date '+%H:%M'))"; snap
  mkdir -p "exp/$tag"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode uncertainty_decoding -e "$tag" \
    --temp "$temp" --eval_data_num "$n" --gamma1 3 --gamma2 3 --gamma 5 "${COMMON[@]}" \
    > "exp_logs/${tag}.log" 2>&1
  echo "  exit=$? ($(TZ=Asia/Shanghai date '+%H:%M'))"
}
run t5a2_cuhlm 0.0 80
run t072_cuhlm 0.7 40
echo "=== GPU快照(跑后) ==="; snap

.venv/bin/python - <<'PYEOF'
import json, glob
print(f"\n{'组':<14} | {'Thr':>6} {'tokfwd':>6} {'Fwd':>5} {'R_acce':>6} {'acc':>6} {'上行MB':>7}")
print("-"*62)
for tag, old in (("t5a2_cuhlm","t5a_cuhlm"),("t072_cuhlm","t07_cuhlm")):
    for t in (tag, old):
        fs = glob.glob(f"exp/{t}/*_metrics.json")
        if not fs: continue
        m = json.load(open(fs[0])); tf = m["target_forward_times"]
        r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
        acc = f"{m['accuracy']:.4f}" if m.get("accuracy") is not None else "—"
        mb = m.get("edge_cloud_data_bytes",0)/1e6
        nm = "新" if t==tag else "旧"
        print(f"{t+'('+nm+')':<14} | {m['throughput']:>6.2f} {m['generated_tokens']/tf:>6.2f} {tf:>5d} {r:>6.1f} {acc:>6} {mb:>7.2f}")
PYEOF
echo "=== CUHLM重测完成 ($(TZ=Asia/Shanghai date '+%F %H:%M')) ==="
