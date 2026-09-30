#!/bin/bash
# O(L²)计费修复验证: DSD/CUHLM 重跑(t5a口径), 字节应从~3000B/tok降到~10B/tok,
# 计数指标应逐位不变(计费不影响rollout), Thr微升(通信时间减少)
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29868}
ACC=src/SpecDec_pp/checkpoints/acc_head
snap() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader --id=$GPU 2>/dev/null | sed "s/^/[$(date '+%H:%M:%S')] /"; }

COMMON=(--data_path data --num_shots 3 --max_tokens 128 --temp 0.0
  --sample_seed 1234 --random_sample --num_samples_per_task 1 --eval_data_num 80
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

run() { local tag=$1 mode=$2
  if ls exp/$tag/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### $tag ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p "exp/$tag"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode "$mode" -e "$tag" \
    --gamma1 3 --gamma2 3 --gamma 5 "${COMMON[@]}" \
    > "exp_logs/${tag}.log" 2>&1
  echo "  exit=$?"
}
run t5a3_dsd   dist_spec
run t5a3_cuhlm uncertainty_decoding

.venv/bin/python - <<'PYEOF'
import json, glob
print(f"{'组':<12} | {'B/tok':>7} | {'Thr':>6} {'tf':>5} {'Fwd':>5} {'R':>5} {'acc':>6} | 旧值对照")
print("-"*76)
for tag, old in (("t5a3_dsd","t5a_dsd"),("t5a3_cuhlm","t5a_cuhlm")):
    fs = glob.glob(f"exp/{tag}/*_metrics.json")
    if not fs: print(f"{tag:<12} | 未产出"); continue
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    b = m.get("edge_cloud_data_bytes",0)/max(1,m["generated_tokens"])
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
    o = json.load(open(glob.glob(f"exp/{old}/*_metrics.json")[0]))
    ob = o.get("edge_cloud_data_bytes",0)/max(1,o["generated_tokens"])
    same = (tf==o["target_forward_times"] and m["generated_tokens"]==o["generated_tokens"])
    print(f"{tag:<12} | {b:>7.1f} | {m['throughput']:>6.2f} {m['generated_tokens']/tf:>5.2f} {tf:>5d} {r:>5.1f} {m.get('accuracy'):>6.4f}"
          f" | 旧 {ob:.0f}B/tok Thr {o['throughput']:.2f}; 计数一致={'✓' if same else '✗'}")
PYEOF
echo "=== 修复验证完成 ($(date '+%F %H:%M')) ==="
