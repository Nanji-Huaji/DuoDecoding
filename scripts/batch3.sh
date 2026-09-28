#!/bin/bash
# 批次3: E1) γ32@temp0最优操作点 E2) 同窗 CEE vs target_only headline
#         E3) isotonic remap A/B  E4) RTT扫描@per_token新操作点
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29866}
ACC=src/SpecDec_pp/checkpoints/acc_head
REMAP=exp/arp_calib_remap.json
snap() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader --id=$GPU 2>/dev/null | sed "s/^/[$(date '+%H:%M:%S')] /"; }

COMMON=(--data_path data --num_shots 3 --max_tokens 128
  --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --transfer_top_k 300 --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm
  --use_rl_adapter --disable_rl_update --use_cuda_graph --task_name gsm8k
  --arp_stop_mode per_token
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth)

run() { local tag=$1 mode=$2 envmap=$3; shift 3
  if ls exp/$tag/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### $tag ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p "exp/$tag"
  env ${envmap:+ARP_CALIB_MAP=$envmap} HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode "$mode" -e "$tag" "$@" "${COMMON[@]}" \
    > "exp_logs/${tag}.log" 2>&1
  echo "  exit=$?"
}

# E1) γ32 @ temp0 最优(θ0.99) —— γ轴在temp0是否重新打开
run e1_t0g32 adaptive_tridecoding "" --temp 0.0 --eval_data_num 40 \
  --gamma1 32 --gamma2 32 --gamma 5 --rl_force_threshold 0.99

# E2) 同窗 headline 对: CEE(θ0.8) 紧接 target_only (SpecEdge式声明)
run e2_ceesd adaptive_tridecoding "" --temp 0.7 --eval_data_num 40 \
  --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold 0.8
run e2_target target_only "" --temp 0.7 --eval_data_num 80 \
  --data_path data --num_shots 3 --max_tokens 128 --sample_seed 1234 --random_sample \
  --num_samples_per_task 1 --draft_model tiny-llama-1.1b --target_model llama-2-13b \
  --little_model llama-68m --use_cuda_graph --task_name gsm8k

# E3) remap A/B: raw θ0.2(旧最优) vs 校准 θ0.6/0.8 (temp0.7, γ16)
run e3_raw02 adaptive_tridecoding "" --temp 0.7 --eval_data_num 40 \
  --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold 0.2
run e3_cal06 adaptive_tridecoding "$REMAP" --temp 0.7 --eval_data_num 40 \
  --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold 0.6
run e3_cal08 adaptive_tridecoding "$REMAP" --temp 0.7 --eval_data_num 40 \
  --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold 0.8

# E4) RTT 扫描 @ 新操作点 (temp0.7, θ0.8, γ16)
for NTT in 5 15 50; do
  TAG="e4ntt_$(printf %02d $NTT)"
  if ls exp/$TAG/*_metrics.json >/dev/null 2>&1; then echo "跳过 $TAG"; continue; fi
  echo "### $TAG ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p "exp/$TAG"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "$TAG" \
    --temp 0.7 --eval_data_num 40 --gamma1 16 --gamma2 16 --gamma 5 \
    --rl_force_threshold 0.8 "${COMMON[@]}" --ntt_ms_edge_cloud "$NTT" \
    > "exp_logs/${TAG}.log" 2>&1
  echo "  exit=$?"
done

.venv/bin/python - <<'PYEOF'
import json, glob
def row(tag):
    fs = glob.glob(f"exp/{tag}/*_metrics.json")
    if not fs: return None
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
    dl = m["draft_generated_tokens"]/max(1,tf)
    ct = m.get("communication_time",0)/max(1,tf)*1000
    return (m["throughput"], m["generated_tokens"]/tf, tf, r, dl, m.get("accuracy"), ct)
def show(title, pairs, ref=""):
    print(f"\n{title}{ref}")
    print(f"{'组':<12} | {'Thr':>6} {'tf':>5} {'R':>5} {'draft':>6} {'acc':>6} {'comm ms/轮':>9}")
    for name, tag in pairs:
        r_ = row(tag)
        if r_: print(f"{name:<12} | {r_[0]:>6.2f} {r_[1]:>5.2f} {r_[3]:>5.1f} {r_[4]:>6.2f} {r_[5]:>6.3f} {r_[6]:>9.1f}")
        else: print(f"{name:<12} | (未产出)")
show("E1 γ轴@temp0最优:", [("γ32θ0.99","e1_t0g32")], "  (γ16θ0.99: 22.87/5.91/30.9/15.87)")
show("E2 同窗headline:", [("CEE θ0.8","e2_ceesd"),("target_only","e2_target")], "  (跨窗旧值 22.14 vs 21.76)")
show("E3 remap A/B:", [("raw θ0.2","e3_raw02"),("cal θ0.6","e3_cal06"),("cal θ0.8","e3_cal08")])
show("E4 RTT扫描:", [("NTT5","e4ntt_05"),("NTT15","e4ntt_15"),("NTT50","e4ntt_50")])
PYEOF
echo "=== 批次3完成 ($(date '+%F %H:%M')) ==="
