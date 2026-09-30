#!/bin/bash
# 方向判定批次: A) temp0 θ阶梯 B) γ扫描@θ0.8 C) 干净基线+target锚 D) isotonic拟合
# 每跑前 GPU 快照审计(lqf 共租检测)
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
PORT=${PORT:-29864}
ACC=src/SpecDec_pp/checkpoints/acc_head
snap() { nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader \
  --id=$GPU 2>/dev/null | sed "s/^/[$(date '+%H:%M:%S')] /"; }

COMMON=(--data_path data --num_shots 3 --max_tokens 128
  --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --transfer_top_k 300 --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm
  --use_rl_adapter --disable_rl_update --use_cuda_graph
  --arp_stop_mode per_token
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth
  --little_rl_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/latest.pth
  --main_rl_best_path checkpoints/rl_agents/main/tiny-llama-1.1b--to--llama-2-13b/best.pth
  --little_rl_best_path checkpoints/rl_agents/little/llama-68m--to--tiny-llama-1.1b/best.pth)

run() { local tag=$1 mode=$2; shift 2
  if ls exp/$tag/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "### $tag ($(date '+%H:%M')) GPU:$(snap)"
  mkdir -p "exp/$tag"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py --eval_mode "$mode" -e "$tag" "$@" "${COMMON[@]}" \
    > "exp_logs/${tag}.log" 2>&1
  echo "  exit=$?"
}

# A) temp0 θ阶梯 (γ16): θ*随温度移动?
for THR in 0.4 0.6 0.8 0.95; do
  run "t0thr_${THR/./}" adaptive_tridecoding --temp 0.0 --eval_data_num 40 \
    --gamma1 16 --gamma2 16 --gamma 5 --rl_force_threshold "$THR"
done

# B) γ扫描 @temp0.7 θ0.8 (γ16是否截断最优)
for G in 8 32; do
  run "t07g_$(printf %02d $G)" adaptive_tridecoding --temp 0.7 --eval_data_num 40 \
    --gamma1 $G --gamma2 $G --gamma 5 --rl_force_threshold 0.8
done

# C) 干净基线 (temp0.7) + target锚
run t072_dsd  dist_spec        --temp 0.7 --eval_data_num 40 --gamma1 3 --gamma2 3 --gamma 5
run t072_dssd dist_split_spec  --temp 0.7 --eval_data_num 40 --gamma1 3 --gamma2 3 --gamma 5
run t072_target target_only    --temp 0.7 --eval_data_num 80

# D) isotonic 拟合 (CPU) + 汇总
.venv/bin/python - <<'PYEOF'
import glob, json
import numpy as np
from sklearn.isotonic import IsotonicRegression

def load(p, key):
    xs, ys = [], []
    for line in open(p):
        try: r = json.loads(line); xs.append(r["arp"]); ys.append(r[key])
        except Exception: pass
    return np.array(xs), np.array(ys)

out = open("exp_logs/direction_batch_summary.txt", "w")
W = out.write

# --- D1: isotonic ---
remap = {}
W("== D) ARP isotonic 重校准拟合 ==\n")
for tag, key in (("t0","gmatch"), ("t07","ratio")):
    xs, ys = load(f"exp_logs/arpcalib_{tag}.jsonl", key)
    if len(xs) < 50: W(f"{tag}: 数据不足\n"); continue
    n = len(xs); idx = np.random.default_rng(0).permutation(n)
    tr, te = idx[: int(n*0.8)], idx[int(n*0.8):]
    iso = IsotonicRegression(out_of_bounds="clip").fit(xs[tr], ys[tr])
    raw_b = np.mean((xs[te]-ys[te])**2); iso_b = np.mean((iso.predict(xs[te])-ys[te])**2)
    W(f"{tag}: n={n} 留出Brier raw={raw_b:.4f} → iso={iso_b:.4f} ({(raw_b-iso_b)/raw_b*100:+.0f}%)\n")
    if tag == "t07":
        grid = [round(g,2) for g in np.arange(0.05, 0.96, 0.05)]
        vals = [round(float(v),3) for v in iso.predict(grid)]
        remap = {"grid_raw": grid, "mapped_true": vals,
                 "bar_theta08_raw0.2_maps_to": round(float(iso.predict([0.2])[0]),3)}
        W(f"  重映射表(raw→true): {dict(zip(grid,vals))}\n")
        W(f"  θ0.8杠杠(raw 0.2) → 真值 {remap['bar_theta08_raw0.2_maps_to']}\n")
json.dump(remap, open("exp/arp_calib_remap.json","w"), indent=1)

# --- 汇总表 ---
def row(tag):
    fs = glob.glob(f"exp/{tag}/*_metrics.json")
    if not fs: return None
    m = json.load(open(fs[0])); tf = m["target_forward_times"]
    r = 100*m["draft_accepted_tokens"]/max(1,m["draft_generated_tokens"])
    dl = m["draft_generated_tokens"]/max(1,tf)
    acc = m.get("accuracy")
    return (m["throughput"], m["generated_tokens"]/tf, tf, r, dl, acc)

W("\n== A) temp0 θ阶梯 (γ16, N=40) — θ*(temp0)=? ==\n")
W(f"{'θ':>5} | {'Thr':>6} {'tf':>5} {'R':>5} {'draft':>6} {'acc':>6}\n")
for thr in ["0.4","0.6","0.8","0.95"]:
    t = f"t0thr_{thr.replace('.','')}"
    r_ = row(t)
    if r_: W(f"{thr:>5} | {r_[0]:>6.2f} {r_[1]:>5.2f} {r_[3]:>5.1f} {r_[4]:>6.2f} {r_[5]:>6.3f}\n")
W("参照 temp0.7: θ* = 0.8 (Thr 22.14, tf 4.64, draft 8.09)\n")

W("\n== B) γ扫描 (temp0.7, θ0.8) ==\n")
W(f"{'γ':>4} | {'Thr':>6} {'tf':>5} {'R':>5} {'draft':>6} {'acc':>6}\n")
for g in ["08","16","32"]:
    t = f"t07g_{g}" if g != "16" else "t07thr_08"
    r_ = row(t)
    if r_: W(f"{g:>4} | {r_[0]:>6.2f} {r_[1]:>5.2f} {r_[3]:>5.1f} {r_[4]:>6.2f} {r_[5]:>6.3f}\n")

W("\n== C) 干净基线 (temp0.7, N=40) + 锚 (N=80) ==\n")
W(f"{'组':<12} | {'Thr':>6} {'tf':>5} {'Fwd':>5} {'R':>5} {'draft':>6} {'acc':>6}\n")
for t, old in (("t072_dsd","t07_dsd"),("t072_dssd","t07_dssd"),("t072_target",None)):
    r_ = row(t)
    if r_ is None: W(f"{t:<12} | (未完成/模式缺失)\n"); continue
    W(f"{t:<12} | {r_[0]:>6.2f} {r_[1]:>5.2f} {r_[2]:>5d} {r_[3]:>5.1f} {r_[4]:>6.2f} {r_[5]:>6.3f}")
    o = row(old) if old else None
    if o: W(f"   (污染窗旧值 Thr {o[0]:.2f})\n")
    else: W("\n")
out.close()
print(open("exp_logs/direction_batch_summary.txt").read())
PYEOF
echo "=== 方向判定批次完成 ($(date '+%F %H:%M')) ==="
