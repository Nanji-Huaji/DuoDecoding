#!/bin/bash
# 训练结束后的自动接力：两个判据性探针 + 对照汇总。
#   A: t5a 精确协议（固定 NTT 50ms）→ 与 旧策略 3.88 / 钉死0.4 的 4.43 直接可比
#   B: 同协议 + --stochastic_ntt → 新策略在动态时延域的鲁棒性
set -u
cd "$(dirname "$0")/.."
GPU=${GPU:-1}
NEWCKPT=${NEWCKPT:-checkpoints/rl_inregime_46mbps_sn}
TAGA=${TAGA:-t5a_ceesd_inregime}
TAGB=${TAGB:-t5a_ceesd_inregime_sn}
ACC=src/SpecDec_pp/checkpoints/acc_head

# ---- 1) 等训练进程退出 ----
echo "[watch] $(date '+%F %T') 等待训练进程退出..."
while pgrep -f "[e]val_mixed.py -e rl_inregime" > /dev/null 2>&1; do sleep 30; done
sleep 10   # 等 checkpoint 落盘
echo "[watch] $(date '+%F %T') 训练结束，best.pth:"
ls -l "$NEWCKPT"/main/*/best.pth "$NEWCKPT"/little/*/best.pth 2>/dev/null

COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num 80
  --temp 0.0 --sample_seed 1234 --random_sample --num_samples_per_task 1
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05
  --comm_round_trip_mode per_round --no-charge_residual_payload --transfer_top_k_cap 0
  --transfer_top_k 300 --small_draft_threshold 0.6 --draft_target_threshold 0.7
  --uncertainty_threshold 0.8 --use_stochastic_comm
  --use_rl_adapter --disable_rl_update --use_cuda_graph --task_name gsm8k
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --main_rl_path "$NEWCKPT/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth"
  --little_rl_path "$NEWCKPT/little/llama-68m--to--tiny-llama-1.1b/latest.pth"
  --main_rl_best_path "$NEWCKPT/main/tiny-llama-1.1b--to--llama-2-13b/best.pth"
  --little_rl_best_path "$NEWCKPT/little/llama-68m--to--tiny-llama-1.1b/best.pth"
  --gamma1 16 --gamma2 16 --gamma 5)

probe() { local tag=$1; shift
  if ls exp/${tag}/*_metrics.json >/dev/null 2>&1; then echo "跳过 $tag"; return 0; fi
  echo "[watch] $(date '+%F %T') 探针 $tag 启动"
  mkdir -p "exp/${tag}"
  HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=$GPU \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port ${PORT:-29790} \
    eval/eval_gsm8k.py --eval_mode adaptive_tridecoding -e "${tag}" \
    "${COMMON[@]}" "$@" > "exp_logs/${tag}.log" 2>&1
  echo "[watch] $tag exit=$? ($(date '+%F %T'))"
}

probe "$TAGA"
probe "$TAGB" --stochastic_ntt

# ---- 3) 对照汇总 ----
.venv/bin/python - <<'EOF' > ${SUMMARY:-exp_logs/t5a_inregime_probe_summary.txt} 2>&1
import json, glob

def load(tag):
    try:
        return json.load(open(glob.glob(f"exp/{tag}/*metrics.json")[0]))
    except Exception:
        return None

REF = {  # t5a 已有对照（见 docs/paper_table5_alignment.md）
    "旧策略(1444步,0.5Mbps域)": {"thr": 22.35, "fwd": 2696, "tokfwd": 3.88},
    "钉死thr=0.4":              {"thr": 24.23, "fwd": 2373, "tokfwd": 4.43},
}
print(f"{'配置':<28} {'Thr':>7} {'Fwd':>6} {'tok/fwd':>8} {'Cloud$':>8} {'R_acce':>7} {'acc':>7}")
for name, r in REF.items():
    print(f"{name:<28} {r['thr']:>7.2f} {r['fwd']:>6} {r['tokfwd']:>8.2f} {'—':>8} {'—':>7} {'—':>7}")
for tag in __import__("os").environ.get("PROBE_TAGS","t5a_ceesd_inregime t5a_ceesd_inregime_sn").split():
    m = load(tag)
    if not m:
        print(f"{tag:<28} (未完成或失败)")
        continue
    gen, fwd = m["generated_tokens"], m["target_forward_times"]
    tokfwd = gen / fwd
    cloud = 4.05 * m["wall_time"] / 3600
    racce = 100 * m["draft_accepted_tokens"] / max(1, m["draft_generated_tokens"])
    print(f"{tag:<28} {m['throughput']:>7.2f} {fwd:>6} {tokfwd:>8.2f} {cloud:>8.2f} {racce:>7.1f} {m.get('accuracy','—'):>7}")
print("\n判据: 新策略 tok/fwd >= 4.43 (钉死0.4) ⇒ C3 翻案; < 3.88 ⇒ 回炉加步数")
EOF
cat "${SUMMARY:-exp_logs/t5a_inregime_probe_summary.txt}"
echo "[watch] 全部完成 $(date '+%F %T')"
