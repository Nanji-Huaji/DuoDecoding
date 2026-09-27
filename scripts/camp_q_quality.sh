#!/bin/bash
# Campaign Q：论文协议下的**任务质量**评测（补 docs/rl_controller_diagnosis.md 顶部标的最高优先级缺口）
#
# 协议 = 论文默认（run_baseline_table.sh）：temp 0.0（贪心！）、N=40、3-shot、128 tok、
#   bf16、4bit、0.5 Mbps WAN + 563 Mbps LAN + NTT 20ms、seed 20260911
# 关键：所有臂同种子同样本 ⇒ jsonl 的 sample_idx 可逐样本对齐 ⇒ 顺带量化 F24
#   （投机流水线 vs target_only 的逐 token 一致率）
#
# 臂：target_only(质量天花板/一致性基准) dsd cuhlm ours_hist ours_full rule(thr0.4,k64,γ16)
# 通信计费自 2026-09-25 起默认=诚实口径（per_round+残差计费+cap16，见 src/utils.py）；
# ours_hist 臂显式钉住遗留三开关，含义不随仓库默认漂移。
# 用法: bash scripts/camp_q_quality.sh [GPU]
set -u
cd "$(dirname "$0")/.."
GPU=${1:-1}; N=${N:-40}; PORT=$((29700 + GPU * 10))
ACC=src/SpecDec_pp/checkpoints/acc_head
COMMON=(--data_path data --num_shots 3 --max_tokens 128 --eval_data_num "$N"
  --target_model llama-2-13b --draft_model tiny-llama-1.1b --little_model llama-68m
  --target_quantization 4bit --model_dtype bf16 --seed 20260911 --temp 0.0
  --min_bandwidth_mbps 0.1 --edge_end_bandwidth 563 --edge_cloud_bandwidth 0.5
  --ntt_ms_edge_cloud 20
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3")
run() { local tag="q_$1" mode=$2; shift 2
  ls "exp/$tag"/*_metrics.json >/dev/null 2>&1 && { echo "  跳过 $tag"; return 0; }
  echo "### $tag mode=$mode extra=$* ($(date +%H:%M:%S))"; mkdir -p "exp/$tag"
  CUDA_VISIBLE_DEVICES=$GPU HTTPS_PROXY=http://127.0.0.1:7890 \
    .venv/bin/accelerate launch --num_processes 1 --main_process_port $PORT \
    eval/eval_gsm8k.py -e "$tag" --eval_mode "$mode" "${COMMON[@]}" "$@" > "exp_logs/$tag.log" 2>&1
  echo "  exit=$?"; }
run target_only target_only
run ours_full   adaptive_tridecoding --charge_residual_payload --comm_round_trip_mode per_round \
                --transfer_top_k_cap 16 --gamma1 16 --gamma2 16 --rl_force_threshold 0.4
run rule        adaptive_tridecoding --charge_residual_payload --comm_round_trip_mode per_round \
                --transfer_top_k_cap 64 --gamma1 16 --gamma2 16 --rl_force_threshold 0.4
run ours_hist   adaptive_tridecoding --no-charge_residual_payload --comm_round_trip_mode per_transfer --transfer_top_k_cap 0
run dsd         dsd
run cuhlm       cuhlm
echo "=== Campaign Q 完成 ($(date +%H:%M:%S)) ==="
