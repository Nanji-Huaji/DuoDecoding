#!/bin/bash
# Phase 0.1 校准双跑 —— golden 固件的稳定性裁决（需 GPU，用户终端执行）
#
# 用途：
#   ① 字段稳定性实证：同配置跑两遍，LOCKED 字段应零漂移
#      （若有漂移 → Phase 0.5 加确定性开关，或修订 TIMING 表）
#   ② golden 复现验证：run1 对照 golden q15_free_g5——
#      worktree 快照代码 + v3 checkpoint 快照应完整复现 campaign 数字
#
# 配置：与 golden q15_free_g5 完全一致（qwen15 系列, γ1=γ2=5, 20 样本,
#       seed=1234, temp=0, paper_table5 协议, cuda graph, RL 冻结）
# 耗时：每 run 约 1-2 分钟；共约 3-5 分钟
# 执行：cd /home/tiantianyi/code/DuoDecoding/.refactor/wt && bash scripts/golden_calibrate.sh
#       （GPU 选择：GPU=0 bash scripts/golden_calibrate.sh，默认 1——
#         与 gammar campaign 同卡，排除跨卡归约差异）
set -euo pipefail
cd "$(dirname "$0")/.."

GPU=${GPU:-1}
VENV=/home/tiantianyi/code/DuoDecoding/.venv   # 解释器共享主树 venv（只读）
MAIN_P=checkpoints/rl_agents/main/qwen1.5-1.8b-chat--to--qwen1.5-7b-chat/latest.pth
LITTLE_P=checkpoints/rl_agents/little/qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat/latest.pth
ACC_LD=src/SpecDec_pp/checkpoints/acc_head/qwen1.5-0.5b-chat--to--qwen1.5-1.8b-chat/exp-weight-layer3
ACC_DT=src/SpecDec_pp/checkpoints/acc_head/qwen1.5-1.8b-chat--to--qwen1.5-7b-chat/exp-weight6-layer3

run_once () {  # run_once <run_name>
  local name=$1
  CUDA_VISIBLE_DEVICES=$GPU \
  "$VENV/bin/accelerate" launch --num_processes 1 \
      --main_process_port $((29300 + RANDOM % 400)) \
    eval/eval_gsm8k.py \
    --eval_mode adaptive_tridecoding --protocol paper_table5 --comm_accounting honest \
    --draft_model Qwen/Qwen1.5-1.8B-Chat --target_model Qwen/Qwen1.5-7B-Chat \
    --little_model Qwen/Qwen1.5-0.5B-Chat \
    --max_tokens 128 --temp 0.0 --gamma1 5 --gamma2 5 --gamma 3 \
    --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46 \
    --min_bandwidth_mbps 4.6 --comm_bw_model fluid --transfer_top_k 300 \
    --num_samples_per_task 1 --eval_data_num 20 --sample_seed 1234 --random_sample \
    --small_draft_threshold 0.6 --draft_target_threshold 0.7 --uncertainty_threshold 0.8 \
    --num_shots 3 --exp_name "golden_calib/$name" \
    --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05 \
    --use_stochastic_comm --use_rl_adapter --disable_rl_update --use_cuda_graph \
    --small_draft_acc_head_path "$ACC_LD" \
    --draft_target_acc_head_path "$ACC_DT" --acc_head_path "$ACC_DT" \
    --main_rl_path "$MAIN_P" --main_rl_best_path "$MAIN_P" \
    --little_rl_path "$LITTLE_P" --little_rl_best_path "$LITTLE_P" \
    > "golden_calib_$name.log" 2>&1
  ls -dt exp/golden_calib/${name}*/ | head -1
}

echo "=== Phase 0.1 校准双跑（GPU=$GPU，worktree 隔离区）==="
echo "--- run 1/2 ---"
R1=$(run_once run1)
echo "run1 → $R1"
echo "--- run 2/2 ---"
R2=$(run_once run2)
echo "run2 → $R2"
echo ""
echo "=== ① 稳定性（run1 vs run2）：LOCKED 字段应零漂移 ==="
python3 test/golden/compare.py --stability \
  "${R1}adaptive_tridecoding_gsm8k_metrics.json" \
  "${R2}adaptive_tridecoding_gsm8k_metrics.json" || true
echo ""
echo "=== ② golden 复现（run1 vs golden q15_free_g5）：应 PASS ==="
python3 test/golden/compare.py q15_free_g5 \
  "${R1}adaptive_tridecoding_gsm8k_metrics.json" || true
echo ""
echo "完成。两份判定若全绿 → golden 分类表生效，Phase 1 可以开工；"
echo "若 ① 红灯 → 存在 GPU 非确定性，走 Phase 0.5（确定性开关）；"
echo "若 ① 绿 ② 红 → 复现链有缺口（环境/checkpoint/数据），先查再动重构。"
