#!/bin/bash
# 验证图修复的**一站式验证**：守门 → 冒烟 → 配对 A/B → 微基准。
#
# 为什么串成一个脚本：本机 GPU 长期被第三方任务占用，空闲窗口出现的时间
# 不可预测。把「冒烟（接线是否通）→ 配对 A/B（性能/确定性结论）→ 微基准
# （归因分解）」排成一个队列，窗口一开就自动依次跑完，不需要人盯着。
#
# 用法: nohup bash scripts/run_verify_graph_validation.sh > exp_logs/vg_validation.log 2>&1 &
#   TOTAL_TIMEOUT_S=21600  总守门时限（默认 6h，超时退出码 3）
set -uo pipefail
cd /home/tiantianyi/code/DuoDecoding

TOTAL_TIMEOUT_S=${TOTAL_TIMEOUT_S:-21600}
SMOKE_N=${SMOKE_N:-2}
SMOKE_MAXTOK=${SMOKE_MAXTOK:-64}
AB_N=${AB_N:-40}
AB_ROUNDS=${AB_ROUNDS:-2}
AB_MAXTOK=${AB_MAXTOK:-128}
PORT_BASE=${PORT_BASE:-29700}

DEADLINE=$(( $(date +%s) + TOTAL_TIMEOUT_S ))
GPU=""

echo "### [$(date +%F' '%T)] 等待任一 GPU 空闲（总时限 ${TOTAL_TIMEOUT_S}s）"
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
  for cand in 0 1; do
    if .venv/bin/python scripts/wait_gpu_idle.py --gpu "$cand" \
         --timeout 45 --samples 2 --interval 10 >/dev/null 2>&1; then
      GPU=$cand; break
    fi
  done
  [ -n "$GPU" ] && break
  sleep 30
done
if [ -z "$GPU" ]; then
  echo "### 守门超时，未放行（exit 3）"; exit 3
fi
echo "### [$(date +%T)] GPU $GPU 空闲，放行"

ACC=src/SpecDec_pp/checkpoints/acc_head
CK=checkpoints/rl_fixedg16_nobp
COMMON=(
  --data_path data
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m
  --temp 0.0 --seed 20260911 --target_quantization 4bit --model_dtype bf16
  --num_shots 3 --eval_data_num "$SMOKE_N" --max_tokens "$SMOKE_MAXTOK"
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 0.5 --min_bandwidth_mbps 0.1
  --ntt_ms_edge_cloud 20
  --small_draft_acc_head_path "$ACC/llama-68m--to--tiny-llama-1.1b/exp-weight6-layer3"
  --draft_target_acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --acc_head_path "$ACC/tiny-llama-1.1b--to--llama-2-13b/exp-weight6-layer3"
  --eval_mode adaptive_tridecoding
  --use_rl_adapter --disable_rl_update --rl_action_space topk_thr
  --rl_force_threshold 0.4 --gamma1 16 --gamma2 16
  --main_rl_path "$CK/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth"
  --main_rl_best_path "$CK/main/tiny-llama-1.1b--to--llama-2-13b/latest.pth"
  --little_rl_path "$CK/little/llama-68m--to--tiny-llama-1.1b/latest.pth"
  --little_rl_best_path "$CK/little/llama-68m--to--tiny-llama-1.1b/latest.pth"
)

# ── 1. 冒烟：图接线是否通（护栏打印 3 次 = 三个缓存都接上）──────────────────
echo "### [$(date +%T)] 冒烟 N=$SMOKE_N MAXTOK=$SMOKE_MAXTOK --use_cuda_graph"
CUDA_VISIBLE_DEVICES=$GPU \
HTTPS_PROXY=http://127.0.0.1:7890 HTTP_PROXY=http://127.0.0.1:7890 \
.venv/bin/accelerate launch --num_processes 1 --main_process_port $((PORT_BASE)) \
  eval/eval_gsm8k.py -e vg_smoke "${COMMON[@]}" --use_cuda_graph \
  2>&1 | tee exp_logs/vg_smoke.log | grep -E "GSM8K Accuracy|cuda-graph|error:|Error|Traceback"

GUARD_COUNT=$(grep -c "已启用图回放" exp_logs/vg_smoke.log || true)
if [ "$GUARD_COUNT" -ne 3 ] || ! grep -q "GSM8K Accuracy" exp_logs/vg_smoke.log; then
  echo "### 冒烟失败：护栏打印 ${GUARD_COUNT}/3 或无准确率输出（exit 2）"
  exit 2
fi
if grep -q "Traceback" exp_logs/vg_smoke.log; then
  echo "### 冒烟失败：出现异常（exit 2）"; exit 2
fi
echo "### 冒烟通过 ✓（护栏 3/3）"

# ── 2. 配对 A/B：唯一变量 --use_cuda_graph（交替顺序，确定性校验）──────────
echo "### [$(date +%T)] 配对 A/B N=$AB_N ROUNDS=$AB_ROUNDS MAXTOK=$AB_MAXTOK"
GPU=$GPU N=$AB_N ROUNDS=$AB_ROUNDS MAXTOK=$AB_MAXTOK \
PORT_BASE=$((PORT_BASE + 10)) TAG_PREFIX=vg GATE_SKIP=1 \
  bash scripts/run_graph_ab_paired.sh

# ── 3. 微基准：verify 图 vs eager 整段（归因分解，13B 是关键数字）──────────
echo "### [$(date +%T)] 微基准 bench_verify_graph"
CUDA_VISIBLE_DEVICES=$GPU .venv/bin/python scripts/bench_verify_graph.py \
  llama/llama-68m llama/tiny-llama-1.1b llama/llama-2-13b \
  2>&1 | tee exp_logs/vg_bench.log | grep -vE "Warning|torch_dtype"

echo "### [$(date +%T)] 全部完成"
.venv/bin/python scripts/summarize_graph_ab.py "exp/vg_*_r*"
