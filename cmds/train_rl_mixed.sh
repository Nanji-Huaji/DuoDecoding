#!/bin/bash

# 确保在项目根目录
cd "$(dirname "$0")/.." || exit

echo "Starting Mixed Multi-Task RL Agent Training..."
echo "Datasets: mt_bench, gsm8k, cnndm, xsum, humaneval (Mixed)"
echo "Metrics: TPS (Tokens Per Second)"

# 环境变量设置
GPU_ID=${CUDA_VISIBLE_DEVICES:-0}
PORT=$((29500 + RANDOM % 1000))

# 混合训练脚本路径
SCRIPT="eval/eval_mixed.py"

# 模型设置 (优先使用环境变量)
DRAFT_MODEL=${DRAFT_MODEL:-"tiny-llama-1.1b"}
TARGET_MODEL=${TARGET_MODEL:-"Llama-2-13b"}
LITTLE_MODEL=${LITTLE_MODEL:-"llama-68m"}

# RL Adapter 路径 (优先使用环境变量)
MAIN_RL_PATH=${MAIN_RL_PATH:-"checkpoints/rl_adapter_main.pth"}
LITTLE_RL_PATH=${LITTLE_RL_PATH:-"checkpoints/rl_adapter_little.pth"}

# Accuracy Head 路径 (优先使用环境变量)
ACC_HEAD_PATH=${ACC_HEAD_PATH:-"src/SpecDec_pp/checkpoints/llama-13b/exp-weight6-layer3"}
SMALL_DRAFT_ACC_HEAD_PATH=${SMALL_DRAFT_ACC_HEAD_PATH:-"src/SpecDec_pp/checkpoints/llama-1.1b/exp-weight6-layer3"}
DRAFT_TARGET_ACC_HEAD_PATH=${DRAFT_TARGET_ACC_HEAD_PATH:-"src/SpecDec_pp/checkpoints/llama-13b/exp-weight6-layer3"}

# 训练参数
TOTAL_SAMPLES=2000 # 混合训练的总样本数
BW=34.6
LATENCY=0
END_BW=563

# 加速启动命令
# 使用 exec 替换当前 shell 进程，这样 PID 会保持一致，且信号能直接传递
exec accelerate launch \
    --num_processes 1 \
    --main_process_port $PORT \
    $SCRIPT \
    --eval_mode adaptive_tridecoding \
    -e train_rl_mixed_tasks \
    --draft_model $DRAFT_MODEL \
    --target_model $TARGET_MODEL \
    --little_model $LITTLE_MODEL \
    --max_tokens 128 \
    --temp 0.0 \
    --use_rl_adapter \
    --use_stochastic_comm \
    --edge_cloud_bandwidth $BW \
    --edge_end_bandwidth $END_BW \
    --ntt_ms_edge_cloud $LATENCY \
    --main_rl_path "$MAIN_RL_PATH" \
    --little_rl_path "$LITTLE_RL_PATH" \
    --acc_head_path "$ACC_HEAD_PATH" \
    --small_draft_acc_head_path "$SMALL_DRAFT_ACC_HEAD_PATH" \
    --draft_target_acc_head_path "$DRAFT_TARGET_ACC_HEAD_PATH" \
    --eval_data_num $TOTAL_SAMPLES
