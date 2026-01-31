#!/bin/bash

# 确保在项目根目录
cd "$(dirname "$0")/.." || exit

# 激活环境 (如果需要，取消注释)
# source /home/tiantianyi/miniconda3/bin/activate duodec_revise

echo "Starting Multi-Task RL Agent Training..."
echo "Mode: adaptive_decoding"
echo "Metrics: TPS (Tokens Per Second)"

# 运行训练的函数
run_training() {
    SCRIPT=$1
    TASK=$2
    SAMPLES=$3
    MODE=${4:-"adaptive_decoding"}
    BW=${5:-34.6}
    LATENCY=${6:-0}
    END_BW=${7:-563}
    
    echo ""
    echo "################################################################"
    echo "Running RL Training on Task: $TASK"
    echo "Mode: $MODE | Bandwidth: $BW Mbps | Latency: $LATENCY ms | Edge-End BW: $END_BW Mbps"
    echo "################################################################"
    
    # 动态分配端口以防冲突
    PORT=$((29500 + RANDOM % 1000))
    
    # 使用环境变量中的 GPU ID，如果没有设置则默认为 0
    GPU_ID=${CUDA_VISIBLE_DEVICES:-0}

    # 为 Tri-decoding 分配模型
    EXTRA_ARGS=()
    if [[ "$MODE" == "adaptive_tridecoding" ]]; then
        EXTRA_ARGS=(
            "--little_model" "llama-68m" 
            "--gamma1" "6" 
            "--gamma2" "4"
            "--small_draft_acc_head_path" "src/SpecDec_pp/checkpoints/llama-1.1b/exp-weight6-layer3"
            "--draft_target_acc_head_path" "src/SpecDec_pp/checkpoints/llama-13b/exp-weight6-layer3"
        )
    fi

    CUDA_VISIBLE_DEVICES=$GPU_ID accelerate launch \
        --num_processes 1 \
        --main_process_port $PORT \
        $SCRIPT \
        --eval_mode $MODE \
        -e train_rl_${TASK} \
        --draft_model tiny-llama-1.1b \
        --target_model Llama-2-13b \
        --max_tokens 128 \
        --temp 0.0 \
        --use_rl_adapter \
        --use_stochastic_comm \
        --task_name "${TASK}" \
        --edge_cloud_bandwidth $BW \
        --edge_end_bandwidth $END_BW \
        --ntt_ms_edge_cloud $LATENCY \
        --acc_head_path src/SpecDec_pp/checkpoints/llama-13b/exp-weight6-layer3 \
        "${EXTRA_ARGS[@]}" \
        --eval_data_num $SAMPLES || echo "Warning: Task $TASK failed or completed with errors."

    echo "Finished training on $TASK"
}

# 1. MT Bench (多轮对话)
run_training "eval/eval_mt_bench_noeval.py" "mt_bench" 50 "adaptive_decoding" 34.6 0

# 2. GSM8K (数学推理)
run_training "eval/eval_gsm8k.py" "gsm8k" 50 "adaptive_tridecoding" 34.6 0

# ... (rest of initial cold start)
run_training "eval/eval_cnndm.py" "cnndm" 50 "adaptive_decoding" 34.6 0

echo ""
echo "================================================================"
echo "Starting Infinite Training Loop with Dynamic Network & Modes..."
echo "================================================================"
while true; do
    # 随机打乱任务顺序
    TASKS=("mt_bench" "gsm8k" "cnndm" "xsum" "humaneval")
    
    # 兼容 bash/zsh 的打乱和数组处理
    if [ -n "$ZSH_VERSION" ]; then
        # zsh 强制按列拆分输出到数组
        SHUFFLED_TASKS=( ${(f)"$(shuf -e "${TASKS[@]}")"} )
    else
        # bash 默认按空白字符拆分
        SHUFFLED_TASKS=($(shuf -e "${TASKS[@]}"))
    fi
    
    # 随机选择解码模式
    MODES=("adaptive_decoding" "adaptive_tridecoding")

    for TASK in "${SHUFFLED_TASKS[@]}"; do
        # 随机生成网络环境参数
        RAND_BW=34.6                       # 为了匹配 baseline，将带宽设为 34.6 Mbps
        RAND_LAT=0                         # 为了匹配 baseline，将 Latency 设为 0
        RAND_END_BW=563                    # 为了匹配 baseline，将 Edge-End 带宽设为 563 Mbps
        
        # 统一处理随机索引
        RAND_IDX=$((RANDOM % 2))
        if [ -n "$ZSH_VERSION" ]; then
            # zsh 是 1-indexed
            RAND_MODE=${MODES[$((RAND_IDX + 1))]}
        else
            # bash 是 0-indexed
            RAND_MODE=${MODES[$RAND_IDX]}
        fi
        
        case $TASK in
            "mt_bench") SCRIPT="eval/eval_mt_bench_noeval.py" ;;
            "gsm8k")    SCRIPT="eval/eval_gsm8k.py" ;;
            "cnndm")    SCRIPT="eval/eval_cnndm.py" ;;
            "xsum")     SCRIPT="eval/eval_xsum.py" ;;
            "humaneval") SCRIPT="eval/eval_humaneval.py" ;;
        esac
        
        run_training "$SCRIPT" "$TASK" 100 "$RAND_MODE" "$RAND_BW" "$RAND_LAT" "$RAND_END_BW"
        
        sleep 5
    done
    
    echo "Completed one full round of tasks. Restarting..."
done
