#!/bin/bash

# HumanEval vLLM 评估脚本示例
# 使用方法: bash run_eval_humaneval.sh

# ============ 配置区域 ============

# 模型路径（修改为你的模型路径）
# MODEL_PATH="meta-llama/Llama-3.1-8B-Instruct"
# MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
MODEL_PATH="/home/tiantianyi/code/DuoDecoding/llama/Llama-2-13b-hf"

# GPU配置
TENSOR_PARALLEL_SIZE=1  # 使用的GPU数量

# 采样参数
NUM_SHOTS=3            # Few-shot示例数量（0=zero-shot）
MAX_TOKENS=512         # 最大生成token数
TEMPERATURE=0.0        # 采样温度（0.0=贪婪解码）
TOP_P=0.95            # Nucleus sampling参数
TIMEOUT=3.0           # 代码执行超时时间（秒）

# 输出文件
OUTPUT_DIR=${OUTPUT_DIR:-"../experiment_results"}
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE=${OUTPUT_FILE:-"$OUTPUT_DIR/humaneval_vllm_results.jsonl"}

# 调试选项
# MAX_SAMPLES=10       # 取消注释以限制评估样本数

# ============ 运行评估 ============

echo "🚀 开始HumanEval评估..."
echo "模型: $MODEL_PATH"
echo "配置: shots=$NUM_SHOTS, temp=$TEMPERATURE, max_tokens=$MAX_TOKENS"
echo "=========================================="

# 基础命令
CMD="python eval_humaneval_vllm.py \
    --model_path $MODEL_PATH \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --num_shots $NUM_SHOTS \
    --max_tokens $MAX_TOKENS \
    --temperature $TEMPERATURE \
    --top_p $TOP_P \
    --timeout $TIMEOUT \
    --output_file $OUTPUT_FILE"

# 如果是chat模型，添加chat template选项
if [[ $MODEL_PATH == *"Instruct"* ]] || [[ $MODEL_PATH == *"chat"* ]]; then
    CMD="$CMD --use_chat_template"
    echo "检测到聊天模型，启用chat template"
fi

# 如果设置了MAX_SAMPLES，添加到命令
if [ ! -z "$MAX_SAMPLES" ]; then
    CMD="$CMD --max_samples $MAX_SAMPLES"
    echo "调试模式：仅评估前 $MAX_SAMPLES 个样本"
fi

echo ""
echo "执行命令:"
echo "$CMD"
echo ""

# 执行评估
eval $CMD

echo ""
echo "✅ 评估完成！"
echo "结果文件: $OUTPUT_FILE"
echo "汇总文件: ${OUTPUT_FILE%.jsonl}_summary.json"
