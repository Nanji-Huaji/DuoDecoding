#!/bin/bash

# GSM8K vLLM 评估运行脚本

# 设置默认参数
MODEL_PATH=${MODEL_PATH:-"meta-llama/Llama-3.1-8B-Instruct"}
NUM_SHOTS=${NUM_SHOTS:-8}
MAX_TOKENS=${MAX_TOKENS:-512}
TENSOR_PARALLEL=${TENSOR_PARALLEL:-1}
OUTPUT_DIR=${OUTPUT_DIR:-"../experiment_results"}
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE=${OUTPUT_FILE:-"$OUTPUT_DIR/gsm8k_vllm_results.json"}

echo "=================================================="
echo "  GSM8K Evaluation with vLLM"
echo "=================================================="
echo "Model: $MODEL_PATH"
echo "Few-shot examples: $NUM_SHOTS"
echo "Max tokens: $MAX_TOKENS"
echo "Tensor parallel size: $TENSOR_PARALLEL"
echo "Output: $OUTPUT_FILE"
echo "=================================================="
echo ""

# 运行评估
python eval_gsm8k_vllm.py \
    --model_path "$MODEL_PATH" \
    --num_shots "$NUM_SHOTS" \
    --max_tokens "$MAX_TOKENS" \
    --tensor_parallel_size "$TENSOR_PARALLEL" \
    --output_file "$OUTPUT_FILE" \
    "$@"

echo ""
echo "✅ 评估完成！结果保存在: $OUTPUT_FILE"
