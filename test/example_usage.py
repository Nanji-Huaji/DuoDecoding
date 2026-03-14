"""
使用示例：演示如何在 Python 代码中调用 GSM8K 评估
"""

from eval_gsm8k_vllm import evaluate_gsm8k

# 示例 1: 基本用法
# print("示例 1: 基本评估")
# accuracy = evaluate_gsm8k(
#     model_path="meta-llama/Llama-3.1-8B-Instruct",
#     num_shots=8,
#     max_tokens=512,
#     output_file="results_example1.json"
# )


# 示例 2: 调试模式（只评估 50 个样本）
print("\n示例 2: 调试模式")
accuracy = evaluate_gsm8k(
    model_path="/home/tiantianyi/code/DuoDecoding/llama/Llama-2-13b-hf",
    num_shots=5,
    max_samples=80,
    output_file="results_debug.json",
)


# # 示例 3: 多 GPU 评估
# print("\n示例 3: 多 GPU 评估")
# accuracy = evaluate_gsm8k(
#     model_path="meta-llama/Llama-3.1-70B-Instruct",
#     tensor_parallel_size=4,  # 使用 4 张 GPU
#     num_shots=8,
#     output_file="results_70b.json"
# )


# # 示例 4: 本地模型路径
# print("\n示例 4: 本地模型")
# accuracy = evaluate_gsm8k(
#     model_path="/path/to/your/local/model",
#     num_shots=8,
#     output_file="results_local.json"
# )


# # 示例 5: 自定义采样参数
# print("\n示例 5: 自定义采样")
# accuracy = evaluate_gsm8k(
#     model_path="Qwen/Qwen2.5-7B-Instruct",
#     num_shots=8,
#     temperature=0.0,  # greedy decoding
#     max_tokens=1024,
#     output_file="results_qwen.json"
# )
