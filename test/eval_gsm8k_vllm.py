"""
GSM8K 评估脚本 - 使用 vLLM 加速推理
支持批量推理和高效内存管理
"""

import os
import sys
import json
import re
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

# vLLM imports
from vllm import LLM, SamplingParams

# 添加父目录到路径以导入 few_shot_examples
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from eval.few_shot_examples import GSM8K_FEW_SHOT_EXAMPLES

INVALID_ANS = "[invalid]"


def extract_answer_from_gold(completion: str) -> str:
    """从标准答案中提取数字答案"""
    if "####" in completion:
        ans = completion.split("####")[1].strip()
        return ans.replace(",", "").replace("$", "")
    return INVALID_ANS


def extract_answer_from_output(completion: str) -> str:
    """
    改进的答案提取逻辑，按优先级尝试多种方法
    1. 寻找 #### 标记（官方 GSM8K 格式）
    2. 寻找 "The answer is" 格式
    3. 提取最后一个数字（作为后备）
    """
    # 方法 1: 寻找 #### 标记
    if "####" in completion:
        try:
            answer = completion.split("####")[1].strip()
            answer = answer.split("\n")[0].strip()  # 取第一行
            answer = answer.replace(",", "").replace("$", "")
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                return numbers[0]
        except:
            pass
    
    # 方法 2: 寻找 "The answer is" 格式
    answer_patterns = [
        r"[Tt]he answer is:?\s*([\-\$]?[\d,\.]+)",
        r"[Aa]nswer:?\s*([\-\$]?[\d,\.]+)",
        r"^####\s*([\-\$]?[\d,\.]+)"
    ]
    
    for pattern in answer_patterns:
        match = re.search(pattern, completion)
        if match:
            answer = match.group(1).replace(",", "").replace("$", "")
            return answer
    
    # 方法 3: 提取最后一个数字（作为后备）
    text = completion.replace(",", "").replace("$", "")
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]
    
    return INVALID_ANS


def is_correct(completion: str, answer: str) -> bool:
    """判断模型输出是否正确"""
    gold = extract_answer_from_gold(answer)
    pred = extract_answer_from_output(completion)
    
    if gold == INVALID_ANS or pred == INVALID_ANS:
        return False
    
    try:
        # 转换为浮点数进行比较，处理小数情况
        return abs(float(gold) - float(pred)) < 1e-6
    except:
        # 如果无法转换为数字，进行字符串比较
        return gold.strip() == pred.strip()


def get_few_shot_prompt(num_shots: int = 8) -> str:
    """生成 few-shot 提示"""
    examples = GSM8K_FEW_SHOT_EXAMPLES[:num_shots]
    prompt = ""
    for example in examples:
        prompt += f"Q: {example['question']}\nA: {example['answer']}\n\n"
    return prompt


def build_prompt(question: str, num_shots: int = 8, use_chat_format: bool = True) -> str:
    """
    构建完整的提示
    
    Args:
        question: 问题文本
        num_shots: few-shot 示例数量
        use_chat_format: 是否使用聊天格式（适用于 chat 模型）
    """
    few_shot = get_few_shot_prompt(num_shots)
    
    if use_chat_format:
        # 适用于 Llama-3, Qwen 等 chat 模型
        instruction = "You are a helpful assistant. Solve the math problem step by step and put your final answer after #### at the end."
        return f"{instruction}\n\n{few_shot}Q: {question}\nA: Let's solve this step by step.\n"
    else:
        # 适用于 base 模型
        return f"{few_shot}Q: {question}\nA:"


def evaluate_gsm8k(
    model_path: str,
    num_shots: int = 8,
    batch_size: int = 32,
    max_tokens: int = 512,
    temperature: float = 0.0,
    tensor_parallel_size: int = 1,
    max_samples: int = None,
    output_file: str = "gsm8k_vllm_results.json"
):
    """
    使用 vLLM 评估 GSM8K 数据集
    
    Args:
        model_path: 模型路径或 HuggingFace model ID
        num_shots: few-shot 示例数量
        batch_size: 批处理大小（vLLM 会自动优化）
        max_tokens: 最大生成 token 数
        temperature: 采样温度（0.0 = greedy decoding）
        tensor_parallel_size: 张量并行大小（多 GPU）
        max_samples: 最大评估样本数（用于调试）
        output_file: 结果保存文件
    """
    
    print(f"🚀 初始化 vLLM 模型: {model_path}")
    print(f"📊 配置: shots={num_shots}, batch={batch_size}, max_tokens={max_tokens}, temp={temperature}")
    
    # 初始化 vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=2048,  # 可根据模型调整
    )
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0 if temperature == 0.0 else 0.95,
    )
    
    # 加载数据集
    print("📚 加载 GSM8K 数据集...")
    dataset = load_dataset("gsm8k", "main", split="test")
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✅ 加载 {len(dataset)} 个样本")
    
    # 构建提示
    print("🔧 构建提示...")
    prompts = []
    for item in dataset:
        prompt = build_prompt(item['question'], num_shots=num_shots)
        prompts.append(prompt)
    
    # 批量推理
    print("🤖 开始推理...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 评估结果
    print("📏 评估结果...")
    results = []
    correct_count = 0
    
    for i, output in enumerate(tqdm(outputs, desc="评估进度")):
        question = dataset[i]['question']
        gold_answer = dataset[i]['answer']
        generated_text = output.outputs[0].text
        
        is_correct_flag = is_correct(generated_text, gold_answer)
        if is_correct_flag:
            correct_count += 1
        
        # 保存详细结果
        result = {
            "index": i,
            "question": question,
            "gold_answer": gold_answer,
            "model_output": generated_text,
            "extracted_gold": extract_answer_from_gold(gold_answer),
            "extracted_pred": extract_answer_from_output(generated_text),
            "is_correct": is_correct_flag
        }
        results.append(result)
    
    # 计算准确率
    accuracy = correct_count / len(dataset) * 100
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"📊 评估结果统计")
    print("="*60)
    print(f"总样本数: {len(dataset)}")
    print(f"正确数量: {correct_count}")
    print(f"准确率: {accuracy:.2f}%")
    print("="*60)
    
    # 保存结果
    output_data = {
        "model_path": model_path,
        "config": {
            "num_shots": num_shots,
            "batch_size": batch_size,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "tensor_parallel_size": tensor_parallel_size,
        },
        "metrics": {
            "total_samples": len(dataset),
            "correct_count": correct_count,
            "accuracy": accuracy
        },
        "results": results
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 结果已保存到: {output_file}")
    
    return accuracy


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 vLLM 评估 GSM8K")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径或 HuggingFace model ID")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="张量并行大小（GPU 数量）")
    
    # 评估配置
    parser.add_argument("--num_shots", type=int, default=8,
                        help="Few-shot 示例数量")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="批处理大小")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度")
    
    # 调试选项
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数（用于调试）")
    parser.add_argument("--output_file", type=str, default="gsm8k_vllm_results.json",
                        help="结果保存文件")
    
    args = parser.parse_args()
    
    # 运行评估
    accuracy = evaluate_gsm8k(
        model_path=args.model_path,
        num_shots=args.num_shots,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        tensor_parallel_size=args.tensor_parallel_size,
        max_samples=args.max_samples,
        output_file=args.output_file
    )
