"""
HumanEval 评估脚本 - 使用 vLLM 加速推理
支持批量推理和高效内存管理
"""

import os
import sys
import json
import multiprocessing
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

# vLLM imports
from vllm import LLM, SamplingParams

# 添加父目录到路径以导入 few_shot_examples
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from eval.few_shot_examples import get_few_shot_prompt


def check_correctness_worker(check_program, result_queue):
    """在独立进程中执行代码检查"""
    try:
        exec(check_program, {})
        result_queue.put("passed")
    except Exception as e:
        result_queue.put(f"failed: {str(e)}")


def check_correctness(completion: str, test_code: str, entry_point: str, timeout: float = 3.0) -> bool:
    """
    检查代码完成的正确性
    
    Args:
        completion: 完整的函数代码
        test_code: 测试代码
        entry_point: 入口函数名
        timeout: 超时时间（秒）
    
    Returns:
        bool: 是否通过测试
    """
    check_program = completion + "\n" + test_code + "\n" + f"check({entry_point})"
    
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=check_correctness_worker, args=(check_program, queue))
    p.start()
    p.join(timeout)
    
    if p.is_alive():
        p.terminate()
        p.join()
        return False  # Timeout
    
    if not queue.empty():
        result = queue.get()
        return result == "passed"
    
    return False  # Failed/Empty


def postprocess_completion(prompt: str, completion: str) -> str:
    """
    后处理生成的代码
    
    Args:
        prompt: 原始提示（包含函数签名）
        completion: 模型生成的代码
    
    Returns:
        str: 处理后的完整代码
    """
    # 定义停止词
    stop_words = [
        "\nclass",
        "\ndef",
        "\n#",
        "\n@",
        "\nprint",
        "\nif",
        "\n```",
    ]
    
    # 移除提示部分，只保留生成的代码
    generation = completion
    
    # 在停止词处截断
    for stop_word in stop_words:
        if stop_word in generation:
            next_line = generation.index(stop_word)
            generation = generation[:next_line].strip()
    
    # 组合提示和生成的代码
    # 注意：prompt已经包含了函数签名，generation是函数体
    output_text = prompt + "\n    " + generation
    output_text = output_text.replace("\t", "    ")  # 统一使用空格缩进
    
    return output_text


def build_prompt(prompt: str, num_shots: int = 0) -> str:
    """
    构建HumanEval提示
    
    Args:
        prompt: 问题的函数签名和文档字符串
        num_shots: few-shot示例数量
    
    Returns:
        str: 完整的提示
    """
    few_shot = get_few_shot_prompt("humaneval", num_shots)
    return few_shot + prompt


def evaluate_humaneval(
    model_path: str,
    num_shots: int = 0,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 0.95,
    tensor_parallel_size: int = 1,
    max_samples: int = None,
    output_file: str = "humaneval_vllm_results.jsonl",
    timeout: float = 3.0,
    use_chat_template: bool = False,
):
    """
    使用 vLLM 评估 HumanEval 数据集
    
    Args:
        model_path: 模型路径或 HuggingFace model ID
        num_shots: few-shot 示例数量
        max_tokens: 最大生成 token 数
        temperature: 采样温度（0.0 = greedy decoding）
        top_p: nucleus sampling 参数
        tensor_parallel_size: 张量并行大小（多 GPU）
        max_samples: 最大评估样本数（用于调试）
        output_file: 结果保存文件（JSONL格式）
        timeout: 代码执行超时时间（秒）
        use_chat_template: 是否使用聊天模板格式
    """
    
    print(f"🚀 初始化 vLLM 模型: {model_path}")
    print(f"📊 配置: shots={num_shots}, max_tokens={max_tokens}, temp={temperature}, top_p={top_p}")
    
    # 初始化 vLLM
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        max_model_len=2048,  # 可根据模型调整
    )
    
    # 获取tokenizer以便使用chat template
    tokenizer = None
    if use_chat_template:
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            print("✅ 加载tokenizer用于chat template")
        except Exception as e:
            print(f"⚠️ 无法加载tokenizer，将不使用chat template: {e}")
            use_chat_template = False
    
    # 设置采样参数
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=["\nclass", "\ndef", "\nif", "\nprint"],  # 代码停止标记
    )
    
    # 加载数据集
    print("📚 加载 HumanEval 数据集...")
    try:
        dataset = load_dataset("openai_humaneval", split="test")
    except Exception as e:
        print(f"❌ 加载数据集失败: {e}")
        raise e
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    print(f"✅ 加载 {len(dataset)} 个样本")
    
    # 构建提示
    print("🔧 构建提示...")
    prompts = []
    original_prompts = []  # 保存原始prompt用于后处理
    
    for item in dataset:
        original_prompt = item['prompt']
        original_prompts.append(original_prompt)
        
        full_prompt = build_prompt(original_prompt, num_shots=num_shots)
        
        # 如果使用chat template
        if use_chat_template and tokenizer:
            messages = [
                {"role": "system", "content": "You are a helpful assistant. Please complete the following Python code."},
                {"role": "user", "content": full_prompt}
            ]
            formatted_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompts.append(formatted_prompt)
        else:
            prompts.append(full_prompt)
    
    # 批量推理
    print("🤖 开始推理...")
    outputs = llm.generate(prompts, sampling_params)
    
    # 评估结果
    print("📏 评估结果...")
    results = []
    passed_count = 0
    
    # 使用JSONL格式保存结果（与官方HumanEval评估格式一致）
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, output in enumerate(tqdm(outputs, desc="评估进度")):
            item = dataset[i]
            task_id = item['task_id']
            original_prompt = original_prompts[i]
            test_code = item['test']
            entry_point = item['entry_point']
            
            generated_text = output.outputs[0].text
            
            # 后处理生成的代码
            completion = postprocess_completion(original_prompt, generated_text)
            
            # 检查正确性
            is_correct = check_correctness(completion, test_code, entry_point, timeout=timeout)
            
            if is_correct:
                passed_count += 1
            
            # 保存结果（JSONL格式）
            result = {
                "task_id": task_id,
                "completion": completion,
                "passed": is_correct,
                "raw_generation": generated_text,
            }
            
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
            f.flush()
            
            results.append(result)
    
    # 计算pass@1准确率
    pass_at_1 = passed_count / len(dataset) * 100
    
    # 打印统计信息
    print("\n" + "="*60)
    print(f"📊 HumanEval 评估结果")
    print("="*60)
    print(f"总样本数: {len(dataset)}")
    print(f"通过数量: {passed_count}")
    print(f"Pass@1: {pass_at_1:.2f}%")
    print("="*60)
    
    # 保存汇总统计
    summary_file = output_file.replace('.jsonl', '_summary.json')
    summary_data = {
        "model_path": model_path,
        "config": {
            "num_shots": num_shots,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "tensor_parallel_size": tensor_parallel_size,
            "timeout": timeout,
            "use_chat_template": use_chat_template,
        },
        "metrics": {
            "total_samples": len(dataset),
            "passed_count": passed_count,
            "pass_at_1": pass_at_1
        }
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 详细结果已保存到: {output_file}")
    print(f"💾 汇总统计已保存到: {summary_file}")
    
    return pass_at_1


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="使用 vLLM 评估 HumanEval")
    
    # 模型配置
    parser.add_argument("--model_path", type=str, required=True,
                        help="模型路径或 HuggingFace model ID")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="张量并行大小（GPU 数量）")
    
    # 评估配置
    parser.add_argument("--num_shots", type=int, default=0,
                        help="Few-shot 示例数量（默认为0，即zero-shot）")
    parser.add_argument("--max_tokens", type=int, default=512,
                        help="最大生成 token 数")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="采样温度（0.0 = greedy decoding）")
    parser.add_argument("--top_p", type=float, default=0.95,
                        help="Nucleus sampling 参数")
    parser.add_argument("--timeout", type=float, default=3.0,
                        help="代码执行超时时间（秒）")
    
    # 格式选项
    parser.add_argument("--use_chat_template", action="store_true",
                        help="使用聊天模板格式（适用于chat模型）")
    
    # 调试选项
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大评估样本数（用于调试）")
    parser.add_argument("--output_file", type=str, default="humaneval_vllm_results.jsonl",
                        help="结果保存文件（JSONL格式）")
    
    args = parser.parse_args()
    
    # 运行评估
    pass_at_1 = evaluate_humaneval(
        model_path=args.model_path,
        num_shots=args.num_shots,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        tensor_parallel_size=args.tensor_parallel_size,
        max_samples=args.max_samples,
        output_file=args.output_file,
        timeout=args.timeout,
        use_chat_template=args.use_chat_template,
    )
