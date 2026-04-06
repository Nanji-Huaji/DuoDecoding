# HumanEval vLLM 评估工具使用说明

本目录包含使用 vLLM 评估 HumanEval 代码生成任务的脚本。

## 文件说明

- `eval_humaneval_vllm.py`: 主评估脚本，使用 vLLM 进行批量推理和评估
- `run_eval_humaneval.sh`: 快速启动脚本示例
- `README_humaneval.md`: 本说明文档

## 依赖安装

```bash
# 安装 vLLM
pip install vllm

# 安装其他依赖
pip install datasets transformers torch tqdm
```

## 快速开始

### 方法1：使用shell脚本（推荐）

1. 编辑 `run_eval_humaneval.sh`，修改模型路径：
```bash
MODEL_PATH="your/model/path"  # 例如: "meta-llama/Llama-3.1-8B-Instruct"
```

2. 运行评估：
```bash
cd test
bash run_eval_humaneval.sh
```

### 方法2：直接使用Python脚本

```bash
cd test
python eval_humaneval_vllm.py \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --tensor_parallel_size 1 \
    --num_shots 0 \
    --max_tokens 512 \
    --temperature 0.0 \
    --output_file ../experiment_results/results_humaneval.jsonl
```

## 参数说明

### 模型配置
- `--model_path`: 模型路径或HuggingFace model ID（必需）
- `--tensor_parallel_size`: 使用的GPU数量，默认为1

### 评估配置
- `--num_shots`: Few-shot示例数量，默认为0（zero-shot）
- `--max_tokens`: 最大生成token数，默认512
- `--temperature`: 采样温度，0.0表示贪婪解码，默认0.0
- `--top_p`: Nucleus sampling参数，默认0.95
- `--timeout`: 代码执行超时时间（秒），默认3.0

### 格式选项
- `--use_chat_template`: 使用聊天模板格式（适用于Instruct/Chat模型）

### 调试选项
- `--max_samples`: 限制评估样本数（用于快速测试）
- `--output_file`: 结果保存文件，默认保存到仓库根目录的 `experiment_results/humaneval_vllm_results.jsonl`

## 使用示例

### 示例1：评估Base模型（Zero-shot）
```bash
python eval_humaneval_vllm.py \
    --model_path "deepseek-ai/deepseek-coder-6.7b-base" \
    --num_shots 0 \
    --temperature 0.0 \
    --output_file ../experiment_results/deepseek_base_results.jsonl
```

### 示例2：评估Chat模型（使用chat template）
```bash
python eval_humaneval_vllm.py \
    --model_path "Qwen/Qwen2.5-Coder-7B-Instruct" \
    --use_chat_template \
    --num_shots 0 \
    --temperature 0.0 \
    --output_file ../experiment_results/qwen_chat_results.jsonl
```

### 示例3：Few-shot评估
```bash
python eval_humaneval_vllm.py \
    --model_path "meta-llama/Llama-3.1-8B-Instruct" \
    --use_chat_template \
    --num_shots 3 \
    --temperature 0.0 \
    --output_file ../experiment_results/llama_fewshot_results.jsonl
```

### 示例4：多GPU并行评估
```bash
python eval_humaneval_vllm.py \
    --model_path "meta-llama/Llama-3.1-70B-Instruct" \
    --tensor_parallel_size 4 \
    --use_chat_template \
    --output_file ../experiment_results/llama_70b_results.jsonl
```

### 示例5：调试模式（仅评估前10个样本）
```bash
python eval_humaneval_vllm.py \
    --model_path "deepseek-ai/deepseek-coder-6.7b-base" \
    --max_samples 10 \
    --output_file ../experiment_results/debug_results.jsonl
```

## 输出文件

评估会生成两个文件：

### 1. 详细结果文件（JSONL格式）
例如：`experiment_results/humaneval_vllm_results.jsonl`

每行包含一个样本的结果：
```json
{
  "task_id": "HumanEval/0",
  "completion": "完整的代码实现",
  "passed": true,
  "raw_generation": "模型原始输出"
}
```

### 2. 汇总统计文件（JSON格式）
例如：`experiment_results/humaneval_vllm_results_summary.json`

包含整体评估指标：
```json
{
  "model_path": "模型路径",
  "config": {
    "num_shots": 0,
    "max_tokens": 512,
    ...
  },
  "metrics": {
    "total_samples": 164,
    "passed_count": 98,
    "pass_at_1": 59.76
  }
}
```

## 评估指标

- **Pass@1**: 在单次生成中代码通过所有测试用例的比例
- **Total Samples**: HumanEval数据集包含164个编程问题
- **Passed Count**: 通过测试的问题数量

## 注意事项

1. **内存要求**: 根据模型大小选择合适的GPU。例如：
   - 7B模型: 1个A100 (40GB) 或 1个A6000 (48GB)
   - 13B模型: 1个A100 (80GB) 或 2个A100 (40GB)
   - 70B模型: 4个A100 (80GB)

2. **超时设置**: 某些复杂测试可能需要更长执行时间，可以适当增加`--timeout`参数

3. **Chat模板**: 对于Instruct/Chat模型，建议使用`--use_chat_template`以获得更好的性能

4. **温度参数**: 
   - `temperature=0.0`: 贪婪解码，结果确定性强（推荐用于代码生成）
   - `temperature>0`: 随机采样，可增加多样性但可能降低准确率

5. **Few-shot**: HumanEval通常在zero-shot设置下评估，但您也可以尝试few-shot来提升性能

## 推荐模型

以下是一些在HumanEval上表现良好的开源模型：

- **DeepSeek-Coder系列**: deepseek-ai/deepseek-coder-{6.7b,33b}-{base,instruct}
- **Qwen-Coder系列**: Qwen/Qwen2.5-Coder-{7B,32B}-Instruct
- **StarCoder系列**: bigcode/starcoder{,2}-{3b,7b,15b}
- **CodeLlama系列**: meta-llama/CodeLlama-{7b,13b,34b}-{Python,Instruct}-hf
- **Llama-3系列**: meta-llama/Llama-3.1-{8B,70B}-Instruct

## 故障排除

### 问题1: CUDA内存不足
```
RuntimeError: CUDA out of memory
```
**解决方案**: 
- 减少`tensor_parallel_size`使用更多GPU
- 减小`max_model_len`参数
- 使用量化版本的模型

### 问题2: 代码执行超时
```
大量样本显示 passed: false
```
**解决方案**:
- 增加`--timeout`参数（例如5.0或10.0）
- 检查生成的代码是否包含死循环

### 问题3: 加载数据集失败
```
Failed to load dataset from Hugging Face
```
**解决方案**:
- 检查网络连接
- 使用代理：`export HF_ENDPOINT=https://hf-mirror.com`
- 或手动下载数据集

## 性能基准

典型评估时间（在A100 80GB GPU上）：

| 模型大小 | GPU数量 | 评估时间 |
|---------|--------|---------|
| 7B      | 1      | ~5分钟  |
| 13B     | 1      | ~8分钟  |
| 34B     | 2      | ~12分钟 |
| 70B     | 4      | ~15分钟 |

*实际时间取决于硬件配置和网络速度*

## 进阶使用

### 批量评估多个模型
```bash
#!/bin/bash
models=(
    "deepseek-ai/deepseek-coder-6.7b-base"
    "Qwen/Qwen2.5-Coder-7B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
)

for model in "${models[@]}"; do
    echo "Evaluating $model..."
    model_name=$(basename $model)
    python eval_humaneval_vllm.py \
        --model_path "$model" \
        --output_file "../experiment_results/results_${model_name}.jsonl"
done
```

### 整合到实验流程
可以将此脚本集成到自动化实验管道中，与其他评估任务（如GSM8K）一起运行。

## 相关资源

- [HumanEval论文](https://arxiv.org/abs/2107.03374)
- [vLLM文档](https://docs.vllm.ai/)
- [OpenAI HumanEval数据集](https://github.com/openai/human-eval)

## 许可证

本脚本遵循项目主仓库的许可证。
