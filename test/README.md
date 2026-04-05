# GSM8K 评估脚本 - vLLM 版本

使用 vLLM 进行高效的 GSM8K 数学推理评估。

## 特性

- ✅ **高性能推理**: 使用 vLLM 的 PagedAttention 和连续批处理
- ✅ **官方标准格式**: 使用 `#### X` 答案格式，与官方 GSM8K 一致
- ✅ **多层答案提取**: 优先 `####` 标记 → "The answer is" → 最后数字
- ✅ **批量处理**: 支持大批量并行推理
- ✅ **多 GPU 支持**: 通过 tensor_parallel_size 使用多卡
- ✅ **详细结果**: 保存每个样本的详细推理过程

## 安装依赖

```bash
pip install vllm datasets
```

## 使用方法

### 基础用法

```bash
# 单 GPU 评估
python test/eval_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --num_shots 8 \
    --max_tokens 512

# 多 GPU 评估（例如使用 4 张卡）
python test/eval_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.1-70B-Instruct \
    --tensor_parallel_size 4 \
    --num_shots 8

# 本地模型路径
python test/eval_gsm8k_vllm.py \
    --model_path /path/to/your/model \
    --num_shots 8
```

### 调试模式

```bash
# 只评估前 50 个样本进行快速测试
python test/eval_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --max_samples 50 \
    --output_file debug_results.json
```

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model_path` | 必需 | 模型路径或 HuggingFace ID |
| `--tensor_parallel_size` | 1 | GPU 数量（多卡并行） |
| `--num_shots` | 8 | Few-shot 示例数量 |
| `--batch_size` | 32 | 批处理大小（vLLM 会自动优化） |
| `--max_tokens` | 512 | 最大生成 token 数 |
| `--temperature` | 0.0 | 采样温度（0=贪婪解码） |
| `--max_samples` | None | 最大样本数（调试用） |
| `--output_file` | gsm8k_vllm_results.json | 结果保存文件 |

## 输出格式

结果保存为 JSON 文件，包含：

```json
{
  "model_path": "meta-llama/Llama-3.1-8B-Instruct",
  "config": {
    "num_shots": 8,
    "max_tokens": 512,
    ...
  },
  "metrics": {
    "total_samples": 1319,
    "correct_count": 1100,
    "accuracy": 83.40
  },
  "results": [
    {
      "index": 0,
      "question": "...",
      "gold_answer": "...",
      "model_output": "...",
      "extracted_gold": "6",
      "extracted_pred": "6",
      "is_correct": true
    },
    ...
  ]
}
```

## 性能优化建议

1. **批量大小**: vLLM 会自动优化，通常不需要手动调整
2. **GPU 内存**: 如果 OOM，减少 `max_model_len` 或增加 GPU 数量
3. **多卡推理**: 大模型使用 `--tensor_parallel_size` 并行
4. **量化**: 使用 GPTQ/AWQ 量化模型可减少显存占用

## 与原版 eval_gsm8k.py 的区别

| 特性 | 原版 | vLLM 版本 |
|------|------|-----------|
| 推理引擎 | HuggingFace Transformers | vLLM |
| 批处理 | 单样本或小批量 | 高效连续批处理 |
| 速度 | 较慢 | 快 10-30 倍 |
| 内存效率 | 一般 | PagedAttention 优化 |
| 多 GPU | 需要手动配置 | 原生支持 |
| 适用场景 | 完整评估管道 | 快速基准测试 |

## 常见问题

**Q: vLLM 和原版评估结果一致吗？**
A: 是的，两者使用相同的提示格式和答案提取逻辑，结果应该一致。

**Q: 如何使用本地模型？**
A: 直接传入本地路径：`--model_path /path/to/model`

**Q: 支持哪些模型？**
A: 所有 vLLM 支持的模型，包括 Llama、Qwen、Mistral、Gemma 等。

**Q: 内存不足怎么办？**
A: 1) 增加 GPU 数量 2) 使用量化模型 3) 减少 max_model_len

## 示例结果

使用 Llama-3.1-8B-Instruct 的典型结果：

```
📊 评估结果统计
============================================================
总样本数: 1319
正确数量: 1100
准确率: 83.40%
============================================================
```
