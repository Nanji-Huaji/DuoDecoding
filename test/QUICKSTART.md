# GSM8K vLLM 评估 - 快速启动指南

## 🚀 快速开始

### 1. 安装依赖
```bash
pip install vllm datasets
```

### 2. 运行评估

**方法 A: 使用 Python 脚本**
```bash
cd test

# 评估 Llama-3.1-8B
python eval_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --num_shots 8

# 调试模式（只评估 50 个样本）
python eval_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.1-8B-Instruct \
    --max_samples 50
```

默认输出会保存到仓库根目录下的 `experiment_results/`。

**方法 B: 使用 Bash 脚本**
```bash
cd test

# 使用默认参数
./run_eval.sh

# 自定义模型
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct" ./run_eval.sh

# 多 GPU
TENSOR_PARALLEL=4 MODEL_PATH="meta-llama/Llama-3.1-70B-Instruct" ./run_eval.sh
```

### 3. 分析结果
```bash
# 查看详细分析
python analyze_results.py --result_file ../experiment_results/gsm8k_vllm_results.json

# 比较两个模型
python analyze_results.py \
    --result_file ../experiment_results/results_model1.json \
    --compare_with ../experiment_results/results_model2.json
```

## 📋 常用命令

### 不同模型评估

```bash
# Llama 3.1 8B
python eval_gsm8k_vllm.py --model_path meta-llama/Llama-3.1-8B-Instruct

# Llama 3.1 70B (4 GPU)
python eval_gsm8k_vllm.py \
    --model_path meta-llama/Llama-3.1-70B-Instruct \
    --tensor_parallel_size 4

# Qwen 2.5 7B
python eval_gsm8k_vllm.py --model_path Qwen/Qwen2.5-7B-Instruct

# Mistral 7B
python eval_gsm8k_vllm.py --model_path mistralai/Mistral-7B-Instruct-v0.2

# 本地模型
python eval_gsm8k_vllm.py --model_path /path/to/local/model
```

### Few-shot 变化

```bash
# 0-shot (不使用示例)
python eval_gsm8k_vllm.py --model_path MODEL --num_shots 0

# 5-shot
python eval_gsm8k_vllm.py --model_path MODEL --num_shots 5

# 8-shot (推荐)
python eval_gsm8k_vllm.py --model_path MODEL --num_shots 8
```

### 调试选项

```bash
# 快速测试（10 个样本）
python eval_gsm8k_vllm.py --model_path MODEL --max_samples 10

# 中等测试（100 个样本）
python eval_gsm8k_vllm.py --model_path MODEL --max_samples 100
```

## 🔧 故障排除

### OOM (内存不足)

**问题**: `torch.cuda.OutOfMemoryError`

**解决方案**:
1. 增加 GPU 数量: `--tensor_parallel_size 2`
2. 使用量化模型
3. 减少最大序列长度（修改代码中的 `max_model_len`）

### 模型下载慢

**解决方案**:
```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或使用本地已下载的模型
python eval_gsm8k_vllm.py --model_path /path/to/downloaded/model
```

### vLLM 版本问题

**推荐版本**:
```bash
pip install vllm>=0.5.0
```

## 📊 预期结果

| 模型 | Few-shot | 预期准确率 |
|------|----------|-----------|
| Llama-3.1-8B-Instruct | 8 | ~80-85% |
| Llama-3.1-70B-Instruct | 8 | ~90-95% |
| Qwen2.5-7B-Instruct | 8 | ~80-85% |
| Mistral-7B-Instruct | 8 | ~45-55% |

*注：实际结果可能因模型版本和配置而异*

## 🆚 与原版评估脚本对比

| 特性 | eval/eval_gsm8k.py | test/eval_gsm8k_vllm.py |
|------|-------------------|------------------------|
| 速度 | 基准 (1x) | 10-30x 更快 |
| 内存效率 | 一般 | 优秀 (PagedAttention) |
| 批处理 | 手动实现 | 自动优化 |
| 多 GPU | 需配置 | 原生支持 |
| 使用场景 | 完整评估流程 | 快速基准测试 |

## 💡 最佳实践

1. **首次测试**: 使用 `--max_samples 50` 快速验证
2. **完整评估**: 移除 `--max_samples` 参数
3. **大模型**: 使用 `--tensor_parallel_size` 分布到多卡
4. **结果分析**: 使用 `analyze_results.py` 深入分析
5. **批量测试**: 修改 `run_eval.sh` 循环测试多个模型

## 📁 文件说明

- `eval_gsm8k_vllm.py` - 主评估脚本
- `run_eval.sh` - Bash 运行脚本
- `example_usage.py` - Python 使用示例
- `analyze_results.py` - 结果分析工具
- `README.md` - 详细文档
- `QUICKSTART.md` - 本快速指南
