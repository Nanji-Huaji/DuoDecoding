# 多卡加载优化文档

## 概述

优化了 `src/engine.py` 中的模型加载逻辑，实现了智能的多GPU分配策略，确保大模型可以合理地分配到多个GPU上。

## 主要改进

### 1. 智能设备映射策略 (`_get_device_map_strategy`)

根据模型大小和可用GPU数量，自动选择最优的设备映射策略：

- **单GPU环境**: 使用 `cuda:0`
- **模型 >= 60B**: 使用 `balanced` - 在所有GPU间均衡分配
- **模型 >= 25B**: 使用 `balanced_low_0` - 优先使用GPU 0，其他GPU作为辅助
- **模型 >= 10B + 多GPU**: 使用 `balanced_low_0`
- **小模型**: 使用 `auto` - 让transformers自动决定

### 2. GPU数量检测 (`_get_available_gpu_count`)

智能检测可用GPU数量：
- 优先读取 `CUDA_VISIBLE_DEVICES` 环境变量
- 回退到 `torch.cuda.device_count()`
- 确保在多GPU环境中正确识别可用资源

### 3. 模型设备分配信息打印 (`_print_model_device_info`)

加载完成后自动打印每个模型的设备分配情况：
```
============================================================
Model Device Allocation:
============================================================

Little Model:
  cuda:0: 12 layers

Draft Model:
  cuda:0: 24 layers

Target Model:
  cuda:0: 15 layers
  cuda:1: 15 layers
  cuda:2: 15 layers
  cuda:3: 15 layers
============================================================
```

## 使用示例

### 单GPU环境
```bash
CUDA_VISIBLE_DEVICES=0 python exp.py
```
- 小模型（< 10B）: 使用 `cuda:0`
- 大模型（>= 10B）: 使用 `cuda:0`（可能OOM）

### 双GPU环境
```bash
CUDA_VISIBLE_DEVICES=0,1 python exp.py
```
- 小模型（< 10B）: 使用 `auto`（通常分配到 `cuda:0`）
- 中型模型（10-25B）: 使用 `balanced_low_0`（主要在GPU 0，部分在GPU 1）
- 大模型（25-60B）: 使用 `balanced_low_0`
- 超大模型（>= 60B）: 使用 `balanced`（均匀分配）

### 四GPU环境
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python exp.py
```
- Qwen3-32B: 使用 `balanced`，均匀分配到4个GPU
- Llama-70B: 使用 `balanced`，均匀分配到4个GPU

## 不同模型组合的建议配置

### Qwen系列 (0.6B + 1.7B + 14B)
- **1 GPU**: 可能OOM
- **2 GPU**: 推荐，draft和target分别占用不同GPU
- **4 GPU**: 最优，target模型可以分布到多个GPU

### Qwen系列 (1.7B + 14B + 32B)
- **2 GPU**: 可能紧张
- **4 GPU**: 推荐，32B模型会使用 `balanced` 策略

### Llama-2系列 (68M + 1.1B + 13B)
- **1 GPU**: 可以运行
- **2 GPU**: 推荐，更流畅

### Llama-2系列 (68M + 7B + 70B)
- **4 GPU**: 推荐，70B模型需要多卡
- **8 GPU**: 最优

## 调试建议

1. **查看设备分配**: 运行时会自动打印各模型的设备分配信息
2. **监控GPU内存**: 使用 `nvidia-smi` 或 `watch -n 1 nvidia-smi`
3. **调整策略**: 如果遇到OOM，可以：
   - 增加GPU数量
   - 启用量化（自动对>20B模型启用4bit量化）
   - 减小batch size或序列长度

## 技术细节

### Device Map策略说明

- `cuda:0`: 将整个模型加载到GPU 0
- `auto`: 让transformers自动分配，通常优先填满GPU 0再使用其他GPU
- `balanced_low_0`: 优先使用GPU 0，其他GPU辅助（适合draft+target场景）
- `balanced`: 在所有GPU间均匀分配（适合单个大模型）

### 量化配置

对于参数量 > 20B 且不包含 'awq' 的模型，自动启用4bit NF4量化：
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
```

## 故障排查

### OOM错误
1. 检查GPU数量: `echo $CUDA_VISIBLE_DEVICES`
2. 增加GPU数量或启用量化
3. 对于70B+模型，至少需要4个GPU

### 模型加载慢
1. 检查设备分配是否合理
2. 确保使用SSD存储模型文件
3. 检查网络（如果使用 `local_files_only=False`）

### 设备分配不均
1. 查看打印的设备分配信息
2. 考虑手动指定 `device_map`
3. 确保 `CUDA_VISIBLE_DEVICES` 设置正确
