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

### Gemma-2 系列 (2B + 9B + 27B)
- **2 GPU + target 4bit**: 在当前环境下可能加载更容易，但需要额外警惕 `27B 4-bit` 的数值稳定性问题。
- **2 GPU + target non-4bit**: 往往会触发 disk offload，能够运行但通常非常慢。
- **3 GPU + target non-4bit**: 当前验证过的较可靠方案。`27B` 可以用 `device_map="auto"` 分片到 3 张卡上，并避免 disk offload。
- **4 GPU + target non-4bit**: 更稳妥，适合同时给 draft 与 target 留出更充裕显存。

## 调试建议

1. **查看设备分配**: 运行时会自动打印各模型的设备分配信息
2. **监控GPU内存**: 使用 `nvidia-smi` 或 `watch -n 1 nvidia-smi`
3. **调整策略**: 如果遇到OOM，可以：
   - 增加GPU数量
   - 启用量化（自动对>20B模型启用4bit量化）
   - 减小batch size或序列长度

4. **先验证 target 数值正确性**:
   - 如果大模型 target 在 4-bit 路径下出现异常 top-k、`NaN logits` 或显著异常 acceptance，优先用独立 raw transformers 脚本验证模型 forward 本身。

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

现在也支持显式覆盖量化策略：

- `--draft_quantization auto|4bit|none`
- `--target_quantization auto|4bit|none`
- `--little_quantization auto|4bit|none`

这允许针对特定模型（例如 Gemma 27B）关闭 4-bit 量化，再结合多卡分片验证非量化路径。

### 27B non-4bit 的 2 卡 / 3 卡差异

当前代码已支持在 target 非量化时，尝试对较大的 target 做 `device_map="auto"` 分片。

经验上：

1. **2 卡**
   - target 能跑起来，但容易出现部分层落盘 (`disk offload`)。
   - 一旦发生 disk offload，target 验证速度会明显变慢。

2. **3 卡**
   - 对 `google/gemma-2-27b-it` 这类 target，更容易避免 disk offload。
   - raw transformers forward 的 logits 已验证可恢复正常。

3. **4 卡**
   - 更适合 target non-4bit 与 draft 并存的正式实验场景。

## 故障排查

### OOM错误
1. 检查GPU数量: `echo $CUDA_VISIBLE_DEVICES`
2. 增加GPU数量或启用量化
3. 对于70B+模型，至少需要4个GPU
4. 对于 27B target non-4bit，如果 2 卡仍然触发 disk offload 或 OOM，优先尝试 3 卡分片

### 模型加载慢
1. 检查设备分配是否合理
2. 确保使用SSD存储模型文件
3. 检查网络（如果使用 `local_files_only=False`）
4. 检查是否发生了 `disk offload`；如果 target 有一部分层落盘，推理速度会显著下降

### 4-bit 路径数值异常
如果观测到以下现象：

- target top-k 全是异常 special token
- acceptance 极低且无法用模型规模差解释
- raw logits / probs 出现 `NaN`

建议按以下顺序排查：

1. 用独立 raw transformers 脚本验证 target forward 是否已经异常。
2. 若 4-bit 异常、non-4bit 正常，则说明问题主要在量化路径，而不是 speculative decoding 逻辑本身。
3. 对该模型优先改用 non-4bit + 多卡分片验证 correctness。

### 设备分配不均
1. 查看打印的设备分配信息
2. 考虑手动指定 `device_map`
3. 确保 `CUDA_VISIBLE_DEVICES` 设置正确
