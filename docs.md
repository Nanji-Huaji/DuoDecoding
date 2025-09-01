# 文档


## 实验运行

在后台运行实验脚本：

```bash
nohup zsh cmds/exp.sh > experiment_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## 核心解码方法

### `tridecoding`

三级推测性解码方法，支持端侧、边缘侧和云端三级架构。

**函数签名：**
```python
tridecoding(self, prefix) -> Tuple[torch.Tensor, DecodingMetrics]
```

**参数：**
- `prefix` (torch.Tensor): 输入序列前缀

**返回值：**
- `torch.Tensor`: 生成的完整序列
- `DecodingMetrics`: 解码指标，包含前向传播次数、生成token数量、通信时间等

**功能：**
- 利用端侧小模型（little model）和边缘侧中等模型（draft model）生成候选序列
- 云端大模型（target model）进行验证和最终采样
- 支持动态的 gamma1 和 gamma2 参数调整
- 包含完整的通信时间模拟

### `tridecoding_with_bandwidth`

带宽感知的三级推测性解码方法。

**函数签名：**
```python
tridecoding_with_bandwidth(
    self,
    prefix,
    edge_cloud_bandwidth: float | None = None,
    edge_end_bandwidth: float | None = None,
    cloud_end_bandwidth: float | None = None,
) -> Tuple[torch.Tensor, DecodingMetrics]
```

**参数：**
- `prefix`: 输入序列前缀
- `edge_cloud_bandwidth`: 边缘-云端带宽 (Mbps)
- `edge_end_bandwidth`: 边缘-端侧带宽 (Mbps)  
- `cloud_end_bandwidth`: 云端-端侧带宽 (Mbps)

**功能：**
- 在 tridecoding 基础上加入了实际的带宽约束
- 根据带宽限制动态调整传输策略
- 提供详细的通信时间分析

### `uncertainty_decoding`

基于不确定性的推测性解码方法。

**函数签名：**
```python
uncertainty_decoding(self, prefix) -> Tuple[torch.Tensor, DecodingMetrics]
```

**功能：**
- 实现论文 "Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission" 中的方法
- 根据模型输出的不确定性决定是否需要传输概率分布
- 支持概率分布压缩以减少通信开销
- 使用 KV-Cache 机制提高效率

### `uncertainty_decoding_without_kvcache`

不使用 KV-Cache 的不确定性解码方法。

**函数签名：**
```python
uncertainty_decoding_without_kvcache(self, prefix) -> Tuple[torch.Tensor, DecodingMetrics]
```

**功能：**
- 与 `uncertainty_decoding` 相同的核心逻辑
- 不使用 KV-Cache，每次都重新计算全序列
- 适用于内存受限的场景

## 通信模拟器

### `CommunicationSimulator`

基础通信模拟器类，提供不同链路的带宽模拟。

**主要参数：**
- `bandwidth_edge_cloud`: 边缘-云端带宽
- `bandwidth_edge_end`: 边缘-端侧带宽  
- `bandwidth_cloud_end`: 云端-端侧带宽

**核心方法：**
- `simulate_transfer()`: 模拟数据传输时间
- `transfer()`: 执行实际的数据传输模拟

### `CUHLM` (Communication-efficient Uncertainty-aware Hybrid Language Model)

高级通信模拟器，支持基于不确定性的自适应传输策略。

**初始化参数：**
```python
CUHLM(
    bandwidth_edge_cloud,
    bandwidth_edge_end=float('inf'),
    bandwidth_cloud_end=float('inf'), 
    uncertainty_threshold: float = 0.8,
    vocab_size: int = 32000,
)
```

**核心方法：**

#### `calculate_uncertainty()`
计算模型输出的不确定性。

**函数签名：**
```python
@staticmethod
calculate_uncertainty(
    logits: torch.Tensor | None, 
    M: int = 20, 
    theta_max: float = 2.0, 
    draft_token: Optional[int] = None
) -> float
```

**参数：**
- `logits`: 模型输出的 logits
- `M`: 扰动采样次数
- `theta_max`: 最大温度参数
- `draft_token`: 草案 token

**返回值：**
- `float`: 不确定性分数 (0-1)

#### `determine_transfer_strategy()`
根据不确定性决定传输策略。

**函数签名：**
```python
determine_transfer_strategy(
    self, 
    uncertainty: float, 
    current_probs: torch.Tensor | None
) -> Tuple[bool, int]
```

**返回值：**
- `bool`: 是否需要传输概率分布
- `int`: 传输的词汇表大小

#### `_calculate_compressed_vocab_size()`
计算压缩后的词汇表大小。

**功能：**
- 严格按照论文公式(24)实现：k(t)* = arg min {k(t) | U_TV(au(t) + b) ≤ θ}
- 根据不确定性和概率分布计算最优的 top-k 大小

#### `rebuild_full_probs()`
重建完整的概率分布。

**函数签名：**
```python
@staticmethod
rebuild_full_probs(compressed_probs: torch.Tensor) -> torch.Tensor
```

**功能：**
- 将压缩的 top-k 概率分布恢复为完整的词汇表概率分布
- 将剩余概率质量均匀分配给未包含的 token

#### `transfer()`
执行数据传输并返回传输时间。

**函数签名：**
```python
transfer(
    self, 
    tokens: Optional[torch.Tensor], 
    prob_history: Optional[torch.Tensor] = None, 
    logits: Optional[torch.Tensor] = None, 
    link_type: LinkType = "edge_cloud"
) -> float
```

**功能：**
- 根据不确定性决定传输策略
- 支持概率分布压缩
- 返回实际的传输时间

## 解码指标 (DecodingMetrics)

记录解码过程中的各种性能指标：

- `little_forward_times`: 小模型前向传播次数
- `draft_forward_times`: 草案模型前向传播次数  
- `target_forward_times`: 目标模型前向传播次数
- `generated_tokens`: 总生成 token 数
- `little_generated_tokens`: 小模型生成的 token 数
- `draft_generated_tokens`: 草案模型生成的 token 数
- `little_accepted_tokens`: 小模型被接受的 token 数
- `draft_accepted_tokens`: 草案模型被接受的 token 数
- `wall_time`: 总墙钟时间
- `throughput`: 吞吐量 (tokens/秒)
- `communication_time`: 总通信时间
- `computation_time`: 总计算时间
- `edge_end_comm_time`: 边缘-端侧通信时间

## 数据类型

### `LinkType`
定义通信链路类型：
- `"edge_cloud"`: 边缘-云端链路
- `"edge_end"`: 边缘-端侧链路
- `"cloud_end"`: 云端-端侧链路

## 使用示例

```python
# 初始化解码器
decoder = Decoding(args)

# 使用三级推测性解码
output, metrics = decoder.tridecoding(input_prefix)

# 使用不确定性解码
output, metrics = decoder.uncertainty_decoding(input_prefix)

# 使用带宽感知的三级解码
output, metrics = decoder.tridecoding_with_bandwidth(
    input_prefix,
    edge_cloud_bandwidth=100.0,  # 100 Mbps
    edge_end_bandwidth=50.0,     # 50 Mbps
    cloud_end_bandwidth=200.0    # 200 Mbps
)
```

## 项目结构

```
DuoDecoding/
├── src/                    # 源代码目录
│   ├── engine.py          # 核心解码引擎
│   ├── communication.py   # 通信模拟器
│   └── ...
├── cmds/                   # 实验脚本
├── data/                   # 数据集文件
├── eval/                   # 评估脚本  
├── exp/                    # 实验结果
└── docs.md                # 本文档
```

## 相关论文

1. **DuoDecoding**: "Hardware-aware Heterogeneous Speculative Decoding with Dynamic Multi-Sequence Drafting" (2025)
2. **CUHLM**: "Communication-Efficient Hybrid Language Model via Uncertainty-Aware Opportunistic and Compressed Transmission"

## 依赖环境

- Python 3.10+
- PyTorch
- llama-cpp-python
- 其他依赖见 `requirements.txt`
