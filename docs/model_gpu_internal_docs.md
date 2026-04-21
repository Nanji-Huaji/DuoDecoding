# KVCacheModel 核心架构与底层机制指南

本文档专为 **DuoDecoding** 团队内部开发者编写，深度剖析了 `src/model_gpu.py` 中 `KVCacheModel` 的设计初衷、状态流转以及显存分配上的核心优化逻辑。

## 1. 宏观定位与职责

在投机解码（Speculative Decoding）及其衍生工作流中，`KVCacheModel` 是连接高层推理逻辑（如 `src/engine.py`）与底层 Hugging Face Transformer 架构的核心桥梁。

- **状态化管理器（Stateful Wrapper）**：与单次无状态调用的模型不同，`KVCacheModel` 会跨步级（Step-by-step）持久化保管网络前向传递所产生的 Key-Value Cache、以及历史概率分布（`_prob_history`）和前向分类层输出（`logits_history`）。
- **双模并行调用**：在核心引擎（如 `_verify_draft_sequence` 所在阶段），它往往作为 `draft_model_cache` 和 `target_model_cache` 成对出现。两者状态在对齐、验证和裁切阶段深度联动。

---

## 2. 内存视图与连续预分配机制 (O(1) 显存管理)

在早期的序列生成中，持续拼接张量（即每通过一次前向网络就调用 `self.logits_history = torch.cat(...)`）会导致极其严重的内存碎片生成和随着序列长度增加的 $\mathcal{O}(N^2)$ 张量拷分开销。

为了解决这一痛点，现在的版本实现了以 **池化预分配** (Pool pre-allocation) 为特征的连续内存策略。

### `_ensure_buffer_size` 的底层机制

```python
def _ensure_buffer_size(self, batch_size: int, seq_len: int, device: torch.device, dtype: torch.dtype):
    ...
```

1. **首次初始化**：
   在首次进入 `_forward_with_kvcache` 时，代码会利用 `torch.empty` 以非置零（非初始化）模式，根据一个有足够安全裕量的长度（默认 `seq_len + 1024`，且下限为 `2048`）开辟两大核心张量物理内存块：
   - `_prob_buffer`: `(batch_size, max_length, vocab_size)`
   - `_logits_buffer`: `(batch_size, max_length, vocab_size)`
2. **零拷贝步进**：
   在增量解码（Cached Forward）阶段，不再进行 `cat` 操作，而是直接通过索引切片写入新生成的词特征：
   `self._prob_buffer[:, self._current_seq_len:end_pos, :] = probs`
3. **动态扩容 (Dynamic Resizing)**：
   如果极端的长文本导致 `seq_len > self.max_length`，模型将申请更大的连续内存 `max(self.max_length * 2, seq_len + 1024)`，并通过原地大批量浅拷贝（Shallow-copy）将旧版张量映射入新池。这显著减少了对 PyTorch 显存分配器 (CUDA Caching Allocator) 的唤醒频率。

---

## 3. 向后兼容性与状态视图 API

为了保证 `engine.py` 高阶应用模块在引用 `.shape` 及 `.size()` 时不出错，而无需重写庞大的外部依赖，底层实际的大内存块使用了 `@property` 视图进行了封装屏蔽。

```python
@property
def _prob_history(self) -> torch.Tensor | None:
    if self._prob_buffer is None:
        return None
    return self._prob_buffer[:, :self._current_seq_len, :]
```

**设计意图：**
- **视角欺骗**：虽然 `_prob_buffer` 长度可能是由 `torch.empty` 生成到的 `2048`，但对外该属性被严格定界到了真实的逻辑上下文长度 `self._current_seq_len` 截断。上游 `engine.py` 获取此属性时认为它的维度严丝合缝，保持了极高的工程兼容性。
- **废弃的 Write Setters**：针对历史遗留的老式 `self.logits_history = ...` 或回滚操作，类内安排了 `pass` 操作的 `@setter`。这意味着外部对这些变量强行重绑定的指针修改等效于只读失败，防御了高维度的指针污染。我们完全依靠内部 `_current_seq_len` 指针的前后调整来控制当前有效生命周期。

---

## 4. 推测解码的核心：KV Cache 回滚 (Rollback)

当 Draft 模型多看走眼生成的 Token 与 Target 大模型分布发生违和（Rejection Sampling），框架会直接触发 `rollback(end_pos)` 斩断失效时间线，复用前面的已有生成状态而不是从头预填充。

```python
def rollback(self, end_pos: int):
    # 此处负责清空端点后的所有数据预判
    ...
```

**回滚流水线细节：**
1. **清理有效逻辑长度**：直接回拉内部游标指针 `self._current_seq_len = end_pos`。（请注意物理侧在 `torch.empty` 缓冲里生成的脏数据不需要被主动清除，因为未来的覆写会覆盖它们；这进一步节约了算力）。
2. **`DynamicCache` 官方支持优先**：针对 Hugging Face 较新的 `.crop(end_pos)` 进行了支持并优先调用——这会依靠底层 C++ 或算子自行解决 KV 的丢弃。
3. **元组 `list(tuple)` 降级裁剪**：如果在旧版 transformer 框架下运行，代码会遍历所有深层网络层传回的元组 `(k, v)`，针对注意力存储规则中的 Sequence 维度（通常是对齐标准的左起第 `3` 维，即 `dim=2`，对应 `[batch, num_heads, seq_len, head_dim]`）进行物理维度定焦 `[:, :, :end_pos, :]`。

---

## 5. 潜在隐患与后续扩展限制 ⚠️

当前优化对于大多数右侧补齐（Right-Padding）的串行或批处理系统有效，但在后续进一步探索框架升级时，需要留意以下盲点：

1. **`hidden_states` 未纳入连续预分配池**
   当前 `self.hidden_states = outputs.hidden_states` 是粗暴的引用转交逻辑。若需在特征空间而不仅仅是词表分布中执行距离判定（例如深层 Hidden State 投机），需要专门为 `hidden_states` 实现同等逻辑的 `_ensure_buffer_size` 缓冲系统以防显存暴涨。
2. **Left-Padding 与动态批计算**的冲突
   如果是 vLLM 或 TGI 那样的完全连续批处理 (Continuous Batching)，且引入了**前缀左填充**模式（Left-Padding），则当前 `rollback` 工具中旧版降级的 `[:, :, :end_pos, :]` 切片器会直接破坏未对齐的注意力矩阵。动态批计算引入前，必须解耦并重构该 Tuple Fallback 操作。
3. **Setter 带来的静默失效问题**
   虽然 `setter` 置为了 `pass` 以防御外部非法修改，但它屏蔽了告警。如若新加入本项目的组员误以为对 `logits_history` 重新赋值是有效的流，会在 debug 环境陷入苦恼。请确保核心推理流程仅通过原生的 `generate` 和 `rollback` 来干涉隐层数据。

---

## 6. 2026 KV Cache 重构说明

这一轮重构的核心目标不是“再补一个 Gemma 特判”，而是将 `KVCacheModel` 从“手写 cached decode 输入协议”的模式，迁移到更贴近 Hugging Face 官方 generation/cache 语义的实现。

### 6.1 触发重构的实际问题

在 `google/gemma-2-9b-it -> google/gemma-2-27b-it` 的 `dist_split_spec` 调试过程中，暴露了三类结构性问题：

1. **模型族输入协议差异**
   - 旧实现通过手工构造 `input_ids + past_key_values + attention_mask` 调用 cached decode。
   - Gemma 这类模型对 `cache_position`、`position_ids`、mask 切片与 cache 长度对齐更敏感。
   - 同样的输入准备方式在 Llama/Qwen 上可能勉强可用，但在 Gemma 上会触发 device-side assert 或错误的 cached decode 行为。

2. **`model.device` 单设备假设不成立**
   - 一旦 target 通过 `device_map="auto"` 做多卡分片，模型不再有单一 `.device`。
   - 旧代码中大量 `self.target_model.device` / `self.draft_model.device` 的访问，在 sharded target 上会失效或产生隐式错误。

3. **Gemma 27B 4-bit 路径的数值不稳定**
   - 通过独立的 raw transformers 脚本验证，`google/gemma-2-27b-it` 在当前 4-bit 加载路径下，最后一位 logits 会直接变成全量 `NaN`。
   - 这意味着即使 DSSD/DSD 框架本身逻辑正确，target 验证分布也已经不可用。

### 6.2 重构后的核心思路

当前实现优先遵循 Hugging Face 官方 generation 输入准备流程：

1. **内部 cache 默认使用 `DynamicCache`**
   - `KVCacheModel` 在 prefill 前显式创建 `DynamicCache(config=model.config)`。
   - 这样做的目的，是将底层 KV 状态与各模型族的 mask/cache 规则尽量交给 transformers 统一处理。

2. **cached decode 统一走 `prepare_inputs_for_generation(...)`**
   - 相比直接手写 `past_key_values + attention_mask + input_ids`，官方 generation 逻辑会自动进行：
     - cache 依赖的输入切片
     - `cache_position` 传递
     - 必要时的 `position_ids` 派生
     - compilable cache 的 mask 处理

3. **保留外层 `KVCacheModel` 的工程语义**
   - 本项目仍然需要：
     - `prob_history`
     - `logits_history`
     - speculative rollback
     - `generate_with_rebuilt_topk`
   - 因此不会直接放弃 `KVCacheModel`，而是把它从“手写缓存协议”收敛成“围绕 HF cache 的状态包装层”。

### 6.3 为什么先选 `DynamicCache`

当前阶段不建议直接切到 `StaticCache`，原因如下：

1. **本项目高度依赖 rollback**
   - `dist_spec` / `dist_split_spec` 的 reject path 需要频繁执行 `crop(end_pos)`。
   - `DynamicCache.crop()` 与当前 speculative decoding 语义更加匹配。

2. **当前目标是先跑正确，而不是先做 compile 优化**
   - `StaticCache` 更适合固定长度、compile/export 优化场景。
   - 本轮重构优先保证多模型、多设备、rollback-heavy 工作流的正确性。

3. **先验证协议，再讨论静态缓存**
   - 如果 `DynamicCache` 路径在 Llama/Qwen/Gemma 上都稳定，再评估是否引入 `StaticCache` 作为第二阶段优化。

### 6.4 这轮重构没有改变的东西

以下行为仍然保留：

1. **连续预分配的 `_prob_buffer` / `_logits_buffer`**
   - 依旧由 `KVCacheModel` 管理，避免反复 `torch.cat` 带来的 O(N^2) 开销。

2. **rollback 对外语义**
   - 外部调用仍然只关心 `rollback(end_pos)`。
   - 内部优先使用 `Cache.crop(end_pos)`，必要时再兼容 legacy tuple cache。

3. **speculative verification 依赖的历史概率视图**
   - `prepare_verification_inputs`、`compute_acceptance_result`、`build_draft_probs_override` 等上层逻辑不需要了解底层 cache 类型。

---

## 7. Gemma 相关结论与工程启示

### 7.1 `gemma-2-27b-it` 的 4-bit 问题是底层数值问题

通过独立脚本（不复用项目中的 `KVCacheModel` 或 DSSD 逻辑）验证，可以确认：

1. `google/gemma-2-27b-it` 在当前 4-bit 路径下，raw transformers forward 就可能产出全 `NaN` logits。
2. 因此：
   - target top-k 异常
   - acceptance ratio 无意义
   - DSSD 低接受率不再能被解释为“仅仅是模型对不匹配”

换言之，Gemma 27B 当前 4-bit 路径不能作为可靠的 target 分布来源。

### 7.2 非 4-bit 的 27B 可以正常，但需要更多 GPU

独立验证结果表明：

1. `google/gemma-2-27b-it` 在 **3 张 A6000 非量化分片** 下，raw logits 是正常的。
2. 在 2 卡环境下若关闭 4-bit，通常会出现：
   - 部分层落到磁盘 offload
   - target 前向极慢

因此，Gemma 27B non-4bit 的正确性路径存在，但资源成本明显更高。

### 7.3 不是所有 Gemma pair 都差

对 draft-target 近似 acceptance 的离线分析表明：

1. `google/gemma-2-9b-it -> google/gemma-2-27b-it`
   - 即使 target non-4bit 且数值正常，下一 token 对齐度仍然极低，top-k overlap 几乎为 0。

2. `google/gemma-2-2b-it -> google/gemma-2-9b-it`
   - 同样的近似 acceptance 检查显示，这对模型在小样本上明显更有希望。

因此，Gemma 家族的问题不能简单归纳成“整个系列不适合 speculative decoding”；更准确的说法是：**不同尺度 pair 的对齐质量差异非常大，必须逐对验证。**

---

## 8. 后续路线图

### Phase 1: 稳定 `DynamicCache` 主路径

1. 补齐 Llama / Qwen 的回归验证。
2. 清理为 Gemma 定位问题而加入的临时 guard，只保留真正必要的 fail-fast 检查。
3. 将“模型输入设备解析”统一成一个稳定接口，彻底消除 `.device` 的单设备假设。

### Phase 2: 完善多卡与量化策略

1. 将 `target_quantization=none` 与 2 卡 / 3 卡 / 4 卡 target 分片策略文档化并回归测试。
2. 对“disk offload 是否可接受”建立明确规则，避免实验 silently 退化到极慢配置。

### Phase 3: 评估 `StaticCache`

只有在以下前提全部满足后，才建议尝试 `StaticCache`：

1. `DynamicCache` 路径在主要模型族上稳定。
2. rollback/crop 语义已有测试覆盖。
3. 当前瓶颈已证明确实来自 cache 结构，而不是目标模型数值问题或跨设备搬运。

`StaticCache` 应被视为第二阶段优化手段，而不是本轮重构的首选方案。
