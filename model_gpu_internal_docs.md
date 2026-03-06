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