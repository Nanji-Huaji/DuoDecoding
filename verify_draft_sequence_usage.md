# `_verify_draft_sequence` 使用示例文档

`_verify_draft_sequence` 是 `Decoding` 类（核心解码引擎）中的核心静态方法，主要用于在投机解码（Speculative Decoding）及其衍生算法中对草稿Token进行基于目标模型概率的拒绝采样（Rejection Sampling）验证逻辑。

## 原理概述

在投机解码中，由较小的“草稿模型”快速生成连续的 `gamma` 个 Tokens，然后交给较大的“目标模型”进行一次并行前向传播以获得目标概率分布。`_verify_draft_sequence` 的作用是使用标准的拒绝采样算法，依次循环这 `gamma` 个生成的 Tokens 并判定它们是否能被接受。

为了支持复杂的分布式推断模拟（如端云协同场景），此函数还集成了不同的通信模拟策略，用来准确计算 Token 传递及概率分发过程中的延迟和带宽开销。

## 参数详解

- `draft_model_cache`: `(KVCacheModel)` 较小的草稿模型包装实例，包含其生成的概率历史（`_prob_history`）和缓存。
- `target_model_cache`: `(KVCacheModel)` 较大的目标模型包装实例，包含其验证阶段提供的真实概率历史。
- `x`: `(torch.Tensor)` 已经生成的候选 Token 序列，形状通常为 `[batch_size, seq_len]`。
- `prefix_len`: `(int)` 此轮草稿 Token 生成前的前缀长度。
- `gamma`: `(int)` 本次尝试验证的草稿 Token 数量。
- `comm_simulator`: `(Optional[CommunicationSimulator])` 用于网络传输耗时计算的模拟器实例（非分布式场景下为 `None`）。
- `comm_link`: `(Literal)` 当前使用的通信链路标识，可选 `"edge_cloud"`, `"edge_end"`, `"cloud_end"`。
- `transfer_mode`: `(Literal)` Token和概率数据传输模式的开销计算策略。可选：
  - `"none"`: 不评估传输耗时（适用于本地普通模式或开销已在外部统筹）。
  - `"serial"`: 校验每个词时，串行统计对应 Token 的传输开销。
  - `"batch_before"`: 开始循环校验前，将所有 `gamma` 个 Tokens 的传输开销一次性结算。
- `send_reject_message`: `(bool)` 若拒绝采样失败，是否触发拒绝模拟中断消息至通信模块。
- `draft_probs_override`: `(Optional[torch.Tensor])` 若指定，在此次拒绝采样检验中强行覆盖默认草稿模型的概率矩阵，用于诸如 Uncertainty Decoding 等变体。
- `decoding_metrics`: `(Optional[DecodingMetrics])` 性能字典。若提供，本轮中验证了多少 Token 及接受了多少 Token 将会自动累加进去。

## 返回值说明
返回格式：`Tuple[int, int]` (目前代码定义上的签名)。
该函数的直接操作在于更新序列边界状态，即当前截尾并确认的最终下标位置 `n`。若提供 `decoding_metrics` 则会自动记录接受数量。后续常配合 `rollback_kvcache()` 一同使用。

---

## 常见使用示例

### 示例 1: 本地标准投机解码 (禁用通信模拟)

当仅在单机或多卡运行基础版 Speculative Decoding 时，不需要涉及通信模拟，此时 `transfer_mode` 设为 `none` 即可。

```python
# 假设 draft_model 已经完成了 gamma 次推理，target_model 做了一次前向校验
accepted_count, final_n = Decoding._verify_draft_sequence(
    draft_model_cache=draft_model_cache,
    target_model_cache=target_model_cache,
    x=input_sequence_tensor,
    prefix_len=current_prefix_len,
    gamma=gamma,
    transfer_mode="none",           # 禁用通信传输出模拟
    send_reject_message=False       # 无需发送云端拒绝切断信号
)

# 验证后，常紧接着调整正确的输入序列长度并回滚多余的 KV Cache
# Decoding.rollback_kvcache(approx_model_cache=.., n=final_n, ...)
```

### 示例 2: 带有网络延迟模拟的端云协作推断 (Edge-Cloud)

在评估端边缘协同框架 (`dist_spec`) 时，往往边缘设备算概率，云端负责检验，遇到被拒绝的 Token，系统应当发送截断信令终止接连不断的串行验证或传输，同时记录通信延迟。

```python
accepted_count, final_n = Decoding._verify_draft_sequence(
    draft_model_cache=edge_draft_model,
    target_model_cache=cloud_target_model,
    x=input_sequence_tensor,
    prefix_len=current_prefix_len,
    gamma=gamma,
    comm_simulator=self.comm_simulator,
    comm_link="edge_cloud",         # 边缘向云端发送
    transfer_mode="serial",         # 序列化通信，逐一检验
    send_reject_message=True,       # 触发中断反馈
    decoding_metrics=self.metrics   # 将 Token 生成指标保存于对象内
)
```

### 示例 3: 优化变体（批量传输草稿再验证）

若某个自适应方法设定打包发送草稿模型输出（由于云端具有高带宽但延迟极大的特性而更划算）则可以使用 `batch_before`：

```python
accepted_count, final_n = Decoding._verify_draft_sequence(
    draft_model_cache=small_model_cache,
    target_model_cache=large_model_cache,
    x=input_sequence_tensor,
    prefix_len=current_prefix_len,
    gamma=gamma,
    comm_simulator=self.comm_simulator,
    transfer_mode="batch_before",   # 在校验第一步之前，合并并模拟全部传输耗时
    send_reject_message=True
)
```

### 示例 4: 使用覆盖概率采样

若是算法内部预先进行了分布校对（例如温度变换退火或使用了特殊权重），则可以将推导过的新分布传给 `draft_probs_override` 而非依赖普通的 Cache 版本。

```python
# customized_probs: 提前运算出来的校正分布 [batch, seq_len, vocab_size]
accepted_count, final_n = Decoding._verify_draft_sequence(
    draft_model_cache=small_model_cache,
    target_model_cache=large_model_cache,
    x=input_sequence_tensor,
    prefix_len=current_prefix_len,
    gamma=gamma,
    transfer_mode="none",
    draft_probs_override=customized_probs  # 自定义采样的底层概率参考
)
```