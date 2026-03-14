# `_finalize_verification` 使用示例文档

`_finalize_verification` 是对先前的 `rollback_kvcache` 函数改进后的新静态方法。除了简单执行 KV Cache 的回滚之外，它还完美内聚了“重新采样”及“动态张量重组”功能。使用它，可以将不同子类下的投机解码中大量的样板代码压缩为一个简单的调用。

## 作用原理

投机解码经过多验证步骤后，我们得到了 `gamma` 个草稿词的拒绝位置（或者全部接受），由游标 `n` 表示。对于接下来的验证收尾工作，分为以下关键步骤：
1. **统一裁剪截断**：把由于并行前向多出来的草稿 Token（`x` 的多余部分）剔除。
2. **KV 状态回滚**：通知 `approx_model_cache` 与 `target_model_cache` 将它内部用来加速推理的历史向量回滚切割到长度 `n + 1`，保证与前缀准确对齐。
3. **补充采样新 Token**：
    - **遇到拒绝分支**：即 `n` 还没有走完整个生成的预测（`n < prefix_len + gamma - 1`）。此时根据当前 `n` 位置的模型差异度 `max_fn(target_prob - approx_prob)`，重采样恢复该位置最符合原目标视角的应有 Token。
    - **全部接受分支**：顺利接受一切，将依据最后验证位置上真实的 target 边概率简单抽样得到下一个延伸 Token。
4. **张量组装**：重新把有效部分前缀与补充的新 Token 沿第二个维度拼接 (`torch.cat`)。

## 参数详解

- `approx_model_cache`: `(KVCacheModel)` 小模型/草稿模型的自带缓存接口实例。
- `target_model_cache`: `(KVCacheModel)` 大模型/目标模型的自带缓存接口实例。
- `x`: `(torch.Tensor)` 拒绝验证循环前生成的最初的超量待测组合序列矩阵 `[batch_size, seq_len]`。
- `prefix_len`: `(int)` 进入当前验证回合之前的基础 Token 长度。
- `gamma`: `(int)` 这一循环内使用草稿模型进行连续臆想（draft）的 Token 数量。
- `n`: `(int)` 从验证结果得出、代表最后被接受连续 Token 的截尾游标缩址（若是第1个被抛弃则 `n = prefix_len - 1`）。

**返回**:
`torch.Tensor`: 返回经过缩水截掉拒绝部分、并补充了下一个精确 Token 后的完整合法张量。您可以直接将其赋归到推断主循环的上下文 `prefix`。

---

## 优化示例与对比

### ⚠️ 原逻辑（未封装时的繁琐判断代码）

```python
# 确定了最后一个允许留存的索引 n 之后
this_step_accepted_tokens = n - prefix_len + 1
prefix = x[:, :n + 1]

# 手动回放
approx_model_cache.rollback(n + 1)

if n < prefix_len + current_gamma - 1:
    # reject someone, sample from the pos n
    t = sample(
        max_fn(
            target_model_cache._prob_history[:, n, : self.vocab_size].to(draft_device)
            - approx_model_cache._prob_history[:, n, : self.vocab_size]
        )
    )
    target_model_cache.rollback(n + 1)
else:
    # all approx model decoding accepted
    t = sample(
        target_model_cache._prob_history[:, -1, : self.vocab_size]
    ).to(draft_device)
    target_model_cache.rollback(n + 2)

prefix = torch.cat((prefix, t), dim=1)
```

### ✅ 推荐用法（改用 `_finalize_verification`）

```python
# 当你从验证代码中拿到了有效的游标 n （例如：n=prefix_len+i-1）
this_step_accepted_tokens = n - prefix_len + 1

# 一行直出！所有状态（KV Cache裁切与采样拼接）在这里闭环。
# 注意需要重新赋值更新 prefix 以及判断 max_tokens 阈位边界卡断情况：

if n + 1 >= max_tokens:
    # 临界区安全打断
    prefix = x[:, :max_tokens]
    break

prefix = self._finalize_verification(
    approx_model_cache=approx_model_cache,
    target_model_cache=target_model_cache,
    x=x,
    prefix_len=prefix_len,
    gamma=current_gamma,
    n=n,
)
```

## 注意细节
1. 在向 `_finalize_verification` 提交处理前，请您确认 `n` 的取值始终满足 `n >= prefix_len - 1`，表示哪怕全部拒绝，`n` 最低界也是等于前一轮长度扣减末尾补位。
2. 方法执行结束后会自动合并不同卡、不同 device 的映射，让得到的张量自动适配其原有的输入特征存储设备中。它本身不是就地修改原张量的内存尺寸，由于 PyTorch 设计它本质上通过组装并返回新内存地址 `Tensor` 的方式绕过了就地报错局限。