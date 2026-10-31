# H5 设计：CEE 三层公共核心（cee_dsd / cee_dssd）

> 状态：设计评审稿，未改代码。基线 `main@de91fc9`。
> 关联：`docs/baselines_split_map.md` §5.5。

## 1. 结论

`cee_dsd`（421 行）与 `cee_dssd`（397 行）在**顶层语句**上 93.5% 相同
（72 vs 67 条语句，difflib ratio 0.935），但差异集中在验证路径与 prefix
更新方式，**不能机械合并**。可以按“同一循环 + 策略钩子”重构，预计净减
250–300 行，但需要分阶段、并用同机 A/B 验证。

## 2. 现有差异清单

顶层只有 3 处不同：

| # | 位置 | cee_dsd | cee_dssd |
|---|---|---|---|
| 1 | caches 属性 | `_cee_dsd_caches` | `_cee_dssd_caches` |
| 2 | prefix 存储 | 预分配 `prefix_buffer` + `prefix_len_tracker` | `torch.cat` 追加 |
| 3 | while 循环体 | 见下 | 见下 |

循环体（50 vs 42 条语句）的实质差异：

1. **little 层 proposal**：
   - dsd：`generate_with_topk_metadata_only` 或 `_generate_with_optional_rebuilt_proposal`，拿 `little_topk_history`；
   - dssd：`_generate_with_optional_rebuilt_proposal(..., need_topk_metadata=True)`，拿 `little_rebuilt_probs` + `little_rebuilt_meta`。
2. **验证 payload**：
   - dsd：直接构造 `little_stage_kwargs` / `draft_stage_kwargs`；
   - dssd：`stage_prob_history` + `collect_verification_payload`。
3. **prefix 更新**：
   - dsd：写 `prefix_buffer`，维护 `prefix_len_tracker`；
   - dssd：`prefix = torch.cat((prefix, t), dim=1)`。
4. **reject 重采样**：
   - dsd：在本地 `sample(...)` 后走统一验证；
   - dssd：额外 `comm_simulator.transfer(None, probs, ...)` 下传拒绝位置分布。
5. **计数口径**：
   - dsd 在循环内更新 `n1_plus1`/`n2_plus1`；
   - dssd 使用 `n2 = prefix_len + ...`，部分计数在循环外。

metrics 收尾也不同：dsd 先截断再计 tokens，dssd 先计 tokens；
`_add_per_model_wall_time` 的 `draft_comp_time`/`target_comp_time` 只有 dsd 传。

## 3. 建议结构

```
def cee_dsd(...):   return self._cee_three_layer(..., split=False)
def cee_dssd(...):  return self._cee_three_layer(..., split=True)

def _cee_three_layer(self, prefix, *, split: bool, ...):
    # 公共：设备、caches、comm simulator、计数器、事件、循环骨架
    # 差异点收敛到 4 个钩子：
    #   _cee_make_prefix_buffer(...) / _cee_append_prefix(...)
    #   _cee_little_propose(...)   -> (x, meta)
    #   _cee_build_verify_payload(...)
    #   _cee_reject_resample(...)
```

不建议用大量 `if split:` 内联分支（会把指纹门禁变成不可读的巨函数）；
钩子方法可以是私有的、无副作用的纯数据变换，便于单测。

## 4. 验证计划

因为会改变 `logic_fingerprint` 的事件序列，不能再用当前 fixture 做“零差异”门禁：

1. **重构前基线**：在 enine（同机同 GPU）跑 cee_dsd / cee_dssd 各 20 样本，
   保存 jsonl + metrics；
2. **重构后 A/B**：同命令重跑，要求 **token 序列逐位一致**、metrics 计数一致
   （吞吐允许计时抖动）；
3. 本地 `pytest` 全绿；指纹 fixture 显式重建并在 PR 中说明“本次为结构变更，
   非行为变更，已用同机 A/B 替代”；
4. 合并前跑一次 80 样本 smoke，确认没有 OOM/图回放回归。

## 5. 里程碑

| 阶段 | 内容 | 预计 |
|---|---|---|
| H5a | 抽出 prefix 缓冲/追加（`_cee_append_prefix`） | 0.5 天 |
| H5b | 抽出 little proposal + verify payload 钩子 | 1 天 |
| H5c | 合并循环、两个 wrapper、重建指纹 | 1 天 |
| H5d | 同机 A/B + 80 样本 smoke | 0.5 天 |

## 6. 非目标

- 不在同一 PR 动 `dist_spec` / `dist_split_spec`（另一对 86.6% 相似，但验证路径同样分叉）；
- 不删除 `ceesd_without_arp`（legacy，另一项决策）；
- 不改计费/协议口径。

## 7. 开放问题

1. `ceesd_without_arp` 是否先归档/删除？它 461 行，与 cee_dsd 有 31 个重复窗口；
2. 能否接受“指纹门禁在 H5 合并点重建一次”？否则只能做 H5a/H5b 的局部抽取；
3. 论文 Table VIII/IX 是否还需要 cee_dsd/cee_dssd 的历史逐 token 对照。
