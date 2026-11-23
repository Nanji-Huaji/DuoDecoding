# CUHLM 阶段拆分（2026-11-15）

基点：`850d75e125dadd730a0242d49ef96f665c54ec41`。生产改动限于 [cuhlm.py](<../src/baselines/cuhlm.py>)。

## 结构

公开入口 `uncertainty_decoding` 从 **368 → 217 行**（AST `def..end`，含注释、不含装饰器）；文件从 438 → 554 行。入口保留 EOS/预算、skip/trigger 调度、上行计费、target 前向、计时和轮末结算。签名、注册别名和模块级 monkeypatch 目标不变。

| 私有阶段 | 职责 |
|---|---|
| `_cuhlm_build_comm_simulator` | precise/plain 模拟器、往返模式与协议标签 |
| `_cuhlm_prepare_caches` | draft/target 缓存获取、Graph 身份与词表配置 |
| `_cuhlm_draft_round` | 单 token 起草、原始 q 提案即时校验 |
| `_cuhlm_verify_trigger` | qhat 重建、整窗校验、一次 acceptance uniform |
| `_cuhlm_finalize_trigger` | accept/reject 定稿、残差采样、0B 下行 |
| `_cuhlm_metrics` | 计数与通信账汇总 |

`_CuhlmRunStats` 仅含 7 个计数；冻结的 `_CuhlmDraft` 仅含当轮 tokens、logits、proposal 和 token id。不持有模型、缓存或模拟器，没有新增通用运行上下文。Lead 去掉了定稿 helper 中未使用的 `prefix` 参数。

## 保留的合同

- skip 不调用 target、不通信、不排队；`pending_resync` 随下次触发上行捎带。
- proposal 在策略判定之前校验；接受用原 q，拒绝残差用 codec 重建的 qhat。
- RNG/通信顺序仍是 acceptance → reject residual → downlink；每轮只定稿一个 token，定稿 token 尚未 forward。
- `gamma` 与 `comm_round_trip_mode` 仍由公开方法读取，保留 [protocols.py](<../src/protocols.py>) 的运行时内省证据；缓存 cap、身份、eager 懒 key 和重置行为不变。
- 删除未读的 `total_drafted_tokens`；draft 前向计数移入 draft 阶段，每轮仍加一次。

## 验证

实现代理（`deepseek-v4.1-flash`）运行现有相关测试：**419 passed / 70 subtests**；关键文件 importlib 模式：**157 passed / 70 subtests**。未新增行为测试。覆盖包括 [真实 CPU 契约](<../test/test_cuhlm_cpu_contracts.py>)、验证内核、旧合同、图身份与协议消费。

[完整 include 语料的指纹](<../test/golden/logic/baselines.json>)由 Lead 复算：`CuhlmMixin.uncertainty_decoding` **90 → 92 events**，`7aec9e478ae6d0cd → 2ac52ad224bad771`。新增事件为两个值对象构造；旧序列保持原顺序。子任务最初给出的 85→87 只装载本模块，不能代替完整冻结条目。

子任务日志见 [.refactor 记录](<../.refactor/cuhlm-20261115/tests-after.log>)；当次合并结果见[历史汇总](<https://github.com/Nanji-Huaji/DuoDecoding/blob/43026d08210f3456d4e49710e698108388a9c70b/docs/baselines_refactor_20261115.md>)。

**边界：**本方法本轮仅 CPU 验证，不声称独立 CUHLM 已获 CUDA Graph 实机、Qwen 或质量基准验收。
