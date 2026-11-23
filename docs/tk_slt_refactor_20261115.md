# TK-SLT 阶段拆分（2026-11-15）

基点：`850d75e125dadd730a0242d49ef96f665c54ec41`。生产改动限于 [tk_slt.py](<../src/baselines/tk_slt.py>)。

## 结构与接口

公开入口 `tk_slt` 从 **532 → 340 行**（Lead 按 AST `def..end` 复算，含注释、不含装饰器）；文件从 639 → 825 行。签名、注册别名、模块级 patch 目标不变。

入口保留 EOS、AS² 调度、standalone/remaining-budget 分派、上行/下行/flush 顺序、Graph 获取与最终裁剪。五个私有阶段负责 target-only 单 token、wire 提议、strict 验证、commit/rollback 和 metrics。

`_OdldState` 管理 α/b/c 运行估计、策略选择与指标；`_TkSltRunStats` 管理 10 个计数。两者均不持有模型、缓存或通信模拟器。没有通用解码循环，也没有把全部运行环境装进 context。

## 保留的关键行为

- draft 从实际 FP16 wire payload 重建的律采样；上行计费与验证使用同一 payload。
- 整次调用只有一个私有 `DecodingRng`；顺序为逐 token draft → acceptance uniforms → continuation → rollback。
- standalone 和 remaining-budget 都只采一个保留 token，但前者保留早停/下行/continue，后者仍无下行并 break。
- ODLD 的 T_SLM 包含 build+候选校验；T_LLM 只计验证前向，standalone 不喂入。per_transfer 在上行后读 tx，per_round 在 flush 后读。
- Graph cap、缓存身份、eager 懒 key、metrics 公式与注册方法中的配置读取不变。

可接受的小差异：同一 target 前向的两次壁钟读数合并为一个值；T_SLM 多包一层函数调用开销；commit 中被提前返回保证恒真的条件不再重复判断。未改变随机数抽取或通信次数。

## 验证与审查

实现代理（`glm-5.3`）运行：定向 TK 四文件 **356 passed / 15 subtests**，相关守卫十文件 **217 passed / 70 subtests**，当时的全量（不含指纹测试）**2154 passed / 15 skipped / 163 subtests**。既有 [CPU 契约](<../test/test_tk_slt_cpu_contracts.py>)未修改。

`deepseek-v4.1-flash` 对完整 diff 做只读审查，逐项核对上述顺序和 helper 的输入输出，**无行为回归发现**。未为了审查再跑一遍全量。

Lead 用 [冻结测试](<../test/test_logic_fingerprint.py>)相同的完整 include 语料复算：

- `TkSltMixin.tk_slt`：**384 → 280 events**，`b4ca5ec6ccb35c2a → 6c0182af026aeb02`。这是总事件数；子任务初报的 357→96 不适用于完整冻结条目。
- 解析器不展开局部对象 `odld.select()`，也不展开含早退的 commit helper；入口事件减少不代表运行时少采样或少回滚。
- [冻结台账](<../test/golden/logic/baselines.json>)新增 `_OdldState.ready`、`select`、`write_metrics` 与 `TkSltMixin._tk_slt_commit_window` 四项，覆盖已移出入口指纹的关键门控、策略和定稿分支。不改解析器、不刷新无关条目。

子任务命令见 [测试记录](<../.refactor/tk-slt-20261115/test-results.md>)；当次合并结果见[历史汇总](<https://github.com/Nanji-Huaji/DuoDecoding/blob/43026d08210f3456d4e49710e698108388a9c70b/docs/baselines_refactor_20261115.md>)。

**边界：**本轮 TK-SLT 证据为 CPU；未跑真实 Qwen、GPU/NTT 或质量基准。commit 的预算已耗尽防御分支保留，但当前 γ 钳制使其不可达，不宣称现有测试覆盖了该分支。
