# Baseline 方法拆分汇总（2026-11-15）

> 本文为当日结构重构的历史记录。CEE-CUHLM 后续已获授权修改协议，见 [2026-11-16 对齐说明](<cee_cuhlm_protocol_alignment_20261116.md>)；下列行数与测试数字不描述后续版本。

基点：`850d75e125dadd730a0242d49ef96f665c54ec41`。该轮仅改三个 baseline 实现、相关指纹/源码扫描台账及文档；以下保留当时的验证结果。

## 完成范围

| 方法 | 公开方法行数（前 → 后） | 拆分内容 |
|---|---:|---|
| [CEE-CUHLM](<../src/baselines/cee_cuhlm.py>) | 548 → 285 | 通信、三缓存、Little→Draft、外层不确定度扫描、指标 |
| [CUHLM](<../src/baselines/cuhlm.py>) | 368 → 217 | 通信、缓存、单 token 提议、触发验证、定稿、指标 |
| [TK-SLT](<../src/baselines/tk_slt.py>) | 532 → 340 | wire 提议、验证、commit/rollback、target-only、ODLD 状态、指标 |

行数由 Lead 按 AST `def..end` 统一计算，含注释、不含装饰器。三个文件总行数增加（640→760、438→554、639→825），主要是阶段签名、说明与小型状态类型；目的不是减少总行数，而是缩小公开方法中的职责跨度。

公开签名、注册别名、兼容导出和模块级 monkeypatch 目标保留。未新增公共 API、通用解码循环或承载全部模型/缓存/策略的运行上下文。DSD/DSSD 已有较清楚的轮结构，本次不为进一步去重引入宽接口；adaptive、CEE-SD 主方法及通用 engine/Graph/通信模块均未修改。

## 实现与审查

按用户要求，初步实现、实验和独立审查主要交给 `byr-llm-gw`：

- `glm-5.3`：CEE-CUHLM 拆分及 A6000 实验、TK-SLT 拆分。
- `deepseek-v4.1-flash`：CUHLM 拆分、接缝调查；分别只读审查 CEE-CUHLM 和 TK-SLT 的复杂分支。
- Lead：检查最终源码与原始实验记录，移除 CUHLM 一个未使用形参，处理共享测试、修正报告中的统计口径并运行合并态回归。

两项独立审查未发现运行行为回归。CEE-CUHLM 审查指出 legacy bridge 扫描仍锚定旧方法作用域，已修正。未来风险建议未一律变成新测试或框架。

## 有界测试更新

[legacy bridge 台账](<../test/test_legacy_verification_bridge.py>)仍允许 CEE-CUHLM 的两次调用，只把其中一次归属移到 `_run_little_to_draft_stage`，没有增加允许的算法消费者。

[完整语料指纹](<../test/golden/logic/baselines.json>)仅刷新两个已有条目：

| 条目 | events | sha256_16 |
|---|---:|---|
| `CuhlmMixin.uncertainty_decoding` | 90 → 92 | `7aec9e478ae6d0cd` → `2ac52ad224bad771` |
| `TkSltMixin.tk_slt` | 384 → 280 | `b4ca5ec6ccb35c2a` → `6c0182af026aeb02` |
| `CeeCuhlmMixin.cee_cuhlm` | 139，不变 | `9d843c8a8576d6c1`，不变 |

CUHLM 新增两个值对象构造事件；TK-SLT 的局部 ODLD 对象调用和早退 commit 不被现有解析器展开，因此额外冻结 `_OdldState.ready/select/write_metrics` 与 `TkSltMixin._tk_slt_commit_window` 四个关键叶子。其余已有条目、include 列表、别名映射、解析器均不变。没有为每个简单抽取新增行为测试。

## 最终验证

- Lead 合并态定向守卫：**80 passed / 70 subtests**。
- Lead 完整 CPU 回归：`.venv/bin/python -m pytest -q test`，**2166 passed / 15 skipped / 163 subtests**，214.68 秒，退出码 0。四条 warning 为现有 pynvml/NVML 与单位检查提示；跳过项不计为通过。见 [完整日志](<../.refactor/baselines-20261115/pytest-final.log>)。
- CEE-CUHLM 真实单卡 A6000：**41 个独立配置，72 组前后配对，非计时差异 0**。覆盖 FP32/BF16、eager/Graph、skip/trigger、预算/EOS、NTT 与缓存小模型权重 smoke。Lead 独立复算四份 JSON，结果一致。
- 最终改动文件的 `ruff check`、现有文档不变量检查和 `git diff --check` 均通过。
- 子任务原有 CPU 合同、importlib、lint 与审查依据详见 [CUHLM](<cuhlm_refactor_20261115.md>)、[TK-SLT](<tk_slt_refactor_20261115.md>)、[CEE-CUHLM](<cee_cuhlm_refactor_20261115.md>)。

## 不作的保证

CEE-CUHLM 仍是历史近似协议，不等同于独立 CUHLM 或目标分布无损采样。GPU 记录排除了壁钟相关字段，不据此声称性能不变；只有 25 条记录保存了调用后仍存活的缓存摘要，不能称每个 eager GPU 用例都核验了三缓存端点。

本轮 CUHLM/TK-SLT 没有新增 GPU 验证；CEE-CUHLM 只验单卡，未覆盖多 GPU 拓扑、RL 更新/checkpoint 或大模型质量基准。原始实验目录由 git 忽略，文档可跟踪但原始工件不会自动随提交发布。
