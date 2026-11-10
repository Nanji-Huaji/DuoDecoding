# CEE-SD 本体重构：阶段接口与逻辑契约

基点：`b72a854e238b9276dbbb5c2ce9ac3bd4e7df6fbf`，分支 `refactor/cee-sd-body-20261107`。本轮对象是三模型 `Baselines.adaptive_tridecoding`（注册别名 `cee_sd`），不是两模型 `adaptive_decoding`。

## 1. 当前授权与历史设计的关系

用户本轮要求实质性拆分 CEE-SD 本体，允许为逻辑正确性改变随机流、调用帧和旧错误行为。[第一层历史设计](<adaptive_first_stage_design.md>)保留其日期、基点坐标和当时的狭窄方案；其中“只抽 V4”“通信必须留在原方法”“第一层 EOS 立即停止”“末轮 flush 前读 wall time”等限制，不是本轮验收规范。

本轮仍拒绝通用解码骨架、god-context 和只搬参数的浅 wrapper。公开入口保留两层拓扑、实际配置读取、策略更新、reward 切点、提交与停止顺序。

## 2. 实现形状

[core.py](<../src/baselines/core.py>) 的公开方法从 **888 行缩为 344 行**（按 `def` 到最后一行、包括正常格式化后的空行/注释/docstring；不含装饰器）。不是按压缩排版后的最少行数计算。

| 私有方法 | 实际职责 |
|---|---|
| `_cee_propose` | 生成当前候选，捕获对应概率行、实际草稿长度和计算时间；零扩展不前向、不采样；保留 entropy 张量来源而非提前转成 CPU 浮点 |
| `_cee_first_stage` | little→draft 的 opportunistic 决策或真实验证、上行载荷及可选残差补计费；返回校正/转发结果 |
| `_cee_target_proposal` | 拼接第一层携带行与新 draft 行，检查它们覆盖的 token 窗口，不用新 top-k 重新解释已产生的前缀 |
| `_cee_verify_proposal` | 校验完整候选/bonus 上下文，以同一 wire 分布做接受与残差采样，返回携带行并结算两个 cache 的实际上下文端点 |
| `_cee_finalize` | 在最后通信结算后统一长度、数值指标、分模型时间、checkpoint 守卫及通信账 |

[私有值类型](<../src/baselines/_cee_state.py>)只有 `CeeProposal`、`CeeVerification` 和 `CeeDecodeStats`。前两个含 Tensor 的结果值冻结且禁用自动逐元素相等比较；统计值只存 **12 个数值字段**，不拥有 cache、通信模拟器、adapter、策略或停止决策。没有把整个运行环境装入可变 context。既有 `arp_overhead_time` 指标仍保留常数 0，不为它增加无用途的统计状态。

继续复用已有生成/重建、采样、残差、rollback、缓存和最终通信记账原语；共享采样家族本身不改。两个明确获准的共享接口扩展为：

- `_acquire_three_layer_caches(..., cache_kwargs=...)` 增加可选 keyword-only 构造参数；容量参数与 graph 开关/跨样本复用分开，旧调用默认行为不变。
- [通信工厂](<../src/baselines/simulators.py>)接受可选构造期 `mode`，保证 CEE 的 trace 来源在构造时确定；未提供时不改旧构造默认值。

基点的 43 个 `Baselines` 方法中，仅本体与上述缓存 helper 的 AST 改变，其余 **41 个旧方法 AST 不变**；wrapper 的展开后指纹变化与它自身源码变化是两回事。

原来只在方法内累积、从未输出或消费的 entropy/rank/accept-rate history 工作被删除。真正供策略使用的 entropy 保留；第二层仍在 WAN transfer 和 target 前向之后才求 CPU 数值，不把同步提前到 proposal helper。

## 3. 六组有意修正

| 原问题 | 本轮契约 |
|---|---|
| 第一层 EOS 还未经过 target 就提交；终止分支跳过 target 回复/下行计费 | 第一层结果只是候选，EOS 必须由 target 确认；target 响应与下行先计费，再按 EOS/early stop 停止 |
| `actual_gamma1=0` 时，即使已有第一层候选也跳过外层验证 | 验证整个“携带前缀 + 新 draft”窗口，不能以新 draft 长度代替总候选长度 |
| skip 前缀按 raw draft 分布验证，或把已产生前缀按 live top-k 再重建 | skip 携带实际 little proposal 行；正常第一层校正携带 raw draft-verifier 行；拼接时保留来源 |
| `r<=p/q` 在 `p=0,r=0` 时错接受；bonus/rollback 端点隐含 | CEE 使用严格 `r<p/q`；残差为 `norm((p-q)+)`；bonus 显式按上下文行取，两个 cache 仅保留已 forward 的前缀，新采样 token 不冒充已缓存 |
| residual 计费开关触发另一套稀疏概率表示，改变接受/残差 law | 接受与残差使用同一 wire-q；`charge_residual_payload` 只决定补计费，metadata 不得切换采样分布 |
| wall/throughput/分模型时间在末轮 flush 前读取 | 最后一轮先结算，再读最终通信时间并计算派生指标 |

这是逻辑修正，不声称与冻结旧方法的副作用轨迹、RNG 状态或结果在所有路径上等价。

## 4. 精确提案与有损 wire 协议

`CeeProposal.probabilities` 在 wire 量化前捕获；wire-q 是实际送入本轮验证和残差计算的分布，两者不可混淆。

- **`prob_payload_bits>=16`：** 保持既有 pass-through 语义，按 Tensor 的实际元素宽度计费，并非强制转 FP16。验证使用真实采样/参考分布；第一层正常校正与 skip 的不同来源必须保留。
- **`prob_payload_bits<16`：** 延续既有 `_uplink_prob_payload` 的显式近似协议。解量化并归一化后的 qhat 同时用于接受比与残差；不是“计低位宽字节，却偷偷用未传输的精确 q 验证”。
- 有损模式中，生成 token 时的 q 与 wire-qhat 一般不同；第一层输出也只能近似 draft law。**本轮不宣称该模式是无偏 target 采样，也不在本体拆分中另行实现量化前生成的新传输算法。**
- 两种 residual 计费配置都应走同一个 wire law；不能因为开启补计费，就从 sparse metadata 换回另一套分布。

## 5. 可见编排与通信位置

公开入口仍直接读取 gamma、计费开关及实际深度配置，保留 protocol introspection 的消费证据。static 关闭 RL/ARP；no_arp 保留 RL、关闭 ARP；opportunistic 可以跳过 little→draft 验证，但不能跳过外层 target 验证。

第一层 reward 仍在第一层验证/残差补计费后、校正回复前观察；主 reward 仍在 target 验证/残差补计费后、最终回复/下行前观察。team reward、禁止 RL 更新和 checkpoint 保存守卫保留。预算对最终生成段截断；缓存端点不包含尚未 forward 的 bonus/correction token。

通信工厂内部依赖应 patch `src.baselines.simulators.X`；调用方的其他 core 依赖继续 patch `src.baselines.core.X`。第一层上行、skip 与残差操作现在可由 `_cee_first_stage` 发起；外层上行、最终回复/下行、round/flush 仍由公开入口编排。

`COMM_TRACE.caller` 记录真实 helper/调用点，不伪造旧帧；跨版本不能要求旧行号或逐字节 trace 文本一致。报文种类、账本与因果顺序由行为测试核对，终止计费及旧逻辑错误的修正除外。

## 6. 验收与指纹边界

- [独立算法契约](<../test/test_cee_sd_contracts.py>)以 toy 分布和字面期望覆盖零长度、长前缀、两层接受/拒绝、EOS、skip、wire-q、缓存、预算和最终结算，不从旧输出推导正确答案。
- [编排与辅助 oracle](<../test/test_cee_sd_orchestration.py>)覆盖常规/static/no_arp/opportunistic、策略更新、缓存 eager/graph 接线、entropy/reward/通信顺序；正常路径差分是辅助证据，不能要求保留已确认旧错误。
- [冻结旧方法](<../test/golden/cee_sd/baseline_b72a854.py.txt>)只包含 `b72a854` 的 decorated 方法，SHA-256 为 `4dad0c0dc8583cb58ce19bb3096b5120cb143746c165e55fa0fd82a2122c62fe`；只编译该方法，不执行旧模块，运行时不依赖 Git 历史。helper-poison 测试确保旧 oracle 不调用新增阶段方法。
- [指纹快照](<../test/golden/logic/baselines.json>)仅可更新 CEE 本体及实际依赖它或获准共享接口的入口；其他条目和其他快照保持，不能盲目整体重冻。
- 指纹只反映工具可见的源码足迹，不证明数学、参数或数据流等价。指标通过字面字段循环和 `getattr(stats,key)` 赋值，不能拿减少的重复源码事件数当作指标不变的证明。

### 本轮快照变化

以下六项经逐项批准更新；其余 **9 个冻结条目 deep-equal 不变**，[engine 快照](<../test/golden/logic/engine.json>)与[前缀快照](<../test/golden/logic/cee_prefix.json>)保持字节一致，include 集合不变。

| 入口 | 事件数（旧 → 新） | sha256_16（旧 → 新） | 原因 |
|---|---:|---|---|
| `Baselines.adaptive_tridecoding` | 315 → 287 | `3f1133ec9c65796f` → `760d5b7192f24e4b` | 阶段接口、输入校验、严格接受/残差、缓存端点、死 history 删除及终止/flush 修正 |
| `Baselines.tridecoding` | 2 → 287 | `85cd22843c836269` → `c0405721bac6c3dd` | 原 wrapper 未改；新主方法可被展开 |
| `Baselines.cee_sd_no_arp` | 2 → 287 | `85cd22843c836269` → `c0405721bac6c3dd` | 同上 |
| `Baselines.cee_sd_opportunistic` | 2 → 287 | `85cd22843c836269` → `c0405721bac6c3dd` | 同上 |
| `Baselines.cee_dssd` | 121 → 122 | `fe959e6a96091386` → `f5216fd3d1b36173` | 仅展开获准缓存 helper 后，多一个 `cache_kwargs or {}` 的 `Or` 事件；默认构造行为不变 |
| `Baselines.cee_dsd` | 125 → 126 | `ded8e207f38c6a81` → `58e38295068be9ad` | 同上 |

快照生成前核对的 core SHA-256 为 `162a409e50da76a01a08aae1e4b86e67acab1c1cf955bd5674779b02d20fdc98`。没有为规避指纹差异而改写实现、变更工具规则，或整体重新冻结其他算法。

### 指纹的可见范围

沿用[既有指纹工具](<../tools/logic_fingerprint.py>)，不为新实现另造“等价证明”：

- `_cee_propose` 与 `_cee_first_stage` 有实际早返回，工具不展开它们，只记录调用；其内部生成、uncertainty、通信和 RNG 要靠阶段/公开入口测试检查。快照中的一次 RNG 源码事件不能解释为每轮只消费一次 RNG。
- 工具只把 receiver 名为 `comm_simulator` 的相应调用归为 `comm`。新编排使用 `comm`，不少操作记为普通 `call transfer/simulate_transfer/flush_round`，不是通信消失或免计费。
- 新主方法只含尾返回，三个既有 wrapper 因而能展开本体；旧主方法含内部返回，其 wrapper 只记录一次 callee 调用。这是 wrapper 事件数大增的原因，不是三个 wrapper 各自新增了一套算法。
- 新值模块无需新增 include：Tensor 字段/私有计数 property 本身没有可展开的调用副作用；构造器保留为值构造调用。维持原有 include 集合，避免引入无关的名称解析变化。

### 验证记录

初始既有定向集 **96 passed、75 subtests**；新增独立契约 **31 passed**、新增编排/oracle **39 passed**，各自默认/importlib 两模式通过。更新快照后的定向指纹门禁，默认 **12 passed（1.02 秒）**、importlib **12 passed（1.03 秒）**。旧三文件迁移后与新增用例合跑，两模式各 **108 passed、5 subtests**；另一个 opportunistic 旧夹具迁移后与新增用例合跑，两模式各 **82 passed**。

2026-11-09 最终全量验证由 Lead 在冻结源码上执行：

| 收集模式 | 结果 | 耗时 |
|---|---|---:|
| 默认 | **852 passed、7 skipped、160 subtests passed** | 73.91 秒 |
| importlib | **852 passed、7 skipped、160 subtests passed** | 48.89 秒 |

每种模式均只有两个既有警告：pynvml 包弃用提示与 RL 单位误用测试的预期警告。最终 scoped Ruff 与 `git diff --check` 通过；整个 core 与旧 CEE 测试文件仍有未更改的历史 lint 项，不声称仓库全量 lint 清零。

Standards 与 Spec 独立审阅均无 finding，旧夹具迁移后也分别完成增量复核。另用 `byr-llm-gw/deepseek-v4.1-flash` 做只读封装检查：11 个本地文档链接、344 行主方法、41 个未改方法、冻结 oracle SHA 与 15 文件变更范围均吻合；该检查不冒充算法或 GPU 实验证明。

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider test/test_logic_fingerprint.py
.venv/bin/python -m pytest -q -p no:cacheprovider --import-mode=importlib test/test_logic_fingerprint.py
```

本轮不运行真实 GPU/SSH，不提供真实模型、ARP/RL 权重、CUDA graph 执行或跨设备数值一致性的新增实验证明。上述 graph 证据是 CPU 接线/复用契约，不等同真实 CUDA 执行验证。
