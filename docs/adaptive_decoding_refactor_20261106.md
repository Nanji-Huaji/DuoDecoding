# 两模型 adaptive_decoding：本体拆分与逻辑修正

基点：`a9a61cc`。本轮处理两模型 `adaptive_decoding`；三模型 `adaptive_tridecoding` 保持原实现。

## 结构变化

[core.py](<../src/baselines/core.py>) 的公开方法从 **387 行缩为 179 行**。它保留初始化、每轮调度、reward、草稿回滚、EOS、早停和下行的可见顺序，五个私有方法分别负责：

| 方法 | 职责 |
|---|---|
| `_adaptive_decoding_propose` | 用当前策略生成草稿，随后选择下一配置；保留生成时的实际 q |
| `_adaptive_decoding_verify` | 检查验证上下文，逐 token 接受或拒绝，记录对应上行 |
| `_adaptive_decoding_next_token` | 拒绝载荷计费、残差或 bonus 采样、target 回滚 |
| `_trace_adaptive_decoding_append` | 追加现有逐轮诊断记录 |
| `_adaptive_decoding_finalize` | 最终长度处理、metrics、checkpoint 守卫和通信账 |

[内部状态类型](<../src/baselines/_adaptive_state.py>)只有两种：四字段的 `AdaptiveProposal` 保存 tokens、实际概率、前缀长度和草稿长度；`AdaptiveDecodeStats` 保存八个数值统计项。缓存、通信和策略对象不放入状态包。

现有公共 helper 继续承担通信工厂、缓存获取、top-k 重建、概率载荷收集、残差分布、下行与最终记账。其余原有 **37 个 Baselines 方法的 AST 未变**，兼容导入、79 个 star exports 和 24 个注册项保留。

## 逻辑修正

用户在本轮明确：以逻辑正确性验收，不把旧轨迹当作不可修改的规范。

| 问题 | 本轮处理 |
|---|---|
| 原代码历史越界后发拒绝消息，却仍把 n 留在整段末尾 | 缺失的验证上下文明报 `ValueError`，不冒充全接受 |
| raw draft history 与生成时重建 q 混用 | 验证、拒绝载荷和残差均以 `proposal.probabilities` 为准 |
| target 缺少 bonus 行或带多余 stale 行 | target history 必须恰好覆盖输入草稿的完整上下文 |
| 采样 token 的 q 为零、负值或非有限值 | 收集当前草稿的已选概率后，在验证 RNG/通信前报错 |
| `p=0`、随机数恰为 0 时旧比较会接受 | 使用 `r < p/q` 接受；实际 RNG 范围为 `[0,1)` |
| 全接受 target 回滚请求包含尚未 forward 的 bonus | 两个分支都回滚到 `n+1`，只保留已 forward 的前缀 |
| 缓存领先当前 prefix | 有长度元数据时明确报错；完整 prefix 已预填充的相等长度合法 |

bonus 显式读取 `L+gamma-1` 行。拒绝采样使用现有 `compute_residual_distribution(p,q)`，即 `norm(max(p-q,0))`。

验证只把**本轮 gamma 行**的 target 概率搬到 draft 设备一次，替代每个 token 都搬整段历史。重复计时、两个无用通信时间读取、重复长度分支和仅为计算 prompt 长度保留的 clone 已移除。

## 验收方式

- [独立正确性契约](<../test/test_adaptive_decoding_contracts.py>)直接断言接受边界、真实 q、残差权重、bonus 行、缓存位置和错误输入；不用历史输出推导期望。
- [公开入口回归](<../test/test_adaptive_decoding_rounds.py>)对正常输入保留旧版辅助差分：真实 torch RNG、真实模拟通信账、输出、全部 metrics、RL 和停止动作。只归一时钟观察与等效回滚请求，不丢弃实际缓存 before/after、通信或奖励差异。
- [冻结旧方法](<../test/golden/adaptive_decoding/baseline_a9a61cc.py.txt>)带 SHA-256 校验；只编译该方法，不执行旧模块，也不依赖完整 Git 历史。
- [值对象测试](<../test/test_adaptive_decoding_state.py>)覆盖字段冻结、数值统计隔离和构造不消费 RNG。
- [平均 top-k 测试](<../test/test_decoding_avg_topk_metrics.py>)的旧 fake cache 原来把刚采样但未 forward 的 token 也写入历史；本轮将 fixture 对齐真实缓存协议，原指标断言保留。
- [指纹门禁](<../test/test_logic_fingerprint.py>)与[快照](<../test/golden/logic/baselines.json>)仅更新 `adaptive_decoding` 条目，逐项记录新增验证和有意修正；其余 14 个冻结条目保持。

新增两批协作共五个子代理任务，均使用 `byr-llm-gw / deepseek-v4.1-flash`，分别核对算法、实现契约测试、检查简化机会和进行对抗审阅。审阅覆盖 eager/graph 缓存语义及跨设备静态路径。

`COMM_TRACE.caller` 使用真实调用位置：仍是 `core.py:<line>`，验证和拒绝载荷的行现在分别归属上述 verify 与 next-token 方法。正常输入的报文种类、因果顺序和计费继续核对；跨版本分析不再要求 caller 文本一致。

## 最终验证记录

- 默认收集：**706 passed、7 skipped、160 subtests**，58.15 秒。
- importlib 收集：**706 passed、7 skipped、160 subtests**，50.61 秒。
- 新增算法/入口/值对象及平均 top-k 定向集：36 passed、70 subtests。
- 新增文件与修改的平均 top-k 测试通过 Ruff；core 新方法按范围检查，既有文件的其他 lint 项不在本轮改写。
- 两种收集模式均只有既有的 pynvml 弃用警告与单位误用测试警告。

本轮验证使用 CPU fixture 与真实 torch/通信计算；三模型的既有 GPU 实验记录单独保留。

运行命令：

```bash
.venv/bin/python -m pytest -q -p no:cacheprovider
.venv/bin/python -m pytest -q -p no:cacheprovider --import-mode=importlib
```
