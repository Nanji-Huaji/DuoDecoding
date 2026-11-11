# TK-SLT 模块与 target 前向契约（2026-11-11）

基点：`567cd26cd2b501f533bc2256aa1520f4c1eeeb0f`。本批完成结构迁移和丢弃采样清理，**不代表 TK-SLT 的数值协议已经全部验收**。

## 改动与保留边界

- [tk_slt.py](<../src/baselines/tk_slt.py>) 的 `TkSltMixin` 接管完整算法循环及 `tk_slt` / `tkslt` 注册名；`tkslt` 是注册别名，不是另一个方法。[core.py](<../src/baselines/core.py>) 仅保留该算法的装配与兼容导出。
- ODLD/AS² 的四个数学函数移入纯 [tk_slt_policy.py](<../src/tk_slt_policy.py>)，不依赖 torch、模型、缓存、通信或 RNG。[baselines_helpers.py](<../src/baselines_helpers.py>) 保留旧导出。算术体不变；修正了 Lambert 精度和 FP16 无损性的不实文档表述。
- 保留图缓存身份、reset/rollback、参数默认值、FP16 概率载荷与论文字节口径、预算预留、通信顺序和 metrics 字段。没有修改模型/图实现或历史权重。
- 新模块加入 [通信配置审计清单](<../test/test_comm_config_reads.py>)，没有因搬家而缩小守卫范围。
- facade 仍只兼容 import/getattr/dir，不支持赋值写透；测试改为 patch 新的定义模块。

## 有意改变：不再采样随后丢弃的 target token

三处 `target.generate(..., 1)` 改为公开 `forward_context(...)`：

| 路径 | 新行为 |
| --- | --- |
| AS² standalone | 前向当前前缀，再采一次保留的 token |
| 剩余预算 fallback | 前向当前前缀，再采最后一个 token |
| 正常验证 | 前向完整候选窗口，再走既有 accept/reject 与 continuation |

合法缓存状态下，前向端点和概率历史不变；每次 target 前向减少一个原本丢弃的 multinomial draw。其余采样操作顺序不变。**不承诺旧新随机流、输出序列或计时数值逐位相同**；ODLD 的实测成本也可能随计算量变化。最后采到的 token 不在同轮提前 forward。

完整方法 AST 与不可变基点比较：仅放行这三次调用替换，以及删除三个未读取的 `_` 结果绑定，其他语句、注册装饰器和签名全部一致。

## CPU 契约与差分证据

[CPU 契约测试](<../test/test_tk_slt_cpu_contracts.py>) 使用真实 tiny Llama 与生产 `KVCacheModel`。固定模型参数形成 token-keyed one-hot law；不是脚本化概率历史。只替换 CUDA 计时及通信模拟器，观察真实缓存端点和 RNG。

覆盖预算 0/1、全接受 bonus、首次/中间拒绝、EOS 截断、ODLD standalone、注册别名、target 不调用 `generate`、每次 `forward_context` 不改变 RNG。生产仍用 CUDA Event；这些测试不等于 CPU 原生计时或真实 CUDA Graph 验收。

- 将三处调用恢复为旧 `generate(...,1)`：CPU 契约 **6 failed、2 passed**；恢复新实现后通过。
- TK 与 CPU 契约：**31 passed、15 subtests**。
- 定向八文件回归：**119 passed、15 subtests**；相邻协议七文件：**115 passed、70 subtests**。
- 新审计范围：**5 passed**。
- 不可变旧 helpers 独立装载：四个数学函数及量化器算术 AST 一致，**1210** 个调用结果/异常逐位一致，旧导出指向新函数。包含 **50 次既存异常**，这是迁移等价证据，不是数值正确性背书。
- 唯一整批只读终审（GLM-5.3）：**No findings**，独立 1326 例数学差分零不一致（含 115 个相同旧异常），定向测试 114 passed、15 subtests。这不认证已知的旧数值缺陷。
- 终审后 Lead 仅收紧两处 docstring（量化质量保证及 RMSNorm 的共同尺度），没有改变算术或测试断言；最终冻结严格 CPU 门禁默认/importlib 各 **1575 passed、7 skipped、163 subtests**（125.14 / 119.20 秒）。14 项历史权重指纹均实际通过；7 个 skip 是缺失原始运行工件。源码/子模块快照一致。
- `forward_context` 不执行旧 `generate` 中额外的 debug-only 概率检查；这里验收有效概率行和合法缓存状态下的前向契约，后续严格 wire 接口仍需完整输入校验。

[逻辑指纹](<../test/golden/logic/baselines.json>) 只改变 TK 条目：171 events 不变，`614586e568ebf2eb` → `49ddf61475a7572d`；调用表面仅 `generate` 4→1、`forward_context` 0→3。其他 **14** 条目与基点一致，没有重生成历史 baseline 固件。

本地私有取证目录（不随提交发布）：

```text
.refactor/tk-slt-20261111/
  migration-diff-final.json
  loop-migration-diff.json
  odld-known-issues.json
  final-review.md
  final-cpu-strict-v2/
```

## 尚未修正的协议与数值问题

### FP16 提案/接收分布不一致

提案仍采自 raw top-K Q，接收端使用 FP16 roundtrip 后的 Qhat；专用近似验证桥仍保留。`Q=[.6,.4]`、`P=wire(Q)` 时，wire 接受比均为 1，但首 token 输出仍为原 Q，与 P 的 TVD 约 `9.76502895e-5`；正尾 `1e-8` 也可在 FP16 下溢为 0。不能据此声称 target-exact，也不能用未计费 raw Q 偷换服务器验证。后续将单独对齐采样前的 wire 表示与验证输入。

### ODLD 合法域中的既存数值失败

Lead 对不可变基点用 Decimal120 穷举 `gamma=1..64`，确认：

| alpha | L=b+c | 正确的有界 gamma | 旧实现（本批保留） |
| --- | --- | ---: | --- |
| 0.5 | 0.001 | 8 | 1；固定 Lambert 区间不足 |
| 0.5 | 0.0005 | 9 | z 下溢为 -0，抛 ValueError |
| nextafter(1,0) | 0.999999 | 64 | z 舍入至 -1/e，抛 ValueError |

这些不是坏输入，已列入后续数值修复；不能把“搬家逐位一致”当作它们已正确。当前固定区间/FP64 二分也不具备原文档宣称的 `1e-28` 精度。

[参考论文](<https://arxiv.org/html/2509.04576v1>) 的摘要与正文对 logits/probabilities 术语并不完全一致；VI-B 的 draft 温度固定为 1，而当前实现同用全局温度。后续必须明确所选概率-wire 变体与实验配置差异，不静默宣称完全复现。

真实 GPU/图路径仍待空闲设备验收；阶段 A 使用独立冻结 checkout，见 [远程记录](<stage_a_remote_20261111.md>)。
