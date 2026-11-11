# CUHLM 压缩契约（2026-11-11）

分支 `refactor/cuhlm-compression-contract-20261111`，基点 `412e3da`（PR #35）。
本批修正压缩大小选择，提取共享 codec 和稳定 L1 运算，补齐 CEE 的词表大小与
实际 token 参数。主 codec 算术保持不变；旧 `terminal_prob` 入口改用该 codec。
未改采样/验证循环或 CUDA Graph 实现。

## 已证实的问题

旧求解器假设单位质量的理想尾部，不核对实际低精度重建。无解时返回
`min(300,V//100)`，分母异常返回 `30`，长度不符也用固定值，均不是保守保证。

以下来自真实 codec 与实际 FP32 sampler weights 的独立 Fraction oracle，
`theta=0`、`u=.9`、`d=min(1,V-1)`：

| 输入 | 旧 k | 实际 codec 首次零失真 k |
|---|---|---|
| BF16 `[.1,.2,.7]` | 0 | 3 |
| BF16 均匀 V3 | 0 | 3 |
| FP32 均匀 V3 | 1 | 3 |
| FP32 `[.95,.04,.01]` | 2 | 3 |
| FP32 `[1,1e-30,5e-31]` | 1 | 3 |
| FP32 `[1]` | 0 | 1 |

FP32 均匀 V3 的 k=1/2 实际 L1 约 `3.97e-8`；BF16 均匀 V3 约 `2.61e-3`。
均匀行、k=V-1 都不一定无损。小尾例 k=1 的真实 L1 为
`3.0000000095132306e-30`，直接 double 归一化后相减只能得到一半，漏掉了
主坐标的补偿变化。

## 模块与选择规则

纯模块 [cuhlm_compression.py](<../src/cuhlm_compression.py>) 提供：

```python
compress_rebuild_probs(probs, k) -> Tensor
select_compressed_vocab_size(
    probabilities, *, uncertainty, theta=0.1,
    draft_token=None, vocab_size=None,
) -> int
```

主 codec 的 raw dtype、显式 top-k mask、残差 clamp、均匀尾部与操作顺序均保留；
`k>=V` 返回原对象。[communication.py](<../src/communication.py>) 保留兼容入口。

求解器只接受一整行完整概率，允许 singleton leading dims。错误 shape/质量、
空/None、非法参数明确报错；`theta` 有限非负、`uncertainty` 在 `[0,1]`。
选中概率为零对该 proxy 合法。未给 token 时仍用 top-1 兼容，不代表真实候选。
`determine_transfer_strategy` 对 None/空的原有 skip 分支保留。

1. 按 codec 看到的 raw 行建立一次 double 理想模型，先筛理想预算 theta 的候选。
2. 若候选 k<V，用同一实际 codec 重建；输入与每个重建行各预处理一次。
3. 用实际 sampler law 的稳定 L1，以及实际选中 `x_d` 的式 (26) 分母检查。
4. 首次不能认证时，复用筛选结果，再尝试理想预算 theta/2 的候选；不重复同一个 k。
5. 两次实际检查都直接比较 float64 `L1/denominator <= theta`，不放宽原预算、
   不加 epsilon。仍不能认证则返回 V。

输出始终为 1..V，V=1 返回 1。只排序一次，O(V log V)；最多重建两个不同候选，
额外 O(V)，不逐 k 枚举。允许保守地返回 V，不保证严格最小。

[probability_laws.py](<../src/probability_laws.py>) 共享 centered cross-difference：
`q[pivot]*sum(p)*(p/sum(p)-q/sum(q))`。残差仍按 clamp→mass→除计算；L1 取
该差值绝对值之和，再除同一尺度。这里 L1 是 TVD 的两倍，不可混称。

## 调用者

- `terminal_prob` 仍保留缺 token 时的 top-1 兼容，但改走主 codec；旧两步重建
  用 `>0` 推断 mask，会把 top-k 内的零值当尾部。
- [core.py](<../src/baselines/core.py>) 的 CEE-CUHLM 普通/Precise 构造传入真实
  `vocab_size`，策略调用传已有的 `current_token_id`。
- [cee.py](<../src/baselines/cee.py>) 的机会门控给不确定度与策略传同一候选。
  该调用只使用返回的 bool、丢弃 k；正常输入的阈值判定与计费顺序未改。
- plain CUHLM 本来已传实际词表与 token，循环未动。

## 验证证据

实现前：旧求解器复现上表；新模块尚不存在；两个 CEE 参数透传测试失败。
实现者版本的 17 文件定向测试共 **483 passed**，覆盖独立 Fraction oracle、
四种 dtype、非 top-1/零概率候选、非法输入、无 RNG/无修改、复杂度、兼容入口
及 CEE 接线。

Lead 另做独立检查，未复用生产 L1 算子：

- **122** 个严格验证数值例逐位一致；旧共享模块从不可变 PR #35 单独加载，
  避免旧 verifier 误导入新 helper 造成伪差分。质量校验函数 AST 未变。
- **186** 个主 codec 例逐位一致，含四种 dtype、非连续 view、空输入、k=0/V/V+3；
  输出、错误类型、fullV 对象身份、输入和 RNG 均符合旧实现。
- **103 行、3592 例**实际 codec 检查：Fraction 精确 law、90 位 Decimal
  softplus，含真实边界的 ±1e-10 相对扰动；**0 个越界，2322 例实际压缩**。
  theta=0 要求精确零失真，极小尾部没有用绝对容差掩盖。

原始工件只在本地忽略目录 `.refactor/cuhlm-compression-20261111/` 保存。
[逻辑指纹](<../test/golden/logic/baselines.json>) 未改，15 项通过。新增关键字实参
不改变现有事件提取器的足迹；接线变化由专门测试检查，而非靠指纹证明。

唯一一次 GLM-5.3 批次终审发现一项 P2：首个理想候选贴近预算，codec 舍入使
低精度大词表输入系统性退回 fullV。审查中的 FP16/BF16 各 24 个合成配置全部
退回 V；这不代表真实模型的退化率。Lead 在现有公开接口加入可压缩的实际 codec
见证，**6 failed → 6 passed**，然后改为上述最多两次认证，不放宽实际预算。
复验 8 seeds × 3 budgets × 3 dtypes 的 72 个合成配置，均有 k<V 且精确 L1
proxy 满足原预算；FP16/BF16 各 24 项的 fullV 数均为零。没有启动另一轮外部审查。
另加强 codec 调用前后输入快照断言，并用极小尾部锁住候选认证，防止朴素减法回归。

修复后重新执行严格 CPU 门禁：默认/importlib 各 **1565 passed、7 skipped、
163 subtests**，分别用时 101.24/103.72 秒。14 项历史权重指纹实际通过；
7 项 skip 是缺失原始 run，不计为验证通过。最终工件在 `final-cpu-strict-v2`，
之前的 `final-cpu-strict` 是修复前结果，不作最终源码的验收凭据。

## 限制

认证的是所选行在 beta 拟合/softplus 代理下的 L1 约束，不是协议的全局 TVD。
CUHLM 仍是有意有损算法，不能称 target-exact。候选失败后直接返回 V，可能
牺牲压缩率；主 codec 的 tie 次序与传输字节口径未改。

本批未运行 GPU/CUDA Graph，CPU 结果不证明图路径和真实模型性能。
