# TK-SLT 概率-wire 契约（2026-11-11）

基点 `b02666b115d261661a49f6aff81fe83aa2bf2be4`。本文记录提案/wire 修复；
同批的纯策略数学另见 [ODLD 报告](<tk_slt_odld_numerics_20261111.md>)，
最终验收见[联合报告](<tk_slt_wire_odld_20261111.md>)。

## 旧实现的三个反例

旧提案采自 raw Q，却用 FP16 roundtrip 后的 Q̂ 验证。

| 输入 | 旧行为 |
| --- | --- |
| `Q=[.6,.4], P=wire(Q)` | 接受比全为 1，但首 token 仍来自 raw Q；与 P 的 TVD 为 `9.765028953570009e-05` |
| 存在正尾 `1e-8` | raw 律可能采到它，wire 却已把它舍入为 0 |
| 均匀 logits，V=4、K=2 | [top-K 过滤](<../src/sampling.py>)保留阈值并列项，留下 4 个正项，却只计 2 个值 |

因此不能只修接收比，也不能把未计费的 dense raw Q 偷传到接收端。

## 明确的概率载荷

[tk_slt_proposal.py](<../src/tk_slt_proposal.py>)拥有编码、重建与草稿提案循环，
公开接口为 `TkSltWireRow`、`TkSltProposal`、`encode_proposal_row`、`build_draft_proposal`。

1. 先校验模型行的完整宽度、浮点类型、有限/非负和近似单位质量。
2. 如声明较窄共享词表，先在前 V 列条件化。`K=None/<=0` 使用全 V；否则
   显式选择 `min(K,V)` 个坐标，精确平局按最小词表 id，**先选支撑再采样**。
3. 保留质量按 FP64 重归一化，再编码为真实 FP16 值；不转回模型原 dtype。
   这是 codec 变换，最终 FP16 行才是实际采样输入。
4. 稀疏 payload 只含 FP16 值和严格递增索引；全 V 使用隐式索引。
   接收端不重新 top-K，也不读取原模型 Q。
5. 解码器校验形状、类型、索引范围/唯一性以及**近似单位质量**；不把全零、
   明显截断或非法质量修复成 one-hot。浮点容差仅用于输入可接受性，不进入接受比。
6. `dense()` 返回新建的 dense-V FP16 行，不二次归一化。两侧向同一个
   `sampling_weights` 提供同一种 dense-V 表示，而不是一侧 K 维、一侧 V 维。

容器冻结不等于 Tensor 内容不可写：构造参数是借用张量，调用者不得再修改；
编码器本身生成独立 payload，不与模型缓存别名。后续缓存写入不改变本轮载荷。

## 草稿循环与严格验证

- 每步用现有 `KVCacheModel.forward_context`，编码后从重建 wire 行采样。
  不新增模型回调，不使用 `generate`，token 写入预分配缓冲。
- 非正 gamma、无效 prefix（含 token 范围）或 top-K 类型，在前向前拒绝；
  模型行在抽样前校验。末 token 尚未前向，草稿缓存端点为 `prefix_len+gamma-1`。
- [TK-SLT 循环](<../src/baselines/tk_slt.py>)在 target 设备上仅由 payload 重建 Q，
  经 `ProposalWindow → prepare_verification_window → decide_verification` 验证。
  先校验完整窗口，再抽一次形状 `(1,gamma)` 的接受随机数。
- 不提供 selected-Q override。接受与残差都来自 wire 行的实际 sampler 权重；
  拒绝续采使用共享稳定残差，不再走 legacy `max_fn` 兜底。
- standalone/fallback 的目标行也先校验再抽样。保留预算预留、EOS、通信顺序、
  ODLD 统计字段、rollback 计划、图缓存 key 与身份/复用契约。
- [DecodingRng](<../src/decoding_rng.py>)由一次 TK 调用拥有，贯穿全部轮次及
  提案、接受、standalone/fallback、残差和 bonus。按实际采样张量设备分配不同 key 的
  私有生成器；同设备共用并推进同一生成器，避免同种子默认 CUDA 流复用随机字。
- 第一次真实抽样时从 CPU 默认 RNG 分配一个 int64 种子元数据，不是额外 token；
  构造、零预算及非法模型行在抽样前失败时不取种子。接受随机数仍在 target 设备抽取，
  不承诺新旧随机序列逐位相同。CPU 结构替身不构成真实 CUDA 验收。
- `build_draft_proposal(..., rng=...)`接收该上下文；独立调用这个 helper 时省略参数
  仍用原默认 sampler。完整 TK 路径始终显式传入，不改变其他算法族的 RNG。

## 账本与论文边界

仍按[论文](https://arxiv.org/html/2509.04576v1)的概率值 16-bit、索引及下行
忽略口径模拟：上行是 `gamma*min(K,V)*2` 字节，全 V 为 `gamma*V*2`，
与实际 payload 的值字节数一致。这是理论计费假设，不是实际网络零开销。

这是明确的**概率-wire 变体**，不是对论文含混 logits/probabilities 表述的逐字复现。
草稿与目标仍沿用全局温度，没有静默改成论文实验的 draft T=1。
FP16 仍可丢尾，但丢尾发生在提案采样之前。有限精度下使用既有严格单轮契约，
不宣称实数意义的全局 target-exact，也不宣称不同 GPU 归约逐位一致。

共享词表仍有既有前提：接收端 target 的前 V 列必须保留近似单位质量，
不会像草稿 codec 一样条件化掉显著的 padding 质量；不满足时明确失败。

## 兼容与证据

- [历史 FP16 bridge](<../src/decoding_ops.py>)保留 callable 及旧语义测试，
  活动 TK 已退出这条路径；[白名单测试](<../test/test_legacy_verification_bridge.py>)锁住这一点。
- 实现子任务的定向回归为 244 passed、85 subtests；Lead 随后补了缺失/过量
  payload 质量与前向前参数校验，两组分别先出现 2、4 个失败，再通过。
- 当前 [proposal 契约测试](<../test/test_tk_slt_proposal.py>)有 44 项；
  [真实微型 Llama/缓存测试](<../test/test_tk_slt_cpu_contracts.py>)覆盖预算、接受/拒绝、
  EOS、缓存端点、RNG 与非 one-hot wire 行。
- [循环测试](<../test/test_tk_slt.py>)捕获实际抽样输入、接收权重与真实载荷字节。
  将抽样临时改回 raw 模型行时，两个 law 测试失败；恢复后通过。
- Fraction 小分布枚举验证一步输出律，覆盖 `u=p/q` 严格拒绝与相邻浮点边界；
  非仅靠蒙特卡洛计数。
- [随机流模块测试](<../test/test_decoding_rng.py>)12 项通过；完整解码入口新增
  一项受控角色随机带反例及四项全路径显式生成器回归，先得到 5 个预期失败，再通过。
  零预算和非法模型行还验证了 CPU 默认 RNG 状态不前进。
- 私有 RNG 接线后再次做内存 raw-Q 退化，两个实际 sampler-law 断言仍失败；
  未通过修改接收端期望或共享 sampler 掩盖失配。
- CUDA Graph、真实多设备及 GPU 路径尚未实测；不能用 CPU 证据替代这些验收。
