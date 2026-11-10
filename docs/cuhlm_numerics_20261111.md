# CUHLM 单 token 数值契约（2026-11-11 第三批）

分支 `refactor/cuhlm-numerical-contract-20261111`，基点 `fe76339`。本批只改
plain `uncertainty_decoding` / `cuhlm` 的验证数值路径：把严格 verifier 的三个
数值 kernel 提取复用，新增独立单 token 验证接口，并让 CUHLM 退出 legacy
验证桥。通信 codec、Eq. (26) k solver、CEE-CUHLM 与 GPU 路径不在本批范围内。

## 三个角色不能混用

| 角色 | 来源 | 用途 |
|---|---|---|
| 原 proposal `q` | 草稿缓存本轮实际采样前的原始概率行 | `ProposalWindow` 按实际 `sampling_weights` 规范化；接受用其选中 `q(d)` |
| target `P` | `target.forward_context(旧 prefix)` 后历史的第 `prefix_len-1` 行 | 接受比值 `P(d)/q(d)` |
| 重建 `qhat` | `compress_rebuild_probs(原 q 行, k)` 返回的原始概率 | 整窗先校验；仅拒绝分支使用 `normalize((P - qhat)+)` |

`q` 与 `P` 各走一次 sampler 预处理；不会把已经处理过的 proposal weights 再喂
`sampling_weights`。`qhat` 是 codec 原始输出（不是 proposal，也不是已处理
weights），先校验形状/有限/非负/质量，再走一次同一预处理得到其归一化 law。

## 决策规则

- uniform 严格取 `[0, 1)`，比较是严格 `<`；比值等号拒绝。
- 原 `q(d)` 选中支撑必须为正；`P(d)=0` 合法，且 `u=0` 必须拒绝。
- 整窗（proposal、恰好覆盖旧 prefix 的 target 历史、重建 law）在取 acceptance
  uniform 之前验证；非法数据不会先消费 RNG。
- 每次触发只取一次 uniform。接受不采新 token（保留 draft，无 bonus）；拒绝
  只从规范化残差采一次。RNG 交错顺序保持为 acceptance 随机数 → 残差采样 →
  下行报文（随机通信模式也会消耗 RNG，不能把采样挪到下行之后）。
- target 历史必须恰好是本轮 forward 的旧 prefix 行数；缺少或多余一行都报错，
  不再裁成空窗口、不重复同一 prefix。
- 只在拒绝时要求残差正质量。`P=qhat` 且接受时允许零残差；拒绝且
  `(P-qhat)+ = 0` 时明确 `ValueError`，不补 epsilon/均匀尾部、不走 argmax、
  不静默改用原 `q`。
- 全词表重建 `qhat=Q` 时，CUHLM 残差与严格 verifier 的残差逐位相同（同一
  kernel、同一输入）；有损 `qhat≠Q` 时保留 `qhat` 自己的输出 law。
- 草稿生成后立即构造并验证 `ProposalWindow`，skip 分支也不能把非法 proposal
  （例如选中支撑为零）藏起来；该验证不采样、不耗 RNG。

## 共享数值 kernel

[`probability_laws.py`](<../src/probability_laws.py>) 从
[`verification.py`](<../src/verification.py>) 提取共享运算：

- `validate_probability_rows`：按 dtype 容差校验原始行，不替调用方归一化；
- `selected_probability_law`：先消全局 scale，再做 float64 选中比值；
- `positive_residual_law`：保留原 pivot/叉积残差、FP32→float64 与归一化顺序。

接受标量与残差运算顺序未改。Lead 对照 PR #34 做了 **122 个逐位差分用例**，
覆盖 BF16/FP16/FP32、gamma 0–3、FP32 selected-Q 载荷和 1e-30 极小尾部，
结果逐位一致，输入未改动。CUHLM 侧新增
[`cuhlm_verification.py`](<../src/cuhlm_verification.py>)，只做单 token
`prepare` / `decide`，无模型、缓存、通信或 RNG 状态。

### 大词表的质量校验修正

这不是完全无行为变化的提取。Lead 复现了两个合法 `torch.softmax` 输出被旧
固定容差误拒绝的情况（词表均为 151936）：

- FP32：seed 0 的 `randn(V)*3`，质量为 `1.000004997`，超出旧 `1e-6`。
- FP16：首个 logit 为 0、其余为 `-11.9921875`，质量为 `1.004162`，
  超出旧 `4*eps`；此例尾部是 subnormal，不能只算相对舍入误差。

公开接口反例先得到 **2 failed、4 passed**。现在用 FP64 测量质量，并按词表
规模、至少 FP32 的 softmax 累加、输出精度及 subnormal 绝对误差建立舍入预算。
令 `u_a=eps_acc/2`、`u_o=eps_out/2`、`nu=(V-1)*u_a`、
`gamma=nu/(1-nu)`，预算为
`(1+u_a)^2*(1+u_o)/(1-gamma)-1 + V*smallest_subnormal/2`，
仍保留旧 dtype 容差下限。`nu>=0.5` 时误差界不再可用，明确报错，
不把质量校验放宽成无穷容差。该预算是保守输入校验，不是分布质量指标。

输入没有被修改或预先归一化；决定实际 law 的 sampler 运算未变。
在 V=151936 时，质量 0、0.5、1.25 仍拒绝；V=2 的质量 1.001 也仍拒绝，
不是所有词表一律使用大容差。零残差仍精确失败。
新增六项后，与原 Stage A 数值回归合计 **114 passed**。此修正同时适用于
严格 verifier，不能再声称质量校验完全沿用原契约。

## 两个已证实反例

1. BF16 `q=[.1,.3,.6]` 与 `P=sampling_weights(q)`：两者实际 sampler weights
   逐位相同，新比值约为 1，`u=.999` 接受。旧 raw-q 比值约 `.9976`，会误拒绝；
   旧 `max_fn` 随后回退 argmax token 0。新实现不再用 raw q 标量。
2. `q=[.95,.04,.01]`、`k=1`、`qhat=P=[.95,.025,.025]`、候选 `d=1`：比值
   `.625`，`u=.9` 确实会拒绝（拒绝概率约 `.375`），但 `(P-qhat)+ = 0`。旧
   `max_fn` 回退 argmax one-hot token 0；新实现抛
   `ValueError: ... has no positive finite residual mass`。这是论文有损压缩的
   真实退化，本批选择 fail-closed，不声称拒绝不可能。

另以真实 codec 核对固定触发、k=1 的完整单步输出 law：
`q=[.8,.11,.09]`、`P=[.7,.115,.185]`，输出约为 `[.7,.125,.175]`，
相对 P 的 TVD 约 0.01。这验证有损残差没有被误换成精确 SD；它不是实际
自适应触发循环的整体分布或通用误差保证。

其他回归：BF16/FP16/FP32 同 law 不误拒；近似成比例（2 的幂、FP32 中点）在
`nextafter(1, 0)` 仍接受；极小正尾部不变成假零残差；selected-Q 比值等号拒绝；
零 target 支撑在 `u=0` 拒绝；错 shape / 缺失或多余 history / 非法 `qhat` /
非法 uniform 都在 RNG 前明确报错；kernel 不改输入、不抽随机。

## 测试与指纹

最终完整严格 CPU 门禁启用 `--require-golden-assets`：

| 导入模式 | 结果 | 耗时 |
|---|---|---|
| default | 1473 passed、7 skipped、163 subtests | 86.96s |
| importlib | 1473 passed、7 skipped、163 subtests | 86.39s |

14 项历史权重指纹实际通过；7 项 skip 为缺失原始 run，不能当作验证通过。
GLM-5.3 对收尾版本做了一次只读批次复查，包含共享运算、质量误差预算、
公开接线和测试，未报告发现；另做了差分与边界探针。源码端点一致。
日志和 manifest 位于本地忽略目录
`.refactor/cuhlm-numerics-20261111/final-cpu-strict/`。

实现者版本的 CPU 定向（尚未包含 Lead 的六项质量校验回归）：

| 文件 | 通过 |
|---|---|
| [test_cuhlm_verification.py](<../test/test_cuhlm_verification.py>)（新） | 37 |
| [test_cuhlm_cpu_contracts.py](<../test/test_cuhlm_cpu_contracts.py>) | 16 |
| [test_uncertainty_decoding.py](<../test/test_uncertainty_decoding.py>) | 17 |
| [test_legacy_verification_bridge.py](<../test/test_legacy_verification_bridge.py>) | 3 |
| [test_verification_contracts.py](<../test/test_verification_contracts.py>) | 34 |
| [test_sampling_law_contracts.py](<../test/test_sampling_law_contracts.py>) | 37 |
| [test_logic_fingerprint.py](<../test/test_logic_fingerprint.py>) | 12 |
| [test_baselines_package.py](<../test/test_baselines_package.py>) | 21 |

合计 **177 passed**。把上表与 CEE/通信/protocol/TK-SLT/docs/dual 相邻文件合并
复跑（23 个文件）：**544 passed、85 subtests passed**。真实 tiny Llama 的预算、EOS、缓存端点、
forward 输入长度与 RNG 预算断言仍通过；新增公开入口用例覆盖有损 `qhat` 决定
拒绝 token、非法 `qhat` 在 acceptance RNG 前失败、skip 不隐藏非法 proposal。

[`test/golden/logic/baselines.json`](<../test/golden/logic/baselines.json>) 只改
`CuhlmMixin.uncertainty_decoding` 一项，其余冻结项逐项比对未变：事件数仍为
90，sha `eaf44fd1b5f6baf4` → `7aec9e478ae6d0cd`。变化为移除
`_prepare_legacy_verification_inputs` / `compute_acceptance_result` /
`materialize_acceptance`，RNG 由 `sample_reject_token` 变为一次 `rand` 加一次
`sample`，加入 `ProposalWindow` / `prepare_cuhlm_verification` /
`decide_cuhlm_verification`，并减少一个本地 `ValueError`。

[`test/test_legacy_verification_bridge.py`](<../test/test_legacy_verification_bridge.py>)
移除了 cuhlm 的 legacy allowlist；`src/baselines/cuhlm.py`、新 kernel 模块与
`src/verification.py` 都纳入“不得导入 legacy 数值接口”的限制（限制的是数值
验证入口，不是声称有损协议无偏）。公开循环不再调用旧
`compute_acceptance_result` / `materialize_acceptance` / `sample_reject_token`。

## 仍未解决

- Eq. (26) k solver 未改：搜索不含 `V`，不满足/长度不匹配时返回
  `min(300, V//100)`，该固定回退仍不是已证明的保守策略；BF16
  `[.1,.2,.7]`、`theta=0` 仍可得到 `k=0`。
- codec / `compress_rebuild_probs` 的 top-k + 均匀尾部语义、传输字节与
  “重建 law 是否满足论文假设”均未改；本批只验证其输出并 fail-closed。
- CEE-CUHLM 的 `vocab_size` / `draft_token` 透传及独立组合验证仍待后续批次。
- 未跑 GPU / CUDA Graph；CPU 通过不代表图路径或真实大词表性能已验证。
- 这仍是论文的有意有损协议（原 `q(d)` 接受、`qhat` 残差），不能称为 target
  分布精确；`draft_accepted_tokens` 也仍只是 draft 定稿数，不是目标准确率。
