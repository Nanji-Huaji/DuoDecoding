# `src/baselines.py` 拆分地图（只读分析，2026-10-19）

> 当前实现已在 S2 WIP 中迁到 `src/baselines/core.py`，兼容入口为
> `src/baselines/__init__.py`。前文保留历史路径与度量；最新进展见 §13。

> 基线：`main@b594290`（PR #12 合并后）。本文只做分析，不改代码。
>
> **进展（2026-10-19）**：H1、H2、H2b 已在 `refactor/baselines-shared-helpers` 落地
> （H1 覆盖 11 个方法；H2/H2b 覆盖 8 个标准构造点），逻辑指纹 0 差异。
> H3 评估后跳过（前置段顺序/有无不一致，净收益 ~15 行，见 §5）。H5 经
> 逐语句 diff 复核后判定**不适合机械抽取**（见 §5.5），等设计评审。
>
> **验证与发现（同日）**：golden 固件扩到 7 个方法（见
> `test/golden/baselines/`）；补固件过程中发现 main 上两个真实 bug 并修复：
> `adaptive_decoding` 在 `--use_rl_adapter` 下读未定义的 `_topk_cap`
> （NameError），以及评测收尾绕过 `disable_rl_update` 无条件写
> checkpoint（会把 worktree 的 RL checkpoint 重写）。两者都由
> 新增的 golden/smoke 暴露，与重构本身无关。
> 目标对应："整理超大模块和超大方法，提取共同框架和 helper，减少耦合"。
> 度量方法：AST 解析 + 方法内顶层语句的 5 语句滚动窗口哈希
> （`K=5`，跨方法重复窗口 112 个）；行号以本基线为准。

## 0. 结论摘要

1. **文件本身不是主要问题，方法是。** `baselines.py` 6,849 行里，12 个解码方法
   占 ~5,600 行；单方法中位数 ~450 行，`adaptive_tridecoding` 1,054 行。
2. **重复集中在两块**：仿真器构造（27 个分支点）与 metrics 收尾
   （12 个方法各写一遍 ~15 行）。这两块可以机械提取，预计减 ~350 行。
3. **不建议直接按"方法族"拆文件**，直到 H1/H2/H3 提取完成、指纹门禁稳定；
   拆包会改变 `logic_fingerprint` 的路径键与 `Register` 的导入顺序，风险
   高于收益。
4. **方法对相似度**（重复窗口数）：`cee_dsd ↔ cee_dssd` 54、
   `cee_dssd ↔ tridecoding` 41、`cee_dsd ↔ tridecoding` 38。
   CEE 两兄弟最适合共享核心，但需要先确认"链路方向/验证路径"的差异是参数
   还是结构。

## 1. 规模

| 文件 | 行数 |
|---|---:|
| `src/baselines.py` | 6,849 |
| `src/communication.py` | 1,487 |
| `exp.py` | 1,404 |
| `src/engine.py` | 1,328 |
| `src/utils.py` | 1,283 |

`Baselines` 共 28 个方法；≥200 行 16 个，≥400 行 9 个：

| 行数 | 方法 | 行号 |
|---:|---|---|
| 1,054 | `adaptive_tridecoding` | 4365–5418 |
| 531 | `tk_slt` | 2253–2783 |
| 518 | `tridecoding` | 2787–3304 |
| 516 | `cee_cuhlm` | 5450–5965 |
| 490 | `ceesd_without_arp` | 3308–3797 |
| 451 | `cee_dsd` | 6399–6849 |
| 427 | `cee_dssd` | 5969–6395 |
| 417 | `adaptive_decoding` | 3944–4360 |
| 392 | `dist_split_spec` | 1099–1490 |
| 375 | `dist_spec` | 1495–1869 |
| 375 | `uncertainty_decoding` | 1874–2248 |

## 2. 12 个方法的共同骨架

以 `dist_split_spec`（392 行）与 `adaptive_tridecoding`（1,054 行）为样本，
方法体可切成 7 段。行号是各自方法内的锚点：

| 段 | 内容 | DSSD 锚点 | AdaptiveTri 锚点 | 方法间差异 |
|---|---|---|---|---|
| S1 前置校验 | dtype/参数归一 | — | 4378 | 小 |
| S2 口径读取 | `_comm = self.comm_config`、`comm_env` | 1150–1153 | 4387–4391 | 字段集合不同 |
| S3 设备与深度 | `max_tokens`、little/draft/target device、`draft_top_k` | 1159–1171 | 4393–4406 | tri 多 little |
| S4 仿真器构造 | precise/regular 分支 + `CommEnv` | 1122–1157 | 4498–4543 | 27 个分支点 |
| S5 缓存/图 | `*_caches` 复用、graph kwargs | 1173–1194 | 4425–4497 | 每族不同 |
| S6 主循环 | `while prefix.shape[1] < max_tokens` | 1224–1417 | 4605–5337 | **方法核心，语义不同** |
| S7 收尾 | `flush_round` → metrics 组装 → 返回 | 1419–1490 | 5338–5418 | 字段有交集 |

注：段序在方法间略有差异（DSSD 先构造仿真器再读 `_comm`，
`adaptive_tridecoding` 相反），所以上表的行号只是各自方法内的定位锚点，
不是统一执行顺序。

**S4 与 S7 是重复的主体**；S6 虽然最长，但它是每个方法真正的差异所在，
不应为了"变短"而抽象。

## 3. 逐字重复块（AST 窗口检测）

112 个跨方法重复窗口，按覆盖方法数排序的典型块：

| 重复内容 | 覆盖方法数 | 代表行 | 建议 |
|---|---:|---|---|
| `metrics = get_empty_metrics()` + 基础字段 | 12 | 1436、1807、5353 | H1 |
| `metrics['edge_cloud_bandwidth_history'] = ...` | 11 | 1473、5401 | H1 |
| `metrics['comm_trace_edge_cloud'] = ...` | 11 | 1479、5407 | H1 |
| `comm_simulator.flush_round()` | 11 | 1419、5381 | H1（注意顺序差异） |
| `metrics['edge_end_comm_time']` / `edge_cloud_data_bytes` | 8 | 1855、2189 | H1 |
| `metrics['computation_time'] = elapsed_time` | 7 | 1842、5359 | H1 |
| `_comm = self.comm_config` + 3 个字段读取 | 7 | 1150–1153、4387–4408 | 保留在方法内（内省约束，见 §5） |
| 仿真器构造参数组 | 27 个分支点 | 1122/1131、4498/4512 | H2 |
| `_add_comm_accounting_metrics(...)` 调用 | 12 | 1482、5410 | 已有 helper，保留 |

**相似方法对**（重复窗口数）：

```
54  cee_dsd ↔ cee_dssd
41  cee_dssd ↔ tridecoding
38  cee_dsd ↔ tridecoding
35  cee_dssd ↔ ceesd_without_arp
31  dist_spec ↔ dist_split_spec
31  cee_cuhlm ↔ ceesd_without_arp
```

## 4. 同形不同义（**不要**机械合并）

| 对 | 看起来相同 | 实际差异 |
|---|---|---|
| `dist_spec` ↔ `dist_split_spec` | DSD/DSSD 的循环与计费 | 链路方向与拒绝路径不同（上行 vs 下行拒绝），验证输入不同 |
| `cee_dsd` ↔ `cee_dssd` | 54 个重复窗口，代码近亲 | DSD 走"目标模型上行整行"，DSSD 走"下行整词表拒绝"；是否 `charge_residual` 的触发点不同 |
| `tridecoding` ↔ `adaptive_tridecoding` | 三级骨架 | 后者有 ARP/DRA 自适应、RL 动作、两段验证；共同部分只在 S2/S7 |
| `cee_cuhlm` ↔ 其它 CEE | 计费接口相似 | CUHLM 不消费仓库计费开关（按论文式 (5) 自算），不能套同一 helper |

结论：H1（收尾）与 H2（构造）可以安全共享；**S6 主循环只在语义确认后
才考虑共享**，优先目标是 CEE 两兄弟的"验证/拒绝公共段"。

## 5. 提取候选（按风险从低到高）

### H1 `_finalize_comm_metrics(metrics, comm_simulator, args, *, flush=True)`

覆盖 11–12 个方法的 S7 公共尾巴：`flush_round`、`communication_time`、
`computation_time`、三链路字节、`comm_energy`、`connect_times`、
带宽/topk/草稿长度历史、`_add_comm_accounting_metrics`、`comm_trace_*`。

- 预计收益：~15 行 × 12 ≈ **150 行**；
- 差异处理：`adaptive_tridecoding` 在 `flush_round` 前写部分 metrics（顺序
  不同）→ 用 `flush: bool` 或拆成 `_record_comm_stats(flush=False)`；
- **不需要改方法体里的 `_comm.*` 读取**，内省不受影响；
- 门禁：逻辑指纹应**不变**（helper 原地展开）；若变，说明顺序被改，停止。

### H2 `_make_comm_simulator(...)` 工厂

27 个分支点（18 `CommunicationSimulator` + 9 `PreciseCommunicationSimulator`）
的公共参数：`min_bandwidth_mbps`、`bw_model`、三路带宽、`use_stochastic`、
`stochastic_ntt`、`ntt_ms_*`。

- 签名建议显式收值，不吞 `_comm.*` 读取：
  `_make_comm_simulator(self, *, precise, use_stochastic_comm, ntt_ms_edge_cloud, ntt_ms_edge_end, bandwidth_edge_end=inf, bandwidth_cloud_end=inf, trace=...)`；
- 预计收益：~10 行 × 27 ≈ **200 行**；
- 风险：precise 分支参数不同（`bandwidth_hz/channel_gain/send_power/noise_power`），
  工厂里保留 `precise` 分支即可；
- 门禁：golden + 指纹。

### H3 `_prepare_decode_context(prefix, transfer_top_k, *, need_little)`

S3 公共段：`max_tokens`、三个 device、`draft_top_k`。

- 预计收益：~8 行 × 12 ≈ **80 行**；
- 风险低，但收益也小；可在 H1/H2 之后顺手做。

### H4 metrics 基础字段

`get_empty_metrics()` 之后各方法重复写 forward/token/comp 字段，字段集不同，
只能做"按需参数化"（例如 `_fill_token_metrics(metrics, **kwargs)`），收益
~50 行，但可读性提升有限，优先级低。

### H5 CEE 公共核心（**2026-10-19 复核后判定不适合机械抽取**）

`cee_dsd` 与 `cee_dssd` 在 AST 顶层语句上相似度 92%，但那是假象：
两者各只有 ~35 条顶层语句，整段 `while` 循环只算 1 条。把循环体拆开逐语句
比对后，**循环内相似度只有 56%**：

| 差异点 | `cee_dsd` | `cee_dssd` |
|---|---|---|
| prefix 维护 | 预分配 `prefix_buffer`（性能） | `torch.cat` |
| 上行/下行传输点 | 先传 `x`、再传 speculated tokens | 按 `little_all_accepted` / `draft_all_accepted` 分支传 stitched 载荷 |
| 两级验证的输入构造 | `stage_prob_history` 的调用位置不同 | 同左但索引基准不同 |
| 循环结束后的 prefix 语义 | 由 buffer 截断 | 由 cat 拼接 |

共享的前置段（~35 条语句）里只有 3 处差异（docstring、cache 属性名、
DSD 的 buffer 预分配），但这段持有 **19 个循环内变量**（devices、
caches、counters、events、prefix、comm_simulator…）。抽成 helper 必须：
要么返回 19 元组（不可读），要么引入 dataclass 并把循环体里上百处局部
变量改成 `st.xxx`（大 diff、可读性反而下降）。

结论：**不建议做机械抽取**。若要做，正确形态是"三层的公共基类 + 两个
子类的 hook（传输点/验证输入构造/prefix 维护）"，属于设计变更，需要单独
评审；当前的 golden 固件（`test/golden/baselines/` 的 cee_dsd/cee_dssd
两个 tag）已经就位，可以直接支撑那次评审。

### H6 不建议：主循环整体抽象

12 个方法的循环体差异是"方法定义"本身；抽象成状态机会让每个方法变成
回调集合，违反可读性目标。

## 6. 模块拆分（第二阶段，包化）

以下是历史的完整拆包方向（接近 S3），不是本轮 S2 的实际布局。
S2 只建立 `__init__.py` + `core.py`，不拆方法族，见 §13。
在 H1–H3 之后，`Baselines` 可移到包：

```
src/baselines/__init__.py   # 保留 from src.baselines import Baselines 兼容
src/baselines/base.py       # 共享 helper、RL 构建、metrics 收尾
src/baselines/dsd_dssd.py   # dist_spec, dist_split_spec
src/baselines/tri.py        # tridecoding, adaptive_tridecoding, ceesd_without_arp
src/baselines/cee.py        # cee_dsd, cee_dssd, cee_cuhlm, cee_sd_opportunistic
src/baselines/adaptive.py   # adaptive_decoding, uncertainty_decoding
src/baselines/tk_slt.py     # tk_slt
```

约束（都必须先验证）：

1. **注册**：`@Register.register_decoding` 在导入时执行；`__init__.py` 必须
   import 所有子模块，否则 `parse_arguments` 才发现的"方法不存在"；
2. **内省**：`protocols.live_mode_consumption` 只跟随"注册方法之间的调用"，
   不跟普通 helper。把 `_comm.*` 读取抽进 helper 会让该模式少报消费开关，
   所以这些读取必须留在方法体或改为"方法内传值"；
3. **指纹**：`tools/logic_fingerprint.py` 以文件路径为键，拆包后
   `test/golden/logic/baselines.json` 需按子模块重建，测试也要改；
4. **导入面**：`src.baselines` 的模块级名字（`Baselines`、`_uplink_prob_payload`、
   `_reward_comm_s`）被测试/工具直接 import，`__init__.py` 要 re-export。

## 7. 实施顺序与门禁

| 步骤 | 内容 | 提交数 | 门禁 |
|---|---|---|---|
| 0 | 本文（只读地图） | 1 | — |
| 1 | H1 `_finalize_comm_metrics`（11 个方法） | 1 | ✅ 指纹不变 + q15 golden |
| 2 | H2 仿真器工厂（tri/cee 4 + 对称 4） | 2 | ✅ 指纹不变 + 3 次 smoke |
| 3 | H3 上下文准备 | — | ⏭ 跳过（前置段差异大，净收益 ~15 行） |
| 4 | H5 CEE 公共核心 | — | ⏸ 判定不适合机械抽取，等设计评审 |
| 5 | 包化（shim 优先） | — | S2 WIP 已完成，见 §13；S3 未开始，尚未提交 |

每一步都遵循：**一次一个逻辑提交；指纹变化必须能逐事件解释；golden
LOCKED 零漂移、tokens 逐位一致**。

## 8. 开放问题

1. `adaptive_tridecoding` 的 1,054 行里，S6 循环体能否按"两级验证"拆成
   两个方法（不动语义）？需要先用指纹确认抽取边界。
2. `cee_dsd`/`cee_dssd` 的"链路方向"差异，能否用 `link="edge_cloud"|"edge_end"`
   参数表达，而不是复制循环？需要对照 `docs/protocol.md` 的计费定义。
3. 包化后 `test/golden/logic/baselines.json` 的路径键怎么组织（每子模块一份
   fixture，还是保留一个合并文件）？建议每子模块一份，评审时 diff 更小。

---

**下一步建议**：先做 H1（收益确定、风险最低），每次只迁移 3–4 个方法，
指纹不变后再继续。本文档本身不改代码，可直接作为 PR 讨论材料。


## 8. 后续（2026-10-31）

- H5 细化设计见 `docs/h5_cee_core_design.md`：顶层 93.5% 相同，但验证/payload/prefix 更新分叉，不能机械抽取；建议按钩子分阶段重构，并用 enine 同机 A/B 替代指纹门禁。

## 9. 2026-10-31 进展

- H5b/H5c：CEE 第一/第二层块钩子已落地；`cee_dsd`/`cee_dssd` 421/397 → 224/221 行；
- S0：指纹 fixture 支持 `include` 多文件语料；
- S1：26 个模块级 helper 搬到 `src/baselines_helpers.py`，`baselines.py` 6,161 → 5,732（−429）；
- 验证：4090 同机 A/B 三轮均 20/20 token 一致；本地 543 tests；
- 未做：S2/S3 包目录化、H5c wrapper/protocols、`adaptive_tridecoding`（1,054 行）。

## 10. 2026-11-02 WIP：`adaptive_tridecoding` 诊断段原点抽取（未提交）

工作树基于 `main@50c43ba`；§1/§7/§9 的历史数字保持原样，本节保留诊断抽取阶段的快照，最新进展见 §13。
这一步**不是** Phase 0.3 的 `verify_and_commit_round` 拆分（该函数尚不存在），
只是把方法里的三段纯诊断代码原地搬到同模块 helper。

| 项 | 抽取前 | 抽取后 | 依据 |
|---|---:|---:|---|
| `adaptive_tridecoding` | 1,062 行（3094–4155） | 944 行（3094–4037） | AST `end_lineno - lineno + 1` |
| `_trace_adaptive_cache_preverify` | — | 38 行（4039–4076） | 同上 |
| `_trace_adaptive_calibration` | — | 114 行（4078–4191） | 同上 |
| `src/baselines.py` | 5,732 行 | 5,768 行 | `wc -l` |

两个 helper 合计 152 行（含 def/docstring/类型标注），方法跨度净减 118 行，
文件净 **+36** 行（还包含 def 前后空行）。**别把总行数当收益**：这是同模块原位抽取，收益在
方法体可读性，文件反而略增。§1 的 1,054 是 2026-10-19（b594290）口径，本节
按当前 AST 重数为 1,062。

搬走的三段（执行点与语义原样）：

1. `CACHE_CHECK_TRACE` 缓存一致性自检：仍在 `actual_gamma1 > 0` 分支内、
   `resolve_stage_verification` 之前，写 `where="pre_verify"`；
2. `ARP_CALIB_TRACE`：逐位 `(arp, tok, pd, pt, ratio, gmatch)`，ARP 取
   `draft_target_adapter.step_acc_probs`（只取 `draft_target` 主级头）；
3. `TOPK_TRACE`：top-k 累计质量/目标 token 秩/熵；`logits_history` 有值时用
   softmax 真分布，否则用 `prob_history` 重建并标 `obj="rebuilt"`。

口径不变（逐条核对过）：

- 内省：helper 不读 `self.args` / `_comm.*`，`live_mode_consumption` 不受影响；
- 计费：helper 体内无 `transfer` / `simulate_transfer` / `_residual_payload_bytes`；
- 采样：无 `sample*` / `torch.rand`，RNG 次数不变；
- 异常：三处各自的 `try/except Exception: pass` 作用范围原样保留；
- 接口：两个 helper 都用 keyword-only 参数，显式区分 draft/target cache，
  不引入通用诊断框架，不新增跨文件依赖。

门禁状态：

- 指纹：`diff src/baselines.py --base main` 判定 PASS，冻结条目
  `Baselines.adaptive_tridecoding` 的事件序列一致；两个 helper 以展开形式
  计入该条目，**不单独冻结**，`test/golden/logic/baselines.json` 未改（15 条）。
- 轮级行为门禁：新文件 `test/test_adaptive_tridecoding_rounds.py`（untracked）
  只走公开入口，真实 `resolve_stage_verification` + 真实
  `CommunicationSimulator`（测试子类在 `simulate_transfer` 记请求账、
  `_charge_transfer` 记结算账，含 round/link/bytes/topk/draft_len，
  实际计费仍由父类完成），
  KV 用脚本化 fake 但保留 `prob_history` 行语义与 rollback 截断。已写断言：
  `per_transfer` + `residual=False` 与默认 `per_round` + `residual=True`
  各自的三轮全接受和两阶段拒绝；同时锁定请求/结算账、残差字节、rollback、
  flush 顺序及默认口径的确定性通信时间。
- 诊断用例：`test/test_adaptive_tridecoding.py` 用非对称且相邻不同的概率行
  区分 proposer/verifier 和行索引；覆盖两轮 JSONL 追加、true logits、rebuilt
  降级，以及路径不可写时继续解码。交换缓存（两种）、错移概率行、覆盖写
  四种内存变异均被断言捕获；不改变共享的单轮 fake。

缺口（不夸大）：

- Phase 0.3 的 round 固件（固定张量 → `verify_and_commit_round` 精确输出）
  **没有**落地，函数不存在；本轮也没动两级验证/拒绝/缓存推进与计费路径，
  §9 的「`adaptive_tridecoding` 未做」只在诊断段这一小块上推进，S6 主循环原样。
- 轮级测试当前只跑 `_ablation="static"`，不证明 RL/ARP、no_arp、
  opportunistic、EOS 或图缓存复用等价；下一批触碰这些路径前需补相应用例。
  未跑 gammar golden，无远端 GPU 验收；本节不写测试通过数量。
- 行号以 2026-11-02 工作树为准，后续任何再抽取都会移位；新文件尚未提交。

## 11. 2026-11-02 续做 WIP：主级 RL 动作选择与应用

本节保留 921 行阶段的快照，最新进展见 §13。

承接 §10，不做 S2/S3、不提交，也不引入完整 layer-step 或
`verify_and_commit_round`。下一刀选在第二层生成及 target 前向之后、验证之前：
把主侧 `bandwidth/latency/acc_probs/task → select_config → top-k cap → ARP 阈值
→ 下一轮 gamma` 连续块原位抽为 `_select_adaptive_main_config`。

- Interface：五个 keyword-only 输入（通信仿真器、已计算的熵、本轮 gamma、
  top-k cap、no-ARP 标志），只返回 `transfer_top_k`；副作用仍是更新主 ARP
  阈值和可选的 `self._next_gamma1`。不引入通用 action/state 对象。
- 顺序：外层 RL/static 条件、`dra_start` 和耗时累计仍在调用方。当前 proposal
  已按旧 top-k/gamma/threshold 生成；选出的 top-k 供本轮后续结算及下一轮
  proposal 使用，gamma 仅下一轮生成生效。little 侧策略与奖励路径未抽取。
- 内省：所有 `_comm.*` 与 `args.gamma1/gamma2` 读取留在入口；唯一随
  `select_config` 移入 helper 的 `args.disable_rl_update` 不是通信/深度消费键，
  且调用方奖励段仍读取它。与 main 的 code-object 消费集比较保持四个通信键
  加 `gamma1/gamma2`，注册别名及两个 wrapper 的运行时内省也保持不变。
- 不移动两级验证、残差载荷、rollback、EOS、下行与 flush。尤其保留原行为：
  第一层 EOS 在 edge-end 下行计费后退出；第二层 EOS 在 edge-cloud 合并消息
  和 edge-end 下行之前退出；本批不修正历史计费口径。

AST 度量（同一工作树，不混用历史口径）：方法 944 → **921 行**
（3094–4014），新 helper **46 行**（4016–4061）；含此前诊断抽取，
方法从 main 的 1,062 行共缩短 141 行。文件 5,768 → 5,792 行，净增 24 行，
不是总行数压缩。

结构验证不只看指纹：将三个 helper 的 body 在调用点展开（去 docstring 和
尾返回、核对实参同名绑定及返回目标），整个 `adaptive_tridecoding` body 的
AST（含条件、参数和表达式，不含行号）与 main **完全相同**。具体归一化是
`ast.dump(ast.Module(body=method.body, type_ignores=[]), include_attributes=False)`，
以其 UTF-8 编码计算 SHA256（main 与展开后两侧相同）：
`9632a2c687de9d92aaa22940d4ee2a7638c9db350e1bccaa02c4081a60189932`。
这仍需人工核对局部变量活跃范围，不是 Python 任意重构的通用等价证明。
逻辑指纹 PASS，冻结 fixture 不变；后续门禁及审阅结果见
`docs/verification_gates.md` §8。

## 12. 2026-11-02 第三批 WIP：little 策略与两层拒绝概率计费

本节保留包化前的快照；S2 的最新布局和兼容合同见 §13。

在 §11 的 921 行基础上继续拆层内连续块，不引入通用 stage 框架：

| 抽取 | Interface / 职责 | 调用点 |
|---|---|---|
| `_select_adaptive_little_config` | 4 个 keyword-only 输入，无局部输出；选择 little 动作并写阈值/`_next_gamma2`，故意不应用 little top-k | 第一级生成后、机会门控前 |
| `_adaptive_rejection_prob_bytes` | 概率行 + 当轮 top-k → 拒绝概率载荷字节；共享元素宽度/top-k 估算、量化调用及 bits 覆盖 | 两层各一个拒绝分支 |

两处 caller 仍各自选择概率行、写 `reject_overhead=6`、执行 EOS/传输；后者
不是完整残差载荷/通信结算模块，**量化张量仍按历史实现只在局部使用，不回传**。
`prob_payload_bits` 在公共 helper 的原执行点经 `self.comm_config` 读取；它仍
在方法里的两个上行点显式读取，四个协议消费键与 `gamma1/gamma2` 未移出。
little 的 RL/static 条件及两个计时采样点留在 caller，机会门控仍能在同轮
看到新阈值，gamma 下一轮才生效。

| 项 | 上批 WIP | 本批 WIP |
|---|---:|---:|
| `adaptive_tridecoding` | 921 行 | **888 行**（3094–3981） |
| little 策略 helper | — | 37 行（4011–4047） |
| 拒绝概率字节 helper | — | 27 行（3983–4009），两处复用 |
| `src/baselines.py` | 5,792 行 | 5,825 行 |

方法本批净减 33 行，累计从 main 的 1,062 行减少 174 行；文件本批净增
33 行。收益是策略职责命名和重复计费内芯收拢，**不是整层拆分完成**。
机会判定仍留在入口：它依赖刚写入的可变阈值，且计算不确定度会消费 RNG；
只包成 5 入参 bool helper 会隐藏这两条合同。整层验证目前需要十多个输入
并会牵涉通信 trace 的栈帧记录，也不在本批硬拆。

验证使用 `.venv/bin/python`（Python 3.10.18）：五个 helper 共六个调用点展开
后，整个方法 body AST 与 main 相同，归一化 SHA256 仍为 §11 的值；AST dump
序列化依赖 Python 版本，不把别的版本哈希不同当行为漂移。调用方独立单文件
足迹仍为 308 事件，指纹 PASS，冻结 fixture 未改。全量 **569 passed、7 skipped、
90 subtests passed**，默认/importlib 两种模式一致；扩展用例换回 main 原方法也
通过（32 passed、5 subtests passed）。细项与剩余边界见 `docs/verification_gates.md` §9。

该批结束时尚未完成：完整第一/第二层步骤模块、真实 GPU/gammar 验收、S2/S3 包化；未提交。

## 13. 2026-11-02 S2 WIP：最小兼容包化

本批只改变物理位置与导入/测试接线，不做 S3，不继续拆方法、不改算法。
包化前比较基点是 §12 的 5,825 行工作树，而不是 main 的 5,732 行：
归一化源码 SHA256 为 `a642db78916433cabe05bb11251634fa0df8ebf133f074118a91e950708f71a6`。

```text
src/baselines/__init__.py   # eager 导入 core，保留历史导入面
src/baselines/core.py       # 原 5,825 行实现，adaptive 仍为 888 行
src/baselines_helpers.py    # S1 helper，位置不动
```

旧 `src/baselines.py` 已移走。core 文本只改 20 个相对 import（19 个顶层、1 个
legacy 方法内导入）各增加一级；逐字归一化和**完整模块 AST**对照均相同。
公开方法、装饰器与调用顺序不变，`Baselines.__module__` 现在为
`src.baselines.core`，这是包化的预期元数据变化，不伪造旧模块身份。

### 兼容合同

- `from src.baselines import Baselines` 保持；历史下划线 helper、模块属性读取、
  `dir()` 和星号导入均保留。包入口 eager 导入 core，24 个注册名及别名对象
  身份不变；重载入口不会重复执行 core 注册。
- 包化前冻结的 **105 个显式可读名字 / 79 个星号名字**已持久化到
  `test/golden/baselines_exports.json`，不是拿迁移后的导出集合自证。
- 兼容层是导入 Adapter，不是模块 globals 写透代理；patch 实现依赖应使用
  `src.baselines.core.X`，测试的原依赖替换点已迁移，公开 import 保持旧入口。
  `vars(src.baselines)` / 包 `__dict__` 不包含虚拟导出名，这一边界明确保留。
- 新 pickle 引用 core；旧的 `src.baselines` 全局 pickle 名仍能解析。
  包 `__file__` 指向 `__init__.py`，真实定义/inspect 源码指向 core。
- `COMM_TRACE.caller` 的文件名从 **baselines.py → core.py**，本批行号不变、
  调用栈深度及 kind 判据不变。跨版本 trace 聚合需归一化 caller，不宣称文本相同。

### 门禁和工具

四个源码/AST 守卫扫描真实 core 而非空壳；旧源码门禁的工作区文件枚举同时
处理未暂存删除与未忽略新文件，不靠 `git add` 才变绿。
逻辑指纹 fixture 仅 `path` 改为 core，`include/rev/15 条 entries` 全部保持。
工具 `diff --base-path` 仅映射旧侧入口，使当前 core 能与 main 的旧文件比较；
新增 CLI 用例锁定改名、真实漂移、路径错误及旧侧 include 的 git 来源。

全量与兼容性验收、独立审阅见 `docs/verification_gates.md` §10。
未提交/暂存；未跑 GPU/gammar。S2 只完成包入口与实现迁移，S3 方法族拆分、
完整第一/第二层步骤模块均仍未完成。
