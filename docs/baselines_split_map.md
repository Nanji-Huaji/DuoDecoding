# `src/baselines.py` 拆分地图（只读分析，2026-10-19）

> 基线：`main@b594290`（PR #12 合并后）。本文只做分析，不改代码。
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

### H5 CEE 公共核心

`cee_dsd` 与 `cee_dssd` 共享 54 个窗口段。建议在 H1/H2 落地后，把
"验证 + 拒绝载荷计费"抽成带 hook 的核心（`_cee_verify_step(...)`），
两方法保留各自的链路方向与停止条件。**这是收益最大、也最需要 golden 的
一步**，单独 PR。

### H6 不建议：主循环整体抽象

12 个方法的循环体差异是"方法定义"本身；抽象成状态机会让每个方法变成
回调集合，违反可读性目标。

## 6. 模块拆分（第二阶段，包化）

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
| 1 | H1 `_finalize_comm_metrics`（先覆盖 4 个方法，再推广到 11 个） | 2 | 指纹不变 + golden q15 |
| 2 | H2 仿真器工厂（先 2 个方法，再全量） | 2 | 指纹不变 + golden q15 |
| 3 | H3 上下文准备 | 1 | 指纹不变 + golden q15 |
| 4 | H5 CEE 公共核心 | 2–3 | 指纹**会变**，需逐事件说明 + golden |
| 5 | 包化（shim 优先） | 2 | 全量测试 + 指纹重建 + golden |

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
