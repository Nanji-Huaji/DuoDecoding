# 通信计费开关的 call site 清单（迁移前基线 + 迁移后状态）

基线：`refactor/logic-gates@5ce0f28`（2026-10-18）。行号会随后续提交漂移，
以"方法 + 开关"为准，行号只用于当前定位。

> **2026-10-19 已全部迁移。** 下表是迁移前的审计快照，保留它是为了让
> 每个读取点的旧回退值可查。现状：**32/32** 读取点经
> `Decoding.comm_config`（基类；Baselines 与 engine.py 的
> `speculative_decoding_with_bandwidth` 共用）；`_uplink_prob_payload`
> 改为收 `bits: int`，调用点传 `self.comm_config.prob_payload_bits`；
> 守卫测试 `test/test_comm_config_reads.py` 扫描两个文件、禁止回退。
>
> **通信数值层（2026-10-19）**：带宽 / `min_bandwidth_mbps` /
> `comm_bw_model` / `stochastic_ntt` / `batch_delay` / NTT trace
> 收进 `CommEnv`（`Decoding.comm_env`，**每次访问重建**：curriculum 会在
> 同一 decoder 上逐样本改带宽，缓存会沿用首样本条件）。迁移前这些值在 ~20 处
> `CommunicationSimulator(...)` / `PreciseCommunicationSimulator(...)`
> 构造点各自 `getattr(self.args, ..., 默认)`；现在默认值只有
> `comm_accounting.ENV_FALLBACKS` 一份，守卫同样禁止 args 直读。

## 1. 结论

- 5 个开关、**32 个读取点**、12 个方法 + 1 个模块助手
  （初版审计漏了 engine.py 的 `speculative_decoding_with_bandwidth`，2026-10-19 补齐）；
- 真正改数字的路径只有 5 条（见 §3），其余全是重复的 `getattr`；
- `force_full_vocab_transfer` 只有 `adaptive_tridecoding` 读（2 处），
  与 `src/protocols.py::_STATIC_ACCOUNTING` 的快照一致；
- CUHLM / TK-SLT 系只读 `comm_round_trip_mode`（统一往返），
  `cee_cuhlm` 一个都不读——与静态快照一致；
- `prob_payload_bits` 只有一处（`_uplink_prob_payload`），所有方法共用；
  迁移后该 helper 收 int，8 个调用点传 `self.comm_config.prob_payload_bits`
  是 2026-10-19 的快照，不是后续 helper 搬移后的源码调用点计数，见 §9。

## 2. 读取点总表

| 方法 | 行 | 开关 | 代码里的回退值 |
|---|---:|---|---|
| `_uplink_prob_payload` | 249 | `prob_payload_bits` | 16 |
| `Baselines.dist_split_spec` | 1146 | `comm_round_trip_mode` | per_transfer |
| `Baselines.dist_split_spec` | 1149 | `charge_residual_payload` | False |
| `Baselines.dist_split_spec` | 1150 | `transfer_top_k_cap` | 0 |
| `Baselines.dist_spec` | 1531 | `comm_round_trip_mode` | per_transfer |
| `Baselines.dist_spec` | 1534 | `charge_residual_payload` | False |
| `Baselines.dist_spec` | 1535 | `transfer_top_k_cap` | 0 |
| `Baselines.uncertainty_decoding` | 1920 | `comm_round_trip_mode` | per_transfer |
| `Baselines.tk_slt` | 2319 | `comm_round_trip_mode` | per_transfer |
| `Baselines.tridecoding` | 2837 | `comm_round_trip_mode` | per_transfer |
| `Baselines.tridecoding` | 2840 | `charge_residual_payload` | False |
| `Baselines.tridecoding` | 2841 | `transfer_top_k_cap` | 0 |
| `Baselines.ceesd_without_arp` | 3379 | `comm_round_trip_mode` | per_transfer |
| `Baselines.ceesd_without_arp` | 3382 | `charge_residual_payload` | False |
| `Baselines.ceesd_without_arp` | 3383 | `transfer_top_k_cap` | 0 |
| `Baselines.adaptive_decoding` | 3984 | `comm_round_trip_mode` | per_transfer |
| `Baselines.adaptive_decoding` | 3987 | `charge_residual_payload` | False |
| `Baselines.adaptive_decoding` | 3988 | `transfer_top_k_cap` | 0 |
| `Baselines.adaptive_tridecoding` | 4381 | `force_full_vocab_transfer` | False |
| `Baselines.adaptive_tridecoding` | 4401 | `charge_residual_payload` | False |
| `Baselines.adaptive_tridecoding` | 4525 | `comm_round_trip_mode` | per_transfer |
| `Baselines.adaptive_tridecoding` | 4527 | `transfer_top_k_cap` | 0 |
| `Baselines.adaptive_tridecoding` | 4528 | `force_full_vocab_transfer` | False |
| `Baselines.cee_dssd` | 6033 | `comm_round_trip_mode` | per_transfer |
| `Baselines.cee_dssd` | 6036 | `charge_residual_payload` | False |
| `Baselines.cee_dssd` | 6037 | `transfer_top_k_cap` | 0 |
| `Baselines.cee_dsd` | 6462 | `comm_round_trip_mode` | per_transfer |
| `Baselines.cee_dsd` | 6465 | `charge_residual_payload` | False |
| `Baselines.cee_dsd` | 6466 | `transfer_top_k_cap` | 0 |

`target_only`、`cee_cuhlm` 不读任何计费开关。

## 3. 每个开关计的是哪一笔

| 开关 | 生效点 | 计费语义 |
|---|---|---|
| `comm_round_trip_mode` | `comm_simulator.coalesce_rounds`（1145/1530/1919/2318/2836/3378/3983/4524/6032/6461） | `per_round` 时同一轮内每条链路只付一次 NTT；`per_transfer` 逐报文付 |
| `charge_residual_payload` | `if charge_residual:`（1357/1750/3008/3170/3557/3700/4222/4629） | 拒绝位置残差采样的提案分布载荷是否计入 |
| `transfer_top_k_cap` | `apply_transfer_top_k_cap(transfer_top_k, cap)`（1151/1536/2842/3384/3989/4527…） | `cap>0` 时把 k 收敛到 cap，直接压低上行字节 |
| `force_full_vocab_transfer` | 4381 / 4528 | 强制传完整词表分布（绕过 top-k 压缩） |
| `prob_payload_bits` | `_uplink_prob_payload`（249） | 概率载荷位宽；`<16` 时真量化并计 `bits/8` |

## 4. 两条硬约束（迁移前必须知道）

### 4.1 运行时内省靠"方法代码里出现开关名"

`src/protocols.py::live_mode_consumption` → `_collect` → `_feature_names(code)`，
读的是 code object 的 `co_names` 与字符串常量。方法体里不再出现开关名
⇒ 运行时判定"不消费" ⇒ `test/test_protocol_spec.py` 的
`live.accounting == snapshot.accounting` 断言失败。

已验证：改成属性访问后名字仍在 `co_names` 里——

```python
self.comm_config.comm_round_trip_mode      # LOAD_ATTR → co_names 命中
getattr(cfg, "comm_round_trip_mode")       # 字符串常量 → 命中
```

因此迁移后**字段名必须与开关名逐字相同**，不能用 `cfg.round_trip` 之类的别名，
除非同时改内省与静态快照（那是另一个 PR）。`proposal_utils.apply_transfer_top_k_cap`
的 docstring 也写着同一约束。

### 4.2 代码里的回退值 ≠ CLI 默认值

| 开关 | 方法 `getattr` 回退 | argparse 默认 |
|---|---|---|
| `comm_round_trip_mode` | `per_transfer` | `per_round` |
| `charge_residual_payload` | `False` | `True` |
| `transfer_top_k_cap` | `0` | `0`（2026-04 起不再全表钳位） |

`parse_arguments()` 保证属性总是存在，所以回退值在生产路径不可达；
但手工构造 `Namespace` 的测试会踩到。**`CommConfig.from_args` 保留这组
回退值**（`src/comm_accounting.FALLBACKS`），避免合成 args 静默换口径。

这里有个实证：`Baselines.comm_config` 初版用模块里的 `DEFAULTS` 当缺失属性
的回退，而 `DEFAULTS["transfer_top_k_cap"]` 曾误写成 16（CLI 实际是 0）；
于是 `test_decoding_avg_topk_metrics` 里 dsd 的 0 被钳成 16 而红。
修法不是改测试，而是把 `from_args` 的回退换成迁移前方法的字面值
（`FALLBACKS`），同时把 `DEFAULTS` 改回 0——两套常量因此必须分别定义。

因此模块里有两套常量，职责不同：

| 常量 | 用途 | 值 |
|---|---|---|
| `DEFAULTS` | 协议投影时缺失键的回退 | = argparse 默认 |
| `FALLBACKS` | 从 args 读且属性缺失时的回退 | = 迁移前各方法 `getattr` 的字面值 |

## 5. 迁移配方

每个方法一个 PR：

1. `Baselines.comm_config` 属性（惰性构造 + 缓存）：方法首次访问时调
   `CommConfig.from_args(self.args)`，不改 `__init__`；
2. 方法内把 `getattr(self.args, "<switch>", <fallback>)` 换成
   `self.comm_config.<switch>`——**字段名 = 开关名**；
3. 跑 `python3 tools/logic_fingerprint.py diff src/baselines/core.py --base main --base-path src/baselines.py --include src/baselines_helpers.py --include src/baselines/simulators.py --entries <方法>`
   ——必然红（flag_getattr 事件消失），逐条列出变化；
4. 因为 4.1，`test_protocol_spec` 与 `live_mode_consumption` 必须保持不变；
5. 数字验收：同一条固定命令跑 before / after 两次，除 timing 与 git_*
   之外全部精确相等；
6. 测试 + fingerprint fixture 更新（只改被迁移方法的条目，并附解释）。

## 6. 第一批（本阶段）

只迁 `dist_spec` 的 3 行（1531/1534/1535），验证配方；`dist_split_spec`
留给下一个 PR——它俩代码 81% 相同，一起改会让 diff 难以逐条解释。

## 7. 计费接口的最终形状（与本轮实现一致）

计划里写的是"5 个方法"，实施时改成**字段 + 纯 helper**，原因还是 §4.1：

- 若方法体写 `cfg.per_round`，`"comm_round_trip_mode"` 会从该方法
  的 `co_names` 里消失，`live_mode_consumption` 立刻判它不消费，
  `test_protocol_spec` 直接红；
- 保留名字的唯一无副作用写法，就是让方法继续出现开关名——
  也就是直接读 `cfg.comm_round_trip_mode`。

因此接口定义为：

| 接口 | 位置 |
|---|---|
| 5 个口径字段（名字 = 开关名） | `CommConfig` |
| top-k 钳位 | `proposal_utils.apply_transfer_top_k_cap`（已有） |
| 概率载荷位宽与量化 | `baselines._uplink_prob_payload`（已有） |
| 协议投影 | `CommConfig.from_values` / `PROFILES` |

alias 方法（如 `cfg.per_round()`）看起来更整齐，代价是内省失效，
或者在方法里留一个没人调用的字符串常量——两者都不值得。

## 8. 本轮验证记录（2026-10-18，GPU 1）

- `q15_free_g5` 复现：metrics 判定 PASS（8 个 timing 允许漂移），
  tokens 20 样本逐位一致；
- `dist_spec` before/after：64 个字段除 timing 与 `git_*` 外全等；
  20 条 jsonl 只有 `time` 不同，`gen_ids` 相同；
- 全套 pytest 427 passed；`test_protocol_spec` 31 passed；
- 命令与产物：`.refactor/logs/q15_g5_calib.log`、
  `dist_spec_{before,after}.log`、`dist_spec_before_metrics.json`；
  worktree 的 `exp/gammar/` 下有三次 run 的完整产物。

## 9. 2026-11-02 WIP：adaptive 两处拒绝概率计费共同段

`adaptive_tridecoding` 的两处拒绝分支改为调用
`_adaptive_rejection_prob_bytes(prob_data=..., transfer_top_k=...)`；原来两次
`self.comm_config.prob_payload_bits` 读取各自在 helper 展开后的位置执行，
不预读、不移动到入口，也不改成裸 args 读取。方法体里的两个上行载荷读取
仍保留，四个协议计费键与深度键没有挪入普通 helper，运行时消费集保持不变。

该 helper 只返回**拒绝概率部分**的计费字节，量化张量不返回给 caller；真实
验证/上行载荷在原位置另行处理，不能把这个返回值当成“已量化的传输载荷”。
索引/token/6 B 拒绝开销、残差消息和 `simulate_transfer` 仍由 caller 持有。
因此通信 trace 的 caller 归属函数与 kind 判据不变；caller 字符串中的源码
行号仍会随抽取缩短而变化，不宣称 trace 文本逐字节相同。本批 CPU 门禁新增
bits=8 两层拒绝及 bits=16 对照，不代替 GPU/gammar 数值验收。

## 10. 2026-11-02 S2：实现路径与 trace 标签

实现从 `src/baselines.py` 原样移到 `src/baselines/core.py`，兼容入口保留旧导入。
S2 相对包化前的 5,825 行 WIP 只改相对 import 层级，计费调用、参数、顺序、
行号和栈深不变；若与 main 比较，还包含此前 helper 抽取造成的行号移动。
但 `COMM_TRACE.caller` 使用代码文件 basename，因此明确从
**`baselines.py:<line>` 变为 `core.py:<line>`**。`scripts/analyze_comm_trace.py`
按 caller 分组，跨版本 join/对照应先归一化这一标签，不直接比较 trace 文本。
`kind` 仍由原调用栈判定，不为保留旧标签修改通信实现。

模块全局依赖替换应 patch `src.baselines.core.X`，而不是只改兼容包上的 `X`。
源码读取/指纹入口也指向 core；旧 main 的映射用 `--base-path src/baselines.py`，
`--include src/baselines_helpers.py` 仍需保留。详细兼容边界见
`docs/baselines.md` 与 `docs/verification_gates.md` §10。

## 11. 2026-11-03 S3a：构造工厂与真实定义模块

仅两个构造工厂迁到 `src/baselines/simulators.py`，所有 `Baselines` 方法体保持；
方法内 transfer/simulate_transfer 的帧位置和 kind 不变，caller basename 仍
`core.py`，但整类源码行号统一前移 89 行。旧 S2“行号不变”只针对包化前 WIP，
不针对整个重构或这次 S3a。

工厂内部的 constructor patch 应指 `src.baselines.simulators.X`；
`adaptive_tridecoding`/CUHLM 的直接构造点仍指 `src.baselines.core.X`。
`src.baselines._make_comm_simulator` 与 core 私有 re-export 都是同一工厂对象，
不因此获得 globals 写透语义。新的指纹调用必须同时保留
`--include src/baselines_helpers.py --include src/baselines/simulators.py`。

同机真实模型 A/B 的 byte/connect/逐消息账与 token 都严格比较，通信模拟时间
也属于 LOCKED，而非一概把名字带 time 的字段忽略；只有明确列出的墙钟/模型
计算/策略开销与吞吐字段可漂移。实验覆盖与限制见夜间验收记录。

## 12. 2026-11-06：两模型 adaptive_decoding 的阶段接口

计费开关和 gamma 读取仍在注册入口；[core.py](<src/baselines/core.py>)的真实通信位置改为：

| 报文/操作 | 所属阶段 |
|---|---|
| 每个待验证 token 及其 proposal 概率 | `_adaptive_decoding_verify` |
| 真正发生 token 拒绝后的拒绝消息 | `_adaptive_decoding_verify` |
| 拒绝位置的 q 载荷与可选 tail scalar | `_adaptive_decoding_next_token` |
| round 设置、最终 flush、下行调度 | 公开入口 |

`COMM_TRACE.caller` 仍是 `core.py:<line>`，行号现在落在相应阶段函数内。
正常样本核对报文种类、上行先于拒绝的因果顺序和实际账本；跨版本分析按阶段归类，使用真实帧位置。

本轮按逻辑正确性修正了旧 bounds 分支：验证上下文不完整时直接报错，
不会发送拒绝后又将整段计为接受。q 使用生成时的 proposal 分布，合法重建 q
不受 raw draft history 长度误判影响。完整说明与测试见[本轮记录](<docs/adaptive_decoding_refactor_20261106.md>)。
