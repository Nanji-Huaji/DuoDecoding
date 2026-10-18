# 通信计费开关的 call site 清单（Phase 1 迁移前）

基线：`refactor/logic-gates@5ce0f28`（2026-10-18）。行号会随后续提交漂移，
以"方法 + 开关"为准，行号只用于当前定位。

## 1. 结论

- 5 个开关、**29 个读取点**、11 个方法 + 1 个模块助手；
- 真正改数字的路径只有 5 条（见 §3），其余全是重复的 `getattr`；
- `force_full_vocab_transfer` 只有 `adaptive_tridecoding` 读（2 处），
  与 `src/protocols.py::_STATIC_ACCOUNTING` 的快照一致；
- CUHLM / TK-SLT 系只读 `comm_round_trip_mode`（统一往返），
  `cee_cuhlm` 一个都不读——与静态快照一致；
- `prob_payload_bits` 只有一处（`_uplink_prob_payload`），所有方法共用。

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
| `transfer_top_k_cap` | `0` | `16` |

`parse_arguments()` 保证属性总是存在，所以回退值在生产路径不可达；
但手工构造 `Namespace` 的测试会踩到。**`CommConfig.from_args` 保留这组
回退值**（`src/comm_accounting.FALLBACKS`），避免合成 args 静默换口径：
`test/test_decoding_avg_topk_metrics.py` 的 Namespace 就没有
`transfer_top_k_cap`，若按 CLI 默认 16 钳位，`dsd` 的 0 会被钳成 16 而红。

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
3. 跑 `python3 tools/logic_fingerprint.py diff src/baselines.py --entries <方法>`
   ——必然红（flag_getattr 事件消失），逐条列出变化；
4. 因为 4.1，`test_protocol_spec` 与 `live_mode_consumption` 必须保持不变；
5. 数字验收：同一条固定命令跑 before / after 两次，除 timing 与 git_*
   之外全部精确相等；
6. 测试 + fingerprint fixture 更新（只改被迁移方法的条目，并附解释）。

## 6. 第一批（本阶段）

只迁 `dist_spec` 的 3 行（1531/1534/1535），验证配方；`dist_split_spec`
留给下一个 PR——它俩代码 81% 相同，一起改会让 diff 难以逐条解释。
