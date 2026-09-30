# 最终口径定义（`paper_table5` 协议）

日期: 2026-09-29 · 状态: 决策已定；**机制层已落地**（§5），统一计费待实施（§3）
配套: `docs/param_ledger.md`（参数来源与覆盖度分析）

## 1. 核心原则

1. **口径 = ours 的计费实现**。`adaptive_tridecoding`（`baselines.py:3300-4290`）已经是
   论文协议（`per_round` 往返 + 残差计费 + top-k 上限）的完整实现，把它作为唯一基准，
   其余方法改为复用同一套 helper，而不是各自内联。这样"同口径"不再需要逐方法讨论
   字节模型（同时也就关闭了 B17 那个悬置的问题：基准即定义）。
2. **L0 只能在协议里改**。任何 L0 参数被命令行单独覆盖时，该 run 不再属于本协议。
3. **协议不可比就不出表**。落不到同一口径的列，宁可不报。

## 2. L0 冻结值（`paper_table5`）

| # | 项 | 冻结值 | 落地状态 |
|---|---|---|---|
| 1 | 模型 | `llama-68m` / `tiny-llama-1.1b` / `llama-2-13b` | 已可 |
| 2 | 数据集 / N / seed | GSM8K（主表）· 80 · 1234 随机采样 | 已可 |
| 3 | num_shots / max_tokens / temp | 3 / 128 / **0.0** | 已可（必须显式传 `--temp 0.0`） |
| 4 | 链路带宽 | edge-end **563** / edge-cloud 46 / cloud-end 46 Mbps | 已可 |
| 5 | NTT | edge-cloud **50** / edge-end 0.317 ms | 已可 |
| 6 | 往返口径 | `per_round`（每轮每链路一次往返） | **需改代码** |
| 7 | 残差计费 | 开（`charge_residual_payload=True`） | **需改代码** |
| 8 | top-k 上限 | 16（`transfer_top_k_cap`） | **需改代码** |
| 9 | 随机带宽 | `--use_stochastic_comm`（配 trace 时同时推进 NTT） | 已可 |
| 10 | 投机深度规则 | **待讨论**（见 §4） | 待决策 |
| 11 | `use_early_stopping` | 关（生成长度须由 max_tokens 决定） | 已可 |
| 12 | RL adapter | 只按 `mode_features.py` 的规定挂载，**不作用于基线** | **需改代码/改扫描** |
| 13 | 统计口径 | warmup 实跑声明次数；失败样本整体剔除；重跑速度只统计本次行 | 已可（B29/B31/B35） |
| 14 | 图加速 | 开，且**必须成对报告**（图开/图关各一列或脚注） | 已可 |

第 12 条的依据：`docs/paper_table5_alignment.md` 记录，把 RL adapter 传给所有 mode 后
DSD 的 R_acce 从论文 32.11% 变成 60.6%，而前向数在 γ=3 时本来就吻合。基线挂 RL 会
改变其 top-k 行为，不属于同口径比较。

## 3. 统一口径要做的事（对应 §2 的"需改代码"）

以 `adaptive_tridecoding` 的路径为基准，抽出并让所有方法复用：

1. **往返合并**：`comm_simulator.coalesce_rounds = (comm_round_trip_mode == "per_round")`，
   并在每轮边界调用 `set_round(idx)`、解码结束调用 `flush_round()`。现在只有
   `adaptive_tridecoding` 会做（`baselines.py:3507/3586/4287`），其余 8 个方法恒为
   `per_transfer`，即一轮付 3.2+3.0 次 NTT（`communication.py:443-461` 逐次累加）。
2. **残差计费**：所有方法统一走 `_residual_payload_bytes`（`baselines.py:2840`）+
   `charge_residual_payload` 开关；不再出现"dsd/adaptive_decoding 只记 `INT_SIZE`"。
3. **top-k 上限**：`transfer_top_k_cap` 对所有方法生效，而不是只对 `adaptive_tridecoding`。
4. **拒绝路径的载荷**：统一为 `k*(4+元素大小)+元素大小` 字节/拒绝位置/链路。

改完后，`--comm_accounting honest` 才真正等于"全表同口径"，`legacy` 才真正等于
"复现 2026-09-24 前的历史数字"。**在此之前任何 run 都是混合口径**，且
`eval/utils.py:123-135` 仍会把标称值写进 metrics JSON —— 这一点必须先修掉，
否则口径标签会继续骗人。

## 4. 唯一待定项：投机深度（γ）

现状：`--gamma` 只被 `dist_spec`/`dist_split_spec`/`uncertainty_decoding`/
`adaptive_decoding`/engine 的 speculative 读取；三级方法只读 `gamma1`/`gamma2`。
扫描没给 `--gamma` ⇒ 落到 `create_config(gamma=5)` 的**签名默认值** ⇒ 基线实际跑
γ=5，而 `align_paper_t5_gsm8k.sh` 的头注释写的是"基线 γ=3"。

过渡期规则（在讨论清楚之前立即生效，避免再次误读）：

- `--gamma`、`--gamma1`、`--gamma2` **一律显式传**，不允许靠默认值兜底；
- run 启动时打印 `[protocol] γ: 单-γ 方法=5 / 三级方法=3/3` 并标注
  `γ 规则未定（协议 paper_table5 不含该项）`；
- 结果文件名/JSON 记录实际用的 γ 与规则名。

讨论时需要的数据（建议一次跑完，其余全部固定为 §2 的值，逐 γ 报
`Fwd`、`Tcomm`、`tok/fwd`、`R_acce`、准确率）：`γ ∈ {3,5,8,16}`，每个方法一组。
这样"matched 还是 per-method"就有表可依，而不是凭感觉。

## 5. 强制机制（Step 1 已落地，不改动任何数字）

| 机制 | 状态 | 位置 |
|---|---|---|
| 唯一真源（协议冻结集） | ✅ 已建 | `src/protocols.py` 的 `PROTOCOLS`；`--protocol`（默认 `none`） |
| 启动自述（含"声明 vs 实际消费"） | ✅ 已用 | `emit_effective_report()`，`src/utils.py` 在 `model_zoo` 前调用 |
| 偏离告警（CLI 覆盖 L0） | ✅ 已验证 | `apply_protocol()`；`protocol_deviations` |
| metrics 记**实际消费值** | ✅ 已接 | `eval/utils.py:get_save_dict` 新增 `protocol` / `protocol_deviations` / `comm_accounting_consumed` / `depth_keys` / `consumption_source` |
| 删除 12 个无消费者参数 | ✅ 已删 | `src/utils.py`（含 `--controlled_*` 整簇） |
| 修复 `--help` 崩溃 | ✅ 已修 | `--use_cuda_graph` / `--rl_charge_queue` 的 help 里 `%` 未转义导致 argparse 抛 `ValueError`（既有 bug，退出码 1）。**110 个参数的 CLI 此前连帮助都打不出来** |
| "不消费就报错" | ⏳ Step 2 | 现在直接报错会让默认扫描的 5 个模式跑不起来；须先统一计费（§3） |
| `exp.py` / `scripts/*.sh` 改引用协议名 | ⏳ Step 2 | 会改数字（NTT 76.3→50、带宽 941→563），属 Step 2 |

消费真相由**运行时内省**给出（`mode_consumption()`：读 `Register` 注册表，
穿过 `@torch.no_grad()` 包装、并跟随跨方法委托如
`cee_sd_opportunistic → adaptive_tridecoding`），注册表未填充时退化为
`STATIC_MODE_CONSUMPTION` 快照；`test/test_protocol_spec.py` 断言两者一致，
所以快照不会悄悄漂移。

实测自述（`--protocol paper_table5`，且故意用 CLI 覆盖 `--ntt_ms_edge_cloud`）：

```
[protocol] 口径自述：protocol=paper_table5
    通信数值   NTT edge_cloud=76.3ms / edge_end=0.317ms · 带宽 563/46/46 Mbps
    计费       round_trip='per_round' · charge_residual=True · topk_cap=16
    ⚠ 当前 eval_mode='dsd' [不消费] 上述计费开关：它们只被 adaptive_tridecoding,
      cee_sd, cee_sd_opportunistic 读取，本 run 的实际字节/往返由该方法内联实现决定
    投机深度   读取键=gamma · gamma=4
    ⚠ γ 规则未定（docs/protocol.md §4）：单-γ 与三级方法读的不是同一组键
    ⚠ CLI 覆盖了协议项 ⇒ 本 run 不属于该协议：ntt_ms_edge_cloud=76.3（协议值 50）
```

## 6. 死参数（L3）

**15 个已于 2026-09-29 删除**（设了无效、也不报错）：

- 从未被传过的 12 个：`--level`、`--guess`、`--max-token-span`、`--num-draft`、
  `--dtype_comm`、`--adaptive_debug_log`、`--controlled_eval_task`、
  `--controlled_topk_values`、`--controlled_topk_step`、
  `--controlled_entropy_quantile`、`--controlled_entropy_threshold`、
  `--controlled_max_high_entropy_states`；
- 被传但无人读的 3 个：`--window`、`--datastore-path`、`--task_name`。
  `--task_name` 有 17 处 call site（`exp.py` + 12 个 `scripts/*.sh` +
  `cmds/train_rl.sh`）却零处读取，全部一并清掉——留着定义而清 call site，
  或清定义而留 call site，都会让跑批当场报错或继续误导。

CLI 现有 103 个参数。参数按"谁消费它"的分层清单见 `docs/param_inventory.md`。

## 7. 现在就能用的命令（已符合 §2 的第 1-5、9、11、13 条）

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/accelerate launch --num_processes 1 \
  eval/eval_gsm8k.py --eval_mode <mode> -e <tag> \
  --gamma 3 --gamma1 3 --gamma2 3 \
  --data_path data --num_shots 3 --max_tokens 128 --eval_data_num 80 \
  --temp 0.0 --sample_seed 1234 --random_sample --num_samples_per_task 1 \
  --draft_model tiny-llama-1.1b --target_model llama-2-13b --little_model llama-68m \
  --edge_end_bandwidth 563 --edge_cloud_bandwidth 46 --cloud_end_bandwidth 46 \
  --ntt_ms_edge_cloud 50 --ntt_ms_edge_end 0.317 --batch_delay 0.05 \
  --comm_accounting honest --transfer_top_k 300 \
  --small_draft_threshold 0.6 --draft_target_threshold 0.7 --uncertainty_threshold 0.8 \
  --use_stochastic_comm --use_cuda_graph
```

注意：在 §3 实施完成前，`--comm_accounting honest` 只对 `adaptive_tridecoding` 生效，
所以这条命令跑出来的基线列**仍不是**同口径；γ 一栏也必须按 §4 的讨论结果改。
