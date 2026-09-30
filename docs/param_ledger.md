# 参数台账（谁决定数字、默认是什么、是否真的生效）

日期: 2026-09-29 · 生成方式: AST 解析 `src/utils.py` 的 `parse_arguments()`（117 个
`add_argument`）+ 全仓读取点扫描（排除 `SpecDec_pp/`）。目的：回答"参数太多、对不上"。

## 0. 参数有三个来源，规则不一样

| 来源 | 位置 | 规则 |
|---|---|---|
| CLI 默认 | `src/utils.py:248+` | 命令行没给就用它 |
| 扫描配置 | `exp.py:760-840` 的 `create_config(...)` | **布尔只在为 True 时才转发**（`exp.py:186-260` 的 `if config.get(x): add_args(...)`） |
| 命令字面量 | `exp.py:100-145` `cmd_temp` | 无条件写死，覆盖 CLI 默认 |

两个必须记住的陷阱：

1. **配置里写 `False` ≠ 传 `--no-x`**，而是"什么都不传、用 CLI 默认"。目前
   `use_early_stopping` / `use_precise` / `use_stochastic_comm` / `use_cuda_graph` /
   `random_sample` 都是 `store_true`（默认 False），所以恰好一致；一旦谁把某个 CLI
   默认改成 True，扫描配置里的 `False` 会被静默忽略。
2. `cmd_temp` 里的字面量：`--temp 0.0`（CLI 默认 0.2）、`-e llama`。

## 1. 默认扫描（`python exp.py`）的真实取值

来源：`exp.py:760-840` + `create_config` 的签名默认（`exp.py:525-570`）。

| 项 | 实际值 | 备注 |
|---|---|---|
| 模型 | `llama_series` | 其它序列全部注释掉 |
| 数据集 | mt_bench_noeval, gsm8k, humaneval | cnndm 注释掉 |
| eval_mode | dsd, dssd, cuhlm, cee_cuhlm, ceesd | **没有** cee_dsd/cee_dssd/adaptive_decoding/adaptive_tridecoding |
| edge_cloud_bandwidth | 46.0 Mbps | 列表里只有它没被注释 |
| edge_end_bandwidth | 941 | 563 那行被注释 |
| cloud_end_bandwidth | 46.0 | = edge_cloud |
| NTT | edge_cloud **76.3 ms** / edge_end 0.317 ms | `exp.py:98-99` 模块常量 |
| `--gamma` | **5** | 扫描没显式传 ⇒ 落到 `create_config(gamma=5)` 的默认 |
| `--gamma1` / `--gamma2` | 3 / 3 | 显式传；cee_cuhlm 被强制成 1/1 |
| transfer_top_k | 300 | |
| max_tokens / num_shots / eval_data_num | 128 / 3 / 80 | |
| random_sample / sample_seed | True / 1234 | |
| small_draft / draft_target / uncertainty 阈值 | 0.6 / 0.7 / 0.8 | 第三个没传 ⇒ CLI 默认 0.8 |
| 开关 | `use_stochastic_comm=True`、`use_cuda_graph=True`、`use_rl_adapter=True`、`disable_rl_update=True`、`use_early_stopping=False`、`use_precise=False` | 后两个是"不传=默认 False"，不是显式 False |
| batch_delay | 0.05 | |

### ⚠️ γ 口径：扫描里基线和三级方法不是同一个投机深度

按读取点（本轮精确扫描，含 `getattr` 形式）：

- 读 `--gamma`：`dist_spec`(dsd)、`dist_split_spec`(dssd)、`uncertainty_decoding`(cuhlm)、
  `adaptive_decoding`、engine 的 `speculative_decoding*`。
- 读 `--gamma1/--gamma2`：`tridecoding`、`ceesd_without_arp`、`adaptive_tridecoding`、
  `cee_cuhlm`、`cee_dsd`、`cee_dssd`。**这些方法完全不读 `--gamma`。**

于是默认扫描实际是：**dsd/dssd/cuhlm 跑 γ=5，ceesd 跑 γ1=γ2=3**。而 γ=5 是
`create_config` 的**签名默认值**（不是扫描里显式写的"最优值"），`gamma1/gamma2=3` 才是
显式选择。`scripts/align_paper_t5_gsm8k.sh:42` 同样传 `--gamma 5`，而该脚本头注释写的是
"基线 γ=3（R4 前向数与论文吻合）"——**注释与实参矛盾**（dsd/dssd/cuhlm 读的是 `gamma`，
拿到的是 5）。全仓没有任何后处理把 `gamma` 改写成 `gamma1`。

这一条足以解释"同一张表里数字对不上"的一大部分：基线列的 γ 与文档声称的不是同一个数。

## 2. 通信计费口径：四个开关的论文协议默认 vs 实际覆盖 ★

| 参数 | CLI 默认 | 语义 | 唯读取点 |
|---|---|---|---|
| `--comm_round_trip_mode` | `per_round` | 每轮每链路合并成一次往返 | `baselines.py:3507` |
| `--transfer_top_k_cap` | 16 | 给传输 top-k 设上限 | `baselines.py:3510` |
| `--charge_residual_payload` | True | 拒绝位置残差载荷计费 | `baselines.py:3386` |
| `--force_full_vocab_transfer` | False | 强制全词表传输 | `baselines.py:3366/3511` |
| `--comm_accounting` | None | 一次设前三项（honest/legacy） | `src/utils.py:1133-1151` |

**这四个开关只被 `adaptive_tridecoding` 读取**（连同 `coalesce_rounds`、`set_round`、
`flush_round` 也只在它内部：`baselines.py:3507/3586/4287`）。`src/engine.py` 零读取。

后果，按重要性排：

1. 默认扫描的 5 个 mode（dsd/dssd/cuhlm/cee_cuhlm/ceesd）**全都不读这四个开关** ⇒
   在扫描里 `per_round`、cap=16、残差计费**通通无效**，各方法只按自己内联的字节模型
   计费（`dsd:1501`/`adaptive_decoding:3282` 只记 `INT_SIZE`；`tridecoding:2126/2284/2286`、
   `ceesd_without_arp:2608/2748/2750`、`cee_dssd:5041/5170/5172` 每轮 2-3 次；
   `cee_cuhlm:4598/4681/4767` 计残差字节）。
2. `_charge_transfer` 是 `transfer_time += ntt` **逐次**累加（`communication.py:443-461`）⇒
   `per_transfer` 下"一轮 3.2+3.0 次调用"就是 3.2+3.0 倍 RTT。这是
   `docs/paper_table5_alignment.md` 里基线 Tcomm 45/53/102 ms/tok 与论文 14-17 的差距中
   **NTT 那一部分**的来源，而不只是它归因的"字节计费"。
3. `eval/utils.py:123-135` 会**无条件**把 `charge_residual_payload` / `transfer_top_k_cap`
   写进每次 run 的 metrics JSON ⇒ **run 的"口径标签"声称的值与实际执行的口径不一致**。
   这正好是 `--comm_accounting` 当初想解决的"口径不可辨"，现在以相反方向复现了。

好消息：`--prob_payload_bits < 16` 对不支持它的 mode 是显式 `parser.error`
（`src/utils.py:1125-1134`）。同样做法可以直接套到上面四个开关。

## 3. 论文表格对齐脚本 vs 默认扫描：两套参数并存

| 项 | `python exp.py` 默认扫描 | `scripts/align_paper_t5_gsm8k.sh` |
|---|---|---|
| NTT edge_cloud | 76.3 | **50** |
| edge_end_bandwidth | 941 | **563** |
| edge_cloud / cloud_end | 46 / 46 | 46 / 46 |
| `--comm_round_trip_mode` | 不传（默认 per_round） | 显式 per_round |
| `--charge_residual_payload` | 不传（默认 True） | 显式 **关**（`--no-...`） |
| `--transfer_top_k_cap` | 不传（默认 16） | 显式 **0** |
| 跑到的 mode | dsd/dssd/cuhlm/cee_cuhlm/ceesd | dsd/dssd/cuhlm/**adaptive_tridecoding** |
| γ | gamma 5 / gamma1,2 3 | gamma 5 / gamma1,2 3 或 5 或 16 |

两者只有 NTT(50)/带宽(563) 是"论文值"，其余差异来自默认值穿透。注意对齐脚本里的
"CEE-SD" 实际是 `adaptive_tridecoding`（`run ceesd_g5 adaptive_tridecoding 5 5`），
所以那两行才真正吃到了 per_round/无残差/cap0；三个基线行依旧 per_transfer。

## 4. 完全没有消费者的参数（15 个，已全部删除）

**第一批（12 个）**：`--level`、`--guess`、`--max-token-span`、`--num-draft`、
`--dtype_comm`、`--adaptive_debug_log`，以及整簇废弃的 `--controlled_*`
（6 个）。这一批从来没人传。

**第二批（3 个）**：`--window`、`--datastore-path`、`--task_name`。

`--task_name` 值得单说，因为它此前被**误判为"保留"**：它被 `exp.py`（每条扫描
命令都推导并传入）、12 个 `scripts/*.sh`、`cmds/train_rl.sh` 共 17 处传着，看起来
很"在用"。但全仓没有任何 `args.task_name` 读取——真正给 RL adapter 提供 task
one-hot 的是 `self.task`，由各评测类硬编码（`src/baselines.py:578` +
`eval/eval_gsm8k.py:80` 等）。**即：一个 17 处传参、零处读取的空操作。**

这也暴露了我先前扫描方法的漏洞：用模糊的 `\btask_name\b` 匹配时，`eval/eval_mixed.py`
的局部变量、`src/rl_adapter.py` 的函数形参都会被误当成消费者。改用精确的
`args.<dest>` / `getattr(args, "<dest>")` 模式后才查实（复现脚本见
`docs/param_inventory.md` 开头的生成方式）。

设置它们不会有任何效果，也不会报错。15 个全部删除；CLI 现有 **103** 个参数。
删除时一并清掉了 `--task_name` 的 17 处 call site——否则那些命令会直接
argparse 报错（`scripts/batch3.sh` 等每个都是）。回归测试：
`test/test_protocol_spec.py::TestCliSurface::test_removed_args_have_no_call_sites`，
已用变异测试验证它会失败。

### 4b. 顺带发现：`--help` 之前是崩的

`--use_cuda_graph` 与 `--rl_charge_queue` 的 help 里有未转义的 `%`，argparse 会对
help 做 `%`-格式化，于是 `--help` 直接 `ValueError: unsupported format character`
并退出码 1（HEAD 上就如此，非本次引入）。**110 个参数的 CLI 此前无法打印帮助**——
这大概是"参数太多又看不清"的一个隐藏原因。已修复，并加了回归测试
（`test/test_protocol_spec.py::test_help_strings_escape_percent`）。

### 4c. 幽灵模式

`MODE_FEATURES` 里有 `speculative_decoding_with_bandwidth_full_prob`，但**没有任何
解码实现注册**它（21 个注册方法里没有，全仓无引用）。它通过能力表校验却会在
`get_decoding_method()` 抛 `NotImplementedError`。两者留一：要么删条目，要么补实现。

## 5. 三种口径怎么跑

```bash
# A. 当前默认扫描（论文数字的现状来源）
python exp.py

# B. 论文 Table V 对齐行（NTT 50 / edge-end 563 / 显式 per_round + 遗留字节）
GPU=0 bash scripts/align_paper_t5_gsm8k.sh

# C. 口径标签（一次设定三个子开关，日志会回显）
--comm_accounting honest     # 残差计费 + per_round + cap16（论文协议，= 仓库默认）
--comm_accounting legacy     # --no-charge_residual_payload + per_transfer + cap0
```
注意 C 目前**只对 `adaptive_tridecoding` 生效**（见 §2）。

## 6. 四项决策的结果（2026-09-29）

1. **通信数值**：以 **NTT edge-cloud 50 ms + edge-end 563 Mbps** 为准（论文正文建模值）；
   默认扫描的 76.3/941 视为 Table II 实测值，只在敏感性分析里出现。
2. **基线计费**：**与 ours 完全同口径**（`per_round` + 残差计费 + cap 16 对所有方法生效）。
   基准即 `adaptive_tridecoding` 的实现；这要求把 §2 的四个开关接线到所有方法。
3. **投机深度 γ**：**待讨论**。过渡期规则：γ 一律显式传、run 启动时打印实际 γ 并标注
   "规则未定"（详见 `docs/protocol.md` §4）。
4. **15 个死参数**：全部删除，含曾被 17 处传参却无人读取的 `--task_name`
   （见 §4；`docs/protocol.md` §6 有清单）。

最终口径的冻结值、落地状态与强制机制见 **`docs/protocol.md`**。
