# 最终口径定义（`paper_table5` 协议）

日期: 2026-09-29 · 状态: 决策已定；**机制层已落地**（§5），**统一计费已于 2026-04
落地**（§3；CUHLM 系按 CU-HLM 论文自身口径计费，见 §3.1）
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
| 4 | 链路带宽 | edge-end **563**；edge-cloud / cloud-end 走**阶梯 [46, 10, 5] Mbps**（46 为 Table II 实测对齐点，10/5 为 2026-10 追加的带宽受限点，见下） | 已可 |
| 5 | NTT | edge-cloud **50** / edge-end 0.317 ms | 已可 |
| 6 | 往返口径 | `per_round`（每轮每链路一次往返） | **已落地**（CUHLM 系按论文口径，§3.1） |
| 7 | 残差计费 | 开（`charge_residual_payload=True`） | **已落地**（CUHLM 系按论文口径，§3.1） |
| 8 | top-k 压缩 | **按方法区分**：CEE-SD `300`（自身设计）；dsd/dssd `0`（原文无压缩 ⇒ 整词表载荷）；`cap=0` 不钳位 | **已落地**（2026-04 全表解钳；按方法区分见 §3 条目 3） |
| 9 | 随机带宽 | `--use_stochastic_comm`（配 trace 时同时推进 NTT） | 已可 |
| 10 | 投机深度规则 | **待讨论**（见 §4） | 待决策 |
| 11 | `use_early_stopping` | 关（生成长度须由 max_tokens 决定） | 已可 |
| 12 | RL adapter | 只按 `mode_features.py` 的规定挂载，**不作用于基线** | **需改代码/改扫描** |
| 13 | 统计口径 | warmup 实跑声明次数；失败样本整体剔除；重跑速度只统计本次行 | 已可（B29/B31/B35） |
| 14 | 图加速 | 开，且**必须成对报告**（图开/图关各一列或脚注） | 已可 |

第 12 条的依据：`docs/paper_table5_alignment.md` 记录，把 RL adapter 传给所有 mode 后
DSD 的 R_acce 从论文 32.11% 变成 60.6%，而前向数在 γ=3 时本来就吻合。基线挂 RL 会
改变其 top-k 行为，不属于同口径比较。

**条目 4 的带宽阶梯（2026-10-09 追加）**：46 Mbps 下 NTT（50 ms/轮）主导通信，
载荷是二阶量——llama/GSM8K 实测每轮载荷：dsd（全词表窗口）≈46 ms、dssd（拒绝
行下发）≈12 ms、tk_slt（K=320）≈1.2 ms、cuhlm ≈0.7 ms。压缩类与全词表类的
机制差异在 46 下几乎不可见。追加 10/5 Mbps 两个受限点（均在论文 Fig.4 的
5–25 Mbps 鲁棒性扫描范围内），按载荷 ∝ 1/BW 外推：@10 Mbps 时 dsd −45%、
dssd −16%、tk_slt −2%、cuhlm −1%。注意三点：① 带宽已进 `exp_name`
（`_bw46/_bw10/_bw5`）与 `RUN_IDENTITY_FIELDS`，阶梯点互不冲突；② 低于 5 会被
链路模拟器 `min_bandwidth_mbps=5` 钳位（`cmd_temp` 不透传该参数）；③ 低带宽下
dsd/dssd 的最优 γ 会下移（载荷 ∝ 草稿数/拒绝行数），各带宽的 γ 用
`scripts/gamma_sweep.py --bandwidth {10,5}` 重扫后再定。

## 3. 统一口径要做的事（对应 §2 的"需改代码"）—— **已于 2026-04 落地**

以 `adaptive_tridecoding` 的路径为基准，抽出并让所有方法复用：

1. **往返合并** ✅：`comm_simulator.coalesce_rounds = (comm_round_trip_mode == "per_round")`，
   每轮边界 `set_round(idx)`、解码结束 `flush_round()`。已接线 8 处：dsd/dssd/
   tridecoding/ceesd_without_arp/adaptive_decoding/cee_dssd/cee_dsd（`src/baselines.py`）
   + engine `speculative_decoding_with_bandwidth`。**CUHLM 系（uncertainty_decoding/
   cee_cuhlm）不接仓库统一开关，改按 CU-HLM 论文自身的口径计费（见下方小节）**。
2. **残差计费** ✅：统一走 `reject_residual_payload_bytes`（总量式，用于 legacy
   拒绝零计费的 dsd/cee_dsd）或 `reject_tail_scalar_bytes`（补差式，用于已按压缩行
   计费的 tri 系/adaptive_decoding/cee_dssd/engine sd），二者拼合出同一统一式；
   `dssd` 的整行 V×元素拒绝计费在 honest 下改为统一式（legacy 路径原样保留）。
   CUHLM 系例外：论文的 reject 在验证方基于已持有的分布重采样（论文式 17），
   不产生额外传输，故无残差计费项。
3. **top-k 上限** ✅（2026-04 修正为**全表不钳位**）：`apply_transfer_top_k_cap`
   （`src/proposal_utils.py`）的接线保留在所有方法里，但 honest/legacy 两档
   预设与 `--transfer_top_k_cap` 的 CLI 默认均为 0（不钳位）。原因：cap=16
   原是 `adaptive_tridecoding` 自己的方法设计（给 RL/自适应选出的 top-k 设
   上限、压低自家拒绝载荷），统一口径时曾把它升格为全表开关——但基线的
   proposal 也吃 `transfer_top_k`（top-k 重构提议），钳到 16 会改变基线的
   提议分布与接受率，那不是"同计费口径"该做的事。口径只管**怎么计费**
   （per_round + 残差），top-k 取多少是各方法自己的超参。要复现 ours 的
   历史钳位行为，显式传 `--transfer_top_k_cap 16`（会记一条协议偏离）。

   **2026-04 追加（按方法区分）**：这里此前只做对了一半——cap 解开了，但
   `exp.py` 仍把 `transfer_top_k=300` 一律传给所有方法，而 `transfer()`
   的 `is_compressed = (k is not None and k > 0)`、`reject_residual_payload_bytes`
   的 `0 < k < vocab` 都以它为准。于是基线也被按 top-300 计费，等于把
   "压缩传输"这项**本该被验证的贡献免费送给了基线**：
   - DSSD 原文（§3 / 式 8）：拒绝时下行是**整词表分布** `P_j(x)`，即 `|V|·bprob`；
   - DSD 原文：上行是 γ 个整词表分布（DSSD 论文 Alg.1 / 式 4）。

   现在按各自原文计费：`exp.py` 里 `TRANSFER_TOP_K_OURS=300`（CEE-SD）、
   `TRANSFER_TOP_K_PAPER=0`（dsd/dssd）。取 0 时 `proposal_top_k(0)` 返回
   `None`、`rebuild_topk_uniform_probs` 原样返回——这是**对齐**而非副作用：
   "top-k + 均匀尾部"的重建本身就是压缩机制的一环，原文没有。`temp=0` 下草稿
   token 取 argmax，top-300 与全量分布给出同一个 token，故 token 序列不变。
   CUHLM 仍走它自己论文的式 (5)（`comm_accounting='paper'`），不受该项影响。
   `transfer_top_k` 也因此从 `paper_table5` 的冻结值里移出（单个全局值表达不了
   按方法区分的取值）。
4. **拒绝路径的载荷** ✅：统一为 `k*(4+元素大小)+元素大小` 字节/拒绝位置/链路
   （无有效 top-k 压缩时为整行 `V×元素`——与 ours 的 `_residual_payload_bytes`
   口径一致）。字节恒等式 `k×元素 + (k×4+元素) == k*(4+元素)+元素` 由
   `test/test_unified_comm_accounting.py` 锁定。

**legacy 口径逐位可复现**：cap=0 / per_transfer / charge_residual=False 时所有
新代码路径都是纯透传或不可达分支（`test_unified_comm_accounting.py` 的透传测试）。
消费面由 `_STATIC_ACCOUNTING` 快照 + 运行时内省锁定（`test_protocol_spec.py`）。

改完后，`--comm_accounting honest` 对**除 CUHLM 系外的全部方法**等于"同口径"，
`legacy` 等于"复现 2026-09-24 前的历史数字"。CUHLM 系的口径标签在
`eval/utils.py` 里如实记为 `paper`（论文自身口径），无通信模式记 `n/a`——
标称值不再冒充实际行为。

### 3.1 CUHLM 系：按 CU-HLM 论文自身的口径计费（2026-04 落地）

CUHLM 系（`uncertainty_decoding`/`cuhlm`、`cee_cuhlm`）是**有原文可依的已发表
方法**，其计费口径以论文为准，而不是仓库统一开关。落地映射
（`src/communication.cuhlm_uplink_payload_bytes` + `src/baselines.py` 各 F-CUHLM
标注点）。**2026-10-09 修订（§3.4）**：载荷字节仍按论文，但**往返次数**已接入
统一开关 `comm_round_trip_mode`（标签 `paper_rt`）；本节描述的是字节口径。

| 论文条目 | 论文依据 | 仓库实现 |
|---|---|---|
| 上行载荷 = `k·(b_prob+b_index)` bits | 式 (5)；`b_prob=8`（§V 仿真参数）、`b_index=⌈log₂V⌉`（§II-B 二进制编码） | `cuhlm_uplink_payload_bytes(k, V)`：V=32000 时 b_index=15；k=|V| 时 92000 B，正是论文摘要的 "92kB/token"；k*=30 时 86.25 B（论文 "<0.1%"） |
| draft/响应/重同步 token 索引 negligible | §II-B "cost incurred by transmitting token indices is negligible"；§III-B Step 5 重同步 "omitted from the cost analysis" | token 索引一律 0 字节（`_send_downlink_index_only`：0 字节报文，保留一次 NTT） |
| 跳过 = 零通信、零云端计算 | Algorithm 1 line 18-20 | 跳过分支无任何计费调用（cee_cuhlm 旧口径每跳过 token 收 8B + accept 消息 2 次 NTT，已废除） |
| 触发 = 一次上行 | Algorithm 1 line 10 | 一次 `simulate_transfer`（k(t) 为论文式自适应量）；cee_cuhlm 的逐轮 token 同步随触发上行捎带，不再单独上行 |
| reject 重采样在验证方 | 式 (16)/(17) | 无额外传输（区别于仓库统一口径的 reject 残差计费） |

已知模型差异（**链路模型**差异，非字节口径差异）：论文的时延模型是 Shannon
容量（式 (6)，无 per-message 固定成本、无下行项）；仓库链路模型要求设备拿到
响应 token，故下行方向有真实报文。2026-10-09 之前：每个物理报文各收一次
NTT（每触发 2×NTT）；之后（§3.4）：per_round 下 0 字节下行并入触发的合并
往返（每触发 1×NTT），`per_transfer` 仍可回到逐报文口径做敏感性分析。
首轮 prompt 上传沿用全仓一次性约定（论文不建模 prompt）。

回归锁定：`test/test_unified_comm_accounting.py::TestCUHLMPaperPayload`（公式与
92kB 交叉验证）、`test/test_uncertainty_decoding.py`（触发上行字节 + 0 字节下行）、
`test/test_cee_refactor.py::test_cee_cuhlm_bills_cuhlm_paper_payload`（两层完整
字节序列）。

### 3.2 CUHLM 的不确定度阈值：从不退化的工作点（2026-04 追加）

`uncertainty_threshold` 是 **CUHLM 的工作点**（越小越常上云：越准、越慢），不是
全表共享的 L0，因此从 `paper_table5` 的冻结值里移出，由 `exp.py` 按方法给。

- 主表里**只有 CUHLM 消费它**。CEE-SD 也读该参数，但只在
  `cee_sd_opportunistic` 变体（`baselines.py` 的 `_opportunistic_first_stage`
  分支），`adaptive_tridecoding` 走不到，故主表不受影响。
- 默认值 0.8 把 CUHLM 放在一个**退化工作点**：Llama-2-13b/GSM8K 上准确率
  **0.0**，而同期 CEE-SD 0.2375、DSD/DSSD 0.25。它的"快 5.9 倍"几乎全部来自
  "几乎不调用云模型"（target forward 仅 141 次 vs CEE-SD 2759 次），这也与论文
  Table V 自己标注的 **"baselines ≈target"** 直接矛盾。
- 新取值 `exp.py: UNCERTAINTY_THRESHOLD_CUHLM = 0.08`，依据是仓库实测的阈值前沿
  （Llama-2-13b/GSM8K）：

  | thr | 0.08 | 0.30 | 0.50 | 0.80 |
  |---|---|---|---|---|
  | GSM8K acc | **0.25** | 0.1375 | 0.0375 | 0.0 |

  0.08 是**等准确率工作点**（≈target 水平，与 DSD/DSSD 相同、略高于 CEE-SD），
  这样 CUHLM 才是在可比质量下参与吞吐比较，而不是拿质量换速度。
- **待办**：Qwen1.5 / Qwen3 两个系列还没有阈值前沿，0.08 是按同一判据外推的；
  重跑后需核对这两个系列 CUHLM 的准确率是否与其他方法同量级，不同则各自标定。
- 回归锁定：`test/test_exp_resume_merge.py::TestPerModeTransferTopK`
  （取值与按方法分发的断言）。

### 3.3 TK-SLT 系：按其论文自身的口径计费（2026-10 落地）

TK-SLT 系（`tk_slt`/`tkslt`，"Communication-Efficient Collaborative LLM Inference
via Distributed Speculative Decoding"，WCSP'25 Zheng & Yang）与 CUHLM 系同待遇：
**有原文可依的已发表方法**，计费以论文为准，不接仓库统一开关的**载荷部分**
（**2026-10-09 修订（§3.4）**：往返次数已接入 `comm_round_trip_mode`，标签从
`paper` 升级 `paper_rt`）。落地映射
（`src/communication.tk_slt_uplink_payload_bytes` + `src/baselines.py` 的
`tk_slt`）：

| 论文条目 | 论文依据 | 仓库实现 |
|---|---|---|
| 草稿分布 = softmax 只作用于 top-K logits（采样自该稀疏分布） | §III Solution 1 | 草稿缓存 `KVCacheModel(draft, temp, top_k=K, top_p)`：`norm_logits` 的 top-k 过滤即稀疏化，`prob_history` 行 = 稀疏分布（temp=0 时 one-hot，行为与其它基线一致） |
| 上行载荷 = `γ·K·b_prob` bits，`b_prob=16`（FP16） | 式 (2) + §VI-B "logits are quantized to half-precision (FP16) and transmitted" | `tk_slt_uplink_payload_bytes(K, γ, V)`：K=320、γ=3 ⇒ 1920 B；K≤0/None ⇒ 整词表 `γ·V·2`（V=32000 即 64000 B/token = 512 kbit，论文 §I "about 500 kbit per token"） |
| token 索引 negligible（草稿 token id、K 个词表索引、下行 (x_j, j)） | §II-B "index size is insignificant … only the uplink transmission latency associated with the vocabulary distribution" | 0 字节报文（`_send_downlink_index_only`：保留一次 NTT）；交叉验证见下 |
| 概率值真实 FP16 量化 | §VI-B 的通信模式 | `_quantize_probs_fp16`：fp16 roundtrip + 按行重归一化 ⇒ 验证判据与计费看到**同一个** q̂（B15/B16 原则）；temp=0 的 one-hot 值逐位不变 |
| 拒绝重采样在验证方、基于稀疏 Q̂ | §II-A 3b：`norm(max(0, P−Q))` | `sample_reject_token(P, Q̂)`：稀疏 Q̂ ⇒ 非 top-K 位置拿到完整 P；无额外传输（同 CUHLM，区别于统一口径的 reject 残差计费） |

**索引不计字节的交叉验证**（Table II 逐位吻合）：论文实测 c≈0.07、b_full≈0.23
（L=0.300 的两项分解），Table II 的 L 满足 `L(K) = c + b_full·(K/32000)`：
K=3→0.0700、32→0.0702、320→0.0723、3200→0.093、32000→0.300——即上行载荷严格
∝ `K·b_prob`、不含索引项（若计 ⌈log₂V⌉ bits 索引，K=320 应得 ≈0.0839 而非
0.0723）。

与 CUHLM 相同的**链路模型差异**（非字节口径差异）：论文时延模型只计上行分布的
发射时长（式 (3)，无 per-message 成本、无下行项）；仓库链路模型要求下行方向
有真实报文（结果 token + 位置 j）。2026-10-09 之前：逐报文各收一次 NTT（每轮
2×NTT）；之后（§3.4）：per_round 下 0 字节下行并入轮末合并往返（每轮 1×NTT）。
首轮 prompt 上传沿用全仓一次性约定（论文不建模 prompt）。AS²/ODLD（论文 §V
Algorithm 1/2）是**可选开关**
`--tk_slt_odld`（默认关 = 固定 γ，与其它基线同口径可比）：在线估计
α̂/b̂/ĉ（接受率、上行纯发射时长/T_LLM、草稿单 token 时长/T_LLM 的运行均值），
按 Theorem 2 的 Lambert-W 闭式逐轮取 γ*；S*≤1 的轮次退回 standalone LLM
（target 直出、无上行分布，只回 0 字节下行）。

主表取值：`exp.py: TRANSFER_TOP_K_TKSLT = 320`（论文 §VI-B Fig 4 的最优
工作点，T=0/T=1 两个温度下最大加速比都在 K=320 取得）。draft 模型用端侧小模型
（论文 §VI-B：68M 草稿 + 7B 验证）。

回归锁定：`test/test_tk_slt.py`（解码循环字节序列、FP16 量化进入验证判据、
ODLD/AS² 接线、Table I 逐格复现）、
`test/test_unified_comm_accounting.py::TestTkSltPaperPayload`（字节公式、
500 kbit 与 Table II 交叉验证）与 `TestAccountingLabelTruth`（paper 标签）。

### 3.4 统一往返计费 + 流体带宽模型（2026-10-09 落地）

两项网络仿真口径修正。动机：46 Mbps 主矩阵实测发现 (a) dsd/dssd/CEE-SD 走
per_round 合并（每轮 1×NTT），而 tk_slt/cuhlm 按论文逐报文计费（每轮 2×NTT
= 上行 50ms + 下行 50ms）——同一主表内存在纯记账差异，tk_slt 合并后实测
**+31% 吞吐**；(b) instant 带宽模型把整条载荷按**起始时刻的瞬时 trace 采样**
计费（Jensen 偏差 E[S/B] > S/E[B]），5 Mbps 档一个整词表窗口的排空要
~600ms、横跨 3+ 个 0.2s 采样间隔，单采样冻结既失真又系统性多扣。

**① 统一往返（NTT 语义：1×NTT(50ms) = 一次完整云请求往返）。**
`comm_round_trip_mode=per_round`（paper_table5 冻结值）现在对**所有方法**生效：
tk_slt 的"上行载荷 + 0B 下行"在轮末 `flush_round()` 合并成一次往返（cuhlm
按触发边界同理：每次云端交互 1×NTT）。载荷字节**不变**——两系仍按各自论文
公式计（`tk_slt_uplink_payload_bytes` / `cuhlm_uplink_payload_bytes`），变的
只有往返次数。标签从 `paper` 升级为 **`paper_rt`**（论文载荷 + 统一往返）；
`cee_cuhlm`（三级变体，不在主矩阵）暂未接线，保持 `paper`。敏感性分析仍可
`--comm_round_trip_mode per_transfer` 回到逐报文口径（每轮 2×NTT）。
tk_slt 的 ODLD 读数（`_last_edge_cloud_tx_seconds`）随之按模式分流：
per_transfer 读上行直充后的 stats 尾，per_round 读轮末 flush 后的合并单元。

**② 流体带宽模型（`--comm_bw_model fluid`，paper_table5 冻结值）。**
`_charge_transfer` 的流体排水：载荷从**连续仿真时钟**的当前时刻起，按
trace 逐 0.2s 间隔积分排空（每采样先过 `min_bandwidth_mbps` 地板），时钟
按"排空时长 + NTT"推进（模 trace 全周期），游标取 `int(时钟/间隔)`；stats
的带宽历史记**有效排水速率** S/tx_time（ODLD 的 b̂ 估计口径）。连续时钟是
预实验踩坑后的设计（两个已修复的 bug）：小载荷虽在单间隔内排空，时钟仍
前进——否则 trace 冻结在起始采样上（dsd 通信时间虚高 +140%）；时钟贴着
间隔边界时必须按下一间隔完整容量计——否则浮点 `t % interval` 返回 ≈interval
得到零容量，排水原地空转直到 `max_intervals` 兜底按 1B/s 计费（通信时间
虚高百万秒）。小载荷（< 采样间隔 × 速率）与 instant 同值；46 Mbps 下载荷
1-12ms，数字几乎不动（实测 ±2-8%：时间游标与取整游标的采样差异），**5
Mbps 档通信下修**（实测：dsd −19.6%/dssd −8.6%，吞吐 +19.5%/+4.1%——
此前的外推 −65%/−31% 高估了：两种模型都会吃到深衰采样，流体只是不再
多扣 Jensen 偏差项）。`instant` 保留为
CLI 默认，历史直跑逐位可复现。配套地板缩放：主矩阵/扫描按
`min_bandwidth_mbps = max(0.5, 带宽/10)` 传（固定 5 会把 5 Mbps 档的 trace
削成近似常数）。

**影响面**：`comm_bw_model`/`min_bandwidth_mbps` 进 `RUN_IDENTITY_FIELDS`，
旧（instant/2×NTT）与新（fluid/1×NTT）run 身份不互撞、resume 不混。**所有
stochastic edge_cloud 的既有结果（主矩阵已跑格 + γ 扫描 42 格）通信数字按
新口径全部重跑**；γ 扫描的新汇总走 `_fluid` 后缀文件
（`gamma_sweep_..._fluid.json`），历史无后缀文件 = instant 口径存档。
`scripts/gamma_sweep.py` 显式传 fluid（cmd_temp 显式传 `--comm_bw_model`，
`apply_protocol` 只填未显式设置的项，不显式传会被 create_config 默认 instant
压过协议冻结）。低带宽档的 γ* 会左移（dsd/dssd 尤其），46 Mbps 档 tk_slt
γ* 右移（每轮省 50ms 使长草稿更划算；instant 口径下 γ=16 仍在爬升，
12.88 tok/s @γ=16 vs 12.68 @γ=8）。

回归锁定：`test/test_comm_simulation.py::TestFluidBwModel`（单/跨间隔排水、
地板、零字节、非随机回退、有效速率记录、时钟推进不冻结、
浮点边界不停滞、60 轮长跑无爆炸）、
`test/test_unified_comm_accounting.py::TestUnifiedRoundTrip`（per_round 恰好
1×NTT/轮、字节中立、协议冻结 fluid）、`test/test_protocol_spec.py`
（内省快照：tk_slt/cuhlm 消费 `comm_round_trip_mode` 且仅此一个开关）。

**A/B 实测**（`scripts/pilot_ab_network.py`，llama/gsm8k 20 样本，
`experiment_results/pilot_ab_network.md`）：46 Mbps 下 dsd/dssd 仅 ±2-8%
（游标采样差异，符合预期）；tk_slt/cuhlm 通信 −42~50%、吞吐 **+26~33%**
（NTT 合并主导，往返数精确 2:1——1968→974）；5 Mbps 下 fluid 再给
dsd −19.6%/dssd −8.6% 的通信下修（吞吐 +19.5%/+4.1%）；所有格两臂
accuracy 逐位一致、纯计算时长匹配 ≤2%（口径不改解码的完整性检查通过）。

**离线重放（`scripts/rebill.py`，2026-10-09 追加）**：A/B 验证的"口径
不改解码"直接兑现成基础设施——每次跑落 `comm_trace_edge_cloud`
（`TransferUnit` 逐消息 [字节, 轮号]；`src/metrics.py` 注册、
`eval/utils.py` 默认落盘），带宽阶梯/NTT 敏感性/口径对照不再占 GPU：

```
.venv/bin/python scripts/rebill.py exp/.../*_metrics.json --bandwidth 5
.venv/bin/python scripts/rebill.py exp/a.json --ntt-ms 76.36        # Table II 档
.venv/bin/python scripts/rebill.py exp/a.json --bw-model instant \
    --round-trip per_transfer                                       # 口径对照
```

重放语义（端到端验证：46→5 Mbps 跨带宽重放与真实 5 Mbps 跑的通信时间
**逐位一致** 51.6242s，吞吐差 0.5% = 两次真实跑的 CUDA 计时噪声）：样本
边界 = 轮号回退处**重建模拟器**（真实实验逐样本构造，连续时钟不跨样本
延续——不归零会让后续样本排水相位偏移 ~1%）；样本内轮号相等同组、变化
切组；`wall = elapsed + comm + queue` 重算，`queue = target_forward_times ×
batch_delay`。**边界**：① 有通信反馈的方法（RL 适配器开启的 CEE-SD、
ODLD 开启的 tk_slt）解码依赖网络，重放只是"同一动作序列在新网络下的
成本"近似；② γ/数据集/模型是解码参数，换了必须重跑；③ 只重放
edge_cloud（矩阵基线只用它）。回归锁定：
`test/test_comm_simulation.py::TestCommTraceReplay*`（双口径重放等价、
轮号标注、样本边界时钟归零）。

## 4. 唯一待定项：投机深度（γ）

现状：`--gamma` 只被 `dist_spec`/`dist_split_spec`/`uncertainty_decoding`/
`tk_slt`/`adaptive_decoding`/engine 的 speculative 读取；三级方法只读 `gamma1`/`gamma2`。
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
| "不消费就报错" | ⏳ 剩 CUHLM 系 | §3 落地后只剩 CUHLM 系不消费仓库开关（它按论文自身口径计费，§3.1，是有意为之而非缺口）；其余模式已可启用该检查 |
| `exp.py` / `scripts/*.sh` 改引用协议名 | ⏳ Step 2 | 会改数字（NTT 76.3→50、带宽 941→563），属 Step 2 |

消费真相由**运行时内省**给出（`mode_consumption()`：读 `Register` 注册表，
穿过 `@torch.no_grad()` 包装、并跟随跨方法委托如
`cee_sd_opportunistic → adaptive_tridecoding`），注册表未填充时退化为
`_STATIC_ACCOUNTING` 快照；`test/test_protocol_spec.py` 断言两者一致，
所以快照不会悄悄漂移。

实测自述（`--protocol paper_table5`，且故意用 CLI 覆盖 `--ntt_ms_edge_cloud`）：

```
[protocol] 口径自述：protocol=paper_table5
    通信数值   NTT edge_cloud=76.3ms / edge_end=0.317ms · 带宽 563/46/46 Mbps
    计费       round_trip='per_round' · charge_residual=True · topk_cap=0
    · eval_mode='dsd' 消费上述计费开关（统一口径已接线，§3）
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

注意：§3 已落地，`--comm_accounting honest` 对除 CUHLM 系（`uncertainty_decoding`/
`cuhlm`/`cee_cuhlm`，按 CU-HLM 论文自身口径计费，见 §3.1）外的全部方法同口径
生效；CUHLM 系的口径标签如实记为 `paper`。γ 一栏仍须按 §4 的讨论结果显式传。
