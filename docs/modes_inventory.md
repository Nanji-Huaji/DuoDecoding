# 口径与模式总盘点（2026-10-10 快照，2026-10-19 更新 L0 清单）

> 互补文档：`docs/param_inventory.md` 按**旗标**清点（108 个参数谁在消费）；
> 本文按**模式轴**清点——每条轴有哪些取值、默认是什么、当前论文口径用哪个、
> 哪些是历史遗留/消融专用。回答"口径和模式到底有多少"。

## 0. 分层总览

| 层 | 轴数 | 一句话 |
|---|---|---|
| L0 解码方法 | 1（23 个注册名 / 16 个函数） | 跑哪个算法 |
| L1 通信数值 | ~6 | 链路多快、多抖 |
| L2 计费口径 | 4（捆绑成 4 协议） | 字节/时间怎么记账 |
| L3 方法内参数 | ~6 | γ、top-k、阈值、ODLD |
| L4 RL 训练 | ~15 | 动作空间、奖励、课程 |
| L5 评测采样 | ~7 | 样本、温度、shot |
| L6 回放 | 3 | 真跑 / 位精确重放 / 近似预览 |
| L7 checkpoint | 5 层 | 权重存在哪、哪一版 |

## 1. L2 计费口径（最核心，2026-10-09 冻结）

| 轴 | 取值 | 默认 | 主表用 | 状态 |
|---|---|---|---|---|
| `comm_bw_model` | fluid / instant | instant | **fluid** | instant 留作对照 |
| `comm_round_trip_mode` | per_round / per_transfer | per_transfer | **per_round** | per_transfer=legacy |
| `charge_residual_payload` | bool | False | **True** | 拒绝时残差计费 |
| `transfer_top_k_cap` | int | 0 | **0**（不钳位） | 2026-04 全表解钳 |

**4 个协议名**（`src/protocols.py`，`--protocol`）把上面的轴捆成预设：

| 协议 | 内容 | 用途 |
|---|---|---|
| `paper_table5` | 上表 4 轴 + 全部通信数值 + 推理设置（temp 0 / 128 tok / 3-shot / 80 样本 / random seed 1234 / 阈值 0.6/0.7 / stochastic comm） | **论文主表唯一入口** |
| `honest` | 只收敛 4 个计费轴 | 只换账本的对照 |
| `legacy` | 不收残差 + per_transfer | 复现 2026-09-24 前数字 |
| `smoke` | 2 样本 / 32 tok / 0-shot | 联调冒烟，不出表 |

注意：`--comm_accounting honest` 与 `--protocol paper_table5` 在 utils.py
后解析收敛到同一套计费轴；bw_model 协议内冻结为 fluid（有断言）。
`transfer_top_k` **故意不进协议**——按方法区分（见 §3）。

## 2. L1 通信数值

| 项 | 主表值 | 说明 |
|---|---|---|
| 带宽三元组 | edge_end 563 / edge_cloud 46 / cloud_end 46 Mbps | Table II 的 76.3/941 留作敏感性分析 |
| NTT | edge_cloud 50ms / edge_end 0.317ms | |
| batch_delay | 50ms | 排队=每次 target 前向 |
| min_bandwidth_mbps | bw/10（下限 0.5） | 随机 trace 钳位 |
| `use_stochastic_comm` | True | ±带宽抖动 trace |
| `comm_trace_mode` | static | 另有 driving/walking（5G 车载，实验性） |
| `stochastic_ntt` + trace 文件 | 关 | NTT 也可走 trace（实验性） |

## 3. L0 解码方法（`@Register.register_decoding`）

2026-10-19 按注册表实测重写：**23 个注册名 / 16 个函数**（旧版只列了 9 个，
漏了 CEE+ 组合与 `adaptive_decoding`）。`status` 是语义定位，不是代码成熟度：

| 函数 | 注册名 / 别名 | status | 主表 | 备注 |
|---|---|---|---|---|
| `dist_spec` | `dsd`, `dist_spec` | baseline | ✓ | transfer_top_k=**0**（原文无压缩） |
| `dist_split_spec` | `dssd`, `dist_split_spec` | baseline | ✓ | top_k=0 |
| `uncertainty_decoding` | `cuhlm`, `uncertainty_decoding` | baseline | ✓ | top_k=300；工作点阈值 0.08（`exp.py` 常量） |
| `tk_slt` | `tkslt`, `tk_slt` | baseline | ✓ | top_k=**320**；`--tk_slt_odld`（实验性） |
| `adaptive_tridecoding` | `cee_sd`, `adaptive_tridecoding` | **core** | ✓（CEE-SD） | top_k=300；γ1/γ2 见 §4 |
| `target_only` | `target_only` | reference | 对照 | 纯目标模型 |
| `cee_cuhlm` | `cee_cuhlm` | combination（Table VI CEE+） | — | 经 `--extra-modes` 附载 |
| `cee_dsd` | `cee_dsd` | combination（Table VI CEE+） | — | 同上 |
| `cee_dssd` | `cee_dssd` | combination（Table VI CEE+） | — | 同上 |
| `cee_sd_opportunistic` | `cee_sd_opportunistic` | variant（真消融） | — | CEE 循环 + little 策略开关；**转发参数**的正确形态 |
| `tridecoding` | `tridecoding` | **ablation（Table VIII Static，保留）** | — | 静态三级消融（无 RL/无 ARP）。当前是旧验证路径的实现；若要让消融与主方法同源，需在 `cee_sd` 上实现 `_ablation="static"` 并重跑该表 |
| `ceesd_without_arp` | `ceesd_w/o_arp`, `ceesd_without_arp` | **legacy（已归档）** | — | 旧验证路径；已从 `exp.py` 矩阵移除，CLI 直跑会打印 `[legacy]` 告警 |
| `adaptive_decoding` | `adaptive_decoding` | **独立方法（待定）** | — | 2 模型 + ARP，不是 CEE 消融；论文不报则归档 |
| `autoregressive_sampling`（engine） | `small`, `large` | reference | — | 自回归基线 |
| `speculative_decoding`（engine） | `sd` | baseline | — | 标准投机解码 |
| `speculative_decoding_with_bandwidth`（engine） | 同名 | legacy（engine） | — | 历史方法，按带宽建模；保留用于对照 |

> 判定"消融"的标准：**同一个实现上关掉某个模块**（如 `cee_sd_opportunistic`）。
> 换了一条代码血缘的实现不能自称消融——那测的是"另一个历史实现"。

## 4. L3 γ 体系（三种语义，别混）

| 轴 | 谁用 | 语义 |
|---|---|---|
| `--gamma` | dsd/dssd/cuhlm/tk_slt | 每轮投机深度，固定 |
| `--gamma1/--gamma2` | CEE-SD 三级 | draft→target / little→draft 窗口**初值**（topk_thr 下 agent 不控窗口，γ1 固定生效） |
| `rl_gamma_candidates` | 仅 topk_thr_gamma | γ 作为动作（实验动作空间） |

主表 γ* 分档（exp.py `GAMMA_BY_MODE_BANDWIDTH`，2026-10-10 曲线验证）：
dsd {46:3, 10:1, 5:1}，dssd/tk_slt 8，cuhlm 3（γ 盲），cee_sD γ1=γ2=5。
**已知问题**：训练用 γ1=γ2=16、评测 5（历史配方沿用）；有效深度实际由
ARP 阈值早停决定（见 docs 内 ceesd_gamma_threshold_probe.md）。

## 5. L4 RL 层（最大的一层，15+ 轴）

### 动作空间与动作格
| 轴 | 取值 | 主表/生产 |
|---|---|---|
| `rl_action_space` | topk_thr / topk_thr_gamma | **topk_thr**（γ 不归 agent） |
| 阈值格 | registry per-role：main [0.05..0.4]，little [0.4..0.9]；`--rl_threshold_candidates` 覆盖；**随 checkpoint 持久化**（10-10 新增） | v3 重训用 0.1..0.99 |
| top-k 格 | TOPK_CANDIDATES 11 档 {1..1024}（=上行压缩候选） | |
| `rl_factored_q` | bool | True（分支决斗 Q 头） |

### 奖励（5 模式 × 4 成本开关）
| 轴 | 取值 | 生产 |
|---|---|---|
| `rl_reward_mode` | legacy / linear / lagrangian / slo / energy | **lagrangian**（λ=12） |
| `rl_comm_cost_mode` | **nominal**（10-10 新默认）/ settled（旧 bug 复现）/ off | nominal |
| `rl_charge_queue` | bool | v3 起 True |
| `rl_byte_price` | float | 0（物理时间已含） |
| `rl_compute_time_mode` | wall / model | **model**（configs/compute_cost_model.json，主机无关） |

### 状态与课程
| 轴 | 取值 | 生产 |
|---|---|---|
| `state_bw_scaling` | linear / log | log |
| `state_latency_scaling` | linear / centi / log | centi |
| curriculum bw | 0.5-50 Mbps loguniform | |
| curriculum ntt | 0→100ms 爬坡 | |
| `rl_init_strategy` | fresh / resume | v3 用 **fresh**（旧格 checkpoint 在新格下会被重标） |

### 消融钉死（只做反事实实验，不出表）
`rl_force_threshold` / `rl_force_threshold_little` / `rl_force_topk` /
`rl_force_gamma` + `arp_stop_mode`（cumulative=历史且惰性 F18 /
**per_token**=有效）。

## 6. L5 评测采样

temp **0**（重放有效性的前提）/ 3-shot / 80 样本 / `random_sample` seed
1234 / 数据集 {gsm8k, mt_bench_noeval, humaneval} / 系列 {llama, qwen1.5,
qwen3} × 带宽 {46, 10, 5}。`use_cuda_graph` 协议不冻结（成对报告）。
`use_early_stopping=False`（EOS 早停口径全表一致）。

## 7. L6 回放体系（本 session 建立）

| 层级 | 工具 | 精度 | 有效性 |
|---|---|---|---|
| 真跑 | eval_*.py + GPU | — | 一切 |
| 位精确重放 | `scripts/rebill.py`（metrics 里 `comm_trace_edge_cloud`，10-09 起必落盘） | bit-exact | **基线四方法**（temp 0、无 RL 反馈 ⇒ 解码与网络无关） |
| 近似预览 | 均匀字节 + 名义成本 | ≤1%（12 锚点实测） | 同上 |
| **CEE-SD** | — | — | **不可重放**（动作依赖网络状态），逐带宽真跑 |

10-08 前的旧工件无 trace ⇒ 只能做预览级分析。

## 8. L7 checkpoint 层次

| 层 | 路径 | 内容 |
|---|---|---|
| 生产 | `checkpoints/rl_agents/{main,little}/<pair>/{latest,best}.pth` | v2 流体重训后晋升（矩阵在用） |
| v2 重训源 | `checkpoints/rl_fluid_retrain/{llama,qwen15,qwen3}/` | 旧格 + comm=0 奖励 |
| **v3 重训** | `checkpoints/rl_fluid_retrain_v3/` | 新格 0.1-0.99 + nominal comm + queue（10-10 进行中） |
| instant 备份 | `checkpoints/rl_agents_instant_20261009_2336/` | 旧口径全量备份 |
| acc heads | `src/SpecDec_pp/checkpoints/acc_head/<pair>/exp-weight6-layer3` | 3 个模型对 |

附：`.buffer` 回放缓冲 sidecar；best/latest 双档（`prefer_latest` 语义）；
**10-10 起动作格随 checkpoint 持久化**（旧 checkpoint 无此字段 → registry
默认格兜底，这也是 v3 必须 fresh 的原因）。

## 9. 当前论文主表 = 这一张清单

```
--protocol paper_table5        # 计费 4 轴 + 数值 + 推理设置全冻结
--comm_bw_model fluid          # 协议内断言
--comm_accounting honest       # = per_round + residual + cap0
γ* 分档（exp.py 表）            # dsd 3/1/1, dssd 8, tk_slt 8, cuhlm 3, ceesd 5
transfer_top_k 按方法           # 0/0/320/300/300
CUDA_VISIBLE_DEVICES=1 + 4bit target + bf16 + cuda graph
```

## 10. 已知坑（跨轴交互）

1. **旧 checkpoint × 新格 = 静默重标**（v3 fresh 的原因；新 checkpoint 已防）
2. settled 奖励口径在 fluid+per_round 下恒 0（根因①，已修，默认 nominal）
3. cumulative 停止规则让阈值维度惰性（F18），必须 per_token
4. per_transfer 方法的 connect_times 数的是传输不是轮（旧扫描 rounds=connect/2）
5. exp 目录名 `g13_g23` 是 γ1/γ2 模板，单 γ 方法也带（纯装饰）
6. humaneval 准确率两个时代都是 0.0（未解）；qwen3 gsm8k 0.05（chat 模板，
   方法间一致）
7. CEE-SD 跨带宽重放无效；基线可重放但旧工件（<10-09）无 trace
