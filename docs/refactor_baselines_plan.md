# baselines.py 重构计划

分支：`refactor/baselines`（worktree `.refactor/wt`，与主树运行区隔离）
基线：f232c0e + 运行态快照（campaign 实际执行的代码，行为完全对齐）
状态：计划已定，Phase 0 待启动

---

## 0. 背景：为什么动它

`src/baselines.py` 6831 行，占业务代码约 36%。解剖：

| 成分 | 行数 | 判断 |
|---|---|---|
| 30 个模块级 helper（载荷公式/量化/tk_slt 数学/metrics 拼装） | ~620 | 健康，是新模块的坯子 |
| `Baselines.__init__` + 缓存获取 + acc_head | ~460 | 已被 MODE_FEATURES 收敛过一轮 |
| 11 个解码方法 | ~5000 | 病灶 |

病灶的两个组成部分：

1. **骨架复制**：每个方法都是同一骨架的拷贝（构造 simulator → 获取 caches →
   while 循环 + `_stop_at_eos` → verify/rollback/accept → 40 行 metrics 结算）。
   simulator 构造 10 份近似拷贝、metrics 结算 7 份近同。报告 B5/B8/B9/B10/
   B15/B17/B18/B19 全部分叉于此——修一处漏九处。
2. **口径内联**：103 处 `getattr(args, ...)` 口径读取散布在方法体内。
   `adaptive_tridecoding`（1052 行）单函数读 16 种开关、约 50 个 if/elif，
   模式/口径/RL/阶段状态四类分支混排。

方法清单（重构迁移顺序按此复杂度递增）：

```
target_only 80 · cee_sd_opportunistic 24（已是薄编排，终态预演）
dist_split_spec 392 · dist_spec 375 · uncertainty_decoding 376
adaptive_decoding 414 · cee_dssd 427 · cee_dsd 451 · ceesd_without_arp 492
tridecoding 518 · cee_cuhlm 516 · tk_slt 532
adaptive_tridecoding 1052（论文主方法，最后动）
```

## 1. 目标形状

```
comm_accounting.py (~300行)          decoding_context.py (~400行)
┌────────────────────────┐          ┌──────────────────────────┐
│ interface: 5 方法        │          │ interface: draft /        │
│  uplink_probs           │          │  verify_and_commit /     │
│  uplink_residual        │◄────────►│  settle / done           │
│  downlink_token         │          │ implementation:          │
│  round_trip / rl_cost   │          │  simulator 构造×1        │
│ implementation:          │          │  caches ×1 · metrics×1   │
│  16 flag → 构造函数      │          │  停止判据 ×1             │
│  数字口径→字段           │          └──────────┬───────────────┘
│  算法口径→2 adapter      │                     │
└────────────────────────┘                     ▼
                                    verify seam: 3 个真实 adapter
                                    Speculative / Cuhlm / Dsd
                                              │
                                              ▼
baselines.py (~1800行)：薄编排 + Register 注册
每个方法 518 行 → ~60 行（cee_sd_opportunistic 已证明可行）
```

**刻意不做的**：
- 不造"万能解码算法 + 20 个策略参数"——阶段拓扑（单级/两级/three-layer）
  是编排职责，留在方法体里；只有真实存在 2+ 实现的变化轴才设 seam
  （验证 ×3、载荷公式 ×4+、带宽模型 ×2——都是实轴）
- RL decision/learn 分离（报告 D4）不并入——正交改动，叠在同一 diff 里
  会让回滚不可能

## 2. 口径处置决策（2026-10-11 审计结论）

| 口径 | 本质 | 处置 |
|---|---|---|
| `rl_comm_cost_mode=settled` | bug（坏观测，已致 v2 重训事故） | **Phase 1 删**，零用户 |
| `--no-charge_residual_payload` | bug 冻结（残差被无声省掉） | 保留为 `ours_hist` profile 数据直到论文冻结，之后删 |
| `per_transfer`（2×NTT） | 口径（老实现确实发两条消息） | 保留，数据化 |
| `instant` 带宽模型 | 口径（明示简化 + 历史数字复现锚点） | 保留为 adapter |
| `fluid`、`force_full_vocab`、`prob_payload_bits`、NTT 三级、trace | 口径 | 保留，数据化/adapter |

已有地基（不重造）：`MODE_FEATURES`（模式注册表）、`--comm_accounting
honest`（三子开关统一覆盖，utils.py:1169-1185）、`--protocol paper_table5`
（协议冻结）。Phase 1 是把这套 CLI 层收敛**下沉到方法体**。

**删除的通用前提**：实验产物记录 git SHA（`get_save_dict` 落
`git rev-parse HEAD`）+ 论文冻结 commit 打 tag。落地后"复现历史数字"的
锚点从代码内分支转移到 commit，legacy 路径的存在理由才过期。

## 3. 分阶段执行

### Phase 0 — 基线锁定（先行，后续一切的安全网）

| 步骤 | 内容 | 产出 |
|---|---|---|
| 0.1 | 校准双跑：同配置原样跑两遍（机器空闲时，GPU 由用户调度） | 字段稳定性表：确定性（token 计数/接受数/comm bytes/time/connects）vs 实测噪音（wall_time/computation_time/throughput） |
| 0.2 | golden 固件：7 份 gammar metrics 只读复制进 `test/golden/gammar/` + 对应完整命令行 | eval 级端到端大网 |
| 0.3 | round 级固件：固定张量 → `verify_and_commit_round` 精确输出 | 单元粒度，无计时噪音 |
| 0.4 | `get_save_dict` 记 git SHA（Step 0，加在 refactor 分支，随下个窗口合并） | 解锁口径删除 |
| 0.5 | 若 0.1 发现 token 级漂移：加确定性开关（`torch.use_deterministic_algorithms` 等）并双跑验证 | golden 立得住的前提 |

锁：无（本阶段就是造锁）。风险：GPU 非确定性导致 golden 不可行 → 0.5 兜底。

### Phase 1 — CommAccounting（口径收拢）

- 新建 `src/comm_accounting.py`：16 flag → 构造函数；5 方法接口；
  `PROFILES = {"ours_full", "ours_hist", "paper_table5", ...}`
- **删 `settled`**：`_reward_comm_s` 收敛为单 nominal 路径
- 103 处内联读取逐方法迁移，**一个方法一个 PR**
- `--comm_accounting` / `--protocol` 语义不变，内部改读 profile
- 锁：`test_unified_comm_accounting` + `test_prob_bits_wiring` + eval 级 golden
- 预计：8~10 个 PR；baselines.py 每方法瘦身 20-30%

### Phase 2 — DecodingContext（骨架收拢）

- 新建 `src/decoding_context.py`：simulator 构造（10→1）、caches、
  metrics 结算（7→1）、停止判据
- 迁移顺序：target_only → dist_spec → dist_split_spec →
  uncertainty_decoding → tk_slt → ceesd_without_arp → cee_dsd → cee_dssd
  → tridecoding → cee_cuhlm → adaptive_decoding
- 锁：golden 全绿 + 各方法既有测试
- 预计：10~12 个 PR

### Phase 3 — verify seam（最高风险）

- `verify_and_commit_round()` + 3 adapter（Speculative/Cuhlm/Dsd）
- rollback 逻辑动刀；golden 必须全程绿；round 级固件在此阶段逐 adapter 建立
- B10 类 bug（EOS/截断在方法间接缝漏）的结构性终点

### Phase 4 — adaptive_tridecoding + profile CLI

- 1052 行方法最后分解（模式已被 10 个方法验证）；graph cache 复用、
  双 RL 接线、16 开关全部落在 context/accounting 上
- CLI：`--protocol` 成为主入口，16 个 flag 降级为 override（兼容旧脚本）
- 完成判定达成（见 §5）

## 4. 横切规则

1. **棘轮原则**：每 PR golden 绿才落地；红则修复或回滚，不留"暂时漂移"
2. **CLI 表面冻结**：campaign 脚本全程零改动
3. **合并窗口 = 编排器间隙**：主树运行区冻结由用户确认时机；
   `git commit` 只写 .git 元数据，campaign 期间安全
4. **worktree 纪律**：`git add` 显式路径（SpecDec_pp 是软链 + skip-worktree）；
   输出落 worktree 自己的 exp/；主树 exp/ 只读引用
5. **行为变更必须能被 golden diff 解释**：解释不了的漂移 = bug

## 5. 完成判定

- [ ] baselines.py ≤ ~2000 行（编排 + 注册）
- [ ] 新解码方法 = ~60 行编排 + profile 一行（cee_sd_opportunistic 形态）
- [ ] 口径变更成本：103 处分支核对 → 1 个构造函数
- [ ] eval 级 golden 全绿（确定性字段零漂移）
- [ ] `settled` 已删；`ours_hist` 以 profile 数据存在（一行 dict + 一个测试）
- [ ] campaign 脚本零改动、实验期间零中断
