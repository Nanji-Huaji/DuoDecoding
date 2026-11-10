# 第一层验证步骤设计（adaptive_tridecoding）

> **历史方案说明（2026-11-07 更新）：** 下文保留原基点、坐标与当时的狭窄授权，不作为当前实现规范。用户后续已授权完整 CEE-SD 本体重构，以逻辑正确性而非旧轨迹验收；当前接口与修正见[本体重构记录](<cee_sd_body_refactor_20261107.md>)。尤其旧 §3 的第一层 EOS 直接停机、末轮 flush 前读取 wall time、通信必须留在原方法等限制已被替代：EOS 须经 target 确认，终止回复/下行须计费，最后结算先于派生时间指标；helper 使用真实新调用帧，不伪造旧位置。原实验状态及以下设计论证仅描述当时版本。

> 基点：`59b80ec`（本地分支 `refactor/baselines-s2-nightly-20261103` 的独立 commit；**未 push、未合 main**）。只读设计：不改源/测试/既有文档，不提交。
> 在 `59b80ec` 中 `src/baselines/core.py` 的 `adaptive_tridecoding` 为 888 行（3094–3981）；`_adaptive_rejection_prob_bytes` 3983、`_select_adaptive_little_config` 4011、`_select_adaptive_main_config` 4049。`_make_comm_simulator` / `_make_symmetric_comm_simulator` 在 `src/baselines/simulators.py:13/69`（S3a 中间态已移出；core 141–145 为私有 re-export）。
> S3a 的未提交中间态把上述 factory 移走，当前方法暂为 3005–3892（方法体净 −89），块内顺序/语义未变。**本文所有方法内行号均为 `59b80ec` 坐标**；当前中间态只能对方法体做 −89 映射，**不能对所有文件/跨文件引用统一偏移**（factory 已是 `simulators.py:13/69`）。S3a 落定后应整体重排一次行号，不要把中间态数字写进验收。
> 未完成验收：无真实 ARP/RL 权重、无 CUDA graph、无真实模型/GPU 对照；本文不包含也不虚构任何实验结果。
> 范围：只讨论 V4（第一层验证）的小返回值与调用边界；不提议 `verify_and_commit_round` 大重构，不改 layer/递归，不引入新框架。

## 1. 实际连续块：阶段一生成后 → EOS 前

| 段 | 行（`59b80ec`） | 内容 | 是否验证步骤 |
|---|---:|---|---|
| G0 | 3357–3393 | adapter reset、gamma、little 生成、entropy、RL select/apply | 生成/策略，不属于 |
| V0 | 3395–3402 | `actual_gamma2`、forward/token 计数、`n1` 初值、accepted/all 初值 | 前置 |
| V1 | 3404–3435 | opportunistic 阈值写回 + uncertainty + transfer 决策 | 策略，不属于 |
| V2 | 3437–3445 | skip：accept=gamma、空 token、8 B 消息、accept message | 分支 |
| V3 | 3446–3479 | draft 预启动 + 第一层上行概率载荷（量化/收集/transfer） | 通信，不属于 |
| V4 | 3481–3519 | `resolve_stage_verification`、残差补计费、gamma=0 fallback | **候选核心** |
| V5 | 3521–3565 | accepted/rate 计数、rank 记录、第一层 reward/step | 记账/策略 |
| V6 | 3567–3568 | `prefix = x[:, : n1 + 1]` | 提交 |
| V7 | 3570–3600 | 拒绝概率行、`prob_bytes`、`reject_overhead=6` | 计费 |
| V8 | 3602–3611 | 索引 + token + reject 合并的 edge_end 计费 | 通信 |
| V9 | 3613–3627 | 校验、`cat`、EOS break、`new_generated_token` | 终止/下一层输入 |

**关键结构事实：这不是一个连续步骤。** V5 的 reward 在 V7/V8 的拒绝与下行计费**之前**读取 `step_comm_time`/`energy_j`；V8 又在 V5 之后。把 V3–V9 收进一个大 helper **可以**在内部保持原序（reward 不必重排），但会把 V3/V4b/V8 的 comm、EOS、reward 和 14 个 live 变量一起吞进去；本设计只切 V4，V5–V9 留在方法体，并保留“reward 不含 V7/V8 字节”的切点合同。否决大 helper 的是接口/帧/内省/测试面（见 I-A），不是 reward 顺序。

## 2. V4 的自由变量 / live-out

| 名字 | 读写 | 说明 |
|---|---|---|
| `x` | 读 | little 生成结果（含 `actual_gamma2` 个 token）；V4 后仍用于 V6/V9 |
| `prefix_len` | 读 | 本轮起点；V4 后仍用于 V5/V6/V9 |
| `actual_gamma2` | 读 | `x.shape[1] - prefix_len`；V5/V7 仍用 |
| `little_stage_probs` | 读 | 可能为 None（gamma=0）；V4 的 override、V4b 残差、V5 rank、V7 拒绝行共用 |
| `little_rebuilt_meta` | 读 | 只用于 F43 top-k history；**可能为 None**（`proposal_top_k is None` 时即使 `charge_residual=True` 也不产 meta）；V4 后 dead |
| `little_model_cache` | 读+写 | proposer；`resolve` 内部 rollback，caller 不得二次 rollback |
| `draft_model_cache` | 读+写 | verifier；**V3 已用本轮 `x` 做 `generate(x, 1)` 预热**（gamma=0 fallback 依赖其最后一行）；`resolve` 内部 rollback |
| `little_device` | 读 | `output_device` |
| `charge_residual` | 读 | 方法入口读的计费开关；只能以 bool 值传入 helper，不能让 helper 读 `_comm` |
| `transfer_top_k` | 读 | 第 1 轮：`force_full_vocab_transfer` 时覆盖为全词表且不 cap，否则入口 cap（cap>0 时）；之后是上一轮 stage2 `_select_adaptive_main_config` 的选值（仅非 None 且 cap>0 时再 cap，否则原样）；只有 RL 关闭（`rl_adapter is None` 或 `_static`）才恒入口 |
| `current_proposal_top_k` | 读 | loop 起点由 `transfer_top_k` 计算的 `proposal_top_k(...)`；stage1 用的是上一轮 stage2 选值/入口值，**不是本轮 stage2 稍后写入的新值**；仅 V4b 残差字节用 |
| `comm_simulator` | 写 | 仅 V4b `simulate_transfer`；不能把它移进 helper（§3.9） |

**live-out**：`little_accepted_this_iter`、`n1`、`t`、`little_all_accepted` 四项（V5/V6/V7/V8/V9 读）；`little_stage_probs`（V5/V7 读）；`x`/`prefix_len`；两个 cache 的新长度。`n1` 在 3400 预置为 `prefix_len + actual_gamma2 - 1`，gamma=0 的 fallback 不覆写它（值同为 `prefix_len-1`），验证返回的 `n` 可以更小。

**V4 的最小边界**：第一句是 3481 的 `if actual_gamma2 > 0:`（若 helper 自带 fallback，则边界提前到 3481 的判断本身），最后一句是 3519 的 `little_all_accepted = True`；V4b 的 3505–3513 残差 `simulate_transfer` 必须留在 caller，因为它是通信副作用且位于 reward 之前。V3 的 payload/量化、V5 的 rank/reward、V6 的 prefix、V7/V8 的拒绝/下行、V9 的 EOS 都不属于该边界。**若把 V3 收进来，helper 需要 `comm_simulator.transfer`，帧归属改变；若把 V5/V7 收进来，虽然可在 helper 内保持原序，但 live-out、comm 和 reward 切点合同会一起进入接口**——这是 I-A 被否的实质原因，不是 reward 顺序。

## 3. 手工合同（按顺序）

1. **阈值顺序**：`_select_adaptive_little_config`（3387）写 `small_draft_adapter.threshold`；opportunistic 在 3412–3415 把它写进 `cuhlm.uncertainty_threshold`；3431 的 `determine_transfer_strategy` 读该字段。任何 helper 不得把这三步拆开或重排；CUHLM API 不接受显式阈值，所以“纯决策函数”无法不依赖可变状态。
2. **概率行与两次量化**：`stage_prob_history(cache, prefix_len, rebuilt_probs)` 返回的 stage 张量按**绝对行号**索引：cache 前缀行 index=logits 位置；rebuilt 段第 `i` 个 draft token（token 位置 `prefix_len+i`）使用绝对行号 `prefix_len-1+i`（`build_draft_probs_override` 从 `prefix_len-1` 开始拼）；V7 拒绝行 `little_stage_probs[:, n1]` 的 `n1` 是最后一个接受位置，也是被拒 token 的前一行，`reject_row` 必须用绝对 `n`，不能减 `prefix_len`。量化有**两处**，不能合并成“只一次”：① 3465 `_uplink_prob_payload(..., prob_payload_bits)` 量化整段 stage 行，返回张量供 3474 transfer、3494 verify override、3525 rank、3592 拒绝行；② 3999 `_adaptive_rejection_prob_bytes` 对选中的拒绝行**再次** `_uplink_prob_payload`，只为按位宽算 bytes，局部量化张量不回传、不流入验证/transfer。②是历史 B15 行为，**不得删除**；Step 0 按调用序号验证（先整段，后每 reject 一次），并确认②的结果没有被下游使用。rollback 后不得从 cache 重建 stage 行。
3. **V3 前提与验证/rollback 所有权**：非 skip 分支必须先执行 V3 的 `draft_model_cache.generate(move(x), 1)`（3449）；gamma=0 fallback 从该 cache 最后一行 `prob_history[:, -1, :V]` 采样，若预热没做或做在 helper 之后，fallback 是错行。`charge_residual` 只保证尝试生成 meta，不是 meta 非 None 的充分条件：`proposal_top_k is None` 时（cache/adapter 路径）meta 必为 None，`stage_topk_proposal_history(None, gamma)` 也返回 None；helper 必须原样传 `None`，不得合成空 history。`resolve_stage_verification`（`src/decoding_ops.py:586`）内部完成接受判定、采样和 `apply_rollback`；caller 不得二次 rollback，也不能在 rollback 后重新切 cache 行。
4. **stage1 top-k 进展**：第 1 轮若 `force_full_vocab_transfer` 为真，`transfer_top_k` 被覆盖为全词表且**不做入口 cap**；否则入口 `transfer_top_k_cap` 生效（cap>0 时）。之后每轮 stage2 的 `_select_adaptive_main_config` 写回选值：仅 `next_topk is not None` 且 `cap>0` 时取 `min(int(next_topk), cap)`，否则原样（可 `None`），下一轮 loop 起点的 `current_proposal_top_k` 与 V7 拒绝字节都用这个值。`no_arp` 只禁用 ARP head/阈值，**不**冻结 top-k：RL adapter 仍在时 stage2 照常 select；只有 `rl_adapter is None` 或 `_static` 才恒入口值。little 策略的 `next_topk` 不应用（helper 内是死变量）。
5. **拒绝行与字节**：3590 的 `prob_data = little_model_cache.prob_history[...]` **每次 reject 都会执行**，不是死语句；因为 reject 只在 `actual_gamma2 > 0` 的 `resolve` 后发生，3591–3594 的 `if actual_gamma2 > 0` 必定覆盖它，最终值恒为 `little_stage_probs[:, n1]`。但删除 3590 需要证明 `prob_history` 属性访问与 `[:, n1, :V]` 切片在 rollback 后不抛（当前未证明），所以不能称“不可达/已证行为中性”；Step 2 必须补证明或保留。`prob_bytes` 由 `_adaptive_rejection_prob_bytes` 按当轮 `transfer_top_k` 计算，`reject_overhead=6.0`；合并计费只在 `not skip` 时发一次 edge_end 消息。
6. **计费/计时切点**：V4b 残差在 reward 之前（3508）；V5 reward 的 `step_comm_time`/`energy_j` 在 3539/3551 读取，**不含** V7/V8 的拒绝与下行字节；V8 在 3611。`per_round` 下这些消息由 `set_round`/`flush_round` 结算，ledger 顺序由现有测试锁定。移动 reward 或把 V8 提前都是行为变更。
7. **EOS**：stage1 的 `_stop_at_eos` 在 V8 合并计费之后（3619），EOS 命中时截断到该 token 并 break，第二层不发生；`_tri_prompt_len` 在入口固定。不得把 EOS 移到 V8 前或移进验证 helper。stage2 的 EOS 位置不同（在 cloud 合并/下行前），不要合并。
8. **RNG 与 helper 副作用**：opportunistic 的 `calculate_uncertainty` 内含 `torch.rand(20)`（3425），发生在 draft 预启动/验证之前；V4 的 `resolve` 恰好采样一个 token（接受或残差），gamma=0 fallback 恰好一次 `sample_accept_token`。`_verify_first_stage` 因此**不是纯值计算**：gamma>0 时它调用 `resolve`（采样 + rollback），gamma=0 时它采样；helper 不能增加/减少/提前任何采样，caller 不能重复 rollback/sample。
9. **计时口径**：`step_start_time`（3347）、`dra_start`（3386）、`little_comp_time`（3367–3378）的区间不能吸收 helper 之外的时间；helper 只允许增加一次普通调用开销，且必须在原位置调用。`wall_time` 仍在末轮 flush 前读取（历史顺序，不修）。
10. **帧/内省/门禁**：`src/communication.py::_trace_comm` 用 `sys._getframe(2)` 记 caller，`simulate_transfer` 用 `sys._getframe(1).f_code.co_name == "transfer"` 判 trace_kind；`transfer`/`simulate_transfer`/`send_accept_message` 必须留在方法体，除 `_verify_first_stage` 自身的采样/rollback 外 helper 不发起通信。`live_mode_consumption` 只跟随注册方法、不跟随普通 helper；四个计费开关与 gamma 读取留在方法体（`charge_residual` 以 bool 传入）。helper 必须只有单个尾 `return`（或无 return），否则指纹不展开；引入 `StageVerification` 构造会新增 `call` 事件，必须走快照评审，不能再声称与 main 的 `ast.dump` 严格相等。

## 4. 现有测试覆盖与缺口

**已覆盖（公开入口，真实 `resolve` + 真实 `CommunicationSimulator` 子类）**

| 合同 | 用例 |
|---|---|
| 三轮全接受：token/metrics/逐消息 ledger/rollback | `test_adaptive_tridecoding_rounds.py:1004` |
| 两级拒绝：拒绝载荷后 rollback | `:1030` |
| bits=8 量化拒绝（两级，k=2 与整行） | `:1062`、`:1106` |
| per_round + residual：残差字节/flush/确定性通信时间 | `:1143`、`:1176` |
| per_transfer + residual：逐消息结算 | `:1209` |
| stage1 EOS（结算后 break）、per_round EOS | `:1271`、`:1300` |
| stage2 EOS（merge/下行前）、第二轮 EOS | `:1327`、`:1357`、`:1384` |
| little select/no_arp/gamma/fallback/static | `test_adaptive_tridecoding_policies.py:344/561/693/754` |
| main top-k cap/gamma | `:454` |
| ARP 阈值下一轮生效、predict 只见旧阈值 | `:805`、`:1025` |
| opportunistic skip/transfer/同轮阈值复用 | `:1221`、`:1288`、`:1349` |
| 两级 `resolve` 配对（patch 记录 proposer/verifier/gamma） | `test_adaptive_tridecoding.py:232` |
| 向量化载荷、短 history 容错 | `:332`、`:382` |

**缺口（不是“没测所以一定错”，是设计/验收边界）**

- `resolve_stage_verification` 的关键 kwargs 没被断言：`draft_probs_override` 是否就是 3465 返回的量化张量、`draft_topk_history` 的 `charge_residual`/meta/top_k 组合（`top_k=None` 时即使 residual=True 也必须 None）、`output_device` 是否 little device；patch 用例只记了 4 个字段。
- 没有按调用序号断言 `_uplink_prob_payload` 的两处语义：3465 整段一次、3999 每个 reject 局部一次，且 3999 的局部量化结果不回传；bits=8 用例只覆盖最终字节。
- 没有断言 stage1 后续轮的 top-k 来自上一轮 stage2 的 RL 选值（再 cap、可 None）；现有 policy 用例锁的是 main 选择/本轮结算，没有把下一轮 stage1 的 `current_proposal_top_k`/拒绝字节作为观察点。`no_arp` 只禁 ARP head，不冻结 top-k，缺“no_arp + RL 开启 + 第二轮 stage1”的显式用例。
- gamma=0 的 V4 fallback 没有直接用例；`logits_history is None` 的 ValueError 没有直接用例。stage1 EOS 用例的 EOS 是 draft bonus token（`draft` 行 1），不是“接受块中间出现 EOS”；该组合未覆盖。
- 没有断言拒绝行必须来自预量化 stage 行（现有测试走的是活分支，不会发现 3590 的 cache 读取被绕过/改写）。
- 没有锁定“reward 不含本层拒绝/下行字节”的合同（ledger 有顺序，但没有显式把 reward 与两笔计费切开）。
- 全部是脚本化 fake cache/CPU；无真实模型分布、真实 ARP/RL 权重、CUDA graph、GPU 数值。无 stochastic/precise simulator、batch>1、`force_full_vocab_transfer`、正式 `transfer_top_k_cap` 组合、`max_tokens` 截断与 helper 边界组合、COMM_TRACE caller 明细。
- 没有新模块的直接单测（目标接口的测试面尚未建立）。

## 5. Interface 候选

### I-A（拒绝）大 context：`FirstStageContext` + `run_first_stage(ctx)`

字段约 14 个：`x/prefix_len/actual_gamma2/little_stage_probs/little_rebuilt_meta/两个 cache/两个 device/comm_simulator/transfer_top_k/current_proposal_top_k/charge_residual/三个 history 与计数`；`run` 会把 V4–V9 一起吞进 helper，返回 5–8 个值或原地改 ctx。

拒绝理由：① 14 字段可变 context，live-out 隐式、生命周期不清；② 若把 comm 也放进去，`transfer/simulate/EOS` 的帧归属、`_comm`/args 消费、测试替身全变；若把 comm 留 caller，则 helper 要返回多笔消息/字节/多段状态，接口比 context 更差；③ 方法体只剩“填 ctx/读 ctx”，可读性反转；④ 测试面仍是整轮循环，不能单独隔离；⑤ EOS/reward 被包进 helper 后更难单独验证。注意：它**可以**在内部保持 V4–V9 原序，reward 不必重排；否决点是接口/帧/内省/测试面，不是 reward 顺序。这不是“更深的模块”，是把方法搬进结构体。

### I-B（目标）`StageVerification` + `_verify_first_stage`

```python
# src/decoding_types.py
@dataclass(frozen=True, eq=False)  # 含 Tensor，禁用自动 __eq__ 的逐元素比较
class StageVerification:
    accepted: int
    n: int
    token: torch.Tensor                 # (B, 1)
    all_accepted: bool
    proposal_rows: Optional[torch.Tensor]  # 量化后的 stage 行；gamma=0 时 None

    @property
    def rejected(self) -> bool: ...
    def reject_row(self, vocab_size: int) -> torch.Tensor:
        """仅 rejected 时；否则 raise；返回 pre-rollback 提案行 [:, n, :vocab_size]。"""

# src/baselines/core.py（放在 _select_adaptive_little_config 之后）
def _verify_first_stage(
    self, *, little_model_cache, draft_model_cache, x, prefix_len, gamma,
    little_stage_probs, little_rebuilt_meta, output_device, charge_residual,
) -> StageVerification:
```

9 个 keyword-only **值** → 1 个返回值；不是 9 字段的可变 state。调用顺序固定为：V3 量化/上行 transfer → `_verify_first_stage`（**有副作用**：gamma>0 调 `resolve` 做真实采样+`apply_rollback`，gamma=0 调 `sample_accept_token`）→ 若 `actual_gamma2 > 0 and charge_residual and verification.rejected` 则 V4b 残差 → V5 reward；helper 内部不得发消息。`gamma > 0` 时 `little_stage_probs` 必须非 None 且已在 V3（3465）量化，`_verify_first_stage` 不再量化；gamma=0 时忽略它，但依赖 V3 已用本轮 `x` 预热 draft cache。`charge_residual` 必须是入口读出的 bool，helper 内不得出现 `_comm`；`little_rebuilt_meta is None` 合法（`proposal_top_k is None` 时即使 residual=True 也为 None），helper 必须原样传 `None`。`proposal_rows` 保存**完整行张量（绝对行号索引）**；`reject_row(vocab_size)` 与现有 `little_stage_probs[:, n1]` 同义。它拥有：gamma=0 fallback、F43 `draft_topk_history` 门控、cache/device/override 配对、拒绝行来源。caller 保留：V1 策略、V3 上行 payload、V4b 残差 `simulate_transfer`、V5 reward、V6 prefix、V7 拒绝字节、V8 下行、V9 EOS。**按 deep module 标准这不是即时杠杆**：单 callsite、约 30 行、9 个入参，实际是一个有名字的合同 + 可复用返回结构；价值在澄清输入/副作用/拒绝行来源，以及将来 stage2 可复用，不宣传成行数收益。删除测试（deletion test）：删掉它，这些规则会在 V4 重新散开，调用点只剩“少写 30 行”。

目标 caller 形状（只移除 V4 内部 `if actual_gamma2 > 0: resolve ... else: sample`；外层 `skip`/V3 transfer 分支、V4b 残差、V5–V9 原样）：

```python
if skip_first_stage_draft:
    ...                                    # V2 原样
else:
    _ = draft_model_cache.generate(_move_token_tensor(x, draft_device), 1)  # V3 预热
    ...                                    # V3 计时代码原样
    if actual_gamma2 > 0:
        ...                                # 3465 量化 + 3474 transfer 原样
    verification = self._verify_first_stage(
        little_model_cache=little_model_cache,
        draft_model_cache=draft_model_cache,
        x=x, prefix_len=prefix_len, gamma=actual_gamma2,
        little_stage_probs=little_stage_probs,
        little_rebuilt_meta=little_rebuilt_meta,
        output_device=little_device,
        charge_residual=charge_residual,
    )
    little_accepted_this_iter = verification.accepted
    n1 = verification.n
    t = verification.token
    little_all_accepted = verification.all_accepted
    if actual_gamma2 > 0 and charge_residual and verification.rejected:
        comm_simulator.simulate_transfer(       # 残差仍在 caller、reward 之前
            self._residual_payload_bytes(little_stage_probs, current_proposal_top_k),
            "edge_end",
        )
```
注意：条件必须是 `actual_gamma2 > 0 and charge_residual and verification.rejected`（原代码是 `charge_residual and not little_all_accepted`，且在外层 `gamma>0` 内）；写成 `not verification.rejected` 会把“接受”当成需要补计费，gamma=0 也会误补。放在 `decoding_types.py` 是因为那里已有 `VerificationInputs` / `AcceptanceResult` / `TopKProposalHistory` 同类值对象。

### I-C（拒绝）把 V4 移到 `decoding_ops.py` 的模块级函数

会绕过 `test_adaptive_tridecoding.py:232` 对 `src.baselines.core.resolve_stage_verification` 的 patch，改测试目标；跨文件还会改指纹语料/路径键。除非同时迁移 patch 与 fixture，否则不划算。

**选择：I-B。** 先只引入 `StageVerification` 类型和 `_verify_first_stage`，不动 V5–V9。若评审不接受 return-shape 变更，则本夜停在“设计+补测”，不要退化成只搬 `resolve` 调用的浅 wrapper（严格 AST 下它只能做纯参数重排，deletion test 不过）。`_verify_first_stage` 保持为 `Baselines` 方法而不是 `decoding_ops` 模块函数：fallback 要读 `self.vocab_size`，而且现有测试 patch 的是 `src.baselines.core.resolve_stage_verification`；若评审更想要纯函数，可加第 10 个 `vocab_size` 参数并留在 core，但这会扩大接口，不推荐。

## 6. 迁移步（分阶段、可停）

1. **Step 0（无人值守安全）**：不改生产代码，补特征测试：同时 patch `resolve_stage_verification` 与 `_uplink_prob_payload`，**按调用序号**记录——每个 stage 先整段量化（stage1 3465 / stage2 3730），每个 reject 再局部量化（3999）；断言 3999 的局部张量/返回不流入验证/transfer，且 `draft_probs_override` 是 3465 返回的同一个张量。`draft_topk_history` 只有在 `charge_residual=True` **且** `little_rebuilt_meta is not None` **且** `proposal_top_k` 有效时才可能非 None；`top_k=None` 时即使 `charge_residual=True` 也必须传 `None`，不能断言“只要 charge 就非 None”。另断言 `output_device` 是 little device；用 fake cache 触发 gamma=0 fallback 和 `logits_history=None`；用 fake policy 的 `compute_reward` 入参 + ledger 前缀锁定“reward 不含 V7/V8 字节”。先让这些测试在现状上绿。
2. **Step 1（需人确认）**：在 `decoding_types.py` 加 `StageVerification`，在 core 加 `_verify_first_stage`，替换 V4；caller 用命名字段，fallback 移入 helper。**门禁变化**：指纹预期新增 `call StageVerification`（若本步同时切 V7 用 `reject_row`，再加 `call reject_row`），其余事件不变；严格 AST 回读不再适用，改为“逐事件 diff + 行为测试”评审。不得顺手改 V5–V9。若严格最小，Step 1 先只带 4 个字段（`accepted/n/token/all_accepted`），`proposal_rows` / `reject_row` 与 Step 2 一起加；不要为了少一次 diff 留未使用字段。
3. **Step 2（Step 1 后，需人确认）**：给结构加 `proposal_rows` / `reject_row`；在替代 3590–3594 之前，先证明 `little_model_cache.prob_history` 的属性访问与 `[:, n1, :V]` 切片在 rollback 后不抛（当前**未证明**，不能称行为中性）。证明后再用 `verification.reject_row(self.vocab_size)` 取代 3590–3594；未证明则保留 3590，或只改 3591–3594 的取行。跑 rounds/policies/adaptive 全套；新增指纹事件/合同，仍算一次评审。
4. **Step 3（另案）**：stage2 复用 `StageVerification`（需处理 preverify trace、bonus prefix 的 topk history、`draft_device`）；本文不设计、不排期。

**明确不做**：大 context；把 `transfer/simulate_transfer/send_accept_message` 移进 helper；把 reward/EOS 移进 helper；helper 内读 `_comm.*`；早退 return；为了行数拆 V4 成多个浅 helper；改 layer/递归结构。

## 7. Human 确认 vs 无人值守

| 事项 | 归属 |
|---|---|
| 接受 `StageVerification` 作为新返回类型、放在 `decoding_types.py` | Human |
| 接受指纹新增 `call StageVerification`、放弃该入口的严格 AST 相等 | Human（快照评审） |
| 是否删除/改写 3590（cache 属性访问与切片是否安全尚未证明） | Human（需专门验证） |
| 真实 ARP/RL 权重、CUDA graph、GPU/gammar 验收 | Human（本夜不做） |
| 是否让 stage2 复用该结构 | Human（另案） |
| Step 0 特征测试、跑现有测试/指纹/文档行号 | 可无人值守 |
| 任何 comm 调用位置、EOS/reward 顺序、`_comm` 读取位置 | 不得无人值守改 |

## 8. 验收状态（截至本文）

- 已完成：块边界、自由变量/live-out、10 条合同、现有测试映射、三个 Interface 候选、分阶段迁移与 gate 代价。
- 未完成：无真实模型/GPU/ARP/RL/graph 结论；无 `StageVerification` 实现；未跑全量 pytest；未提交。所有行号以 `59b80ec` 当前树为准；S3a 落定后只重排方法内行号（本中间态 −89），factory/跨文件引用按 `simulators.py:13/69` 单独维护，不能做全文统一偏移。
