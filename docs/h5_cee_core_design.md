# H5 设计：CEE 三层公共核心（cee_dsd / cee_dssd）

> 状态：设计评审修订稿（2026-10-31）。基线 `main@0d8cca7`。
> 关联：`docs/baselines_split_map.md` §5.5。

## 1. 结论

`cee_dsd`（421 行）与 `cee_dssd`（397 行）在**顶层语句**上 93.5% 相同
（72 vs 67 条语句，difflib ratio 0.9353），但差异集中在验证路径、prefix
更新、拒绝计费与 rollback，**不能机械合并**。

可以采用“同一循环 + 块级钩子”重构，但净收益没有原估的 250–300 行，
更现实的量级是 **150–200 行**；且钩子必须按“块”设计，不能假设成纯数据变换。

## 2. 已核实的差异清单

顶层只有 3 处不同：

| # | 位置 | cee_dsd | cee_dssd |
|---|---|---|---|
| 1 | caches 属性 | `_cee_dsd_caches` | `_cee_dssd_caches` |
| 2 | prefix 存储 | 预分配 buffer + tracker | `torch.cat` 追加 |
| 3 | while 循环体 | 见下 | 见下 |

> 已核实：metrics 收尾并**没有**差异。两边的 52 行 epilogue 逐字节相同
> （`src/baselines.py:5884-5935` vs `5460-5511`），都是先截断再计 tokens，
> 也都没有向 `_add_per_model_wall_time` 传 `draft_comp_time`/`target_comp_time`。
> 原稿中“dsd 先截断、dssd 先计 tokens”的说法是错误的，已删除。

循环体（50 vs 42 条语句）的实质差异，至少 **7 个切分点**：

1. **第一层 little proposal**
   - dsd：`generate_with_topk_metadata_only` 或 `_generate_with_optional_rebuilt_proposal`，返回 `little_topk_history`；
   - dssd：`_generate_with_optional_rebuilt_proposal(..., need_topk_metadata=True)`，返回 `little_rebuilt_probs` + `little_rebuilt_meta`；
   - dssd 的 `little_rebuilt_probs/meta` 后续要喂给 `stage_prob_history` 和
     `draft_probs_override`，钩子的返回签名不能只写 `(x, meta)`。
2. **第一层验证 payload**
   - dsd：直接构造 `little_stage_kwargs`；
   - dssd：`stage_prob_history` + `collect_verification_payload`。
3. **第一层 gamma clamp / cache rollback**
   - dsd 有 `l1_actual_gamma` 截断与 rollback（`5660-5671`），dssd 没有；
4. **第一层额外上行**
   - dssd 在 `5650` 有额外 `comm_simulator.transfer(...)`。
5. **拒绝计费**
   - dsd：`reject_residual_payload_bytes`（`5702`）；
   - dssd：压缩行 + 尾部标量（`5285-5300`），并多一次下行 `transfer(None, probs, ...)`。
6. **第二层 draft proposal / payload**
   - 与第一层同构地分叉（`5737-5753` vs `5304/5420`），原稿漏了这个钩子。
7. **第二层 clamp / rollback / 额外上行**
   - dsd 的 `l2_actual_gamma` 截断、rollback（`5781-5792`），dssd 无；
   - dsd 在 `5763` 有额外上行。

另外，prefix 的 append 会原地改 buffer、proposal 会改 cache、reject 会调
`comm_simulator` 计费——**钩子不是纯函数**，不能按“无副作用数据变换”来切。

## 3. 建议结构

```
def cee_dsd(...):   return self._cee_three_layer(..., split=False)
def cee_dssd(...):  return self._cee_three_layer(..., split=True)

def _cee_three_layer(self, prefix, *, split: bool, ...):
    # 公共：设备、caches、comm simulator、计数器、事件、循环骨架
    # 差异点按“块”收敛：
    #   _cee_prefix_*          （已由 CEETriPrefix 承接，见 §5）
    #   _cee_little_stage(...)  -> 含 proposal + 第一层 verify payload
    #   _cee_layer1_reject(...) -> 含 rollback / 计费 / 重采
    #   _cee_draft_stage(...)   -> 第二层 proposal + payload
    #   _cee_layer2_reject(...)
```

约束：

- 钩子保持**同模块私有方法**，且尽量**无 early return**；`tools/logic_fingerprint.py`
  只展开同模块、无 early return 的 `self.*` helper，这样核心搬迁仍可保持
  指纹 0 差异（原稿误判为必须重建）。
- 如果某个钩子必须有早返回，把它限制在很小的块，并在最终合并点只重建
  一次指纹；不要让整个循环变成 value-returning 大钩子。
- 公开 wrapper 会丢 `co_names`：`src/protocols.py` 的 `live_mode_consumption`
  只递归注册方法，`return self._cee_three_layer(...)` 会让
  `comm_round_trip_mode`/`charge_residual_payload`/`transfer_top_k_cap`/`gamma1`/
  `gamma2` 从观测面消失，`test/test_protocol_spec.py` 会失败。
  两个修法任选：
  1. 在 wrapper 里保留这些属性的只读引用（像 `ceesd_without_arp` 的
     `_wiring` 兼容标记）；
  2. 扩展 `protocols._collect`，允许跟随 `self._cee_*` 辅助方法。
     推荐 2，长期更干净。

## 4. 验证计划

原稿的“必须重建指纹”前提已修正：

1. **本地零差异门禁**：核心与块钩子都放 `src/baselines.py`，保持同模块、
   无 early return；`python tools/logic_fingerprint.py diff src/baselines.py --base main`
   应 PASS。若必须重建，只在最终合并点重建一次，并在 PR 说明。
2. **fake-based 行为门禁**：`test/test_cee_refactor.py:303-388` 已有两个
   方法的 fake 运行，可作为本地首选回归；补充 `gamma=0`、EOS、
   `l1_actual_gamma<=0`、`charge_residual`、`transfer_top_k_cap`、
   stochastic comm 覆盖。
3. **enine 同机 A/B**：旧 `main` vs 重构分支，`cee_dsd`/`cee_dssd` 各 20 样本；
   必须固定 seed（仓库 src 里没有全局 seeding，只有 `rl_adapter.py:197`），
   并核对 token 逐位一致、metrics 计数一致。
4. **80 样本 smoke**：确认无 OOM/图回放回归。
5. 20 样本本身覆盖不到 `gamma=0` 等边界，必须靠第 1/2 条的定向用例补。

## 5. 里程碑（修订）

| 阶段 | 内容 | 状态/预计 |
|---|---|---|
| H5a | prefix 容器 `CEETriPrefix`（buffer/cat 两模式） | **已完成**：`refactor/h5c-prefix`（PR #21，未合） |
| H5b | 同模块核心 `_cee_three_layer` + 第一层块钩子，指纹 0 差异 | 1 天 |
| H5c | 第二层块钩子 + wrapper + protocols 跟随 | 1 天 |
| H5d | 一次指纹重建（若需要）+ enine A/B + 80 样本 smoke | 0.5 天 |

## 6. 非目标

- 不在同一 PR 动 `dist_spec` / `dist_split_spec`（另一对 86.6% 相似，但验证路径同样分叉）；
- 不改计费/协议口径。

## 7. 开放问题

1. `ceesd_without_arp` 现在是 42 行 delegator（`main:2943-2984`，PR #19 已归档实现），
   是否还要删除这个注册名？
2. protocols 的跟随方案选 1（wrapper 只读引用）还是 2（扩展 `_collect`）？推荐 2。
3. 论文 Table VIII/IX 是否还需要 cee_dsd/cee_dssd 的历史逐 token 对照。
