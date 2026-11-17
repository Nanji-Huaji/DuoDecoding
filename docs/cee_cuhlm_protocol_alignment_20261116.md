# CEE-CUHLM 向 CUHLM 对齐（2026-11-16）

本轮由用户明确授权修改协议，接在 [前一轮结构重构](<baselines_refactor_20261115.md>)之后。不是旧 CEE-CUHLM 的输出等价重构，也不修改独立 CUHLM 本身。

## 差异与决策

独立 [CUHLM](<../src/baselines/cuhlm.py>) 是本轮基准。旧 [CEE-CUHLM](<../src/baselines/cee_cuhlm.py>) 只是混入机会传输的三级组合，并未实现同一个外层验证协议。

| 维度 | 旧 CEE-CUHLM | 本次实现 |
|---|---|---|
| 不确定度触发 | 首次触发直接走残差，相当于强制拒绝 | 触发表示调用 Target，仍以 `u < P(d)/q(d)` 判断接受 |
| 残差 | 用提议行作差；没有构造计费所对应的 k* 重建行 | 接受用原 q，拒绝用 `normalize((P-qhat)+)` |
| 候选覆盖 | 最后候选因索引差一未门控，全跳过后另行重采样 | 每个候选都独立门控、定稿一个 token，无外层 bonus |
| Target 前向 | 整段候选前向，另抽一个被丢弃的样本 | 只 `forward_context` 已定稿 prefix，不 forward 当前候选、不采样 |
| 提议来源 | carried 用 raw Draft，fresh 用 uniform-tail 重建行 | carried/fresh 都对应 Draft 的实际采样律，q 与 qhat 分工明确 |
| 零残差 | legacy 路径可回退到 argmax | 合法接受分支不算残差；拒绝且残差无质量则报错 |
| 往返口径 | 忽略 `comm_round_trip_mode`，逐报文计费 | 每个外层候选独立划轮，支持 per_round / per_transfer |

最直观的反例是 `P = q = qhat`：触发只表示“请 Target 检查”，接受比此时为 1，应该保留候选。旧实现却强制计算零质量残差，再由 fallback 选 token。因此这不是可接受的壁钟或浮点差异，而是协议行为不同。

## 保留 CEE，而不是退化为两模型 CUHLM

Little→Draft 继续做投机加速，`gamma2` 控制 Little 提案数，`gamma1` 控制 fresh Draft 预取数。内层改用现有严格 [验证模块](<../src/verification.py>)，其 accept/correct kernel 输出的边际律为 Draft 的实际采样律 Q_D。

因此，内层 correction 携带的外层 q 是对应前缀上的 Q_D，**不是**条件拒绝残差；carried 也不能直接免经 Target 门控进入最终 prefix。概率行与 logits 必须在缓存变更前保存。外层复用独立 [CUHLM 验证模块](<../src/cuhlm_verification.py>)，不复制另一套判定公式。

skip/accept 后前缀未分叉，预取后缀可以继续逐个门控；reject 后则废弃整个未消费后缀，回滚 Little/Draft，再从新 prefix 提议。Target 始终只前向已定稿 prefix。内层必要的 continuation 与已删除的外层 bonus 是两件事。

## 通信、配置与可比性

- skip 不新增 edge-cloud 消息、Target 前向或云端排队；Little→Draft 的预取成本另计，首轮 prompt 约定另计。
- trigger 为论文载荷上行、验证、接受或残差定稿、0B 下行；per_round 不得把同一预取块中的多次云触发压成一次 RTT。
- `uncertainty_threshold` / `draft_target_threshold` 控制外层门；`small_draft_threshold` 与旧实现一样不是内层 gate，不能把它当作有效双阈值实验。
- 往返标签从旧 `paper` 变为 `paper_rt`，表示消费往返开关，不表示始终使用 per_round。`legacy` 计费选项不恢复旧概率协议；历史 token、RNG 次数、时延与接受率不能被当作新版本必须一致的基准。
- 原 q 接受、重建 qhat 残差仍是独立 CUHLM 自身的有损协议，**不保证 Target 分布无损**。已有压缩 k* 求解器的限制不在本轮修改范围。

## 审查后的边界修正

[独立审查](<../.refactor/cee-cuhlm-alignment-20261116/review/findings.md>) 的四项发现均已处理：

1. 候选在生产模型设备上保存，拼入已定稿 prefix 前显式迁移，避免 CPU prefix / GPU 模型等放置方式下的跨设备拼接错误。
2. 恢复一次原始 prompt 的 edge-cloud 上传，放在第一个候选的计费轮内；零生成预算不上传云端。若触发数为 T 且预算非零，cloud connect 数为：per_transfer = 1 + 2T；per_round = T +「首候选 skip」。Little 链路另外计费。
3. **每个候选定稿后立即检查 EOS**，而非处理完整预取块再截断。不会继续门控 EOS 后的后缀；`generated = draft_accepted + outer_reject` 在 EOS 情形也成立，`disable_eos_stop` 仍有效。
4. 澄清 carried correction 的 q 是边际 Q_D，不是条件残差。另补位置相关权重模型的回归：从真实定稿 prefix 无缓存重算 Draft 的 q/logits，覆盖外层拒绝后的重提议。

## 实现与验证状态

- [实现](<../src/baselines/cee_cuhlm.py>) 已冻结，SHA256：`3ec19e45aabff8be4a4bcdef125cd302ac33ae714aa28d458a9c8ee30846ef69`；独立 CUHLM / TK-SLT 保持本轮开始时的内容。
- 全量 CPU：**2181 passed / 15 skipped / 163 subtests**，216.73 秒，见 [完整日志](<../.refactor/cee-cuhlm-alignment-20261116/integration/full-cpu-tests.log>)。其中相关三文件为 101 个测试，含 43 个真实 CPU 协议合同。
- [逻辑足迹](<../test/golden/logic/baselines.json>) 只更新 CEE 入口（158 events）并补冻结两个不自动内联展开的 CEE 阶段；其余既有条目与本轮前快照相同。指纹守卫 **12 passed**。
- 源码、三个实现相关测试与 legacy bridge 守卫的 Ruff 全绿。元数据三文件有 14 条既有 lint 诊断，与修改前快照对照完全相同，未顺带清理。
- 最终 A6000 单卡复验：**44 次执行、1003/1003 项 case 断言通过**，另有 10 项源码锚点通过。覆盖 FP32/BF16、eager/真实 CUDA Graph、自然及受控门控、零深度与独立 CUHLM 对照、两种 RTT、Graph 复用/重置及 llama-68m 实权重 smoke，见 [实机报告](<../.refactor/cee-cuhlm-alignment-20261116/gpu-final/summary.md>) 与 [完整 JSON](<../.refactor/cee-cuhlm-alignment-20261116/gpu-final/results/aligned_validation_final.json>)。
- 新增实机边界：CPU prefix + GPU 三缓存分别在 FP32 eager/BF16 Graph 下与 GPU-prefix 对照一致；块首 EOS 的 skip/trigger-accept 均只执行 1 次门控；首候选 skip 的 prompt-only 轮不合并到后续触发。42 个 CEE case（另 2 个为独立 CUHLM 对照）均满足 `gates = generated = accepted + rejects`。
- Lead 已逐项核对 JSON 断言、七个生产/测试文件哈希及产物哈希，均与最终工作树一致。首轮 35 次 / 649 项检查对应修正前的 `8c65914d…`，仅保留为过程证据；前一轮 72 组旧协议对比不是本轮验收。
- 验证限于 batch=1 单卡协议行为；没有做质量、吞吐或多卡结论。未安装依赖或修改驱动。原始日志、审查记录与实机工件保存在本地忽略目录，上述工件链接仅供本地查阅，不随提交发布。
