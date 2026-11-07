# 2026-11-03 夜间重构验收

## 范围与提交

本夜先收口旧 WIP，再按用户追加授权推进一个 S3a；不是只验证旧代码。
本地分支 `refactor/baselines-s2-nightly-20261103`，未 push、未合 main。

| 提交 | 内容 |
|---|---|
| `59b80ec` | adaptive 五个 helper、行为门禁与 S2 最小兼容包化一起收口 |
| `8836b06` | 两个通信构造工厂原样迁入 `src/baselines/simulators.py` |
| `1f1d5e3` | 真实模型回归采集器与严格比较器，及 82 个 CPU 用例 |

S3a 后 core 5,736 行（−89），新工厂模块 107 行；整类 `Baselines` 文本/AST
与 `59b80ec` 完全相同，adaptive 仍 888 行。包总源码没有减少 89 行：职责搬家
增加了 import/docstring，不把物理移走等同逻辑删除。105 个旧可读名字、79 个
星号名字、24 注册项和别名身份保留，六种冷启动顺序有持久测试。

第一层验证设计另见 `docs/adaptive_first_stage_design.md`：已完成合同与接口候选，
但没有实现 `StageVerification` 或 `_verify_first_stage`，没有移动 reward/EOS/comm。

## 本地门禁

- 默认与 importlib 全量 pytest：均 **672 passed、7 skipped、90 subtests passed**；
  仅两个既有 warning。新增工具与 S3a 的全部测试已包含在内。
- 对 `59b80ec` / `50c43ba` 的多文件逻辑指纹均 PASS（68 入口）。
- 原 15 条 frozen entries 不变；S3a 仅追加工厂模块到 include。
- 五个 adaptive helper 在六个调用点回内联后，与 `50c43ba` 方法体 AST 完全
  相同，SHA256 为 `9632a2c687de9d92aaa22940d4ee2a7638c9db350e1bccaa02c4081a60189932`。
- 工厂函数逐字/AST 相同，整类逐字/AST 相同；源回归定向集 101 passed，
  20 subtests，随后增加了三个 simulators-first 冷启动用例。
- 采集器 39、比较器 43 个 CPU 用例；两者已独立缺陷审阅并复核关闭。
- 本夜不宣称 mypy 全绿，也不把既有类型债务隐瞒为已验收。

## 真实模型 A/B

**已完成，六项严格比较全部 PASS。** 北京时间 02:38:48–02:48:11 完成八臂
采集：96 次样本生成、4,608 个新增 token。每个 case 四臂的 token digest 与全部
非计时 metrics digest 分别相同；计时差异仅记录，不用容差洗平逻辑漂移。

| case | main 自重复 | candidate 自重复 | main vs candidate | 每臂新增 token |
|---|---|---|---|---:|
| legacy_static | PASS | PASS | PASS | 576 |
| honest_int8 | PASS | PASS | PASS | 576 |

只对照同一 case 的两版本，不把两 case 的字节差异解释成压缩/吞吐收益：它们还
同时改变温度、top-k 和结算口径。每臂代表性统计见 `summary.json`。

### 固定条件

- water4090 的物理 GPU1：RTX 4090，UUID `GPU-1081b666-dc44-95ea-ead9-832937f52281`。
  GPU3 正被其他人使用，不分配；0/2 不使用。每臂启动检查零 compute process、
  free≥22000 MiB、util≤5%；加载后只允许自身 PID，发现外来 PID 主动退出。
- 同一 venv：Python 3.10.12、torch 2.6.0+cu124、CUDA 12.4、transformers 4.57.6；
  NVIDIA driver 535.154.05（运行后记录，见 manifest）。
- little=draft=缓存的 Qwen3-0.6B，target=Qwen3-1.7B；bf16、不量化。
  两模型词表映射相同，embedding vocab=151936；7/9 个模型文件逐项 size/SHA256
  固定，未声明文件拒绝。submodule 固定为 `d340008` 的同一份代码，不含 head 权重。
- Graph off，无 ARP head/RL，`disable_rl_update=True`；全部网络访问离线。
- 12 条自造固定 prompt，每条 seed=20261103+sample_id，最多新增 48 token，
  自然 EOS 可以更短。不是 GSM8K、不是既有 gammar golden，不评价 accuracy。
- 两个 case 都是 `tridecoding` static 同一验证路径：

| case | temp | transfer top-k | prob bits | round trip | residual |
|---|---:|---:|---:|---|---|
| legacy_static | 0.0 | 32 | 16 | per_transfer | false |
| honest_int8 | 0.7 | 8 | 8 | per_round | true |

每 case 顺序 A1(main) → B1(candidate) → B2(candidate) → A2(main)。每臂 12 样本；
先分别检查 A1/A2 与 B1/B2 自一致，再对照 A1/B1。不以吞吐或容差替代 token/
非计时 metrics 逐位对照。被测源码是 `50c43ba` vs `8836b06`；后续只提交工具与
报告，不改被测 `src/`。本实验不实际调用 S3a 搬走的工厂族；那些路径由类/工厂
AST 相等与 CPU constructor 命中测试保证，不能混称已做其 GPU 验收。

### 隔离与证据

远端工作目录为 `/home/tiantianyi/duodecoding-nightly/20261103-0143`：
两棵 `git archive` 源码、共享固定 submodule 代码、既有缓存模型只读、全新结果/
日志目录。未在原远端仓库切分支/checkout，未复制或更新原 checkpoint。

- 最初只打包 src/configs，预检发现动态 metrics loader 还依赖 eval/utils.py。
  用仅加载 metrics factory 的 CPU 命令最小复现，然后重建包含 src/configs/eval
  的完整归档并断言必需成员；最小复现转绿。未改生产代码或注入 fake dumper。
- 直接 Python 调用 Accelerator 在 RTX4000 需 `NCCL_P2P_DISABLE=1` 和
  `NCCL_IB_DISABLE=1`；同臂共同显式设置，未修改主机全局环境或安装依赖。
- 修复后的 2 prompt × 8 token 基线预检完成，严格 schema 自比 PASS；这只证明
  harness 真能加载/生成/记账，不当作 main-vs-candidate 等价结论。

采集计划/manifest 记录源码归档 hash、driver hash、模型文件 hash、完整命令、
环境、PID、返回码与时间；原始 JSON 与日志同时保留在隔离目录和本地实验工件。
完整计划、模型资产 SHA、八臂 PID/命令/返回码、raw JSON SHA 与运行时间见
`docs/experiments/nightly-20261103/manifest.json`；六份逐字段比较报告见
`comparisons.json`，紧凑统计见 `summary.json`。原始八份结果与日志位于远端
隔离目录，本地 raw JSON/失败预检日志保留在 `._____temp/nightly-refactor-20261103-0143/`。
模型文件在运行后再次逐项验 SHA；GPU1 已释放，未运行他人的 PID 操作。
`manifest.post_run_verification` 保存了逐文件后验 SHA 和零 compute PID 的快照。
完整本地工件为 `._____temp/nightly-refactor-20261103-0143/nightly-refactor-artifacts.zip`
（2,905,066 B、40 个成员，不含模型权重，内含每个成员的 CHECKSUMS）；SHA256：
`f19932fe453a450a5e2e02eecb5897d28bc121a62b8323f2738675c0d6f0b2f1`。

## Standards

审阅基点 `50c43ba...59b80ec`，以下为独立 Standards 轴结论，未与 Spec 轴混排：

**文档/标准项**

1. S2 文档“行号不变”与同提交抽取使行号移动的措辞有歧义。
   已修：只对包化前 5,825 行 WIP 的 S2 文件搬迁成立；与 main 比还有 helper
   抽取的位移，S3a 又统一前移 89。不能跨阶段宣称 caller 文本相同。
2. 新 typed helper 暴露了返回 Any、Optional Tensor/数值赋值的 mypy 信息。
   审阅者注明仓内已有大量诊断、不是当前门禁；本夜没有用类型忽略或改运行表达式
   来换绿，登记后续类型收敛，不把此项当作已修复或新运行时缺陷。

**启发式判断（非硬性违例）**

- Duplicated Code：rounds/policies 测试共享 fake/patch 栈，且 `_RoundFixture`
  手调另一 TestCase 的生命周期。后续应独立出共享夹具，不在本夜混入测试重写。
- Mysterious Name：little 的 `_select_adaptive_little_config` 不返回 top-k，
  main 同族却返回；这是历史语义保留，当前测试锁定，后续应让命名/合同更清楚。
- Data Clumps：策略 helper 持有 simulator 只读 bandwidth/latency；是否改传值需
  同时审计属性读取顺序，不为减少依赖参数而在机械提交里重排观测点。
- Divergent Change：校准 helper 合并两个独立诊断门控；当前原位搬迁且门控有
  独立测试，后续实际变动时再考虑拆开，不预建框架。

兼容 shim 的 Middle Man 由明确 API 合同要求，不当作 smell 缺陷；ruff 可自动
检查的 style 不重复报告。此轴：2 个信息/文档项 + 4 个启发式建议，未有运行时
阻塞项；主要文档歧义已修，类型与夹具/命名建议保留。

## Spec

同一固定基点的独立 Spec 轴没有发现可复现行为/兼容缺陷，文档/范围项如下：

1. P3：legacy wrapper 文档仍指已删 baselines.py。已改为 core.py。
2. P4：baselines_helpers 的 re-export/patch 机制说明过期。已改为包入口适配、
   按函数真实定义模块 patch；factory 另明确在 simulators，不笼统声称都在 core。
3. P4：Register 无 getattr 兜底的既有错误文档顺带纠正，超出纯物理搬家。
   内容已核对正确；本报告明确列为说明修正，Register 代码未变，不伪装成新功能。

此轴：3 个文档/范围项，前两项已修，后一项已明确记入说明；无行为缺失或错误。
S3a 另有独立只读审阅：两工厂/整类逐字与 AST 相同，真实 constructor fake 命中，
多文件指纹正反例成立。设计稿的七项合同问题和工具审阅发现均已修订并关闭；
这些关闭不代表未来 `StageVerification` 实现已获等价性背书。

## 尚未完成

- 真 ARP/RL checkpoint、训练轨迹、CUDA Graph、原模型三元组和 gammar golden。
- 第一层验证的返回类型/快照事件变更：先补调用顺序、量化数据流、gamma=0、
  cache fallback 与 reward 切点特征测试，再确认是否接受 typed result。
- 整体 S3 方法族拆分、共享测试夹具与类型门禁收敛。S3a 不等于 S3 全部完成。
