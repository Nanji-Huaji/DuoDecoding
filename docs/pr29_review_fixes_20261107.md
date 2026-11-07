# PR #29 审阅修复（2026-11-07）

审阅基点：`50c43ba...a9a61cc`。四项独立审阅按 Standards / Spec 分开进行；本轮在基础分支上修复工具层的三个问题。

## Standards

无文档规范违例。一个非阻塞的 Duplicated Code 建议：后续将 rounds/policies 的公共 patch 栈与手工 TestCase 生命周期整理为共享测试支撑。

## Spec：三个 P2

| 问题 | 修复 |
|---|---|
| NVML 的物理 GPU1 与 CUDA ordinal 1 可能不是同一设备，旧身份检查晚于模型分配 | [采集器](../tools/run_refactor_smoke.py)在运行库导入前绑定目标 GPU UUID，并在 Runner/模型构造之前核对实际可见 CUDA 设备 UUID；保留忙碌/PID 等检查 |
| 两边同时缺少接受计数等非计时字段仍可 PASS | [比较器](../tools/compare_refactor_smoke.py)要求生产者完整的非计时字段集合；缺字段按 INVALID 处理 |
| 两边的通信 trace 都错误时，仅互相比对会 PASS | 每个样本单独核对 trace 字节总和与 aggregate，使用与采集器一致的绝对容差 `1e-6`、相对容差 0 |

CPU mock 复现了旧身份检查允许同型号、不同 UUID 的 CUDA 设备进入 Runner 和模型分配；后续 PID 检查即使发现错误，也已经晚于分配。真实旧 A1/B1 数据的内存变异也复现了：删除双方接受计数仍 PASS；将双方首样本 trace 改为零但保留 3160B aggregate 仍 PASS。

原始实验文件保持只读，既有 GPU 实验的 UUID/PID 证据没有被改写。源码归档由 launcher manifest 和文件 SHA 记录归属；revision 标签不单独视为 Git tree 的密码学证明。

## 验证

- 比较器：默认/importlib 各 89 个用例通过；包含逐字段缺失、账本不一致、浮点累计容差、合法 early EOS，以及合法指标漂移 FAIL 与坏数据 INVALID 的区分。
- 六项原始真实数据比较全部 PASS，八份 raw JSON 的 SHA-256 均未变化。
- 采集器：默认/importlib 各 66 个用例通过，覆盖 UUID 绑定顺序、身份缺失/不符、环境改写及既有 GPU 安全门禁。
- 基础分支全量回归：默认模式 **745 passed、7 skipped、90 subtests passed**（57.65 秒）；importlib 模式同为 **745 passed、7 skipped、90 subtests passed**（61.20 秒）。两种模式各有两个既有警告。
- 修复增量经独立 Standards / Spec 复核：比较器与采集器均无新增 finding，三个 P2 已关闭。

本轮新验证全部使用 CPU mock，不执行真实 CUDA/NVML 查询、模型加载或远端任务。
