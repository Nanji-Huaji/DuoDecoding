# 阶段 A：核心契约与模块拆分状态

本页是本轮实施的现行状态入口；目标见[路线图](<refactor_roadmap_20261110.md>)。历史记录保留其原基点，不视作当前源码验收。

## 授权与边界

- 分支：`refactor/core-contracts-modules-20261110`；起点：`fdfd27a7bb71d99d3465f08fc3f869db7f5fef70`。
- 2026-11-10 的实施范围为当前目录新建分支及 CPU 可执行部分，当时未授权 GPU、下载、现有权重写入、提交和推送。
- 2026-11-11 用户追加授权通过 `gh` 提交、创建并合并 PR，以及连接 `tiantianyi@10.160.8.194`、选择空闲 GPU 做真实验收。不停止他人的进程，不覆盖现有权重；后续 GPU 结果与本页 CPU 记录分别保存。
- 最初按偏好请求 `byr-llm-gw/deepseek-v4.1-flash`，相关 workflow 未返回可采纳结果，已请求取消；后续由可通信队友与 Lead 实现，独立复查使用直接 fork 及只读数值审查队友。未提供或假称调整 reasoning effort。废弃 workflow 不计作已完成审查；Lead 负责集成、最终 diff 和测试。
- “CPU 完成”不等于整个阶段 A 完成；真实模型、ARP/RL 权重、CUDA graph 执行和跨设备验证仍需当前代码的独立运行证据。

## 支持与迁移矩阵

下表基于当前[模式能力表](<../src/mode_features.py>)、[注册机制](<../src/register.py>)和[既有模式盘点](<modes_inventory.md>)。精确/近似列描述目标契约或待核条件，不宣称所有现有实现已经满足。

| 规范入口 | 别名 | 模型 | 保留定位 / 迁移批次 | 分布契约与当前证据边界 |
|---|---|---|---|---|
| `large` | — | target | 自回归参考；A 验收对照 | 指定采样参数下的目标分布；不得用不同配置作恒等比较 |
| `small` | — | draft | 小模型参考；保留 | draft 自回归分布，不是 target 精确基线 |
| `target_only` | — | 当前装配为 tri | 参考；保留 | 目标模型参考实现，装配冗余另案，不顺手改变加载 |
| `sd` | — | dual | 活跃基线；A | 严格实际 q 契约已接入；CPU 窗口/分布/真实微型模型验证，GPU 待验 |
| `dist_spec` | `dsd` | dual | 活跃基线；A | 按实际 proposal q 验证；top-k 采样/重建和传输表示必须对应 |
| `dist_split_spec` | `dssd` | dual | 活跃基线；A | 归一化 selected-Q 按 FP32 标量上行并真实参与验收；明确有限舍入，不承诺实数无舍入传输；拒绝回传顺序独立 |
| `adaptive_decoding` | — | dual | 活跃独立方法；A | 真实 proposal q、ARP/RL 时点与冻结边界；已有 CPU 契约，待当前 GPU 验收 |
| `adaptive_tridecoding` | `cee_sd` | tri | 主方法；A | pass-through 与有损 wire 分开；低位宽不承诺无偏；已有 CPU 契约 |
| `tridecoding` | — | tri | 当前 static 消融；A | 与 CEE 主体同源，关闭 RL/ARP，依配置区分精确/近似 |
| `cee_sd_no_arp` | — | tri | 当前无 ARP 消融；A | 与 CEE 同源，保留 RL；不得与旧独立消融混同 |
| `cee_sd_opportunistic` | — | tri | 当前机会路径；A | 须核对候选来源与最终 target 验证；不能仅因 skip 名称断言有偏 |
| `uncertainty_decoding` | `cuhlm` | dual | 活跃论文基线；B | 机会跳过/重建按论文逐配置核对；不预先承诺 target 精确性 |
| `tk_slt` | `tkslt` | dual | 活跃论文基线；B | 实际 q 与 FP16 wire q 分开；专用受限桥处理量化支撑丢失，不承诺无偏；稀疏提案及 ODLD/AS² 保留 |
| `cee_dsd` | — | tri | 保留组合；B | 双层来源、拒绝、通信与共享验证影响需审查 |
| `cee_dssd` | — | tri | 保留组合；B | 同上，保留分裂协议差异 |
| `cee_cuhlm` | — | tri | 保留组合；B | 独立机会协议与通信口径，不强行并入 CEE-SD 骨架 |
| `ceesd_without_arp` | `ceesd_w/o_arp` | tri | 归档迁移；当前不删除 | 旧血缘，仅历史复现；引用审计后退出活跃接口，不静默重定向 |
| `speculative_decoding_with_bandwidth` | — | dual | 历史对照；本轮保留 | 共享验证调用者，需回归；是否退役尚未授权 |

`small` 与 `large` 共用一个实现，因此上述 18 行对应 17 个已注册实现、24 个注册名。所有受支持 batch 范围以真实实现为准；本阶段不增加 batch>1 能力。

## 实施批次与写入协调

| 批次 | 责任范围 | 状态 |
|---|---|---|
| A0 | 依赖/CPU 门禁、支持矩阵、修改前基线 | 入口与门禁完成；2026-11-11 补齐全新环境安装及干净克隆双模式 CPU 验证，见下方后续记录 |
| A1 | ARP adapter 与独立停止契约测试 | 完成实现、Lead 复核与当前模型独立审查；11 项停止契约双模式通过 |
| A2 | runtime/adaptive/CEE 模块迁移及对应测试/指纹映射 | 完成真实迁移、独立补审和双模式 944 项中间集成门禁 |
| A3 | 无采样前向、有效缓存身份与接线 | 全部接线完成并独立复查；最后三处保留算法的缓存温度失效先红后绿，所有角色/存储/配置覆盖；CUDA 执行仍待验 |
| A4 | SD/DSD/DSSD 验证契约与阶段模块 | CPU 范围完成并独立复查；TK FP16、BF16、极小尾部及标量中点发现均已先红后绿关闭；显式 legacy 桥保留既定 B 阶段边界 |
| A5 | 独立审查、CPU 集成、真实模型验收入口 | 工具 5 项 P2 全部关闭；所有代码写入及必需审查停止后，全量双模式各 1379 passed/7 skipped/163 subtests，端点源码一致；真实 GPU 与跨设备验收未执行 |

写入范围分批在代理任务中明确；不同代理不得同时改同一核心文件。此状态文档及路线图由 Lead 单独维护。迁移或纠错批次之后在冻结源码上测试；不把并发修改期间的测试当最终验收。

## 验证记录

### 修改前基线

Lead 在任何业务代码修改前实跑：默认 **852 passed、7 skipped、160 subtests（52.95 秒）**；importlib **852 passed、7 skipped、160 subtests（49.87 秒）**。两模式均仅两个既有 warning（pynvml 弃用、RL 单位误用测试）。使用以下命令，GPU 隐藏、禁用自动插件和 pytest 缓存，避免宿主插件改变证据：

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider test
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider --import-mode=importlib test
```

本地日志保存在忽略的 `.refactor/stage-a-20261110/` 下；它们不是已提交的可移植证据包。最终结果在本页记录实际命令、数量和范围。

### 第一批中间验证（不是阶段最终验收）

- Lead 新增[CPU 验收入口](<../tools/run_cpu_gate.py>)及[单元测试](<../test/test_cpu_gate_runner.py>)：初版双模式各 9 passed；独立审查后加真实子进程反例，共 10 项。入口不安装依赖，设置 GPU 隐藏及 Hugging Face 离线变量（不是通用网络/设备安全沙箱），清除 `PYTEST_ADDOPTS`/`PYTEST_PLUGINS` 并覆盖配置 `addopts`，记录两模式日志/JUnit、解释器/包版本和包括根 Python 入口的前后源码指纹；拒绝复用旧证据目录，端点快照不同或任一测试失败均返回失败。快照不证明中间从未变更，最终验收仍须先停止所有写入。
- Lead 联合运行 ARP 新测试、缓存新测试、既有 graph wiring/rollback/buffer 和上述入口测试：**67 passed、1 个既有 warning，13.58 秒**。这是模块迁移前的局部验证，不是当前工作树全量结论。
- 当前模型通过直接 fork 完成有界独立审查（先前未返回的 workflow 已请求停止，不计其审查为通过）：ARP 无必须修复项；缓存 extend 的真实 KV/history 不一致漏检已先红后绿修复。依赖门禁审查发现的 hash/marker/完整集合、TOML 缺失 skip、版本解析问题也已修复并通过独立复查。
- 修订后的 ARP（11 项）、缓存（25 项）、CPU 入口（10 项）联合 **46 passed**：默认 5.41 秒、importlib 5.02 秒。缓存新增失败保持状态测试确实在修复前失败。
- [严格验证值接口](<../src/verification.py>)与[独立 toy-law 测试](<../test/test_verification_contracts.py>)双模式各 **25 passed**，scoped Ruff 通过；当前模型独立审查无实现 finding。依据审查建议，分布积分现在同时执行实际接受与拒绝分支，不再解析注入接受质量。该次验证时尚未接入算法，不能单凭叶子测试声称整个算法已纠正；后续接线记录见下。
- Lead 已独立核对 A2 中间源码：原 48 个类方法全部 AST 不变且每项只存在一处；15 项指纹值在 6 项真实入口类名映射后全相等。另有独立审查核对 33 个迁出方法的真实 globals、MRO/init、24 个模式消费及旧 oracle，定向 47 passed/24 subtests；未发现阻断项，scoped imports/留白清理未改变 AST。

### A2 合并后的中间全量门禁

使用 [CPU gate](<../tools/run_cpu_gate.py>)，输出目录为 `.refactor/stage-a-20261110/a2-foundation-cpu`：默认 **944 passed、7 skipped、160 subtests，66.75 秒**；importlib **944 passed、7 skipped、160 subtests，63.37 秒**。两次均仅两个既有 warning；manifest 为 passed，前后源码快照相等。此后 A3/A4/A5 继续修改，不能把这个结果当最终工作树结论。

### 依赖门禁复查与后续接线

- [离线导出工具](<../tools/export_locked_requirements.py>)已实际再生成并检查；清单完整 SHA 与初次 146 记录导出相同，生产声明/锁与 HEAD 逐字节未改。[绑定记录](<../test/golden/dependency_export.json>)同时锚定生产输入和完整清单；三个篡改负例均被阻止。[依赖测试](<../test/test_dependency_contract.py>)默认/importlib 各 **15 passed**，独立复查再次各 15 passed，无剩余实质 finding。隔离工具安装失败的限制仍保留，详见[依赖与 CPU 说明](<cpu_validation.md>)。
- A3 首批接线由可通信队友完成：默认定向 **324+3 passed**，importlib **327 passed/78 subtests**。独立审查抽跑 6 项及小 probe；有效身份、eager 不内省、公开前向的授权 AST 替换、oracle 边界均无阻断。Lead 另将非 strided 权重拒绝固化成测试，连同 meta 边界 2 passed。该批当时尚未覆盖三处保留算法 graph 复用，后续已补齐，见下节。
- [真实双模型 CPU 契约](<../test/test_dual_model_cpu_contracts.py>)先得到有效 **14 failed/7 passed**；独立审查以 9 组真实模型/历史 probe 排除夹具造假。算法迁移后 Lead 补充直接 Decoding 子类并经真实 registry 调用，**22 passed**。它只覆盖无压缩 batch1/eager 常量与均匀分布，不代替时变 q、非平凡残差、通信账或 GPU 证据。

### A3/A4 最后增量与独立复查

- [保留算法缓存测试](<../test/test_legacy_graph_cache_identity.py>)通过公开 CUHLM/TK-SLT/CEE-CUHLM 入口先复现 3 个温度改变却复用旧缓存的红例，再补齐全部角色的对象/存储/dtype、词表、实际图配置及采样参数、等价 reset、eager 不内省。三个原构造器组的调用 AST 完全不变；另 10 个保留方法完全不变。相关图测试默认 **138 passed**，加指纹/兼容入口的 importlib **170 passed**。CPU 零 token 资源测试不表示执行过 CUDA 图。
- A4 默认/importlib 定向各 **369 passed/161 subtests**；独立协议审查另 **67 passed** 并核对通信账、注册/MRO、直接子类、未触及指纹。实际 q 与 wire q、显式 bonus 行、gamma=0、严格 `<`、确认 EOS 后的缓存截断分别有独立证据，不以同 seed 旧 token 完全相等作纠错验收。
- 数学审查发现 TK-SLT 的合法实际 `q=1e-8` 经 FP16 后 wire q=0，与新增的严格支持校验冲突。修复采用[专用 FP16 legacy 接口](<../src/decoding_ops.py>)，只由 TK-SLT 调用：正 p/零 wire q 显式为无穷比率，双方零拒绝，仍严格 `<`；不加 epsilon、不改 wire 元数据、不放松实际 q 接口。量化管线与公开 TK 路径均先红后绿；该批默认/importlib 各 **303 passed/91 subtests**。独立复查再跑 **67 passed** 及极小正 p、等号、坏值先于 RNG 的独立 probe，无剩余 finding。
- [legacy 裁短桥白名单](<../test/test_legacy_verification_bridge.py>)限定 core 四处及旧 stage wrappers 两处调用；旧公开名仅为兼容 alias，新的严格模块不得导入旧验证链。Lead 移除这次纯命名/alias 补丁后，两份生产文件 SHA 与上述独立审查快照逐字节相同。默认相关 **145 passed/15 subtests**；加指纹/兼容入口后 importlib **177 passed/15 subtests**。
- 指纹变更有界：A4 仅 SD/DSD/DSSD 入口及其真实模块映射；后续 A3 三项为 CUHLM **88→91**、TK-SLT **168→171**、CEE-CUHLM **133→136**。最后 legacy 命名仅替换对应 1/1/2 个调用名、事件数不变；其它条目保持，解析器不改。指纹只是结构证据，不代替正确性测试。

### A5 工具与最终集成

- [新验收入口](<../tools/stage_a_smoke.py>)及[独立 schema2](<../tools/stage_a_smoke_schema.py>)由旧入口显式 `--stage stage-a` 分流；schema1 历史回放保留。当前仅支持一个明确 GPU UUID，跨设备验收待扩展和授权；真实资源、成功 replay、reset/reuse、端点源码与协议位宽消费状态均须记录。输出 `OBSERVED` 不等于数学正确或整个 A5 通过，详见[验收边界](<stage_a_gpu_validation.md>)。
- 独立审查发现 5 项 P2：fd 相对写绕过保护、可移动包含模型/HF blob 的祖先目录、effective 资源字段缺失仍通过、两侧同样非法 timing 被放过、reset 晚于 replay 仍计复用。全部分别先红后绿修复；强化 schema2 不接受欠字段旧工件，schema1 未改；定向默认/importlib 各 **312 passed**。权重路径负例用无副作用底层替身，并未实际写入/移动权重。
- 工具修后独立复查：5 项全部关闭，无剩余 finding；**263 个内存断言、13 项定向测试**，并以真实 `effective_resources` 覆盖 11 模式、真实 runner 序列化链配惰性资源替身覆盖 4 模式，不只是两份相同 fake 相比较。未调用真实 loader、权重或 GPU。
- 在上述实现停止写入后，全量默认/importlib 各 **1330 passed、7 skipped、163 subtests**（89.35/77.18 秒），两项既有 warning、端点源码相同。工件位于 `.refactor/stage-a-20261110/final-cpu`，但随后独立集成审查发现下述 BF16 P1，因此这个目录名不能代表当前最终验收；修复后须另建目录重跑。

### 最终集成审查新增的 BF16 P1

- 真实 CPU Llama/BF16 + KVCache 在 SD/DSD/DSSD 三个公开入口均复现：原 cache 行近似单位和，但实际 sampler 先转 FP32 归一化；验证直接用 BF16 行可产生假的零正残差并中止。Lead 新增三个有效红例，不把此前全 FP32 小模型测试当成低精度覆盖。
- [统一采样权重预处理](<../src/sampling.py>)仅抽出原 `sample` 预处理，不改变旧 sampler 的 weights/RNG/fallback。严格验证先拒绝非法原始行，再从相同 weights 计算实际 law；归一化总量及残差用 float64，采用 `delta - q * sum(delta)/sum(q)`，不使用容易丢小尾的 `sum(p)-sum(q)`、epsilon 或零残差回退。
- bonus 及 DSSD 拒绝回传保留 raw target 行，只对真正采样的输入预处理一次；DSD full-q 保留 raw dtype、可完整重建 sampler law。DSSD 明确发送和消费同一个 FP32 归一化 selected-Q（目标标量也按 FP32 舍入）：BF16 标量 **2→4 字节**，原 FP32 消息账常数不变。内部 double 算术不等于 FP64 上线协议，亦不承诺 FP32 wire 无舍入。
- [新增数值回归](<../test/test_sampling_law_contracts.py>)五个有效红例覆盖 BF16、`1e-8/1e-20/1e-38/1e-44` 尾部；连同三个真实入口均转绿。真实 BF16 独立字节账为 DSD **70 字节**、DSSD **56 字节**（gamma=1、budget=2、一次拒绝后 target-only），且最后采样 weights 与 AR 逐字节相同。原非平凡协议残差断言仅同步 float64 期望 dtype，不松容差或改 FP32 消息账。
- toy-law 积分改用独立定义的实际 sampler weights，而非 raw cache 行；扩充 FP16/BF16/FP32/FP64，并保留 **1e-12** 误差界及增加接受边界 `nextafter` 检查。初版相关定向 **140 passed**、scoped Ruff 通过。
- 独立协议/兼容复查关闭：从固定点单独编译旧 `sample` AST，对 **47 类输入、94 次调用**逐位核对 weights、输出/异常、RNG 和输入未修改；另 **28 项**真实 BF16/消息/终止测试通过，无 finding。数学复查仍发现全局比例差会让简单 `sum(delta)` 丢失 `1e-38` 小尾；Lead 新增四种尾大小、full/FP32 scalar、前后位置共 16 例，先 **8 failed/8 passed**，未放宽残差 guard。
- 进一步在最大正 Q 的 pivot 上做交叉差 `delta = p*q[k] - q*p[k]` 再中心化；FP32 weights 的乘积在 float64 中精确可表示，pivot 差精确为零，先去掉全局比例差再累计小尾。实数结果为 `q[k]*sum(p)*(P-Q)`，只改变正部归一化前的公共尺度，不是 epsilon/fallback。上述 16 例均转绿，定向共 **156 passed**、Ruff 通过；随后独立复查结果见下。
- 数值复查确认 pivot 修复本身无 finding（额外 CPU/Fraction 矩阵 1620 对、3972 次拒绝，最大残差绝对误差 `1.11e-16`），但又构造出 scalar midpoint 边界：实际 weights 精确同比例时，直接归一化可在 FP32 中点两侧舍入，导致合法假拒绝。项目 PyTorch 2.6.0 持久例先 **4 failed**（3 个真实 abort、1 个 scalar 不一致），再以 `weights/max(weights)` 的 float64 相对表示计算 selected law；同一比例类得到相同表示，未使用接受兜底或残差回退。
- 最后定向 **160 passed**、Ruff 通过；此前含指纹/保留基线的 importlib **257 passed/15 subtests**。非二进制 toy-law 的 `nextafter` 检查先将实际 scalar ratio 与独立真实 law 校准至相对 **1e-15**，再检查该表示的严格边界；独立输出律积分仍保留 **1e-12**，没有用放宽积分容差遮掩末位差异。项目解释器上的实际最终补丁独立复查已关闭，无剩余 finding：PyTorch **2.6.0+cu124、CPU**，128 次 midpoint 接受、289 次拒绝、66 次输出律积分、空候选与 RNG 检查通过；最大残差误差 `1.11e-16`，最大积分误差 `1.313e-16`。前述更广的矩阵使用系统 PyTorch **2.9.1+cu128、CPU**，只算额外证据，未混称项目环境验证。
- 本批仅 DSSD 指纹发生已解释变化：事件数 **96 不变**，`squeeze` 叶调用变成显式 `float`，SHA `d419e3f6ee9ef8eb→49a1e37adf1aa08f`。其它 baseline 条目及 engine/CEE 指纹不变，解析器不改。

### 当前工作树最终 CPU 验收（2026-11-10）

所有实现队友与必需只读审查结束后，Lead 核对最后审查的两份生产文件、两份测试 SHA，停止源码写入，再运行：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_cpu_gate.py \
  --output-dir .refactor/stage-a-20261110/final-cpu-r2
```

| 模式 | passed | skipped | subtests | 用时 | 进程状态 |
|---|---:|---:|---:|---:|---|
| default | 1379 | 7 | 163 | 77.88 s | exit 0 |
| importlib | 1379 | 7 | 163 | 75.21 s | exit 0 |

[默认日志](<../.refactor/stage-a-20261110/final-cpu-r2/default.log>)、[importlib 日志](<../.refactor/stage-a-20261110/final-cpu-r2/importlib.log>)及[manifest](<../.refactor/stage-a-20261110/final-cpu-r2/manifest.json>)保留在本地忽略目录。两次只有原有 pynvml 弃用及 RL 单位测试 warning；manifest 为 `passed`、`source_unchanged=true`，使用项目 Python 3.10.18、torch 2.6.0、transformers 4.57.6、pytest 9.1.1。随后只更新结果文档，没有再改源码。

这完成的是已授权的阶段 A **代码与当前环境 CPU 验收范围**，不是整个阶段 A 验收完成：A0 隔离安装仍因离线缓存缺 pytest 未验证；真实模型/ARP/RL、CUDA capture/replay 和跨设备验收待另行授权及资源。旧 workflow 未返回的只读审查已请求取消，不计作成功证据。截至上述 CPU 门禁结束时，未 commit/push、未使用 GPU、未下载模型、未修改现有权重或重装共享环境。

### 2026-11-11 后续记录

- 阶段 A 已通过 [PR #31](https://github.com/Nanji-Huaji/DuoDecoding/pull/31) 合并，提交为 `58034cd7ce8063ca1046c41c79fcb6c1456e07ef`。
- CUHLM 首批真实迁移及三项决策修复已通过 [PR #32](https://github.com/Nanji-Huaji/DuoDecoding/pull/32) 合并；其双模式 CPU 结果各为 1385 passed，见[批次记录](<cuhlm_refactor_20261111.md>)。
- 远端初次干净克隆得到 18 项资源/路径相关失败，未将其记作通过。后续修复测试可移植性，并在全新依赖环境验证：两模式各 1386 passed、21 skipped、163 subtests；14 项缺失权重明确跳过，严格模式则如期失败。本机完整资产严格门禁各 1400 passed、7 skipped、163 subtests。安装过程、源码哈希核对及结果见[可移植性记录](<cpu_portability_20261111.md>)。

### GPU 门槛

远端 GPU 源码固定在 `58034cd`，22 个配置及所需资产已准备完毕；等待器要求连续两次无计算进程、低显存占用且利用率为零，才启动单卡实验。目前未获得空闲卡，真实验收尚未执行。不得以历史工件或 CPU 测试替代它，也不得占用他人的 GPU 进程或改写既有权重。
