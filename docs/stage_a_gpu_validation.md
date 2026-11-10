# Stage-A 真实验收工具：支持范围与待执行项

**当前仅实现并用 CPU fake 验证工具。没有执行本轮真实模型、ARP/RL 权重或 CUDA graph 验收。** 历史 tinyQwen/static/12-prompt 结果不代表当前源码。本工具输出 `OBSERVED`，不是 A5 完成、数学无偏、质量达标或性能回归通过。

## 入口和版本

- [旧运行入口](<../tools/run_refactor_smoke.py>)不带 stage 参数时保留 schema 1、旧固定 case/GPU1 行为；只用于明确选择的旧 case/历史回放。
- `--stage stage-a` 必须位于参数首部，显式转到[新 runner](<../tools/stage_a_smoke.py>)。也可直接调用新 runner。
- [新 schema](<../tools/stage_a_smoke_schema.py>)使用 `schema_version: 2`、`profile: stage-a`，绝不把新结果伪装成 v1。
- [比较入口](<../tools/compare_refactor_smoke.py>)按 artifact schema 分流。v1 仍需完整 revision，token、非 timing metrics、配对/资产/配置规则不变；v1/v2 混比无效。

## 已实现的选择范围，不代表已验收

| 模式 | 本地模型角色 | 必需 ARP heads | 必需冻结 RL |
|---|---|---|---|
| `sd` / `dsd` / `dssd` | draft、target | 无 | 无 |
| `adaptive_decoding` | draft、target | draft_target | main |
| `adaptive_tridecoding` / `cee_sd` | little、draft、target | small_draft、draft_target | main、little |
| `tridecoding`（static 消融） | little、draft、target | 无 | 无 |
| `cee_sd_no_arp` | little、draft、target | 无 | main、little |
| `cee_sd_opportunistic` | little、draft、target | small_draft、draft_target | main、little |

DSD/DSSD 的现有 `dist_spec` / `dist_split_spec` 别名亦可选择。资产角色按[能力表](<../src/mode_features.py>)校验；实际绑定从[注册表](<../src/register.py>)读取，不依赖旧类名/`__qualname__`。缺 head/RL 就失败，不换 static、不静默关 ARP/RL。

支持 `eager` / `graph` 请求。**首版只支持一个完整 GPU UUID**。空列表、ordinal、两个/三个 UUID、实际模型 offload/非 `cuda:0` 放置均拒绝；跨设备真实验收仍待资源授权和后续工具扩展，不属于单卡成功的覆盖范围。

## 运行前需由用户另行授权

1. 指定完整 GPU UUID、空闲窗口、明确资产清单、样本/token/缓存/时间预算。
2. 停止源码和资产写入者、冻结本次树；本工具不替用户保证整个运行期间不变。
3. 权重、tokenizer、head、RL 都须已经在本地。HF 离线，无下载/安装、无自动训练或 checkpoint 保存。

显式授权标志不等于替用户申请授权；下面只是**待批准后使用的命令模板，本轮没有执行**：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B tools/run_refactor_smoke.py \
  --stage stage-a --repo-root /home/tiantianyi/code/DuoDecoding \
  --config /absolute/approved-config.json \
  --asset-manifest /absolute/approved-assets.json \
  --output /absolute/new-evidence.json --authorize-gpu-execution
```

配置 JSON 示例（UUID 必须换成用户批准的真实完整 UUID；`behavior` 中未写的字段会解析并全部记录为版本化 profile 默认值）：

```json
{
  "mode": "adaptive_tridecoding",
  "backend": "graph",
  "gpu_uuids": ["GPU-00000000-0000-0000-0000-000000000001"],
  "samples": 2,
  "max_tokens": 48,
  "seed": 20261110,
  "cache_capacity": 512,
  "max_seconds": 600,
  "behavior": {
    "temp": 0.7,
    "gamma": 3,
    "gamma1": 3,
    "gamma2": 2,
    "transfer_top_k": 32,
    "prob_payload_bits": 16,
    "arp_stop_mode": "cumulative"
  }
}
```

`max_seconds` 是加载/样本边界检查的软预算；不能抢占正在执行的模型加载/CUDA kernel，超时不会写成功工件，不自动 kill 进程。样本数 1–12、新 token 数 1–256、容量 64–4096；graph 至少两个样本。固定采用[旧 prompt 集](<../tools/run_refactor_smoke.py#L82-L96>)，batch size 1。`cache_capacity` 整轮固定，不逐 prompt 改构造身份；容量不足拒绝。实际 runner 桶长、verify sizes、capture count 另记录，不能把请求容量当实际捕获容量。

不接受任意 args passthrough。可选 behavior 以[默认 profile](<../tools/stage_a_smoke_schema.py>)为准；资源路径、训练开关、量化、外部 trace 等不能绕过清单传入。`disable_rl_update=True`、EOS stop 开启、无随机通信、未量化固定。执行方法实际 kwargs 单独记录。

### 资产清单

清单顶层必须恰好含 `schema_version: 2`、`models`、`heads`、`rl`、`tokenizer_vocab_sha256`。三个角色映射必须与上表/能力表一致，无需的组用 `{}`。每个 model/head 条目为：

```json
{"path": "/absolute/local/directory", "files": [
  {"name": "config.json", "size": 123, "sha256": "完整64位小写sha256"},
  {"name": "model.safetensors", "size": 456, "sha256": "完整64位小写sha256"}
]}
```

RL 也使用目录条目，另加 `"entrypoint": "best.pth"`。该文件必须在 files 内；latest/best 都显式指向它，不允许生产 loader 自动选另一个 checkpoint。**目录内全部文件**（包括 tokenizer/config、分片、RL `.buffer`）都要声明 size/hash，不能只列示例两个文件。允许嵌套相对文件名和指向 HF blob 的文件符号链接；拒绝目录符号链接、路径穿越、遗漏文件。tokenizer digest 是排序 JSON `get_vocab()` 的 SHA256；所有模型 tokenizer 与运行 tokenizer 均需一致。

清单前后只读重验。[runtime 资源加载](<../src/baselines/runtime.py>)的实际角色、成功的 RL load 路径、[checkpoint provenance](<../src/checkpoint_provenance.py>)哈希、ARP stop mode 和 RL 动作格被记录并核对。RL load 失败转换成非 `RuntimeError` 工具异常，阻止[原 loader](<../src/rl_adapter.py>)的 `.corrupt` rename/legacy fallback；RL update/save 和 torch.save 硬失败，不是 no-op。临时 guards 在 finally 恢复，不拦截 `nn.Module.eval()` 内部的 `train(False)`。文件保护覆盖常用 Python open/rename/unlink 和已声明 blob。带非 `None` 的 `dir_fd` / `src_dir_fd` / `dst_dir_fd` 写操作一律 fail-closed，不按 CWD 猜路径；只读 fd-relative open 仍透传。rename/remove 等 mutation 还保护模型 root 和外部 blob 的祖先目录；不扩大到上级其它文件的普通读写。**这不是任意第三方/native code 的通用文件沙箱**。

## 证据的解释

- 配置/文件守卫先于 torch import；选中 UUID 覆盖 CUDA 可见性，随后核对 torch 实际 UUID/名称/数量；开跑前无 compute、加载及逐样本结束后只有当前 PID，无法识别 PID namespace 就拒绝，永不 kill。
- 使用[CPU gate 的 source_snapshot](<../tools/run_cpu_gate.py#L46-L65>)记录 tracked + untracked 相关源码和 submodule 状态。当前源码身份是完整 snapshot digest，不是只写 HEAD 或旧 core 单文件 hash。前后相同只证明端点；Lead/user 负责冻结期间。
- 记录最终实际参数、registry 绑定、所有包版本、GPU UUID、所有模型实际参数/缓冲区设备、完整资产清单、每样本 prefix/output/metrics/通信账和资源调用计数。effective 模型的 dtype/hf_device_map、head 的 stop_mode/初始 threshold/device、RL 的 SHA/device/action_space/gamma/top-k/threshold 格与 update_count 均为必需字段，并检查类型及配置/资产交叉关系；checkpoint 恢复的 top-k/threshold 格不强行替换为 CLI 默认。欠字段 v2 工件直接无效，不提供旧缺项 fallback。
- 每侧工件先校验所有 common metrics 必需字段及数值类型；8 项 common timing 必需，3 项逐角色 wall time 如出现也须为有限非负数。即使两侧 timing 同为字符串、负数、None 或 bool，也不能获得 exact-replay PASS。
- 依据[protocol 消费查询](<../src/protocols.py>)记录计费开关与 gamma 的 `consumed` / `not_consumed`。prob bits 区分实现读取、能力表允许有损、`pass_through` / `lossy_configured` / `not_consumed`；CLI 传入 8/16/32 不等于实际线上载荷位宽，内省也不是每条条件分支的执行覆盖。能力表不支持的低 bit 配置直接拒绝，含部分别名。
- SD 无通信是合法观测，不套用旧 v1 的 bytes>0 聚合门槛。全部模式仍须非空生成、预算/EOS/前缀/生成计数一致、真实 target Python forward hook 发生。Python forward hook 包括 prefill/warmup/capture，不把它当 CUDA replay 次数。
- Graph 临时 wrapper 调用[现有 runner](<../src/graph_decode.py>)原方法，成功的 `step`、返回非 None 的 `verify` 才记 replay，异常不计，None 记 fallback。记录 `begin_new_sequence` reset、capture count 和弱引用身份序号（runner 与 StaticCache 分别记录）；不同样本同一 live runner/cache、capture count 稳定且实际 replay 才算跨样本复用观测；后续样本必须在该 runner 的**首次 replay 之前**发生匹配 cache/capture count/容量/验证档位的 reset。replay 后才 reset（即使随后又 replay）不能补救这一顺序证据。仅启用 flag/存在对象不算。未触发 replay 或复用会拒绝成功输出。此证据只覆盖记录到的角色/调用，不声称所有模型角色都用 graph。
- CPU fake graph 测试只验证 wrappers/guards/schema，不是 CUDA graph 正确性证据。

### 比较口径

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -B \
  tools/compare_refactor_smoke.py --base /absolute/base.json \
  --candidate /absolute/candidate.json --expect-samples 2 \
  --base-source <base的code.sha256> --candidate-source <candidate的code.sha256> \
  --comparison observations
```

v2 默认 `observations`：严格校验各自工件，匹配配置/资产/实际角色/消费口径/包版本/输入，列 token/metrics 差异；结论 `OBSERVED`（退出 0 仅表示可比观测，**不代表相等**）。`exact-replay` 另要求同源码身份/运行环境，逐 token 和非 timing metrics 严格比较，PASS/FAIL 也只表示该重复回放匹配。不同 RNG 流无需 token 逐位相同；有损概率不能声称无偏。性能、随机 law、近似质量、target-only 对照均需另有预先批准的验收方案；本工具不能单独完成[路线图 A5](<refactor_roadmap_20261110.md#L155-L162>)。

## CPU 定向检查

[新 CPU 测试](<../test/test_stage_a_smoke.py>)覆盖纯配置/schema、假设备/资产、只读/坏checkpoint、recording/restoration/假graph负例和 v2 dispatch；[旧配置测试](<../test/test_refactor_smoke_config.py>)与[旧比较测试](<../test/test_compare_refactor_smoke.py>)继续约束 schema 1。

```bash
CUDA_VISIBLE_DEVICES='' PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  .venv/bin/python -B -m pytest -q -p no:cacheprovider \
  test/test_stage_a_smoke.py test/test_refactor_smoke_config.py test/test_compare_refactor_smoke.py
# 同一命令另加 --import-mode=importlib；不运行真实 runner。
```

五项独立审查修复均先红后绿；最新定向结果：default / importlib **各 312 passed**。上述新/旧工具及新测试的 scoped Ruff 检查通过（使用系统现有 Ruff 0.15.4；共享 `.venv` 无 Ruff 模块，未安装/修改环境）。这些结果仅证明 CPU 工具契约与旧格式兼容，不证明真实 GPU/模型/图已执行。
