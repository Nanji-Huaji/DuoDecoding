# Baselines Module


> [!warning] 本文档由AI生成，未经人工审阅

本仓库中与“对比基线 / 组合方法”相关的解码实现集中在 `src/baselines.py`，核心类为 `Baselines(Decoding)`。评测脚本（例如 `eval/eval_gsm8k.py`、`eval/eval_mt_bench*.py`）会继承 `Baselines`，并通过 `args.eval_mode` 选择具体解码方法执行。

## 入口与调用方式

### 1) 通过 `--eval_mode` 选择方法

`Decoding` 继承自 `Register`，`Register.get_decoding_method()` 会按如下顺序取方法：

1. 优先从注册表 `_DECODING_REGISTRY` 中查找 `eval_mode`（由 `@Register.register_decoding("...")` 注册）。
2. 兜底：如果类上存在同名方法，则用 `getattr(self, eval_mode)`。

因此，`Baselines` 中大量方法会通过装饰器同时注册多个别名（例如 `dssd` 与 `dist_split_spec`）。

### 2) 评测脚本如何传参（推荐参考写法）

以 `eval/eval_gsm8k.py` 为例，会先取到解码函数，再用 `functools.partial` 绑定通信/早停等通用参数：

```python
decoding = self.get_decoding_method()
decoding = partial(
    decoding,
    transfer_top_k=args.transfer_top_k,
    use_precise_comm_sim=args.use_precise,
    use_stochastic_comm=args.use_stochastic_comm,
    ntt_ms_edge_cloud=args.ntt_ms_edge_cloud,
    ntt_ms_edge_end=args.ntt_ms_edge_end,
    use_early_stopping=args.use_early_stopping,
)
output_ids, metrics = decoding(input_ids)
```

所有 `Baselines` 方法通常返回：

- `output_ids: torch.Tensor`：形状 `[batch, seq_len]`
- `metrics: DecodingMetrics`：见本文档 “Metrics 字段” 小节

## 方法总览（eval_mode → 实现）

> 下面的 “模型需求” 指 `load_model()` 会加载哪些模型；2 模型方法使用 `draft_model + target_model`，3 模型方法使用 `little_model + draft_model + target_model`。

| eval_mode（别名） | 模型需求 | 方法名（`src/baselines.py`） | 核心特征（高层） |
|---|---:|---|---|
| `dssd`, `dist_split_spec` | 2 | `dist_split_spec` | DSSD：uplink 仅传 `token id + q(x)` 标量；reject 时下发整行 `P(x)` 并在端侧按 `max(P-Q,0)` 重采样 |
| `dsd`, `dist_spec` | 2 | `dist_spec` | DSD：传 draft token 序列 +（可 top-k）draft 概率窗口，批量验证 accept/reject |
| `cuhlm`, `uncertainty_decoding` | 2 | `uncertainty_decoding` | CUHLM：按不确定度机会传输/压缩传输概率分布（当前实现阈值固定 0.8） |
| `tridecoding` | 3 | `tridecoding` | 三级：Little→Draft→Target，两层投机验证与通信（edge-end + edge-cloud） |
| `ceesd_w/o_arp`, `ceesd_without_arp` | 3 | `ceesd_without_arp` | CEE-SD 去 ARP head 的消融：可启用 RL 仅调 top-k（不依赖 ARP） |
| `adaptive_decoding` | 2 | `adaptive_decoding` | 2 模型 + ARP head：草稿侧每步可提前 stop（自适应 draft 长度） |
| `cee_sd`, `adaptive_tridecoding` | 3 | `adaptive_tridecoding` | 3 模型 + 2 个 ARP head：两层都可自适应 stop；可启用 RL 动态调参 |
| `cee_cuhlm` | 3 | `cee_cuhlm` | 3 模型 + CUHLM：在 CEE 框架内加入不确定度机会传输 |
| `cee_dssd` | 3 | `cee_dssd` | 3 模型 + DSSD 风格：更偏“串行/分裂式”验证与回传 |
| `cee_dsd` | 3 | `cee_dsd` | 3 模型 + DSD 风格：更偏“并行/批量”验证（一次性传窗口概率） |

## 通用参数（CLI / args）

这些参数由 `src/utils.py:parse_arguments()` 提供，通常由评测脚本透传给具体解码方法。

### 模型与采样

- `--draft_model`：草稿模型（小于 target）
- `--target_model`：目标模型（用于验证）
- `--little_model`：更小的模型（仅 3 模型方法用）
- `--max_tokens`：最多生成 token 数（相对 prompt）
- `--temp --top_k --top_p`：采样超参（部分方法会将 `transfer_top_k` 同时用作草稿侧 top-k）

### speculative 长度

- 2 模型：`--gamma`
- 3 模型：`--gamma1`（Draft→Target 层），`--gamma2`（Little→Draft 层）

### 通信模拟（带宽/RTT/压缩）

- `--edge_cloud_bandwidth --edge_end_bandwidth --cloud_end_bandwidth`（单位 Mbps）
- `--transfer_top_k`：通信中传输概率分布时的 top-k（也常被用作草稿侧 top-k）
- `--use_precise`：启用“物理层”精细通信模拟（在代码里对应参数名 `use_precise_comm_sim`）
- `--use_stochastic_comm`：启用随机带宽轨迹（若模拟器支持）
- `--ntt_ms_edge_cloud --ntt_ms_edge_end`：链路 NTT（毫秒）

### 排队延迟与早停

- `--batch_delay`：每次 target forward 叠加的排队延迟（秒），用于 `queuing_time`
- `--use_early_stopping`：开启后，解码循环会在满足停止条件时提前结束（EOS 或 `stop_sequences`）

## ARP Head / 自适应解码相关参数

ARP head（`AcceptancePredictionHead`）用于根据 hidden states 预测“是否停止继续 draft”，并由 `DecodingAdapter` 结合阈值做停止判定。

### `adaptive_decoding`（2 模型 + 1 个 head）

- `--acc_head_path`：head 路径（`AcceptancePredictionHead.from_pretrained`）
- `--draft_target_threshold`：阈值（`DecodingAdapter.threshold`）

### `adaptive_tridecoding` / `cee_sd` / `cee_cuhlm`（3 模型 + 2 个 head）

- `--small_draft_acc_head_path` + `--small_draft_threshold`：Little→Draft 层
- `--draft_target_acc_head_path` + `--draft_target_threshold`：Draft→Target 层

## RL Adapter（可选）

启用方式：`--use_rl_adapter`。

- 主 RL：`--main_rl_path`（默认指向 `checkpoints/best/.../rl_adapter_main.pth`）
- 小 RL：`--little_rl_path`（默认指向 `checkpoints/best/.../rl_adapter_little.pth`）
- `--disable_rl_update`：只推理不更新（避免在线训练）

注意：不同方法中 RL 介入点不同；常见行为包括动态选择 `transfer_top_k`、ARP 阈值，或在 little 层动态调整 `gamma2`。

## 各方法说明（API + 行为）

下面仅描述 `src/baselines.py` 中注册的方法；`sd`（标准 speculative decoding）位于 `src/engine.py`。

### 1) `dssd` / `dist_split_spec`（DSSD，2 模型）

签名要点：

- `dist_split_spec(prefix, transfer_top_k=300, use_precise_comm_sim=False, use_stochastic_comm=False, ntt_ms_edge_cloud=200, ntt_ms_edge_end=20, use_early_stopping=False, stop_sequences=None, ...)`

协议/通信要点（代码内 docstring 也有描述）：

- uplink（端→边/云）：只发送 draft token ids + 对应的标量 `q_j(x_j)`（draft 在该 token 上的概率）
- 验证在 edge 侧用 target probs 进行 accept/reject
- reject 分支：downlink 发送被拒位置的整行 `P_j(x)`（target 分布），端侧用缓存的 `Q_j(x)` 做 `norm(max(P-Q,0))` 重采样
- all-accepted 分支：只回传“下一个 token”

常见 metrics：

- `avg_top_k`, `avg_draft_len`
- `draft_forward_times`, `target_forward_times`
- `draft_generated_tokens`, `draft_accepted_tokens`, `generated_tokens`
- `communication_time`, `edge_cloud_data_bytes`

### 2) `dsd` / `dist_spec`（DSD，2 模型）

签名要点同上（`dist_spec`）。

行为要点：

- 先把 prefix（首次）与 draft 的 token 序列传到 edge-cloud
- 再传一段概率窗口（可选 top-k 压缩）用于验证 accept/reject
- reject 时从 `max(P-Q,0)` 采样新 token；全接收时从 target next probs 采样

### 3) `cuhlm` / `uncertainty_decoding`（CUHLM，2 模型）

签名要点同上（`uncertainty_decoding`）。

行为要点：

- 每步生成 1 个 draft token，并计算该步不确定度 `uncertainty`
- 若不确定度高：传输（压缩的）概率分布用于验证/采样；否则倾向于“机会接受”
- 当前 baselines 实现中 `uncertainty_threshold` 固定为 `0.8`（未从 `args.uncertainty_threshold` 读取）

### 4) `tridecoding`（3 模型）

签名要点：

- `tridecoding(prefix, transfer_top_k=300, use_precise_comm_sim=False, use_stochastic_comm=False, ntt_ms_edge_cloud=10, ntt_ms_edge_end=1, use_early_stopping=False, stop_sequences=None, ...)`

行为要点：

- Layer1（edge-end）：Little draft `gamma2`，由 Draft 验证；必要时传概率用于 reject 重采样
- Layer2（edge-cloud）：Draft draft `gamma1`，由 Target 验证；批量传输 token/prob 窗口以节省 RTT
- 同时统计 edge-end 与 edge-cloud 两条链路的通信耗时/字节数

### 5) `ceesd_without_arp`（3 模型，消融：无 ARP）

行为要点：

- 不加载/不使用 ARP head（但允许 RL adapter 仅调整 top-k 等配置）
- 结构仍是两层（Little→Draft→Target）投机与通信

### 6) `adaptive_decoding`（2 模型 + ARP）

行为要点：

- 草稿侧每步 forward 后取 hidden states，经 ARP head 估计“继续生成会被接受”的概率序列
- `DecodingAdapter` 用阈值（`draft_target_threshold`）做停止判定，从而得到“可变长度”的实际 `gamma`
- 可选 RL：动态选 `transfer_top_k` 与 adapter 阈值

### 7) `adaptive_tridecoding` / `cee_sd`（3 模型 + 2×ARP）

行为要点：

- Little→Draft 层使用 `small_draft_adapter` 控制本层 draft 长度（自适应停止）
- Draft→Target 层使用 `draft_target_adapter` 控制本层 draft 长度（自适应停止）
- 统计中额外包含 `arp_overhead_time` / `dra_overhead_time`（若启用 RL/ARP 相关计时）

### 8) `cee_cuhlm`（3 模型 + CUHLM）

行为要点：

- 在 CEE（两层投机）框架下，将 Draft→Target 或对应链路上的传输策略替换为 CUHLM 的“不确定度机会传输”
- 与 `uncertainty_decoding` 一样，当前阈值固定 0.8

### 9) `cee_dssd`（3 模型，DSSD 风格）

行为要点（高层）：

- 组合 edge-end 与 edge-cloud 两层结构
- 更偏“分裂式/串行”的验证与回传策略（对应 DSSD 的“reject 才下发整行分布”思路）

### 10) `cee_dsd`（3 模型，DSD 风格）

行为要点（高层）：

- 组合 edge-end 与 edge-cloud 两层结构
- 更偏“并行/批量”的验证：一次性传输窗口概率，再在接收端完成验证循环

## Metrics 字段（`DecodingMetrics`）

`metrics` 由 `src/engine.py:get_empty_metrics()` 初始化为 0/空列表，不同方法会填充其中一部分字段。

常见字段：

- forward 计数：`little_forward_times`, `draft_forward_times`, `target_forward_times`
- token 计数：`generated_tokens`, `little_generated_tokens`, `draft_generated_tokens`, `little_accepted_tokens`, `draft_accepted_tokens`
- 时间：`wall_time`（秒）、`communication_time`（秒）、`computation_time`（秒）、`queuing_time`（秒）
- 吞吐：`throughput = generated_tokens / wall_time`
- 通信字节：`edge_cloud_data_bytes`, `edge_end_data_bytes`, `cloud_end_data_bytes`
- 通信能耗/连接：`comm_energy`, `connect_times`
- 统计：`avg_top_k`, `avg_draft_len`
- 历史（用于画图/分析）：`edge_cloud_bandwidth_history`, `edge_cloud_topk_history`, `edge_cloud_draft_len_history`

## 可运行示例（2 模型 / 3 模型）

下面示例以 `eval/eval_gsm8k.py` 为例（其他评测脚本用法类似）。请按你的环境替换模型名与 head 路径。

### 示例 A：2 模型 DSSD（`--eval_mode dssd`）

```bash
accelerate launch --num_processes 1 eval/eval_gsm8k.py \
  --eval_mode dssd \
  -e demo_dssd \
  --draft_model <your_draft_model> \
  --target_model <your_target_model> \
  --max_tokens 256 \
  --gamma 4 \
  --edge_cloud_bandwidth 20 \
  --transfer_top_k 300 \
  --ntt_ms_edge_cloud 200 \
  --ntt_ms_edge_end 20
```

### 示例 B：3 模型 CEE-SD（`--eval_mode cee_sd` / `adaptive_tridecoding`）

```bash
accelerate launch --num_processes 1 eval/eval_gsm8k.py \
  --eval_mode cee_sd \
  -e demo_cee_sd \
  --little_model <your_little_model> \
  --draft_model <your_draft_model> \
  --target_model <your_target_model> \
  --max_tokens 256 \
  --gamma1 5 --gamma2 5 \
  --edge_end_bandwidth 100 \
  --edge_cloud_bandwidth 20 \
  --cloud_end_bandwidth 100 \
  --transfer_top_k 300 \
  --small_draft_acc_head_path <path_to_small_draft_head> \
  --draft_target_acc_head_path <path_to_draft_target_head> \
  --small_draft_threshold 0.8 \
  --draft_target_threshold 0.8
```

## 常见坑 / 备注

- `transfer_top_k` 在多处同时影响“通信 top-k 压缩”与“草稿侧 KVCacheModel 的 top-k”，因此改它会同时改变计算与通信行为。
- `adaptive_*` 模式依赖 `output_hidden_states=True`（`load_model()` 已按这些 mode 自动设置），且对应的 ARP head 路径必须存在。
- `Baselines.load_model()` 会在加载后从实际 embedding 层取 `vocab_size`，避免 `config.vocab_size` / tokenizer 不一致导致的越界问题。
