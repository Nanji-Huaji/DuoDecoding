# CLI 参数清单（103 个，按"谁消费它"分类）

> 生成方式：AST 解析 `src/utils.py` 的 `add_argument`，再按精确模式
> `args.<dest>` / `getattr(args, "<dest>")` 扫描全仓（排除 `SpecDec_pp/` 与
> vendored 的 `src/model/rest/`），最后按消费文件归类。
> **103 = 日常出表 16 + 条件开关 68 + 训练专用 19。**换算路径：117
> （`492488c`）— `7b8c9f0` 删 12 个死参数并新增 `--protocol` → 106 —
> `5b5d3ee` 删 3 个 + 17 处 call site → **103**。
>
> 已知盲区：若代码用 `_pair("name")` 或变量形式读取（如 `auto_train_manager.py`
> 的 `_pair("curriculum_bw_start")`），本扫描会漏报，所以"仅训练消费"是下界。

## 0. `--help` 的 9 个分组：这些参数是干什么的

`src/utils.py` 用 `add_argument_group` 把 103 个参数分成 9 组（只影响 `--help`
排版，不改任何默认值）。这张表就是"这么多参数主要干什么"的答案：

| 组 | 个数 | 干什么 | 你多久动一次 |
|---|---|---|---|
| ① 日常出表 | **16** | 模型、评测模式、投机深度 γ、生成长度、协议名 | **每次都动** |
| ② 数据集与采样 | 7 | 数据集路径、样本数、采样与 seed | 换数据集时 |
| ③ 通信数值与计费口径 | 17 | L0 口径：带宽/NTT/往返计费/残差/top-k cap | 只在定口径时动 |
| ④ trace 回放 | 4 | 用实测 trace 取代解析式带宽时延模型 | 做实测回放时才动 |
| ⑤ 解码方法与准确率头 | 9 | 早停、三个阈值、准确率头路径 | 调方法时动 |
| ⑥ RL adapter | 35 | RL adapter 的挂载、路径、reward 整形 | 多数只在训练里读 |
| ⑦ 部署、精度与显存 | 6 | dtype、量化、多卡、验证图档位 | 换机器时动 |
| ⑧ curriculum | 5 | 训练用的网络条件课程 | 只在训练里动 |
| ⑨ 输出与裁判 | 4 | 结果落盘、MT-Bench 裁判 | 需要细看输出时 |

也就是说：**⑧+⑥ 的大部分属于训练，③+④ 属于"定一次口径就不再动"，
① 是你真正每次要碰的 16 个。** ③ 才是论文相关的那一层（冻结值见
`docs/protocol.md`）。

> 接下来 §1–§4 是**另一个轴**的分类：不看用途，而看"谁读了它"。
> 两个轴都成立，用途轴解释"为什么有这么多"，消费轴回答"哪些其实没用"。

## 1. 日常出表真正要动的（16）

出一次论文表，改这些就够了；其余全部走默认或由协议代填。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--draft_model` | `None` | 必填 |
| `--eval_data_num` | `80` | number of samples to evaluate |
| `--eval_mode` | `'small'` | eval mode |
| `--exp_name` | `'test'` | folder name for storing results |
| `--gamma` | `4` | guess time |
| `--gamma1` | `4` | The number of guesses for the first d |
| `--gamma2` | `4` | The number of guesses for the second |
| `--little_model` | `'vicuna-68m'` | The little model for decoding |
| `--max_tokens` | `1024` | max token number generated |
| `--num_shots` | `0` | number of shots for few-shot evaluati |
| `--protocol` | `'none'` | 命名口径（唯一真源: src/protocols |
| `--random_sample` | `'store_true'` | Randomly sample eval_data_num example |
| `--sample_seed` | `1234` | Random seed used when --random_sample |
| `--target_model` | `None` | 必填 |
| `--temp` | `0.2` | temperature for generating new tokens |
| `--use_cuda_graph` | `'store_true'` | 把定长 decode 前向捕获成 CUDA Graph 回放，绕开 per |

## 2. 死参数（3，已删除）：没有任何代码读 `args.<name>`

已于 2026-09-29 全部删除，连同 call site；CLI 参数 106 → **103**。同一批还删了
先前查实的 12 个（`--level` `--guess` `--max-token-span` `--num-draft`
`--dtype_comm` `--adaptive_debug_log` 及 `--controlled_*` 6 个），它们本就没有
call site。详见 `docs/protocol.md` §6。

| 参数 | 默认 | 删除原因 |
|---|---|---|
| `--datastore-path` | `'datastore/'` | REST 检索遗留，无 call site（同名的 `args.datastore_path` 只存在于 vendored 包自己的 CLI） |
| `--task_name` | `'unknown'` | **17 处传参、0 处读取**：`exp.py` 每条命令都推导传入，12 个 `scripts/*.sh`、`cmds/train_rl.sh` 也都传，但无人读。task 实际由各评测类硬编码 `self.task`（`src/baselines.py:578`） |
| `--window` | `10` | lookahead decoding 遗留，无 call site |

## 3. 只被 RL 训练路径消费（19）：评测完全读不到

只出现在 `auto_train_manager.py` / `src/rl_*.py` / `src/model/rest/**` / `*train*` 里。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--rl_action_space` | `'topk_thr'` | Action space of the RL adapters |
| `--rl_buffer_size` | `5000` | Replay-buffer size of the DDQN agents |
| `--rl_byte_price` | `0.0` | Price of transferred bytes, expressed |
| `--rl_charge_queue` | `'store_true'` | Reward 修复：把每轮排队时延（batch_delay）计入 lagr |
| `--rl_compute_cost_json` | `None` | JSON with seconds per forward pass, e |
| `--rl_compute_time_mode` | `'wall'` | Where the reward's compute time comes |
| `--rl_factored_q` | `'store_true'` | Use a branching (factored) dueling Q- |
| `--rl_force_gamma` | `None` | Ablation: pin the draft length gamma |
| `--rl_gamma_candidates` | `'2,4,8,16'` | Comma-separated draft lengths for --r |
| `--rl_include_gamma_in_state` | `'store_true'` | Append the last chosen draft length t |
| `--rl_reward_deadline_ms` | `0.0` | Per-decision deadline (ms) for --rl_r |
| `--rl_reward_deadline_penalty` | `1.0` | Penalty per second of deadline overru |
| `--rl_reward_energy_weight` | `0.0` | Weight mu of the communication-energy |
| `--rl_reward_lambda` | `0.0` | Shadow price of time (tokens/second) |
| `--rl_reward_log_window` | `200` | Window of the adapter's windowed rewa |
| `--rl_reward_mode` | `'legacy'` | Reward for the RL adapters |
| `--rl_reward_no_alpha2` | `'store_true'` | Drop the (N_acc/gamma)^2 factor from |
| `--state_bw_scaling` | `'linear'` | How the bandwidth state feature is sc |
| `--state_latency_scaling` | `'linear'` | How the latency state feature is scal |

## 4. 条件开关（68）：只在特定运行里才有意义

出现 RL adapter、CUDA Graph、trace 回放、量化、多卡、curriculum 等条件时才需要。

### A 通信与计费口径（20）

| 参数 | 默认 | 消费路径 | 说明 |
|---|---|---|---|
| `--batch_delay` | `0.05` | core解码×2,其它×1 | The delay time added to each batch in |
| `--charge_residual_payload` | `True` | core解码×2,eval×1,测试×1 | F43：如实计入拒绝位置残差采样所需的提案分布载荷 |
| `--cloud_end_bandwidth` | `100.0` | core解码×2,eval×1,其它×1 | The bandwidth between cloud and end d |
| `--comm_accounting` | `None` | 其它×1 | L1 口径收敛总开关：honest = 残差计费 + per_round |
| `--comm_round_trip_mode` | `'per_round'` | core解码×2,eval×1,测试×1 | 通信往返口径：per_round=同一轮内每条链路合并为一次往返（成批实现 |
| `--comm_trace_mode` | `'static'` | core解码×1 | 随机通信 trace 的移动模式（配合 --use_stochastic_ |
| `--curriculum_ntt_end` | `'0,5'` | eval×1 | Curriculum edge-cloud latency range ( |
| `--curriculum_ntt_start` | `'0,5'` | eval×1 | Curriculum edge-cloud latency range ( |
| `--edge_cloud_bandwidth` | `20.0` | core解码×3,eval×2,其它×1,测试×1 | The bandwidth between edge and cloud |
| `--edge_end_bandwidth` | `100.0` | core解码×2,eval×2,其它×1,测试×1 | The bandwidth between edge and end de |
| `--min_bandwidth_mbps` | `5.0` | core解码×2 | Minimum bandwidth floor (Mbps) applie |
| `--ntt_ms_edge_cloud` | `200.0` | core解码×1,eval×9,测试×1 | The network time delay between edge a |
| `--ntt_ms_edge_end` | `20.0` | core解码×1,eval×9 | The network time delay between edge a |
| `--ntt_trace_file` | `''` | core解码×1,eval×1 | 真实 RTT trace 回放（sigcomm ping/ 实测，与 th |
| `--ntt_trace_scale` | `1.0` | core解码×1,eval×1 | RTT trace 回放缩放（1 |
| `--prob_payload_bits` | `16` | core解码×1,测试×1 | 传输概率载荷的位宽（16=与历史一致；8=int8 量化；4=上界数据点） |
| `--stochastic_ntt` | `'store_true'` | core解码×1,eval×1 | L1：edge-cloud NTT 动态化（要求 --use_stocha |
| `--transfer_top_k` | `300` | eval×9,其它×1 | The top k probs to transfer during co |
| `--transfer_top_k_cap` | `16` | core解码×2,eval×1,测试×1 | 给（含 RL 选出的）transfer_top_k 设上限，压低拒绝载荷字 |
| `--use_stochastic_comm` | `'store_true'` | core解码×1,eval×8,测试×1 | Whether to use stochastic communicati |

### B 部署与硬件（6）

| 参数 | 默认 | 消费路径 | 说明 |
|---|---|---|---|
| `--draft_quantization` | `'auto'` | core解码×1 | Quantization mode for the draft model |
| `--graph_verify_sizes` | `''` | core解码×2 | 验证图的捕获档位（逗号分隔升序，如 '4,8,16,32'） |
| `--keep_target_on_single_gpu` | `'store_true'` | core解码×1,测试×1 | Keep the DSD target model on its sele |
| `--little_quantization` | `'auto'` | core解码×1 | Quantization mode for the little mode |
| `--model_dtype` | `'bf16'` | core解码×1 | Compute dtype for all models (and bnb |
| `--target_quantization` | `'auto'` | core解码×1,脚本×1 | Quantization mode for the target mode |

### C 方法、RL 挂载与消融（27）

| 参数 | 默认 | 消费路径 | 说明 |
|---|---|---|---|
| `--acc_head_path` | `None` | core解码×1 | The path of the accuracy head model |
| `--disable_eos_stop` | `'store_true'` | core解码×1,eval×1 | Disable EOS early stopping (restores |
| `--disable_rl_update` | `'store_true'` | core解码×1 | Whether to disable RL adapter update |
| `--draft_target_acc_head_path` | `None` | core解码×1 | The path of the draft-target accuracy |
| `--draft_target_threshold` | `0.8` | core解码×1 | The threshold for the draft-target mo |
| `--little_rl_best_path` | `None` | core解码×1 | The path of the best little RL adapte |
| `--little_rl_path` | `None` | core解码×1 | The path of the little RL adapter mod |
| `--main_rl_best_path` | `None` | core解码×1 | The path of the best main RL adapter |
| `--main_rl_path` | `None` | core解码×1 | The path of the main RL adapter model |
| `--rl_batch_size` | `None` | core解码×1,测试×1 | Override the mode-specific RL batch s |
| `--rl_checkpoint_root` | `'checkpoints/rl_agents'` | core解码×1,测试×1 | Root directory used to resolve pair-s |
| `--rl_epsilon_decay` | `None` | core解码×1,测试×1 | Override the mode-specific RL epsilon |
| `--rl_force_threshold` | `None` | core解码×1,训练RL×1 | Ablation: pin the ARP early-stop thre |
| `--rl_force_threshold_little` | `None` | core解码×1 | Ablation: pin the *little* (edge-end) |
| `--rl_force_topk` | `None` | core解码×1 | Ablation: pin the uplink top-k to the |
| `--rl_init_seed` | `None` | core解码×1,测试×1 | Seed used for deterministic RL networ |
| `--rl_init_strategy` | `'resume'` | core解码×1 | Initialize new RL agents or resume ex |
| `--rl_reward_scale` | `None` | core解码×1,测试×1 | Override the mode-specific RL reward |
| `--rl_team_reward` | `'store_true'` | core解码×1 | Give both adapters the same iteration |
| `--small_draft_acc_head_path` | `resolve_acc_head_path('llama-68m', 'tiny-llama-1.1b')` | core解码×1 | The path of the small draft accuracy |
| `--small_draft_threshold` | `0.8` | core解码×1 | The threshold for the small draft mod |
| `--top_k` | `0` | core解码×2,eval×1,测试×1,脚本×8 | top_k for ungreedy sampling strategy |
| `--top_p` | `0.95` | core解码×2,eval×1,测试×1,脚本×1 | top_p for ungreedy sampling strategy |
| `--uncertainty_threshold` | `0.8` | core解码×1 | The uncertainty threshold for uncerta |
| `--use_early_stopping` | `'store_true'` | core解码×1,eval×6 | Whether to use early stopping during |
| `--use_precise` | `'store_true'` | eval×8 | Use the physics level to simulate the |
| `--use_rl_adapter` | `'store_true'` | core解码×1,eval×1 | Whether to use RL adapter for dynamic |

### D 数据、评测与输出（10）

| 参数 | 默认 | 消费路径 | 说明 |
|---|---|---|---|
| `--data_path` | `'data/'` | eval×4 |  |
| `--dump_network_stats` | `'store_true'` | eval×1 | Whether to dump network statistics du |
| `--dump_outputs` | `None` | eval×1 | Write one JSON line per evaluated sam |
| `--judge_model` | `os.environ.get('JUDGE_MODEL', 'deepseek-v3.1')` | eval×1 | Judge model for MT-Bench |
| `--num_samples_per_task` | `1` | eval×7 | num_samples for a task (prompt) in hu |
| `--openai_api_base` | `os.environ.get('OPENAI_BASE_URL')` | eval×1 | OpenAI API Base for MT-Bench Judge |
| `--openai_api_key` | `os.environ.get('OPENAI_API_KEY')` | eval×1 | OpenAI API Key for MT-Bench Judge |
| `--run_full_dataset` | `'store_true'` | eval×1 | Evaluate the full dataset instead of |
| `--seed` | `1234` | core解码×1,eval×2,脚本×2 | set a random seed, which can makes th |
| `--sub_domain` | `'math_reasoning'` | eval×1 | sub domain in specbench |

### E curriculum（训练概念，评测侧仅 mixed 用）（3）

| 参数 | 默认 | 消费路径 | 说明 |
|---|---|---|---|
| `--curriculum_bw_end` | `'20,50'` | eval×1 | Curriculum bandwidth range (Mbps) at |
| `--curriculum_bw_start` | `'20,50'` | eval×1 | Curriculum bandwidth range (Mbps) at |
| `--curriculum_sampling` | `'uniform'` | eval×1 | How bandwidth is sampled within the c |

### F 其它（2）

| 参数 | 默认 | 消费路径 | 说明 |
|---|---|---|---|
| `--arp_stop_mode` | `'cumulative'` | core解码×1 | Acceptance-prediction early-stop rule |
| `--force_full_vocab_transfer` | `'store_true'` | core解码×1 | 强制传输完整词表分布（不做 top-k 稀疏化） |
