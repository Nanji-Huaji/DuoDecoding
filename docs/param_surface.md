# CLI 参数面快照（2026-10-21，110 个；默认 --help 可见 65 个）

> 生成方式：AST 解析 `src/utils.py::parse_arguments` 的 `add_argument`，
> 再按 `args.<dest>` / `getattr(args, "<dest>")` 扫描 `exp.py` + `src/` +
> `eval/` + `tools/`（排除 `SpecDec_pp/` 与 `test/`）。
> 冻结测试：`test/test_cli_surface.py`（111 个选项字符串，只允许有意识地增删）。
> 2026-10-19：RL（37）与 curriculum（5）归档到 `--help-advanced`；
> 弃用的 `--comm_accounting` 也从 `--help` 隐藏（仍解析）。默认可见 65 项。

## 0. 四个取值来源

同一件事有四个入口，这是"参数太多、对不上"的根因：

| 来源 | 位置 | 规则 |
|---|---|---|
| CLI 默认 | `src/utils.py` | 命令行没给就用它 |
| 命名协议 | `--protocol` → `src/protocols.PROTOCOLS` | 只填**命令行未显式给出**的项 |
| 口径总开关 | `--comm_accounting honest\|legacy` | 强制覆盖 3 个子开关；**与 `--protocol honest\|legacy` 重复，已标弃用、help 隐藏（仍解析）** |
| 扫描配置 | `exp.py` 的 `create_config` / `cmd_temp` | 布尔只有在为 True 时转发；部分字面量写死 |

## 1. 分组统计

| 组 | 参数数 | 性质 |
|---|---:|---|
| ① 日常出表（这些就够） | 16 | 每次出表要动 |
| ② 数据集与采样 | 7 | 换数据集时动 |
| ③ 通信数值与计费口径（常态交给 --protocol） | 18 | 定一次口径；常态交给 --protocol |
| ④ trace 回放 | 4 | 实测 trace 回放时用 |
| ⑤ 解码方法与准确率头 | 10 | 调方法/准确率头时动 |
| ⑥ RL adapter | 39 | RL adapter（仅 --use_rl_adapter 时有效） |
| ⑦ 部署、精度与显存 | 6 | 换机器时动 |
| ⑧ curriculum | 5 | 训练 curriculum |
| ⑨ 输出与裁判 | 4 | 输出与裁判 |
| ⑩ 帮助 | 1 | `--help-advanced`（查看归档的 ⑥/⑧） |

## 2. 完整台账

| 选项 | 默认 | 分组 | 读取点 | 主要读取文件 |
|---|---|---|---:|---|
| `--data_path` | `'data/'` | ② | 4 | eval/eval_mixed.py, eval/eval_mt_bench.py, eval/eval_mt_bench_noeval.py … |
| `--draft_model` | `None` | ① | 54 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--target_model` | `None` | ① | 57 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--exp_name` | `'test'` | ① | 17 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--eval_mode` | `'small'` | ① | 38 | eval/collect_confidence.py, eval/eval_cnndm.py, eval/eval_gsm8k.py … |
| `--num_samples_per_task` | `1` | ② | 7 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--seed` | `1234` | ② | 6 | eval/eval_gsm8k.py, eval/eval_mixed.py, src/engine.py |
| `--max_tokens` | `1024` | ① | 25 | eval/collect_confidence.py, eval/eval_cnndm.py, src/baselines.py … |
| `--temp` | `0.2` | ① | 27 | eval/collect_confidence.py, src/baselines.py, src/engine.py … |
| `--top_k` | `0` | ② | 18 | eval/collect_confidence.py, src/baselines.py, src/engine.py |
| `--top_p` | `0.95` | ② | 19 | eval/collect_confidence.py, src/baselines.py, src/engine.py … |
| `--gamma` | `4` | ① | 15 | eval/utils.py, src/baselines.py, src/engine.py |
| `--use_cuda_graph` | `False` | ① | 3 | src/baselines.py, src/graph_decode.py, src/protocols.py |
| `--graph_verify_sizes` | `''` | ⑦ | 2 | src/baselines.py, src/graph_decode.py |
| `--eval_data_num` | `80` | ① | 9 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--run_full_dataset` | `False` | ② | 2 | eval/utils.py, src/utils.py |
| `--random_sample` | `False` | ① | 2 | eval/utils.py, src/protocols.py |
| `--sample_seed` | `1234` | ① | 2 | eval/utils.py, src/protocols.py |
| `--num_shots` | `0` | ① | 12 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--sub_domain` | `'math_reasoning'` | ② | 8 | eval/eval_specbench.py |
| `--openai_api_key` | `os.environ.get('OPENAI_API_KEY')` | ⑨ | 2 | eval/eval_mt_bench.py |
| `--openai_api_base` | `os.environ.get('OPENAI_BASE_URL')` | ⑨ | 1 | eval/eval_mt_bench.py |
| `--judge_model` | `os.environ.get('JUDGE_MODEL', 'deepseek-v3.1')` | ⑨ | 1 | eval/eval_mt_bench.py |
| `--little_model` | `'vicuna-68m'` | ① | 19 | eval/utils.py, src/baselines.py, src/engine.py … |
| `--gamma1` | `4` | ① | 34 | eval/collect_confidence.py, eval/utils.py, src/baselines.py … |
| `--gamma2` | `4` | ① | 42 | eval/collect_confidence.py, eval/utils.py, src/baselines.py … |
| `--edge_cloud_bandwidth` | `20.0` | ③ | 17 | eval/collect_confidence.py, eval/eval_mixed.py, src/baselines.py … |
| `--edge_end_bandwidth` | `100.0` | ③ | 10 | eval/collect_confidence.py, eval/eval_mixed.py, src/baselines.py … |
| `--cloud_end_bandwidth` | `100.0` | ③ | 9 | eval/collect_confidence.py, src/baselines.py, src/protocols.py |
| `--uncertainty_threshold` | `0.8` | ⑤ | 6 | src/baselines.py |
| `--tk_slt_odld` | `False` | ⑤ | 1 | src/baselines.py |
| `--transfer_top_k` | `300` | ③ | 10 | eval/collect_confidence.py, eval/eval_cnndm.py, eval/eval_gsm8k.py … |
| `--use_precise` | `False` | ③ | 8 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--ntt_ms_edge_end` | `20.0` | ③ | 10 | eval/collect_confidence.py, eval/eval_cnndm.py, eval/eval_gsm8k.py … |
| `--ntt_ms_edge_cloud` | `200.0` | ③ | 11 | eval/collect_confidence.py, eval/eval_cnndm.py, eval/eval_gsm8k.py … |
| `--acc_head_path` | `None` | ⑤ | 2 | src/baselines.py, src/utils.py |
| `--small_draft_acc_head_path` | `resolve_acc_head_path('llama-68m', 'tiny-llama-1.1b')` | ⑤ | 2 | src/baselines.py, src/utils.py |
| `--draft_target_acc_head_path` | `None` | ⑤ | 3 | src/baselines.py, src/utils.py |
| `--small_draft_threshold` | `0.8` | ⑤ | 2 | src/baselines.py |
| `--draft_target_threshold` | `0.8` | ⑤ | 3 | src/baselines.py |
| `--comm_trace_mode` | `'static'` | ④ | 1 | src/baselines.py |
| `--use_stochastic_comm` | `False` | ③ | 9 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--ntt_trace_file` | `''` | ④ | 2 | eval/utils.py, src/baselines.py |
| `--ntt_trace_scale` | `1.0` | ④ | 2 | eval/utils.py, src/baselines.py |
| `--stochastic_ntt` | `False` | ④ | 13 | eval/utils.py, src/baselines.py |
| `--protocol` | `'none'` | ① | 2 | eval/utils.py, src/utils.py |
| `--comm_accounting` | `None` | ③ | 3 | src/utils.py（**help 已隐藏**，仅为兼容旧命令保留解析） |
| `--min_bandwidth_mbps` | `5.0` | ③ | 26 | eval/utils.py, src/baselines.py, src/engine.py |
| `--comm_bw_model` | `'instant'` | ③ | 25 | eval/utils.py, src/baselines.py, src/engine.py |
| `--curriculum_bw_start` | `'20,50'` | ⑧ | 1 | eval/eval_mixed.py |
| `--curriculum_bw_end` | `'20,50'` | ⑧ | 1 | eval/eval_mixed.py |
| `--curriculum_ntt_start` | `'0,5'` | ⑧ | 1 | eval/eval_mixed.py |
| `--curriculum_ntt_end` | `'0,5'` | ⑧ | 1 | eval/eval_mixed.py |
| `--curriculum_sampling` | `'uniform'` | ⑧ | 2 | eval/eval_mixed.py |
| `--state_bw_scaling` | `'linear'` | ⑥ | 1 | src/rl_adapter.py |
| `--state_latency_scaling` | `'linear'` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_reward_mode` | `'legacy'` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_charge_queue` | `False` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_comm_cost_mode` | `'nominal'` | ⑥ | 2 | src/baselines.py |
| `--rl_threshold_candidates` | `None` | ⑥ | 3 | src/baselines.py |
| `--rl_reward_lambda` | `0.0` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_reward_deadline_ms` | `0.0` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_reward_deadline_penalty` | `1.0` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_reward_energy_weight` | `0.0` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_force_threshold_little` | `None` | ⑥ | 2 | src/baselines.py |
| `--model_dtype` | `'bf16'` | ⑦ | 1 | src/engine.py |
| `--disable_eos_stop` | `False` | ⑤ | 2 | eval/eval_mixed.py, src/baselines.py |
| `--comm_round_trip_mode` | `'per_round'` | ③ | 9 | eval/utils.py, src/baselines.py, src/engine.py … |
| `--transfer_top_k_cap` | `0` | ③ | 7 | eval/utils.py, src/baselines.py, src/engine.py … |
| `--force_full_vocab_transfer` | `False` | ③ | 2 | src/baselines.py |
| `--prob_payload_bits` | `16` | ③ | 2 | src/baselines.py, src/utils.py |
| `--charge_residual_payload` | `True` | ③ | 7 | eval/utils.py, src/baselines.py, src/engine.py … |
| `--dump_outputs` | `None` | ⑨ | 1 | eval/eval_mixed.py |
| `--arp_stop_mode` | `'cumulative'` | ⑤ | 3 | src/baselines.py |
| `--rl_force_topk` | `None` | ⑥ | 1 | src/baselines.py |
| `--rl_force_threshold` | `None` | ⑥ | 2 | src/baselines.py, src/rl_adapter.py |
| `--rl_byte_price` | `0.0` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_compute_time_mode` | `'wall'` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_compute_cost_json` | `None` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_reward_no_alpha2` | `False` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_action_space` | `'topk_thr'` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_gamma_candidates` | `'2,4,8,16'` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_force_gamma` | `None` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_factored_q` | `False` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_include_gamma_in_state` | `False` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_buffer_size` | `5000` | ⑥ | 1 | src/rl_adapter.py |
| `--rl_team_reward` | `False` | ⑥ | 2 | src/baselines.py |
| `--rl_reward_log_window` | `200` | ⑥ | 1 | src/rl_adapter.py |
| `--use_rl_adapter` | `False` | ⑥ | 2 | eval/eval_mixed.py, src/baselines.py |
| `--main_rl_path` | `None` | ⑥ | 3 | src/baselines.py, src/utils.py |
| `--main_rl_best_path` | `None` | ⑥ | 3 | src/baselines.py, src/utils.py |
| `--little_rl_path` | `None` | ⑥ | 3 | src/baselines.py, src/utils.py |
| `--little_rl_best_path` | `None` | ⑥ | 3 | src/baselines.py, src/utils.py |
| `--rl_checkpoint_root` | `'checkpoints/rl_agents'` | ⑥ | 3 | src/baselines.py, src/utils.py |
| `--rl_init_seed` | `None` | ⑥ | 1 | src/baselines.py |
| `--rl_init_strategy` | `'resume'` | ⑥ | 1 | src/baselines.py |
| `--rl_epsilon_decay` | `None` | ⑥ | 1 | src/baselines.py |
| `--rl_reward_scale` | `None` | ⑥ | 1 | src/baselines.py |
| `--rl_batch_size` | `None` | ⑥ | 1 | src/baselines.py |
| `--disable_rl_update` | `False` | ⑥ | 12 | src/baselines.py |
| `--batch_delay` | `0.05` | ③ | 17 | src/baselines.py, src/engine.py |
| `--use_early_stopping` | `False` | ⑤ | 7 | eval/eval_cnndm.py, eval/eval_gsm8k.py, eval/eval_humaneval.py … |
| `--dump_network_stats` | `False` | ③ | 1 | eval/utils.py |
| `--draft_quantization` | `'auto'` | ⑦ | 6 | src/engine.py |
| `--target_quantization` | `'auto'` | ⑦ | 4 | src/engine.py |
| `--keep_target_on_single_gpu` | `False` | ⑦ | 1 | src/engine.py |
| `--little_quantization` | `'auto'` | ⑦ | 1 | src/engine.py |

## 3. 规范规则（本快照之后）

1. 新增/删除参数必须同时更新 `test/test_cli_surface.py` 与本文档；
2. 计费口径的**唯一入口**是 `--protocol`；子开关只用于消融/复现，不再加同类开关；
3. 训练/RL 参数默认不出现在"日常出表"分组；它们在 `--use_rl_adapter` 或
   `exp.py` 下才有意义；
4. 任何"默认值"的副本（`src/comm_accounting.DEFAULTS`、协议冻结值）都必须有
   机械核对测试，硬编码字符串不算。
