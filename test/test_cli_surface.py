"""CLI 参数面冻结：只允许有意识地增删。

EXPECTED 由 src/utils.py 的 add_argument 字面量生成（2026-10-18，110 个
选项字符串 / 107 个参数）。任何新增或删除都会让 test_surface_is_frozen 红；
更新 EXPECTED 必须与参数改动放在同一个提交里，并在提交信息写明原因。

参数太多不是靠这个测试解决的；它只保证"以后不会再悄悄变多"。
"""

import ast
from pathlib import Path

EXPECTED = (
    '--acc_head_path',
    '--arp_stop_mode',
    '--batch_delay',
    '--charge_residual_payload',
    '--cloud_end_bandwidth',
    '--comm_accounting',
    '--comm_bw_model',
    '--comm_round_trip_mode',
    '--comm_trace_mode',
    '--curriculum_bw_end',
    '--curriculum_bw_start',
    '--curriculum_ntt_end',
    '--curriculum_ntt_start',
    '--curriculum_sampling',
    '--data_path',
    '--disable_eos_stop',
    '--disable_rl_update',
    '--draft_model',
    '--draft_quantization',
    '--draft_target_acc_head_path',
    '--draft_target_threshold',
    '--dump_network_stats',
    '--dump_outputs',
    '--edge_cloud_bandwidth',
    '--edge_end_bandwidth',
    '--eval_data_num',
    '--eval_mode',
    '--exp_name',
    '--force_full_vocab_transfer',
    '--gamma',
    '--gamma1',
    '--gamma2',
    '--graph_verify_sizes',
    '--judge_model',
    '--keep_target_on_single_gpu',
    '--little_model',
    '--little_quantization',
    '--little_rl_best_path',
    '--little_rl_path',
    '--main_rl_best_path',
    '--main_rl_path',
    '--max_tokens',
    '--min_bandwidth_mbps',
    '--model_dtype',
    '--ntt_ms_edge_cloud',
    '--ntt_ms_edge_end',
    '--ntt_trace_file',
    '--ntt_trace_scale',
    '--num_samples_per_task',
    '--num_shots',
    '--openai_api_base',
    '--openai_api_key',
    '--prob_payload_bits',
    '--protocol',
    '--random_sample',
    '--rl_action_space',
    '--rl_batch_size',
    '--rl_buffer_size',
    '--rl_byte_price',
    '--rl_charge_queue',
    '--rl_checkpoint_root',
    '--rl_comm_cost_mode',
    '--rl_compute_cost_json',
    '--rl_compute_time_mode',
    '--rl_epsilon_decay',
    '--rl_factored_q',
    '--rl_force_gamma',
    '--rl_force_threshold',
    '--rl_force_threshold_little',
    '--rl_force_topk',
    '--rl_gamma_candidates',
    '--rl_include_gamma_in_state',
    '--rl_init_seed',
    '--rl_init_strategy',
    '--rl_reward_deadline_ms',
    '--rl_reward_deadline_penalty',
    '--rl_reward_energy_weight',
    '--rl_reward_lambda',
    '--rl_reward_log_window',
    '--rl_reward_mode',
    '--rl_reward_no_alpha2',
    '--rl_reward_scale',
    '--rl_team_reward',
    '--rl_threshold_candidates',
    '--run_full_dataset',
    '--sample_seed',
    '--seed',
    '--small_draft_acc_head_path',
    '--small_draft_threshold',
    '--state_bw_scaling',
    '--state_latency_scaling',
    '--stochastic_ntt',
    '--sub_domain',
    '--target_model',
    '--target_quantization',
    '--temp',
    '--tk_slt_odld',
    '--top_k',
    '--top_p',
    '--transfer_top_k',
    '--transfer_top_k_cap',
    '--uncertainty_threshold',
    '--use_cuda_graph',
    '--use_early_stopping',
    '--use_precise',
    '--use_rl_adapter',
    '--use_stochastic_comm',
    '-e',
    '-n',
    '-s',
)


def _options():
    src = Path(__file__).resolve().parents[1] / "src" / "utils.py"
    tree = ast.parse(src.read_text(encoding="utf-8"))
    return sorted(
        arg.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument"
        for arg in node.args
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
    )


def test_surface_is_frozen():
    assert _options() == list(EXPECTED)
