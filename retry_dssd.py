"""重试 R4 中失败的 3 个 dssd（dist_split_spec）实验。

R4 其余 12 个已成功（experiment_summary_20260927_073642.json）。
失败原因：dist_split_spec 漏定义 _tri_prompt_len（已修复）。
"""
from exp import build_matrix, filter_configs_by_mode, run_experiments_parallel

if __name__ == "__main__":
    retry = filter_configs_by_mode(build_matrix(), "dssd")
    print(f"待重试: {len(retry)} 个 dssd 实验")
    run_experiments_parallel(
        retry,
        max_workers=2,
        log_dir="exp_logs",
        summary_file="experiment_results/experiment_summary_dssd_retry.json",
    )
