from exp import *


# Define Exp Args

# 10/27
# Target: Find out optimal threshold for adaptive tridecoding
# Method:
# 1. find the optimal threshold of end-edge level that makes the wall-time shortest
# 2. find the optimal threshold of edge-cloud level that makes the forward times smallest

end_edge_threshold = [i / 10 for i in range(1, 11)]  # 0.1 ~ 1.0
edge_cloud_threshold = [i / 10 for i in range(1, 11)]  # 0.1 ~ 1.0

def create_adaptive_decoding_experiment(base_cmd: str, end_edge_th: float) -> str:
        cmd = base_cmd
        cmd = add_args(cmd, "")

if __name__ == "__main__":
    
    config_to_run = []

    # 创建日志目录
    log_dir = "exp_logs"
    Path(log_dir).mkdir(exist_ok=True)

    # 并行运行实验
    all_results = run_experiments_parallel(
        config_to_run, max_workers=2, log_dir=log_dir
    )

    # 保存汇总结果
    summary_file = (
        f"experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # 打印汇总报告
    print("\n" + "=" * 80)
    print("实验汇总报告:")
    print("=" * 80)

    successful = sum(1 for r in all_results if r["status"] == "success")
    failed = sum(1 for r in all_results if r["status"] == "failed")
    no_result = sum(1 for r in all_results if r["status"] == "no_result")
    exception = sum(1 for r in all_results if r["status"] == "exception")

    print(f"总实验数: {len(all_results)}")
    print(f"成功: {successful}")
    print(f"失败: {failed}")
    print(f"无结果: {no_result}")
    print(f"异常: {exception}")
    print(f"\n汇总结果已保存到: {summary_file}")

    for result in all_results:
        print(f"\n实验: {result['exp_name']}")
        print(f"状态: {result['status']}")
        if result.get("log_file"):
            print(f"日志: {result['log_file']}")
