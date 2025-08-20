import subprocess
import sys
import os
import json
import time
import threading
from queue import Queue, Empty
from typing import List, Dict, Any, Literal, TypedDict, Optional


class ExperimentConfig(TypedDict):
    CUDA_VISIBLE_DEVICES: Optional[str]
    eval_mode: Literal["sd", "tridecoding"]
    e: Literal["llama", "vicuna"]
    little_model: Optional[str]
    draft_model: str
    target_model: str
    max_tokens: int
    temperature: float
    gamma: Optional[List[int]]
    gamma1: Optional[List[int]]
    gamma2: Optional[List[int]]
    exp_name: str


class EvalResults(TypedDict):
    best_gamma: int
    best_wall_time: float
    best_gamma1: int
    best_gamma2: int
    best_target_model_forward_times: int


class RunExperiment:
    def __init__(self, experiments: List[ExperimentConfig], **kwargs: Dict[str, Any]):
        self.experiments = experiments
        self.eval_results = EvalResults(
            best_gamma=0,
            best_wall_time=10**18,
            best_gamma1=0,
            best_gamma2=0,
            best_target_model_forward_times=10**18,
        )
        self.kwargs = kwargs
        self.all_results = []  # 存储所有实验结果

    @staticmethod
    def read_results(exp_name: str, eval_mode: Literal["sd", "tridecoding"], **kwargs) -> Dict[str, Any]:
        result_file = os.path.join("exp", exp_name, f"{eval_mode}_mt_bench_metrics.json")
        try:
            with open(result_file, "r") as f:
                results = json.load(f)
            print(f"✓ Successfully read results from {result_file}")
            return results
        except FileNotFoundError:
            print(f"✗ Results file not found: {result_file}")
            return {}
        except json.JSONDecodeError:
            print(f"✗ Invalid JSON in results file: {result_file}")
            return {}

    def _read_stream(self, stream, queue, stream_name):
        """在单独线程中读取流输出"""
        try:
            for line in iter(stream.readline, ""):
                if line:
                    queue.put((stream_name, line.rstrip()))
            stream.close()
        except Exception as e:
            queue.put((stream_name, f"Error reading {stream_name}: {str(e)}"))

    def _run_single_experiment(
        self, experiment: ExperimentConfig, gamma: int = None, gamma1: int = None, gamma2: int = None
    ) -> Dict[str, Any]:
        """运行单个实验配置并返回详细结果"""
        env = os.environ.copy()
        if experiment.get("CUDA_VISIBLE_DEVICES"):
            env["CUDA_VISIBLE_DEVICES"] = experiment["CUDA_VISIBLE_DEVICES"]

        # 为每个参数组合创建唯一的实验名称
        if experiment["eval_mode"] == "sd":
            unique_exp_name = (
                f"{experiment['exp_name']}_{experiment['draft_model']}_{experiment['target_model']}_gamma_{gamma}"
            )
            param_info = f"gamma={gamma}"
        else:
            unique_exp_name = f"{experiment['exp_name']}_{experiment['little_model']}_{experiment['draft_model']}_{experiment['target_model']}_gamma1_{gamma1}_gamma2_{gamma2}"
            param_info = f"gamma1={gamma1}, gamma2={gamma2}"

        cmd = [
            "accelerate",
            "launch",
            "--num_processes",
            "1",
            "--main_process_port",
            "29051",
            "eval/eval_mt_bench.py",
            "--eval_mode",
            experiment["eval_mode"],
            "-n",
            "1",
            "-e",
            experiment["e"],
            "--use-gpt_fast_model",
            "false",
            "--draft_model",
            experiment["draft_model"],
            "--target_model",
            experiment["target_model"],
            "--max_tokens",
            str(experiment["max_tokens"]),
            "--temp",
            str(experiment["temperature"]),
            "--exp_name",
            unique_exp_name,
        ]

        if experiment["eval_mode"] == "tridecoding":
            cmd.extend(["--little_model", experiment["little_model"]])
            cmd.extend(["--gamma1", str(gamma1), "--gamma2", str(gamma2)])
        else:
            cmd.extend(["--gamma", str(gamma)])

        print(f"\n🚀 Running experiment with {param_info}")
        print(f"📝 Command: {' '.join(cmd)}")
        print("-" * 80)

        # 捕获输出的变量
        stdout_lines = []
        stderr_lines = []
        all_output_lines = []

        try:
            start_time = time.time()

            # 使用 Popen 来实时捕获输出
            process = subprocess.Popen(
                cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, bufsize=1
            )

            # 创建队列来接收输出
            output_queue = Queue()

            # 启动线程来读取 stdout 和 stderr
            stdout_thread = threading.Thread(target=self._read_stream, args=(process.stdout, output_queue, "stdout"))
            stderr_thread = threading.Thread(target=self._read_stream, args=(process.stderr, output_queue, "stderr"))

            stdout_thread.daemon = True
            stderr_thread.daemon = True
            stdout_thread.start()
            stderr_thread.start()

            # 实时处理输出
            while process.poll() is None or not output_queue.empty():
                try:
                    stream_name, line = output_queue.get(timeout=0.1)

                    # 添加时间戳
                    timestamp = time.strftime("%H:%M:%S")
                    formatted_line = f"[{timestamp}] {line}"
                    all_output_lines.append(formatted_line)

                    if stream_name == "stdout":
                        stdout_lines.append(line)
                        print(f"📤 {line}")
                    elif stream_name == "stderr":
                        stderr_lines.append(line)
                        print(f"🚨 {line}")

                except Empty:
                    continue
                except Exception as e:
                    print(f"⚠️  Error processing output: {e}")

            # 等待进程完成
            return_code = process.wait()
            end_time = time.time()
            duration = end_time - start_time

            # 等待线程结束
            stdout_thread.join(timeout=1)
            stderr_thread.join(timeout=1)

            print("-" * 80)

            # 准备返回结果
            result = {
                "success": return_code == 0,
                "return_code": return_code,
                "duration": duration,
                "unique_exp_name": unique_exp_name,
                "stdout": stdout_lines,
                "stderr": stderr_lines,
                "all_output": all_output_lines,
                "command": " ".join(cmd),
                "parameters": {},
            }

            if experiment["eval_mode"] == "sd":
                result["parameters"]["gamma"] = gamma
            else:
                result["parameters"]["gamma1"] = gamma1
                result["parameters"]["gamma2"] = gamma2

            if return_code == 0:
                print(f"✅ Experiment completed successfully in {duration:.2f} seconds")
            else:
                print(f"❌ Experiment failed with return code {return_code} after {duration:.2f} seconds")

            return result

        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            print(f"💥 Exception occurred: {str(e)}")

            return {
                "success": False,
                "return_code": -1,
                "duration": duration,
                "unique_exp_name": unique_exp_name,
                "stdout": stdout_lines,
                "stderr": stderr_lines + [f"Exception: {str(e)}"],
                "all_output": all_output_lines + [f"Exception: {str(e)}"],
                "command": " ".join(cmd),
                "parameters": (
                    {"gamma": gamma} if experiment["eval_mode"] == "sd" else {"gamma1": gamma1, "gamma2": gamma2}
                ),
                "exception": str(e),
            }

    def run(self) -> None:
        print("=" * 80)
        print("🎯 STARTING EXPERIMENT RUNNER")
        print("=" * 80)

        for i, experiment in enumerate(self.experiments, 1):
            print(f"\n📊 Running Experiment {i}/{len(self.experiments)}: {experiment['exp_name']}")
            print("-" * 60)

            if experiment["eval_mode"] == "sd":
                self._run_speculative_decoding(experiment)
            elif experiment["eval_mode"] == "tridecoding":
                self._run_tridecoding(experiment)

        self._print_final_results()
        self._save_results()

    def _run_speculative_decoding(self, experiment: ExperimentConfig) -> None:
        """运行投机解码实验"""
        assert experiment["gamma"] is not None, "gamma is required for speculative_decoding"

        print(f"🔍 Testing gamma values: {experiment['gamma']}")

        for gamma in experiment["gamma"]:
            print(f"\n{'='*20} Testing gamma = {gamma} {'='*20}")

            # 运行实验并获取详细结果
            run_result = self._run_single_experiment(experiment, gamma=gamma)

            if not run_result["success"]:
                print(f"⚠️  Experiment failed, skipping result processing")
                self.all_results.append(
                    {
                        "exp_name": run_result["unique_exp_name"],
                        "eval_mode": experiment["eval_mode"],
                        "gamma": gamma,
                        "success": False,
                        "run_details": run_result,
                    }
                )
                continue

            # 读取评估结果
            eval_results = self.read_results(run_result["unique_exp_name"], experiment["eval_mode"])

            if eval_results:
                wall_time = eval_results.get("wall_time", float("inf"))
                print(f"📊 Evaluation Results: wall_time = {wall_time:.4f}")

                # 记录完整结果
                result_entry = {
                    "exp_name": run_result["unique_exp_name"],
                    "eval_mode": experiment["eval_mode"],
                    "gamma": gamma,
                    "wall_time": wall_time,
                    "success": True,
                    "eval_results": eval_results,
                    "run_details": run_result,
                }
                self.all_results.append(result_entry)

                # 更新最佳结果
                if wall_time < self.eval_results["best_wall_time"]:
                    self.eval_results["best_gamma"] = gamma
                    self.eval_results["best_wall_time"] = wall_time
                    print(f"🏆 NEW BEST RESULT! Wall time: {wall_time:.4f} (gamma = {gamma})")
            else:
                self.all_results.append(
                    {
                        "exp_name": run_result["unique_exp_name"],
                        "eval_mode": experiment["eval_mode"],
                        "gamma": gamma,
                        "success": False,
                        "run_details": run_result,
                        "error": "Failed to read evaluation results",
                    }
                )

    def _run_tridecoding(self, experiment: ExperimentConfig) -> None:
        """运行三重解码实验"""
        assert experiment["little_model"] is not None, "little_model is required for tridecoding"
        assert (
            experiment["gamma1"] is not None and experiment["gamma2"] is not None
        ), "gamma1 and gamma2 are required for tridecoding"

        print(f"🔍 Testing gamma1 values: {experiment['gamma1']}")
        print(f"🔍 Testing gamma2 values: {experiment['gamma2']}")

        valid_combinations = 0
        for gamma1 in experiment["gamma1"]:
            for gamma2 in experiment["gamma2"]:
                # if gamma1 < gamma2:
                #     print(f"⏭️  Skipping gamma1={gamma1}, gamma2={gamma2} (gamma1 < gamma2)")
                #     continue

                valid_combinations += 1
                print(f"\n{'='*15} Testing gamma1={gamma1}, gamma2={gamma2} {'='*15}")

                # 运行实验并获取详细结果
                run_result = self._run_single_experiment(experiment, gamma1=gamma1, gamma2=gamma2)

                if not run_result["success"]:
                    print(f"⚠️  Experiment failed, skipping result processing")
                    self.all_results.append(
                        {
                            "exp_name": run_result["unique_exp_name"],
                            "eval_mode": experiment["eval_mode"],
                            "gamma1": gamma1,
                            "gamma2": gamma2,
                            "success": False,
                            "run_details": run_result,
                        }
                    )
                    continue

                # 读取评估结果
                eval_results = self.read_results(run_result["unique_exp_name"], experiment["eval_mode"])

                if eval_results:
                    target_forward_times = eval_results.get("target_forward_times", 10**18)
                    print(f"📊 Evaluation Results: target_forward_times = {target_forward_times}")

                    # 记录完整结果
                    result_entry = {
                        "exp_name": run_result["unique_exp_name"],
                        "eval_mode": experiment["eval_mode"],
                        "gamma1": gamma1,
                        "gamma2": gamma2,
                        "target_forward_times": target_forward_times,
                        "success": True,
                        "eval_results": eval_results,
                        "run_details": run_result,
                    }
                    self.all_results.append(result_entry)

                    # 更新最佳结果
                    if target_forward_times < self.eval_results["best_target_model_forward_times"]:
                        self.eval_results["best_gamma1"] = gamma1
                        self.eval_results["best_gamma2"] = gamma2
                        self.eval_results["best_target_model_forward_times"] = target_forward_times
                        print(
                            f"🏆 NEW BEST RESULT! Target forward times: {target_forward_times} "
                            f"(gamma1={gamma1}, gamma2={gamma2})"
                        )
                else:
                    self.all_results.append(
                        {
                            "exp_name": run_result["unique_exp_name"],
                            "eval_mode": experiment["eval_mode"],
                            "gamma1": gamma1,
                            "gamma2": gamma2,
                            "success": False,
                            "run_details": run_result,
                            "error": "Failed to read evaluation results",
                        }
                    )

        print(f"✅ Tested {valid_combinations} valid gamma combinations")

    def _print_final_results(self) -> None:
        """打印最终结果"""
        print("\n" + "=" * 80)
        print("🏆 FINAL RESULTS SUMMARY")
        print("=" * 80)

        successful_runs = len([r for r in self.all_results if r.get("success", False)])
        failed_runs = len(self.all_results) - successful_runs

        print(f"📈 Total experiments: {len(self.all_results)}")
        print(f"✅ Successful runs: {successful_runs}")
        print(f"❌ Failed runs: {failed_runs}")
        print()

        if self.eval_results["best_wall_time"] < float("inf"):
            print(f"🥇 Best Speculative Decoding:")
            print(f"   Gamma: {self.eval_results['best_gamma']}")
            print(f"   Wall Time: {self.eval_results['best_wall_time']:.4f}")
            print()

        if self.eval_results["best_target_model_forward_times"] < 10**18:
            print(f"🥇 Best Tridecoding:")
            print(f"   Gamma1: {self.eval_results['best_gamma1']}")
            print(f"   Gamma2: {self.eval_results['best_gamma2']}")
            print(f"   Target Forward Times: {self.eval_results['best_target_model_forward_times']}")
            print()

    def _save_results(self) -> None:
        """保存结果到文件"""
        # 保存最佳结果
        with open("best_results.json", "w") as f:
            json.dump(self.eval_results, f, indent=4)
        print(f"💾 Best results saved to: best_results.json")

        # 保存所有结果（包括子进程输出）
        with open("all_results_detailed.json", "w") as f:
            json.dump(self.all_results, f, indent=4)
        print(f"💾 Detailed results with subprocess output saved to: all_results_detailed.json")

        # 保存简化版本（不包含详细输出）
        simplified_results = []
        for result in self.all_results:
            simplified = {k: v for k, v in result.items() if k != "run_details"}
            if "eval_results" in result:
                simplified["eval_results"] = result["eval_results"]
            simplified_results.append(simplified)

        with open("all_results_summary.json", "w") as f:
            json.dump(simplified_results, f, indent=4)
        print(f"💾 Summary results saved to: all_results_summary.json")

        # 生成详细报告
        self._generate_report()

    def _generate_report(self) -> None:
        """生成详细的实验报告"""
        report_lines = []
        report_lines.append("DETAILED EXPERIMENT REPORT")
        report_lines.append("=" * 50)
        report_lines.append(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Total experiments: {len(self.all_results)}")

        successful_runs = [r for r in self.all_results if r.get("success", False)]
        failed_runs = [r for r in self.all_results if not r.get("success", False)]

        report_lines.append(f"Successful runs: {len(successful_runs)}")
        report_lines.append(f"Failed runs: {len(failed_runs)}")
        report_lines.append("")

        # 失败的实验详情
        if failed_runs:
            report_lines.append("FAILED EXPERIMENTS:")
            report_lines.append("-" * 30)
            for result in failed_runs:
                if "gamma" in result:
                    report_lines.append(f"❌ {result['exp_name']} (gamma={result['gamma']})")
                else:
                    report_lines.append(
                        f"❌ {result['exp_name']} (gamma1={result.get('gamma1')}, gamma2={result.get('gamma2')})"
                    )

                if "run_details" in result and "stderr" in result["run_details"]:
                    stderr_preview = result["run_details"]["stderr"][-3:] if result["run_details"]["stderr"] else []
                    for line in stderr_preview:
                        report_lines.append(f"   Error: {line}")
                report_lines.append("")

        # 按实验模式分组成功的实验
        speculative_results = [r for r in successful_runs if r["eval_mode"] == "sd"]
        tridecoding_results = [r for r in successful_runs if r["eval_mode"] == "tridecoding"]

        if speculative_results:
            report_lines.append("SPECULATIVE DECODING RESULTS:")
            report_lines.append("-" * 30)
            speculative_results.sort(key=lambda x: x.get("wall_time", float("inf")))
            for result in speculative_results:
                duration = result.get("run_details", {}).get("duration", 0)
                report_lines.append(
                    f"✅ Gamma: {result['gamma']}, Wall Time: {result.get('wall_time', 'N/A'):.4f}, "
                    f"Duration: {duration:.2f}s"
                )
            report_lines.append("")

        if tridecoding_results:
            report_lines.append("TRIDECODING RESULTS:")
            report_lines.append("-" * 30)
            tridecoding_results.sort(key=lambda x: x.get("target_forward_times", 10**18))
            for result in tridecoding_results:
                duration = result.get("run_details", {}).get("duration", 0)
                report_lines.append(
                    f"✅ Gamma1: {result['gamma1']}, Gamma2: {result['gamma2']}, "
                    f"Target Forward Times: {result.get('target_forward_times', 'N/A')}, "
                    f"Duration: {duration:.2f}s"
                )
            report_lines.append("")

        report_lines.append("BEST RESULTS:")
        report_lines.append("-" * 15)
        if self.eval_results["best_wall_time"] < float("inf"):
            report_lines.append(
                f"🏆 Best Speculative Decoding - Gamma: {self.eval_results['best_gamma']}, "
                f"Wall Time: {self.eval_results['best_wall_time']:.4f}"
            )
        if self.eval_results["best_target_model_forward_times"] < 10**18:
            report_lines.append(
                f"🏆 Best Tridecoding - Gamma1: {self.eval_results['best_gamma1']}, "
                f"Gamma2: {self.eval_results['best_gamma2']}, "
                f"Target Forward Times: {self.eval_results['best_target_model_forward_times']}"
            )

        with open("experiment_report.txt", "w") as f:
            f.write("\n".join(report_lines))
        print(f"📊 Detailed report saved to: experiment_report.txt")

    def get_experiment_output(self, exp_name: str) -> Dict[str, Any]:
        """获取特定实验的输出"""
        for result in self.all_results:
            if result["exp_name"] == exp_name:
                return result.get("run_details", {})
        return {}

    def print_experiment_output(self, exp_name: str) -> None:
        """打印特定实验的输出"""
        run_details = self.get_experiment_output(exp_name)
        if not run_details:
            print(f"No output found for experiment: {exp_name}")
            return

        print(f"\n{'='*60}")
        print(f"OUTPUT FOR EXPERIMENT: {exp_name}")
        print(f"{'='*60}")
        print(f"Command: {run_details.get('command', 'N/A')}")
        print(f"Duration: {run_details.get('duration', 0):.2f} seconds")
        print(f"Return Code: {run_details.get('return_code', 'N/A')}")
        print(f"Success: {run_details.get('success', False)}")
        print()

        if run_details.get("all_output"):
            print("COMPLETE OUTPUT:")
            print("-" * 40)
            for line in run_details["all_output"]:
                print(line)
        else:
            print("No output captured.")


# 使用示例
if __name__ == "__main__":
    # 示例配置
    experiments = [
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "eval_mode": "tridecoding",
            "e": "llama",
            "draft_model": "tiny-llama-1.1b",
            "target_model": "Llama-2-13b",
            "little_model": "llama-68m",
            "max_tokens": 128,
            "temperature": 0.0,
            "gamma": None,
            "gamma1": [4],
            "gamma2": [24],
            "exp_name": "tridecoding_test",
        }
    ]

    runner = RunExperiment(experiments)
    runner.run()

    # 可以查看特定实验的输出
    # runner.print_experiment_output("spec_decode_test_gamma_2")
