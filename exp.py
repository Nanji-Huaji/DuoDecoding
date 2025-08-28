import subprocess
import sys
import os
import json
import time
import threading
from queue import Queue, Empty
from typing import List, Dict, Any, Literal, TypedDict, Optional, Union


class ExperimentConfig(TypedDict, total=False):
    CUDA_VISIBLE_DEVICES: Optional[str]
    eval_mode: Literal["sd", "tridecoding", "tridecoding_with_bandwidth"]
    e: Literal["llama", "vicuna"]
    little_model: Optional[str]
    draft_model: str
    target_model: str
    max_tokens: int
    temperature: float
    gamma: Optional[List[int]]
    gamma1: Optional[List[int]]
    gamma2: Optional[List[int]]
    edge_cloud_bandwidth: Optional[Union[float, List[float]]]
    edge_end_bandwidth: Optional[Union[float, List[float]]]
    cloud_end_bandwidth: Optional[Union[float, List[float]]]
    exp_name: str


class EvalResults(TypedDict, total=False):
    best_gamma: int
    best_wall_time: float
    best_gamma1: int
    best_gamma2: int
    best_target_model_forward_times: int
    best_throughput: float
    best_edge_cloud_bandwidth: Optional[float]
    best_edge_end_bandwidth: Optional[float]
    best_cloud_end_bandwidth: Optional[float]


class RunExperiment:
    def __init__(self, experiments: List[ExperimentConfig], **kwargs: Dict[str, Any]):
        self.experiments = experiments
        self.eval_results = EvalResults(
            best_gamma=0,
            best_wall_time=float("inf"),
            best_gamma1=0,
            best_gamma2=0,
            best_target_model_forward_times=float("inf"),
            best_throughput=0.0,
            best_edge_cloud_bandwidth=None,
            best_edge_end_bandwidth=None,
            best_cloud_end_bandwidth=None,
        )
        self.kwargs = kwargs
        self.all_results = []

    @staticmethod
    def read_results(
        exp_name: str, eval_mode: Literal["sd", "tridecoding", "tridecoding_with_bandwidth"], **kwargs
    ) -> Dict[str, Any]:
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

    def _normalize_bandwidth_params(self, experiment: ExperimentConfig) -> Dict[str, List[Optional[float]]]:
        """标准化带宽参数为列表格式"""

        def to_list(param):
            if param is None:
                return [None]
            elif isinstance(param, list):
                return param
            else:
                return [param]

        return {
            "edge_cloud_bandwidths": to_list(experiment.get("edge_cloud_bandwidth")),
            "edge_end_bandwidths": to_list(experiment.get("edge_end_bandwidth")),
            "cloud_end_bandwidths": to_list(experiment.get("cloud_end_bandwidth")),
        }

    def _run_single_experiment(
        self,
        experiment: ExperimentConfig,
        gamma: Optional[int] = None,
        gamma1: Optional[int] = None,
        gamma2: Optional[int] = None,
        edge_cloud_bw: Optional[float] = None,
        edge_end_bw: Optional[float] = None,
        cloud_end_bw: Optional[float] = None,
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
        elif experiment["eval_mode"] == "tridecoding_with_bandwidth":
            bandwidth_info = f"ec{edge_cloud_bw}_ee{edge_end_bw}_ce{cloud_end_bw}".replace(".", "_").replace(
                "None", "auto"
            )
            unique_exp_name = f"{experiment['exp_name']}_{experiment['little_model']}_{experiment['draft_model']}_{experiment['target_model']}_gamma1_{gamma1}_gamma2_{gamma2}_{bandwidth_info}"
            param_info = f"gamma1={gamma1}, gamma2={gamma2}, edge_cloud_bw={edge_cloud_bw}, edge_end_bw={edge_end_bw}, cloud_end_bw={cloud_end_bw}"
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

        if experiment["eval_mode"] in ["tridecoding", "tridecoding_with_bandwidth"]:
            if experiment.get("little_model"):
                cmd.extend(["--little_model", experiment["little_model"]])
            if gamma1 is not None and gamma2 is not None:
                cmd.extend(["--gamma1", str(gamma1), "--gamma2", str(gamma2)])

            # 添加带宽参数
            if experiment["eval_mode"] == "tridecoding_with_bandwidth":
                if edge_cloud_bw is not None:
                    cmd.extend(["--edge_cloud_bandwidth", str(edge_cloud_bw)])
                if edge_end_bw is not None:
                    cmd.extend(["--edge_end_bandwidth", str(edge_end_bw)])
                if cloud_end_bw is not None:
                    cmd.extend(["--cloud_end_bandwidth", str(cloud_end_bw)])
        else:
            if gamma is not None:
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
            elif experiment["eval_mode"] == "tridecoding_with_bandwidth":
                result["parameters"].update(
                    {
                        "gamma1": gamma1,
                        "gamma2": gamma2,
                        "edge_cloud_bandwidth": edge_cloud_bw,
                        "edge_end_bandwidth": edge_end_bw,
                        "cloud_end_bandwidth": cloud_end_bw,
                    }
                )
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
            duration = end_time - start_time if "start_time" in locals() else 0
            print(f"💥 Exception occurred: {str(e)}")

            params = {}
            if experiment["eval_mode"] == "sd":
                params["gamma"] = gamma
            elif experiment["eval_mode"] == "tridecoding_with_bandwidth":
                params.update(
                    {
                        "gamma1": gamma1,
                        "gamma2": gamma2,
                        "edge_cloud_bandwidth": edge_cloud_bw,
                        "edge_end_bandwidth": edge_end_bw,
                        "cloud_end_bandwidth": cloud_end_bw,
                    }
                )
            else:
                params["gamma1"] = gamma1
                params["gamma2"] = gamma2

            return {
                "success": False,
                "return_code": -1,
                "duration": duration,
                "unique_exp_name": unique_exp_name if "unique_exp_name" in locals() else "unknown",
                "stdout": stdout_lines,
                "stderr": stderr_lines + [f"Exception: {str(e)}"],
                "all_output": all_output_lines + [f"Exception: {str(e)}"],
                "command": " ".join(cmd) if "cmd" in locals() else "unknown",
                "parameters": params,
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
            elif experiment["eval_mode"] in ["tridecoding", "tridecoding_with_bandwidth"]:
                self._run_tridecoding(experiment)

        self._print_final_results()
        self._save_results()

    def _run_speculative_decoding(self, experiment: ExperimentConfig) -> None:
        """运行投机解码实验"""
        if not experiment.get("gamma"):
            print("⚠️  Warning: gamma is required for speculative_decoding but not provided")
            return

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
        """运行三重解码实验（包括带宽版本）"""
        if not experiment.get("little_model"):
            print("⚠️  Warning: little_model is required for tridecoding but not provided")
            return

        if not experiment.get("gamma1") or not experiment.get("gamma2"):
            print("⚠️  Warning: gamma1 and gamma2 are required for tridecoding but not provided")
            return

        print(f"🔍 Testing gamma1 values: {experiment['gamma1']}")
        print(f"🔍 Testing gamma2 values: {experiment['gamma2']}")

        # 处理带宽参数
        bandwidth_params = self._normalize_bandwidth_params(experiment)
        edge_cloud_bandwidths = bandwidth_params["edge_cloud_bandwidths"]
        edge_end_bandwidths = bandwidth_params["edge_end_bandwidths"]
        cloud_end_bandwidths = bandwidth_params["cloud_end_bandwidths"]

        if experiment["eval_mode"] == "tridecoding_with_bandwidth":
            print(f"📡 Bandwidth combinations to test:")
            print(f"   Edge-Cloud: {edge_cloud_bandwidths} Mbps")
            print(f"   Edge-End: {edge_end_bandwidths} Mbps")
            print(f"   Cloud-End: {cloud_end_bandwidths} Mbps")

        valid_combinations = 0
        total_combinations = (
            len(experiment["gamma1"])
            * len(experiment["gamma2"])
            * len(edge_cloud_bandwidths)
            * len(edge_end_bandwidths)
            * len(cloud_end_bandwidths)
        )

        print(f"🔢 Total combinations to test: {total_combinations}")

        for gamma1 in experiment["gamma1"]:
            for gamma2 in experiment["gamma2"]:
                for edge_cloud_bw in edge_cloud_bandwidths:
                    for edge_end_bw in edge_end_bandwidths:
                        for cloud_end_bw in cloud_end_bandwidths:
                            valid_combinations += 1

                            if experiment["eval_mode"] == "tridecoding_with_bandwidth":
                                print(
                                    f"\n{'='*10} Testing gamma1={gamma1}, gamma2={gamma2}, EC_BW={edge_cloud_bw}, EE_BW={edge_end_bw}, CE_BW={cloud_end_bw} {'='*10}"
                                )
                            else:
                                print(f"\n{'='*15} Testing gamma1={gamma1}, gamma2={gamma2} {'='*15}")

                            # 运行实验并获取详细结果
                            run_result = self._run_single_experiment(
                                experiment,
                                gamma1=gamma1,
                                gamma2=gamma2,
                                edge_cloud_bw=edge_cloud_bw,
                                edge_end_bw=edge_end_bw,
                                cloud_end_bw=cloud_end_bw,
                            )

                            if not run_result["success"]:
                                print(f"⚠️  Experiment failed, skipping result processing")
                                self.all_results.append(
                                    {
                                        "exp_name": run_result["unique_exp_name"],
                                        "eval_mode": experiment["eval_mode"],
                                        "gamma1": gamma1,
                                        "gamma2": gamma2,
                                        "edge_cloud_bandwidth": edge_cloud_bw,
                                        "edge_end_bandwidth": edge_end_bw,
                                        "cloud_end_bandwidth": cloud_end_bw,
                                        "success": False,
                                        "run_details": run_result,
                                    }
                                )
                                continue

                            # 读取评估结果
                            eval_results = self.read_results(run_result["unique_exp_name"], experiment["eval_mode"])

                            if eval_results:
                                # 对于带宽版本，主要关注wall_time和throughput
                                if experiment["eval_mode"] == "tridecoding_with_bandwidth":
                                    wall_time = eval_results.get("wall_time", float("inf"))
                                    throughput = eval_results.get("throughput", 0.0)
                                    print(
                                        f"📊 Evaluation Results: wall_time = {wall_time:.4f}, throughput = {throughput:.4f}"
                                    )

                                    # 记录完整结果
                                    result_entry = {
                                        "exp_name": run_result["unique_exp_name"],
                                        "eval_mode": experiment["eval_mode"],
                                        "gamma1": gamma1,
                                        "gamma2": gamma2,
                                        "wall_time": wall_time,
                                        "throughput": throughput,
                                        "edge_cloud_bandwidth": edge_cloud_bw,
                                        "edge_end_bandwidth": edge_end_bw,
                                        "cloud_end_bandwidth": cloud_end_bw,
                                        "success": True,
                                        "eval_results": eval_results,
                                        "run_details": run_result,
                                    }

                                    # 更新最佳结果（基于wall_time和throughput）
                                    if wall_time < self.eval_results["best_wall_time"]:
                                        self.eval_results.update(
                                            {
                                                "best_gamma1": gamma1,
                                                "best_gamma2": gamma2,
                                                "best_wall_time": wall_time,
                                                "best_edge_cloud_bandwidth": edge_cloud_bw,
                                                "best_edge_end_bandwidth": edge_end_bw,
                                                "best_cloud_end_bandwidth": cloud_end_bw,
                                            }
                                        )
                                        print(
                                            f"🏆 NEW BEST WALL TIME! {wall_time:.4f} (gamma1={gamma1}, gamma2={gamma2}, bandwidths: {edge_cloud_bw}/{edge_end_bw}/{cloud_end_bw})"
                                        )

                                    if throughput > self.eval_results["best_throughput"]:
                                        self.eval_results["best_throughput"] = throughput
                                        print(
                                            f"🚀 NEW BEST THROUGHPUT! {throughput:.4f} (gamma1={gamma1}, gamma2={gamma2}, bandwidths: {edge_cloud_bw}/{edge_end_bw}/{cloud_end_bw})"
                                        )

                                else:
                                    # 传统tridecoding模式
                                    target_forward_times = eval_results.get("target_forward_times", float("inf"))
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

                                    # 更新最佳结果
                                    if target_forward_times < self.eval_results["best_target_model_forward_times"]:
                                        self.eval_results.update(
                                            {
                                                "best_gamma1": gamma1,
                                                "best_gamma2": gamma2,
                                                "best_target_model_forward_times": target_forward_times,
                                            }
                                        )
                                        print(
                                            f"🏆 NEW BEST RESULT! Target forward times: {target_forward_times} "
                                            f"(gamma1={gamma1}, gamma2={gamma2})"
                                        )

                                self.all_results.append(result_entry)
                            else:
                                self.all_results.append(
                                    {
                                        "exp_name": run_result["unique_exp_name"],
                                        "eval_mode": experiment["eval_mode"],
                                        "gamma1": gamma1,
                                        "gamma2": gamma2,
                                        "edge_cloud_bandwidth": edge_cloud_bw,
                                        "edge_end_bandwidth": edge_end_bw,
                                        "cloud_end_bandwidth": cloud_end_bw,
                                        "success": False,
                                        "run_details": run_result,
                                        "error": "Failed to read evaluation results",
                                    }
                                )

        print(f"✅ Tested {valid_combinations} valid combinations")

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
            print(f"🥇 Best Wall Time:")
            print(f"   Gamma1: {self.eval_results['best_gamma1']}")
            print(f"   Gamma2: {self.eval_results['best_gamma2']}")
            print(f"   Wall Time: {self.eval_results['best_wall_time']:.4f}")
            print()

        if self.eval_results["best_throughput"] > 0:
            print(f"🚀 Best Throughput:")
            print(f"   Throughput: {self.eval_results['best_throughput']:.4f}")
            print()

        if self.eval_results["best_target_model_forward_times"] < 10**18:
            print(f"🥇 Best Tridecoding (Target Forward Times):")
            print(f"   Gamma1: {self.eval_results['best_gamma1']}")
            print(f"   Gamma2: {self.eval_results['best_gamma2']}")
            print(f"   Target Forward Times: {self.eval_results['best_target_model_forward_times']}")
            print()

    def _save_results(self) -> None:
        """保存实验结果到文件"""
        try:
            # 确保结果目录存在
            os.makedirs("experiment_results", exist_ok=True)

            # 保存所有结果
            timestamp = time.strftime("%Y%m%d_%H%M%S")

            # 保存详细的所有结果
            all_results_file = f"experiment_results/all_results_{timestamp}.json"
            with open(all_results_file, "w", encoding="utf-8") as f:
                json.dump(self.all_results, f, indent=2, ensure_ascii=False, default=str)
            print(f"📁 All results saved to: {all_results_file}")

            # 保存最佳结果
            best_results_file = f"experiment_results/best_results_{timestamp}.json"
            with open(best_results_file, "w", encoding="utf-8") as f:
                json.dump(dict(self.eval_results), f, indent=2, ensure_ascii=False, default=str)
            print(f"🏆 Best results saved to: {best_results_file}")

            # 保存汇总结果
            summary = {
                "timestamp": timestamp,
                "total_experiments": len(self.all_results),
                "successful_experiments": len([r for r in self.all_results if r.get("success", False)]),
                "failed_experiments": len([r for r in self.all_results if not r.get("success", False)]),
                "best_results": dict(self.eval_results),
                "experiment_modes": list(set(r.get("eval_mode") for r in self.all_results)),
            }

            summary_file = f"experiment_results/summary_{timestamp}.json"
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            print(f"📊 Summary saved to: {summary_file}")

            # 生成详细报告
            self._generate_report()

        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")

    def _generate_report(self) -> None:
        """生成详细的实验报告"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report_file = f"experiment_results/report_{timestamp}.txt"

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
                    if "gamma" in result and result["gamma"] is not None:
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
            speculative_results = [r for r in successful_runs if r.get("eval_mode") == "sd"]
            tridecoding_results = [r for r in successful_runs if r.get("eval_mode") == "tridecoding"]
            bandwidth_results = [r for r in successful_runs if r.get("eval_mode") == "tridecoding_with_bandwidth"]

            if speculative_results:
                report_lines.append("SPECULATIVE DECODING RESULTS:")
                report_lines.append("-" * 30)
                speculative_results.sort(key=lambda x: x.get("wall_time", float("inf")))
                for result in speculative_results:
                    duration = result.get("run_details", {}).get("duration", 0)
                    wall_time = result.get("wall_time", "N/A")
                    wall_time_str = f"{wall_time:.4f}" if isinstance(wall_time, (int, float)) else str(wall_time)
                    report_lines.append(
                        f"✅ Gamma: {result.get('gamma')}, Wall Time: {wall_time_str}, " f"Duration: {duration:.2f}s"
                    )
                report_lines.append("")

            if tridecoding_results:
                report_lines.append("TRIDECODING RESULTS:")
                report_lines.append("-" * 30)
                tridecoding_results.sort(key=lambda x: x.get("target_forward_times", float("inf")))
                for result in tridecoding_results:
                    duration = result.get("run_details", {}).get("duration", 0)
                    target_times = result.get("target_forward_times", "N/A")
                    target_times_str = str(target_times) if target_times != "N/A" else "N/A"
                    report_lines.append(
                        f"✅ Gamma1: {result.get('gamma1')}, Gamma2: {result.get('gamma2')}, "
                        f"Target Forward Times: {target_times_str}, "
                        f"Duration: {duration:.2f}s"
                    )
                report_lines.append("")

            if bandwidth_results:
                report_lines.append("TRIDECODING WITH BANDWIDTH RESULTS:")
                report_lines.append("-" * 40)
                bandwidth_results.sort(key=lambda x: x.get("wall_time", float("inf")))
                for result in bandwidth_results:
                    duration = result.get("run_details", {}).get("duration", 0)
                    wall_time = result.get("wall_time", "N/A")
                    throughput = result.get("throughput", "N/A")
                    wall_time_str = f"{wall_time:.4f}" if isinstance(wall_time, (int, float)) else str(wall_time)
                    throughput_str = f"{throughput:.4f}" if isinstance(throughput, (int, float)) else str(throughput)

                    report_lines.append(
                        f"✅ Gamma1: {result.get('gamma1')}, Gamma2: {result.get('gamma2')}, "
                        f"Wall Time: {wall_time_str}, "
                        f"Throughput: {throughput_str}, "
                        f"EC_BW: {result.get('edge_cloud_bandwidth', 'N/A')}, "
                        f"EE_BW: {result.get('edge_end_bandwidth', 'N/A')}, "
                        f"CE_BW: {result.get('cloud_end_bandwidth', 'N/A')}, "
                        f"Duration: {duration:.2f}s"
                    )
                report_lines.append("")

            report_lines.append("BEST RESULTS:")
            report_lines.append("-" * 15)
            if self.eval_results["best_wall_time"] < float("inf"):
                if any(r.get("eval_mode") == "sd" for r in successful_runs):
                    report_lines.append(
                        f"🏆 Best Speculative Decoding - Gamma: {self.eval_results['best_gamma']}, "
                        f"Wall Time: {self.eval_results['best_wall_time']:.4f}"
                    )
                else:
                    report_lines.append(
                        f"🏆 Best Wall Time - Gamma1: {self.eval_results['best_gamma1']}, "
                        f"Gamma2: {self.eval_results['best_gamma2']}, "
                        f"Wall Time: {self.eval_results['best_wall_time']:.4f}"
                    )
            if self.eval_results["best_throughput"] > 0:
                report_lines.append(f"🚀 Best Throughput: {self.eval_results['best_throughput']:.4f}")
            if self.eval_results["best_target_model_forward_times"] < float("inf"):
                report_lines.append(
                    f"🏆 Best Tridecoding - Gamma1: {self.eval_results['best_gamma1']}, "
                    f"Gamma2: {self.eval_results['best_gamma2']}, "
                    f"Target Forward Times: {self.eval_results['best_target_model_forward_times']}"
                )

            with open(report_file, "w", encoding="utf-8") as f:
                f.write("\n".join(report_lines))
            print(f"📊 Detailed report saved to: {report_file}")

        except Exception as e:
            print(f"❌ Error generating report: {str(e)}")

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
    # 根据提供的带宽表格配置实验
    experiments = [
        # 配置1: 33.2 Mbps (end-edge) + 0.14 Mbps (end/edge-cloud)
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "eval_mode": "tridecoding_with_bandwidth",
            "e": "llama",
            "draft_model": "tiny-llama-1.1b",
            "target_model": "Llama-2-13b",
            "little_model": "llama-68m",
            "max_tokens": 128,
            "temperature": 0.0,
            "gamma1": [4],
            "gamma2": [24],
            "edge_end_bandwidth": 33.2,  # end - edge
            "edge_cloud_bandwidth": 0.14,  # edge - cloud
            "cloud_end_bandwidth": 0.14,  # cloud - end (假设对称)
            "exp_name": "bandwidth_test_33_2_0_14",
        },
        # 配置2: 100 Mbps (end-edge) + 5.0 Mbps (end/edge-cloud)
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "eval_mode": "tridecoding_with_bandwidth",
            "e": "llama",
            "draft_model": "tiny-llama-1.1b",
            "target_model": "Llama-2-13b",
            "little_model": "llama-68m",
            "max_tokens": 128,
            "temperature": 0.0,
            "gamma1": [4],
            "gamma2": [24],
            "edge_end_bandwidth": 100.0,  # end - edge
            "edge_cloud_bandwidth": 5.0,  # edge - cloud
            "cloud_end_bandwidth": 5.0,  # cloud - end (假设对称)
            "exp_name": "bandwidth_test_100_5_0",
        },
        # 配置3: 200 Mbps (end-edge) + 15.0 Mbps (end/edge-cloud)
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "eval_mode": "tridecoding_with_bandwidth",
            "e": "llama",
            "draft_model": "tiny-llama-1.1b",
            "target_model": "Llama-2-13b",
            "little_model": "llama-68m",
            "max_tokens": 128,
            "temperature": 0.0,
            "gamma1": [4],
            "gamma2": [24],
            "edge_end_bandwidth": 200.0,  # end - edge
            "edge_cloud_bandwidth": 15.0,  # edge - cloud
            "cloud_end_bandwidth": 15.0,  # cloud - end (假设对称)
            "exp_name": "bandwidth_test_200_15_0",
        },
        # 配置4: 350 Mbps (end-edge) + 25.0 Mbps (end/edge-cloud)
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "eval_mode": "tridecoding_with_bandwidth",
            "e": "llama",
            "draft_model": "tiny-llama-1.1b",
            "target_model": "Llama-2-13b",
            "little_model": "llama-68m",
            "max_tokens": 128,
            "temperature": 0.0,
            "gamma1": [4],
            "gamma2": [24],
            "edge_end_bandwidth": 350.0,  # end - edge
            "edge_cloud_bandwidth": 25.0,  # edge - cloud
            "cloud_end_bandwidth": 25.0,  # cloud - end (假设对称)
            "exp_name": "bandwidth_test_350_25_0",
        },
        # 配置5: 563 Mbps (end-edge) + 34.6 Mbps (end/edge-cloud)
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "eval_mode": "tridecoding_with_bandwidth",
            "e": "llama",
            "draft_model": "tiny-llama-1.1b",
            "target_model": "Llama-2-13b",
            "little_model": "llama-68m",
            "max_tokens": 128,
            "temperature": 0.0,
            "gamma1": [4],
            "gamma2": [24],
            "edge_end_bandwidth": 563.0,  # end - edge
            "edge_cloud_bandwidth": 34.6,  # edge - cloud
            "cloud_end_bandwidth": 34.6,  # cloud - end (假设对称)
            "exp_name": "bandwidth_test_563_34_6",
        },
    ]

    try:
        runner = RunExperiment(experiments)
        runner.run()

        # 实验完成后的总结
        print("\n" + "=" * 60)
        print("📊 BANDWIDTH EXPERIMENT SUMMARY")
        print("=" * 60)
        print("实验已完成，请查看以下文件获取详细结果：")
        print("- experiment_results/all_results_*.json: 包含所有实验的性能指标")
        print("- experiment_results/report_*.txt: 详细的实验报告")
        print("- experiment_results/best_results_*.json: 最佳性能结果")
        print("- experiment_results/summary_*.json: 实验汇总信息")
        print("\n您可以从这些文件中提取挂钟时间、吞吐量和带宽使用量数据来填充您的表格。")

    except KeyboardInterrupt:
        print("\n❌ 实验被用户中断")
    except Exception as e:
        print(f"\n💥 实验运行时发生错误: {str(e)}")
        import traceback

        traceback.print_exc()
