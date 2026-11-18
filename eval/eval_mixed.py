"""混合任务 RL rollout 入口（旧 CLI：``python eval/eval_mixed.py -e <exp> ...``）。

一个 decoder 上按样本切换任务与网络课程：每个 step 随机抽任务、抽样本，
把 ``decoder.task`` 设为该任务（RL one-hot 依赖它），再按 curriculum 采样
带宽/时延并调用 ``run.generate``。数据加载、metrics 累加、输出文件位置与
计时都在 ``eval.runner``，本模块只保留混合循环特有的采样、prompt 构造、
dump 输出和供外部脚本解析的日志行。
"""

import hashlib
import json
import os
import random
import sys
import time
import traceback

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from fastchat.model import get_conversation_template

from eval.data import load_task_data
from eval.few_shot_examples import get_few_shot_prompt
from eval.model_ids import determine_model_id_mixed
from eval.runner import Evaluation
from src.rl_adapter import KNOWN_TASKS
from src.utils import (
    parse_arguments,
    parse_range_spec,
    sample_curriculum_condition,
    seed_everything,
)


class MixedTask:
    """多任务混合训练循环；数据保持 ``{task: [sample, ...]}`` 的 dict。

    ``save_metrics=False``：本任务不写 metrics JSONL（旧脚本也只打印逐样本
    指标并落 ``--dump_outputs``），因此 runner 不会生成额外的 summary 文件。
    """

    name = "mixed"
    save_metrics = False

    # B34：preprocess_prompt 里会走 chat template（模板自带 BOS）的 model_id 集合。
    # 该集合必须与下面 preprocess_prompt 的模板分支保持一致。
    CHAT_TEMPLATE_MODEL_IDS = ("llama-3.1", "qwen")

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        # D1：判定逻辑单点化到 eval.model_ids（mixed 链规则原样保留）。
        self.model_id = determine_model_id_mixed(args.draft_model, args.target_model)

        # 边界必须覆盖 tokenizer 能产出的全部 id：Qwen 系的
        # tokenizer.vocab_size=151643 只数基础词表，不含 added specials
        # （如 <|endoftext|> id=151643 会合法出现在 3-shot 提示里），用它做
        # 安全校验会把整样本判成 Invalid token ids 而静默丢弃。eval 引擎
        # （src/engine.py）一直用 max(len, vocab_size)，此处对齐。
        self.vocab_size = max(len(tokenizer), tokenizer.vocab_size)
        self.pad_token_id = (
            tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        )
        print(f"[Token Safety] Using tokenizer vocab_size: {self.vocab_size}")
        print(
            f"[Token Safety] Final vocab_size: {self.vocab_size}, "
            f"pad_token_id: {self.pad_token_id}"
        )

    def load_data(self):
        """每个任务单独走共享 loader；任一任务失败都直接报错，不静默降级。"""
        print(">>> Loading datasets...")
        all_data = {}
        for task in KNOWN_TASKS:
            try:
                samples = load_task_data(task, self.args)
            except Exception as exc:
                raise RuntimeError(
                    f"[eval_mixed] failed to load task {task!r}: {exc}"
                ) from exc
            all_data[task] = samples
            print(f"Loaded {len(samples)} samples for {task}.")
        return all_data

    def clamp_token_ids(self, input_ids):
        """将超出 vocab_size 范围的 token id 钳位到 pad_token_id。"""
        max_id = input_ids.max().item()
        min_id = input_ids.min().item()

        if max_id >= self.vocab_size or min_id < 0:
            print(
                "⚠️  [Token Clamp Warning] Found out-of-range token IDs! "
                f"min={min_id}, max={max_id}, vocab_size={self.vocab_size}"
            )
            input_ids = torch.where(
                (input_ids >= 0) & (input_ids < self.vocab_size),
                input_ids,
                torch.full_like(input_ids, self.pad_token_id),
            )
            print(
                f"   Replaced with pad_token_id={self.pad_token_id}. "
                f"New range: min={input_ids.min().item()}, "
                f"max={input_ids.max().item()}"
            )

        return input_ids

    def _uses_chat_template(self) -> bool:
        return self.model_id in self.CHAT_TEMPLATE_MODEL_IDS

    def preprocess_prompt(self, task, item):
        """为不同任务构建合适的 prompt。"""
        few_shot_prompt = get_few_shot_prompt(task, self.args.num_shots)

        if task == "mt_bench":
            prompt_text = few_shot_prompt + item["turns"][0]
        elif task == "gsm8k":
            prompt_text = (
                few_shot_prompt
                + f"Question: {item['question']}\nSolve the following math problem "
                "step by step and end your answer with 'The answer is <number>'.\n"
                "Answer:"
            )
        elif task == "cnndm":
            prompt_text = (
                few_shot_prompt
                + f"Article: {item['article']}\nSummarize the following article "
                "in a few sentences:\nSummary:"
            )
        elif task == "xsum":
            prompt_text = (
                few_shot_prompt
                + f"Article: {item['document']}\nSummarize the following news "
                "article in one sentence:\nSummary:"
            )
        elif task == "humaneval":
            prompt_text = few_shot_prompt + item["prompt"]
        else:
            prompt_text = few_shot_prompt + str(item)

        # 应用模型模板
        if self.model_id in self.CHAT_TEMPLATE_MODEL_IDS:
            messages = [{"role": "user", "content": prompt_text}]
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        elif "vicuna" in self.model_id or "llama-2-chat" in self.model_id:
            # Chat 模型使用 conversation template
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        else:
            # Base 模型（如 llama-2）使用简单格式，避免对话标记导致格式冲突
            return prompt_text

    def _dump_outputs(
        self, run, *, step, task, mode, input_ids, output_ids, item, bw, ntt
    ):
        """落盘生成结果（无损性检验与离线质量评测依赖该 schema）。"""
        dump_path = getattr(run.args, "dump_outputs", None)
        if not dump_path or not run.is_main_process:
            return
        try:
            prompt_len = int(input_ids.shape[1])
            gen_ids = output_ids[:, prompt_len:].tolist()
            # prompt 指纹：恒等检验必须先排除"两次运行输入不同"这一解释
            prompt_sha = hashlib.sha1(
                input_ids[0].tolist().__str__().encode()
            ).hexdigest()[:12]
            record = {
                "step": step,
                "task": task,
                "prompt_sha": prompt_sha,
                "eval_mode": mode,
                "seed": getattr(run.args, "seed", None),
                "prompt_len": prompt_len,
                "gen_ids": gen_ids[0] if gen_ids else [],
                "text": self.tokenizer.decode(
                    output_ids[0, prompt_len:], skip_special_tokens=True
                ),
                "item": item,
                "bw_mbps": float(bw),
                "ntt_ms": float(ntt),
            }
            with open(dump_path, "a") as dump_fh:
                dump_fh.write(json.dumps(record, default=str) + "\n")
        except Exception as dump_err:  # 落盘失败不应中断评测
            print(f"dump_outputs failed: {dump_err}")

    @staticmethod
    def _print_metrics(metrics):
        """逐样本日志；字段与顺序被 scripts/ 下的日志解析器精确匹配。"""
        tps = metrics.get("throughput", 0)
        acc = metrics.get("draft_accepted_tokens", 0) / (
            metrics.get("draft_generated_tokens", 1) + 1e-6
        )
        # 与 auto_train_manager.py 的正则保持一致
        print(f"Average Generation Speed: {tps:.2f} tokens/s")
        print(
            f"   -> Result: Latency={metrics.get('wall_time', 0):.2f}s | "
            f"Speed={tps:.2f} tokens/s | Acc={acc:.1%}"
        )
        # Deterministic, machine-load-independent policy metrics: the simulated
        # communication time / transferred bytes / round-trips depend only on
        # the policy's actions and the configured link, not on how busy this
        # host is. `compute` (wall - comm) is printed precisely so that host
        # contention stays visible instead of silently polluting throughput.
        try:
            comm = float(metrics.get("communication_time", 0.0))
            wall = float(metrics.get("wall_time", 0.0))
            connects = metrics.get("connect_times", {})
            connects_str = (
                "/".join(f"{k}:{int(v)}" for k, v in connects.items())
                if isinstance(connects, dict)
                else str(connects)
            )
            print(
                f"   -> Metrics: comm={comm:.4f}s "
                f"bytes={int(metrics.get('edge_cloud_data_bytes', 0))} "
                f"connects={connects_str} "
                f"energy={float(metrics.get('comm_energy', 0.0)):.3e} "
                f"compute={max(wall - comm, 0.0):.4f}s "
                f"gen={int(metrics.get('generated_tokens', 0))} "
                f"draft_gen={int(metrics.get('draft_generated_tokens', 0))} "
                f"draft_acc={int(metrics.get('draft_accepted_tokens', 0))} "
                f"fwd={int(metrics.get('little_forward_times', 0))}/"
                f"{int(metrics.get('draft_forward_times', 0))}/"
                f"{int(metrics.get('target_forward_times', 0))} "
                f"comp_time={float(metrics.get('draft_computation_time', 0.0)):.3f}/"
                f"{float(metrics.get('target_computation_time', 0.0)):.3f}s"
            )
        except Exception as metric_err:  # never let logging break the run
            print(f"   -> [Metrics Error]: {metric_err}")

    def run(self, run, data):
        args = run.args
        # 训练轮数
        total_steps = args.eval_data_num if args.eval_data_num else 500
        # runner 已按 args.eval_mode 取到解码函数；未知模式在进入本流程前就会报错。
        mode = args.eval_mode

        # 获取所有有数据的任务列表
        available_tasks = [name for name in KNOWN_TASKS if data.get(name)]
        if not available_tasks:
            raise ValueError("[eval_mixed] no task has any loaded samples")

        print(
            f"\n>>> Starting Mixed Multi-Task Training Loop "
            f"({total_steps} total steps) <<<\n"
        )
        print("Sampling Strategy: [Random Task] -> [Random Sample in Task]")
        print(f"Available Tasks: {available_tasks}")

        # 解析课程式环境参数（带宽/延迟范围随训练进度从易到难插值）
        bw_start = parse_range_spec(args.curriculum_bw_start)
        bw_end = parse_range_spec(args.curriculum_bw_end)
        ntt_start = parse_range_spec(args.curriculum_ntt_start)
        ntt_end = parse_range_spec(args.curriculum_ntt_end)
        print(
            f"Curriculum: bw {bw_start} -> {bw_end} Mbps, "
            f"ntt {ntt_start} -> {ntt_end} ms, sampling={args.curriculum_sampling}"
        )

        # 失败可见性：单样本异常原先只打印 traceback 就被静默跳过，既不计数
        # 也不汇总，长跑时"到底少跑了多少样本"无从得知。
        step_errors = 0

        for step in range(total_steps):
            # 1. 先随机选择一个任务种类
            task = random.choice(available_tasks)
            # 2. 再从该任务的数据集中随机抽取一个样本
            item = random.choice(data[task])

            # 3. 设置 RL 当前任务（One-Hot 向量依赖 decoder.task）
            run.decoder.task = task

            # 4. 环境参数课程式采样：带宽/延迟范围随训练进度从易到难扩展
            bw_mbps, ntt_ms = sample_curriculum_condition(
                step,
                total_steps,
                bw_start,
                bw_end,
                ntt_start,
                ntt_end,
                sampling=args.curriculum_sampling,
            )
            args.edge_cloud_bandwidth = bw_mbps
            args.ntt_ms_edge_cloud = ntt_ms
            args.edge_end_bandwidth = random.uniform(300.0, 800.0)  # 300-800 Mbps

            # 5. 构建输入
            prompt = self.preprocess_prompt(task, item)
            # B34：chat 模板路径的 prompt 自带 BOS，必须关 add_special_tokens，
            # 否则 Llama-3/3.2（model_ids 归一到 llama-3.1）会双 BOS。非模板路径
            # （vicuna/llama-2-chat/base）保持原有行为。
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                add_special_tokens=not self._uses_chat_template(),
            ).input_ids.to(run.device)

            # 钳位 token IDs，防止越界
            input_ids = self.clamp_token_ids(input_ids)

            # 6. 执行解码（触发 RL 适配器的 select_config 和 update）
            print(
                f"[{step + 1}/{total_steps}] Task: {task:<10} | Mode: {mode:<20} "
                f"| bw={bw_mbps:.2f}Mbps ntt={ntt_ms:.1f}ms"
            )

            try:
                output_ids, metrics, _elapsed = run.generate(
                    input_ids,
                    # 项目自带 `_check_stopping_criteria`（EOS 检查），但它一直挂在
                    # `use_early_stopping` 开关后面而这个开关**从未被传入**，于是混合
                    # 评测永远跑满 max_tokens（gsm8k 的 "#### 答案" 不出现、humaneval
                    # 被截断，质量无法评测）。这里显式启用。
                    use_early_stopping=not getattr(args, "disable_eos_stop", False),
                )

                self._dump_outputs(
                    run,
                    step=step,
                    task=task,
                    mode=mode,
                    input_ids=input_ids,
                    output_ids=output_ids,
                    item=item,
                    bw=bw_mbps,
                    ntt=ntt_ms,
                )
                self._print_metrics(metrics)
            except Exception as e:
                step_errors += 1
                print(f"   -> [Step Error]: {e}")
                traceback.print_exc()

        if step_errors:
            print(
                f"[eval_mixed] {step_errors}/{total_steps} 个样本在解码或记账阶段"
                "抛错并被跳过（不计入任何指标）。若非预期，请检查上方 traceback。"
            )
        print("\n>>> Mixed Training Finished! <<<")


class EvalMixed(Evaluation):
    task_type = MixedTask


if __name__ == "__main__":
    args = parse_arguments()

    # 如果是 RL 训练模式，且没有手动设置特殊的种子，则使用基于时间的随机种子
    if args.use_rl_adapter and args.seed == 1234:
        new_seed = int(time.time() * 1000) % 10000
        print(f"Detected RL Training mode. Randomizing seed to: {new_seed}")
        seed_everything(new_seed)
        args.seed = new_seed

    EvalMixed(args).eval()
