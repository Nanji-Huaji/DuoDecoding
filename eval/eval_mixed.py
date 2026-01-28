import os
import sys
import torch
import json
import random
import time
import numpy as np
from typing import List, Tuple, Dict, Any
from datasets import load_dataset
from fastchat.model import get_conversation_template
# 添加父目录到路径，确保能导入 src 模块
sys.path.append(os.path.join(sys.path[0], "../"))
from src.utils import seed_everything, parse_arguments
from src.baselines import Baselines, get_empty_metrics
from functools import partial
from eval.few_shot_examples import get_few_shot_prompt
import inspect

# 同步 rl_adapter.py 中的定义
KNOWN_TASKS = ["mt_bench", "gsm8k", "cnndm", "xsum", "humaneval"]

class EvalMixed(Baselines):
    def __init__(self, args):
        super().__init__(args)
        self.device = self.accelerator.device # 显式定义 device 属性
        self.load_tokenizer()
        self.load_model()
        self.all_data = {}
        self.load_data() # 改为调用抽象方法
        self.model_id = self._determine_model_id()
        self.color_print(f"Using Model ID: {self.model_id}", 2)

    def _determine_model_id(self):
        target = str(self.args.target_model).lower()
        draft = str(self.args.draft_model).lower()
        
        if "llama-3" in target:
            return "llama-3.1"
        if "qwen" in target:
            return "qwen"
        if "llama-2" in target:
            return "llama-2-chat"
        return "vicuna"

    def load_data(self):
        """实现抽象方法：一次性加载所有任务的数据集"""
        # 1. MT-Bench
        try:
            mt_data = []
            mt_path = os.path.join(self.args.data_path, "mt_bench.jsonl")
            if os.path.exists(mt_path):
                with open(mt_path) as f:
                    for line in f: mt_data.append(json.loads(line))
            self.all_data["mt_bench"] = mt_data
            self.color_print(f"Loaded {len(mt_data)} MT-Bench samples.", 2)
        except Exception as e:
            self.color_print(f"Error loading MT-Bench: {e}", 1)
            self.all_data["mt_bench"] = []

        # 2. GSM8K
        try:
            ds = load_dataset("gsm8k", "main", split="test")
            self.all_data["gsm8k"] = [dict(item) for item in ds]
            self.color_print(f"Loaded {len(self.all_data['gsm8k'])} GSM8K samples.", 2)
        except Exception as e:
            self.color_print(f"Error loading GSM8K: {e}", 1)
            self.all_data["gsm8k"] = []

        # 3. CNN/DM
        try:
            ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
            self.all_data["cnndm"] = [dict(item) for item in ds]
            self.color_print(f"Loaded {len(self.all_data['cnndm'])} CNN/DM samples.", 2)
        except Exception as e:
            self.color_print(f"Error loading CNN/DM: {e}", 1)
            self.all_data["cnndm"] = []

        # 4. XSum
        try:
            ds = load_dataset("xsum", split="test")
            self.all_data["xsum"] = [dict(item) for item in ds]
            self.color_print(f"Loaded {len(self.all_data['xsum'])} XSum samples.", 2)
        except Exception as e:
            self.color_print(f"Error loading XSum: {e}", 1)
            self.all_data["xsum"] = []

        # 5. HumanEval
        try:
            ds = load_dataset("openai_humaneval", split="test")
            self.all_data["humaneval"] = [dict(item) for item in ds]
            self.color_print(f"Loaded {len(self.all_data['humaneval'])} HumanEval samples.", 2)
        except Exception as e:
            self.color_print(f"Error loading HumanEval: {e}", 1)
            self.all_data["humaneval"] = []

        # 构建扁平化的样本池，以便真正随机地抽取样本
        self.flattened_data = []
        for task_name, samples in self.all_data.items():
            for sample in samples:
                self.flattened_data.append((task_name, sample))
        
        self.color_print(f"Total global sample pool size: {len(self.flattened_data)}", 3)

    def preprocess(self, input_text):
        """实现抽象方法"""
        pass

    def postprocess(self, input_text, output_text):
        """实现抽象方法"""
        pass

    def preprocess_prompt(self, task, item):
        """为不同任务构建合适的 Prompt"""
        few_shot_prompt = get_few_shot_prompt(task, self.args.num_shots)
        
        if task == "mt_bench":
            prompt_text = few_shot_prompt + item["turns"][0]
        elif task == "gsm8k":
            prompt_text = few_shot_prompt + f"Question: {item['question']}\nSolve the following math problem step by step and end your answer with 'The answer is <number>'.\nAnswer:"
        elif task == "cnndm":
            prompt_text = few_shot_prompt + f"Article: {item['article']}\nSummarize the following article in a few sentences:\nSummary:"
        elif task == "xsum":
            prompt_text = few_shot_prompt + f"Article: {item['document']}\nSummarize the following news article in one sentence:\nSummary:"
        elif task == "humaneval":
            prompt_text = few_shot_prompt + item["prompt"]
        else:
            prompt_text = few_shot_prompt + str(item)

        # 应用模型模板
        if self.model_id in ["llama-3.1", "qwen"]:
             messages = [{"role": "user", "content": prompt_text}]
             return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            conv = get_conversation_template(self.model_id)
            conv.append_message(conv.roles[0], prompt_text)
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()

    @torch.no_grad()
    def eval(self):
        # 训练轮数
        total_steps = self.args.eval_data_num if self.args.eval_data_num else 500
        mode = "adaptive_tridecoding"
        
        # 获取所有有数据的任务列表
        available_tasks = [t for t in KNOWN_TASKS if t in self.all_data and self.all_data[t]]
        
        if not available_tasks:
            self.color_print("Critical: No data loaded for any task!", 1)
            return

        self.color_print(f"\n>>> Starting Mixed Multi-Task Training Loop ({total_steps} total steps) <<<\n", 3)
        self.color_print(f"Sampling Strategy: [Random Task] -> [Random Sample in Task]", 3)
        self.color_print(f"Available Tasks: {available_tasks}", 3)
        
        for step in range(total_steps):
            # 1. 先随机选择一个任务种类
            task = random.choice(available_tasks)
            # 2. 再从该任务的数据集中随机抽取一个样本
            item = random.choice(self.all_data[task])
            
            # 3. 正确设置 self.task 保证 RL Adapter 的 One-Hot 向量正确
            self.task = task 
            
            # 4. 模式固定为 tridecoding
            decoding_fn = getattr(self, mode)
            
            # 4. 环境参数随机模拟 (模仿 train_rl.sh)
            # 随机化网络带宽和延迟，模拟真实场景的多样性
            self.args.edge_cloud_bandwidth = random.uniform(20.0, 50.0) # 20-50 Mbps
            self.args.ntt_ms_edge_cloud = random.uniform(0.0, 5.0)      # 0-5 ms 延迟
            self.args.edge_end_bandwidth = random.uniform(300.0, 800.0) # 300-800 Mbps

            # 5. 构建输入
            prompt = self.preprocess_prompt(task, item)
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
            
            # 6. 执行解码 (此过程会触发 RL RLNetworkAdapter 的 select_config 和 update)
            print(f"[{step+1}/{total_steps}] Task: {task:<10} | Mode: {mode:<20}")
            
            fn = partial(
                decoding_fn,
                transfer_top_k=self.args.transfer_top_k,
                use_precise_comm_sim=self.args.use_precise,
                use_stochastic_comm=self.args.use_stochastic_comm,
                ntt_ms_edge_cloud=self.args.ntt_ms_edge_cloud,
                ntt_ms_edge_end=self.args.ntt_ms_edge_end,
            )
            
            try:
                # 运行解码
                output_ids, metrics = fn(input_ids)
                
                # 打印单步结果
                tps = metrics.get('throughput', 0)
                acc = metrics.get('draft_accepted_tokens', 0) / (metrics.get('draft_generated_tokens', 1) + 1e-6)
                # 修改这里的输出格式，以匹配 auto_train_manager.py 中的正则表达式
                print(f"Average Generation Speed: {tps:.2f} tokens/s")
                print(f"   -> Result: Latency={metrics.get('wall_time',0):.2f}s | Speed={tps:.2f} tokens/s | Acc={acc:.1%}")
            except Exception as e:
                print(f"   -> [Step Error]: {e}")
                import traceback
                traceback.print_exc()

        self.color_print("\n>>> Mixed Training Finished! <<<", 3)

if __name__ == "__main__":
    args = parse_arguments()
    
    # 如果是 RL 训练模式，且没有手动设置特殊的种子，则使用基于时间的随机种子
    if args.use_rl_adapter and args.seed == 1234:
        new_seed = int(time.time() * 1000) % 10000
        print(f"Detected RL Training mode. Randomizing seed to: {new_seed}")
        seed_everything(new_seed)
        args.seed = new_seed
        
    evaluator = EvalMixed(args)
    evaluator.eval()
