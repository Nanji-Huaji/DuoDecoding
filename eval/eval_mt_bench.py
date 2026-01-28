import os
import sys

sys.path.append(os.path.join(sys.path[0], "../"))
import torch
import json
import tqdm
import time
import random
import shortuuid
from src.utils import seed_everything, parse_arguments
from src.engine import Decoding
from fastchat.model import get_conversation_template
from typing import List, Tuple


from src.baselines import get_empty_metrics, DecodingMetrics


from src.baselines import Baselines

from few_shot_examples import get_few_shot_prompt

from functools import partial

import inspect

from collections import Counter

import re
import multiprocessing
import asyncio
from openai import AsyncOpenAI

async def grade_mt_bench_async(question, answer, judge_model="gpt-4", api_key=None, api_base=None):
    if not api_key:
        return 0

    client = AsyncOpenAI(
        api_key=api_key,
        base_url=api_base
    )

    prompt_template = """
[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]
"""
    
    prompt = prompt_template.format(question=question, answer=answer)
    
    try:
        response = await client.chat.completions.create(
            model=judge_model,
            messages=[
                {"role": "system", "content": "You are a helpful and precise assistant for checking the quality of the answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=2048,
        )
        content = response.choices[0].message.content
        match = re.search(r"\[\[(\d+)\]\]", content)
        if match:
            return float(match.group(1))
        else:
            return 0
    except Exception as e:
        print(f"Error in grading: {e}")
        return 0

decoding_metrics = get_empty_metrics()


def get_class_methods(cls) -> List[str]:
    """获取类中定义的所有方法（不包括继承的方法和__init__）"""
    methods = []
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # 检查方法是否在当前类中定义（而非继承）
        if name in cls.__dict__ and name != "__init__":
            methods.append(name)
    return methods


def read_results(file_path):
    f = open(file_path)
    data = [json.loads(line) for line in f.readlines()]
    record = {}
    for item in data:
        if item["category"] not in record:
            record[item["category"]] = {"wall_time": [], "num_token": []}
        for choice in item["choices"]:
            record[item["category"]]["wall_time"].extend(choice["wall_time"])
            record[item["category"]]["num_token"].extend(choice["num_token"])
    return record


class EvalMTBench(Baselines):
    def __init__(self, args):
        super().__init__(args)
        # load relative resources

        self.load_tokenizer()
        self.load_model()

        self.load_data()

        self.task = "mt_bench"

        print(self.args.target_model)

        if "Llama-2" in str(self.args.draft_model) and "Llama-2" in str(
            self.args.target_model
        ):
            self.model_id = "llama-2-chat"
        elif "Llama-2" in str(self.args.target_model):
            self.model_id = "vicuna"
        elif "vicuna" in str(self.args.draft_model) and "vicuna" in str(
            self.args.target_model
        ):
            self.model_id = "vicuna"
        elif "Llama-3.1" in str(self.args.draft_model) and "Llama-3.1" in str(
            self.args.target_model
        ):
            self.model_id = "llama-3.1"
        elif "llama" in str(self.args.draft_model):
            self.model_id = "vicuna"
        elif "Qwen" in str(self.args.target_model) or "qwen" in str(self.args.target_model):
            self.model_id = "qwen"
        else:
            raise NotImplementedError

    def load_data(self):
        # * load evaluation data
        self.color_print(f"Loading MT-bench data...", 3)
        data = []
        with open(os.path.join(self.args.data_path, "mt_bench.jsonl")) as f:
            for line in f.readlines():
                datum = json.loads(line)
                data.append(datum)
        self.data = data

    def preprocess(self, input_text):
        pass

    def postprocess(self, input_text, output_text):
        pass

    @torch.inference_mode()
    def eval(self):
        global decoding_metrics
        decoding = self.get_decoding_method()


        decoding = partial(
            decoding,
            transfer_top_k=self.args.transfer_top_k,
            use_precise_comm_sim=self.args.use_precise,
            use_stochastic_comm=self.args.use_stochastic_comm,
            ntt_ms_edge_cloud=self.args.ntt_ms_edge_cloud,
            ntt_ms_edge_end=self.args.ntt_ms_edge_end,
        )

        out_path = os.path.join(
            self.args.exp_name, f"{self.args.eval_mode}_mt_bench.jsonl"
        )
        out_f = open(out_path, "a")

        # warmup
        print(f"Start warm up...")
        n = 10
        for question in tqdm.tqdm(
            self.data,
            total=len(self.data),
            disable=not self.accelerator.is_main_process,
            ncols=50,
        ):
            n -= 1
            if n == 0:
                break
            choices = []
            # set random seed. Ensure each experiment runs with a unique random seed.
            for i in range(1):

                if self.model_id == "llama-3.1" or self.model_id == "qwen":
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                        },
                    ]
                else:
                    conv = get_conversation_template(self.model_id)
                    if self.model_id == "llama-2-chat":
                        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                        conv.system_message = sys_p

                turns = []
                wall_time = []
                num_token = []
                for turn_idx in range(len(question["turns"])):
                    qs = question["turns"][turn_idx]
                    if turn_idx == 0 and self.args.num_shots > 0:
                        few_shot_prompt = get_few_shot_prompt("mt_bench", self.args.num_shots)
                        qs = few_shot_prompt + qs

                    if self.model_id == "llama-3.1" or self.model_id == "qwen":
                        messages.append({"role": "user", "content": qs})
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        input_ids = torch.tensor(
                            self.tokenizer(
                                [prompt],
                                add_special_tokens=False,
                            ).input_ids
                        )

                    else:
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt() + " "
                        input_ids = torch.tensor(
                            self.tokenizer.encode(prompt)
                        ).unsqueeze(0)

                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids = decoding(input_ids)
                    if isinstance(output_ids, tuple) and len(output_ids) == 2:
                        output_ids, _ = output_ids
                    torch.cuda.synchronize()
                    end_time = time.time()

                    output_text = self.tokenizer.decode(
                        output_ids[0], spaces_between_special_tokens=False
                    )

                    for (
                        special_token
                    ) in self.tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output_text = output_text.replace(
                                    special_tok, ""
                                )
                        else:
                            output_text = output_text.replace(special_token, "")
                    output_text = output_text.strip()
                    if self.model_id == "llama-3.1" or self.model_id == "qwen":
                        messages.append(
                            {"role": "assistant", "content": output_text}
                        )
                    else:
                        conv.messages[-1][-1] = output_text
                    turns.append(output_text)

        for question in tqdm.tqdm(
            self.data,
            total=len(self.data),
            disable=not self.accelerator.is_main_process,
            ncols=50,
        ):

            choices = []
            # set random seed. Ensure each experiment runs with a unique random seed.
            for i in range(self.args.num_samples_per_task):
                while self.seed in self.seed_set:
                    self.seed = random.randint(0, 1000000)
                seed_everything(self.seed)

                if self.model_id == "llama-3.1" or self.model_id == "qwen":
                    messages = [
                        {
                            "role": "system",
                            "content": "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.",
                        },
                    ]
                else:
                    conv = get_conversation_template(self.model_id)
                    if self.model_id == "llama-2-chat":
                        sys_p = "You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.\n\nIf a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."
                        conv.system_message = sys_p

                turns = []
                wall_time = []
                num_token = []
                for turn_idx in range(len(question["turns"])):

                    qs = question["turns"][turn_idx]

                    if self.model_id == "llama-3.1" or self.model_id == "qwen":
                        messages.append({"role": "user", "content": qs})
                        prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                        input_ids = torch.tensor(
                            self.tokenizer(
                                [prompt],
                                add_special_tokens=False,
                            ).input_ids
                        )

                    else:
                        conv.append_message(conv.roles[0], qs)
                        conv.append_message(conv.roles[1], None)
                        prompt = conv.get_prompt() + " "
                        input_ids = torch.tensor(
                            self.tokenizer.encode(prompt)
                        ).unsqueeze(0)

                    torch.cuda.synchronize()
                    start_time = time.time()
                    output_ids = decoding(input_ids)
                    if isinstance(output_ids, tuple) and len(output_ids) == 2:
                        output_ids, metrics = output_ids
                        for key in decoding_metrics.keys():
                            if key in metrics and key not in [
                                "little_acceptance_rate",
                                "draft_acceptance_rate",
                                "accuracy"
                            ] and hasattr(metrics[key], "__add__"):
                                decoding_metrics[key] += metrics[key]
                                assert (
                                    decoding_metrics[key] is not None
                                ), f"Metric {key} is None, please check your decoding function."
                            else:
                                # 如果传入一个字典，尝试将字典的值进行累加
                                if isinstance(metrics[key], dict):
                                    try:
                                        for sub_key in metrics[key]:
                                            if sub_key in decoding_metrics[key]:
                                                decoding_metrics[key][sub_key] += metrics[key][sub_key]
                                            else:
                                                decoding_metrics[key][sub_key] = metrics[key][sub_key]
                                    except Exception as e:
                                        print(f"Error updating metric {key}: {e}")
                                        


                    torch.cuda.synchronize()
                    end_time = time.time()

                    output_text = self.tokenizer.decode(
                        output_ids[0], spaces_between_special_tokens=False
                    )

                    for (
                        special_token
                    ) in self.tokenizer.special_tokens_map.values():
                        if isinstance(special_token, list):
                            for special_tok in special_token:
                                output_text = output_text.replace(
                                    special_tok, ""
                                )
                        else:
                            output_text = output_text.replace(special_token, "")
                    output_text = output_text.strip()
                    if self.model_id == "llama-3.1" or self.model_id == "qwen":
                        messages.append(
                            {"role": "assistant", "content": output_text}
                        )
                    else:
                        conv.messages[-1][-1] = output_text
                    turns.append(output_text)
                    wall_time.append(end_time - start_time)
                    num_token.append(output_ids.shape[1] - input_ids.shape[1])
                choices.append(
                    {
                        "index": i,
                        "wall_time": wall_time,
                        "num_token": num_token,
                        "turns": turns,
                    }
                )

            ans_json = {
                "question_id": question["question_id"],
                "category": question["category"],
                "answer_id": shortuuid.uuid(),
                "model_id": self.model_id,
                "choices": choices,
                "tstamp": time.time(),
            }
            if self.accelerator.is_main_process:
                # Calculate Score
                if self.args.openai_api_key:
                    for choice in choices:
                        # Create async tasks for all turns
                        tasks = []
                        for idx, turn in enumerate(choice["turns"]):
                            qs = question["turns"][idx]
                            tasks.append(grade_mt_bench_async(qs, turn, self.args.judge_model, self.args.openai_api_key, self.args.openai_api_base))
                        
                        # Run tasks concurrently
                        async def run_grading_tasks():
                             return await asyncio.gather(*tasks)

                        scores = asyncio.run(run_grading_tasks())
                        choice["score"] = sum(scores) / len(scores) if scores else 0
                        
                        # update metrics
                        if "accuracy" not in decoding_metrics or decoding_metrics["accuracy"] == 0:
                             decoding_metrics["accuracy"] = []
                        decoding_metrics["accuracy"].append(choice["score"])

                out_f.write(json.dumps(ans_json, ensure_ascii=False) + "\n")
                out_f.flush()
        out_f.close()

        self.color_print(f"current eval mode: {self.args.eval_mode}", 0)

        record = read_results(out_path)

        total_num_token, total_wall_time = [], []

        for k in record:
            num_tokens = torch.tensor(record[k]["num_token"])
            wall_times = torch.tensor(record[k]["wall_time"])
            total_num_token.extend(record[k]["num_token"])
            total_wall_time.extend(record[k]["wall_time"])

            speed = torch.sum(num_tokens) / torch.sum(wall_times)
            speed_std = torch.std(num_tokens / wall_times)
            self.color_print(
                f"Generating speed of category {k}: {speed.float().item():.2f} with std {speed_std.float().item()} token / second",
                2,
            )

        total_speed = torch.sum(torch.tensor(total_num_token)) / torch.sum(
            torch.tensor(total_wall_time)
        )
        total_speed_std = torch.std(
            torch.tensor(total_num_token) / torch.tensor(total_wall_time)
        )
        self.color_print(
            f"Average generating speed: {total_speed.float().item():.2f} with std {total_speed_std.float().item()} token / second",
            2,
        )

        # Print debug information
        self.color_print(
            f"wall_time: {torch.sum(torch.tensor(total_wall_time))}, num_token: {torch.sum(torch.tensor(total_num_token))}",
            2,
        )

        # 添加类型检查和默认值
        computation_time = decoding_metrics.get("computation_time", 0.0)
        if not isinstance(computation_time, (int, float)):
            decoding_metrics["computation_time"] = 0.0

        communication_time = decoding_metrics.get("communication_time", 0.0)
        if not isinstance(communication_time, (int, float)):
            decoding_metrics["communication_time"] = 0.0

        if decoding_metrics["wall_time"] != 0:
            decoding_metrics["throughput"] = (
                decoding_metrics["generated_tokens"]
                / decoding_metrics["wall_time"]
            )

        if self.accelerator.is_main_process and "accuracy" in decoding_metrics and isinstance(decoding_metrics["accuracy"], list) and len(decoding_metrics["accuracy"]) > 0:
             avg_score = sum(decoding_metrics["accuracy"]) / len(decoding_metrics["accuracy"])
             decoding_metrics["accuracy"] = avg_score
             self.color_print(f"MT-Bench Average Score: {avg_score:.2f}", 2)

        metrics_str = f"""
        {json.dumps(decoding_metrics, indent = 4)}
        """

        metrics_str += """
        ------------- End of Evaluation Summary -------------
        """

        eval_result = dict(decoding_metrics)
        eval_result["little_model"] = self.args.little_model
        eval_result["draft_model"] = self.args.draft_model
        eval_result["target_model"] = self.args.target_model
        eval_result["eval_mode"] = self.args.eval_mode
        eval_result["gamma"] = self.args.gamma
        eval_result["gamma1"] = self.args.gamma1
        eval_result["gamma2"] = self.args.gamma2

        decoding_metrics_path = os.path.join(
            self.args.exp_name, f"{self.args.eval_mode}_mt_bench_metrics.json"
        )
        os.makedirs(os.path.dirname(decoding_metrics_path), exist_ok=True)
        self.color_print(f"{metrics_str}", 3)
        with open(decoding_metrics_path, "w") as f:
            json.dump(eval_result, f, indent=4)
        self.color_print(
            f"Decoding metrics saved to {decoding_metrics_path}", 2
        )


        self.accelerator.wait_for_everyone()

if __name__ == "__main__":
    args = parse_arguments()
    alg = EvalMTBench(args)
    alg.eval()
