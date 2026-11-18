"""MT-Bench judge/noeval 两个入口共享的对话、解码、速度与 IO 流程。

判分是 judge 入口独有的可选依赖：本模块不在导入期接触 openai；只有
``eval.eval_mt_bench.grade_mt_bench_async`` 真正被调用时才延迟导入。
``eval.eval_mt_bench_noeval`` 因此不依赖 API key、网络或 openai 包。

两个任务都沿用旧入口的 ``mt_bench`` 名称（decoder.task 与输出 basename），
差异只在 warmup 规模/轮数、是否判分和调试采样检查。
"""

from __future__ import annotations

import json
import os
import time
from typing import Any

import shortuuid
import torch
import tqdm
from fastchat.model import get_conversation_template

from eval.data import load_task_data
from eval.few_shot_examples import get_few_shot_prompt
from eval.model_ids import determine_model_id

_MT_BENCH_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe.  Your answers should not include "
    "any harmful, unethical, racist, sexist, toxic, dangerous, or illegal "
    "content. Please ensure that your responses are socially unbiased and "
    "positive in nature.\n\nIf a question does not make any sense, or is not "
    "factually coherent, explain why instead of answering something not "
    "correct. If you don't know the answer to a question, please don't share "
    "false information."
)

# 旧入口中走 tokenizer.apply_chat_template 的模型族（模板自带 BOS）。
_CHAT_TEMPLATE_MODEL_IDS = ("llama-3.1", "llama-3.2", "llama-3", "qwen")


def read_results(file_path, skip_lines=0):
    """只解析本次运行追加的行（skip_lines = 运行前文件已有的行数）。

    兼容旧调用方：仍按 category 汇总 wall_time / num_token。实际评测的
    速度统计直接使用本次写出的行，不再回读整个 append 文件。
    """
    with open(file_path) as f:
        lines = f.readlines()[skip_lines:]
    data = [json.loads(line) for line in lines]
    record = {}
    for item in data:
        if item["category"] not in record:
            record[item["category"]] = {"wall_time": [], "num_token": []}
        for choice in item["choices"]:
            record[item["category"]]["wall_time"].extend(choice["wall_time"])
            record[item["category"]]["num_token"].extend(choice["num_token"])
    return record


def _token_debug_checks_enabled() -> bool:
    return os.environ.get("DUODEC_DEBUG_TOKEN_CHECKS", "0") == "1"


def _strip_special_tokens(text: str, tokenizer) -> str:
    for special_token in tokenizer.special_tokens_map.values():
        if isinstance(special_token, list):
            for special_tok in special_token:
                text = text.replace(special_tok, "")
        else:
            text = text.replace(special_token, "")
    return text.strip()


def _decode_answer(output_ids, input_ids, tokenizer) -> str:
    generated_ids = output_ids[0][input_ids.shape[1] :]
    text = tokenizer.decode(generated_ids, spaces_between_special_tokens=False)
    return _strip_special_tokens(text, tokenizer)


def _filter_invalid_output_ids(output_ids, tokenizer):
    """B21 调试开关：只替换越界 id，不改变正常路径。"""
    tokenizer_size = len(tokenizer)
    if (output_ids < 0).any() or (output_ids >= tokenizer_size).any():
        print(
            "DEBUG: Found invalid tokens! "
            f"tokenizer.vocab_size: {tokenizer.vocab_size}, "
            f"len(tokenizer): {tokenizer_size}"
        )
        invalid_mask = (output_ids < 0) | (output_ids >= tokenizer_size)
        print(f"DEBUG: Invalid tokens: {output_ids[invalid_mask]}")
        output_ids = output_ids.clone()
        output_ids[invalid_mask] = 0
    return output_ids


class MTBenchConversation:
    """一个样本的一次完整多轮对话（跨 choice 不复用）。

    模板分叉原样保留：

    - Llama-3.x / Qwen：长 system prompt 起手，逐轮 append。
    - Gemma：空 messages；第一轮把长 system prompt 拼在 user 内容前。
    - 其余（vicuna / llama-2-chat）：FastChat conversation template，
      llama-2-chat 额外设置旧 system prompt。
    """

    def __init__(self, tokenizer, model_id: str):
        self.tokenizer = tokenizer
        self.model_id = model_id
        self.messages = None
        self.conv = None
        if model_id in _CHAT_TEMPLATE_MODEL_IDS:
            self.messages = [{"role": "system", "content": _MT_BENCH_SYSTEM}]
        elif model_id == "gemma":
            self.messages = []
        else:
            self.conv = get_conversation_template(model_id)
            if model_id == "llama-2-chat":
                self.conv.system_message = _MT_BENCH_SYSTEM

    def build_inputs(self, text: str, turn_idx: int):
        if self.messages is not None:
            content = text
            if self.model_id == "gemma" and turn_idx == 0:
                content = _MT_BENCH_SYSTEM + "\n" + text
            self.messages.append({"role": "user", "content": content})
            prompt = self.tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            input_ids = torch.tensor(
                self.tokenizer([prompt], add_special_tokens=False).input_ids
            )
            return prompt, input_ids

        self.conv.append_message(self.conv.roles[0], text)
        # Package uses wrong type hint
        self.conv.append_message(self.conv.roles[1], None)
        prompt = self.conv.get_prompt() + " "
        input_ids = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        return prompt, input_ids

    def append_answer(self, text: str) -> None:
        if self.messages is not None:
            self.messages.append({"role": "assistant", "content": text})
        else:
            self.conv.messages[-1][-1] = text


def _validate_input_ids(run, input_ids, label: str) -> None:
    validate = getattr(getattr(run, "decoder", None), "validate_input_ids", None)
    if validate is not None:
        validate(input_ids, label)


def _speed_of(num_tokens, wall_times):
    num_tokens = torch.tensor(num_tokens)
    wall_times = torch.tensor(wall_times)
    return (
        torch.sum(num_tokens) / torch.sum(wall_times),
        torch.std(num_tokens / wall_times),
    )


def report_speed(rows: list[dict[str, Any]]) -> None:
    """只统计本次 task.run 写出的 rows，不读回历史 append 文件。"""
    record: dict[str, dict[str, list]] = {}
    for item in rows:
        category = item["category"]
        bucket = record.setdefault(category, {"wall_time": [], "num_token": []})
        for choice in item["choices"]:
            bucket["wall_time"].extend(choice["wall_time"])
            bucket["num_token"].extend(choice["num_token"])

    totals: dict[str, list] = {"wall_time": [], "num_token": []}
    for category, bucket in record.items():
        speed, speed_std = _speed_of(bucket["num_token"], bucket["wall_time"])
        totals["wall_time"].extend(bucket["wall_time"])
        totals["num_token"].extend(bucket["num_token"])
        print(
            f"Generating speed of category {category}: "
            f"{speed.float().item():.2f} with std {speed_std.float().item()} "
            "token / second"
        )

    if not totals["num_token"]:
        return
    total_speed, total_speed_std = _speed_of(totals["num_token"], totals["wall_time"])
    print(
        f"Average generating speed: {total_speed.float().item():.2f} with std "
        f"{total_speed_std.float().item()} token / second"
    )
    print(
        f"wall_time: {torch.sum(torch.tensor(totals['wall_time']))}, "
        f"num_token: {torch.sum(torch.tensor(totals['num_token']))}"
    )


class MTBenchTaskBase:
    """两个入口共用的任务实现；子类只声明差异属性/判分钩子。"""

    name = "mt_bench"
    record_name = "mt_bench"
    save_metrics = True
    model_id_task = "mt_bench"
    warmup_questions = 10
    # None = 原 judge 入口：warmup 也逐轮生成；1 = noeval 只热身第一轮。
    warmup_turns = None
    enable_judge = False
    log_warmup_speed = False
    debug_token_checks = False

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.model_id = determine_model_id(
            self.model_id_task, args.draft_model, args.target_model
        )

    def load_data(self):
        return load_task_data("mt_bench", self.args)

    def _run_turns(self, run, question, *, warmup: bool, few_shot: bool, max_turns):
        conversation = MTBenchConversation(self.tokenizer, self.model_id)
        turns: list[str] = []
        wall_time: list[float] = []
        num_token: list[int] = []
        total_turns = len(question["turns"])
        if max_turns is not None:
            total_turns = min(total_turns, max_turns)
        first_input_len = 0
        for turn_idx in range(total_turns):
            qs = question["turns"][turn_idx]
            if few_shot and turn_idx == 0:
                num_shots = int(getattr(run.args, "num_shots", 0) or 0)
                if num_shots > 0:
                    qs = get_few_shot_prompt("mt_bench", num_shots) + qs

            _, input_ids = conversation.build_inputs(qs, turn_idx)
            input_ids = input_ids.to(run.device)
            if turn_idx == 0:
                first_input_len = int(input_ids.shape[1])
            mode = "warmup" if warmup else "eval"
            _validate_input_ids(
                run, input_ids, f"{self.model_id_task}.{mode}.turn_{turn_idx}"
            )

            output_ids, _, elapsed = run.generate(input_ids, record=not warmup)
            if not warmup and self.debug_token_checks and _token_debug_checks_enabled():
                output_ids = _filter_invalid_output_ids(output_ids, self.tokenizer)

            output_text = _decode_answer(output_ids, input_ids, self.tokenizer)
            conversation.append_answer(output_text)
            turns.append(output_text)
            wall_time.append(elapsed)
            num_token.append(int(output_ids.shape[1] - input_ids.shape[1]))
        return turns, wall_time, num_token, first_input_len

    def _warmup(self, run, data) -> None:
        count = min(int(self.warmup_questions), len(data))
        if count <= 0:
            return
        print("Start warm up...")
        main = run.is_main_process
        for warmup_idx, question in enumerate(
            tqdm.tqdm(
                data[:count],
                total=count,
                disable=not main,
                ncols=50,
            )
        ):
            _, wall_time, num_token, input_len = self._run_turns(
                run,
                question,
                warmup=True,
                few_shot=True,
                max_turns=self.warmup_turns,
            )
            if self.log_warmup_speed:
                generated = num_token[0]
                elapsed = wall_time[0]
                rate = generated / elapsed if elapsed else 0.0
                print(f"[Warmup {warmup_idx + 1}/{count}] Input tokens: {input_len}")
                print(
                    f"[Warmup {warmup_idx + 1}/{count}] Generated {generated} "
                    f"tokens in {elapsed:.2f}s ({rate:.2f} tokens/s)"
                )

    def _grade_choices(self, run, question, choices, scores, failures) -> None:
        """子类钩子：judge 入口在此判分；noeval 不导入/不调用判分依赖。"""

    def _run_eval(self, run, data):
        scores: list[float] = []
        failures = {"api": 0, "unparsed": 0}
        rows: list[dict[str, Any]] = []
        main = run.is_main_process
        num_samples = int(getattr(run.args, "num_samples_per_task", 1) or 1)

        for question in tqdm.tqdm(
            data,
            total=len(data),
            disable=not main,
            ncols=50,
        ):
            choices = []
            for choice_idx in range(num_samples):
                # next_seed 同时负责 seed_everything 与跨样本去重。
                run.next_seed()
                turns, wall_time, num_token, _ = self._run_turns(
                    run,
                    question,
                    warmup=False,
                    few_shot=False,
                    max_turns=None,
                )
                choices.append(
                    {
                        "index": choice_idx,
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
            if main:
                if self.enable_judge:
                    self._grade_choices(run, question, choices, scores, failures)
                run.write(ans_json)
            rows.append(ans_json)

        summary = {}
        if scores:
            average = sum(scores) / len(scores)
            run.metrics["accuracy"] = average
            summary["accuracy"] = average
            print(f"MT-Bench Average Score: {average:.2f}")
        if self.enable_judge and (failures["api"] or failures["unparsed"]):
            print(
                f"[mt_bench] 判分未成功：API/网络错误 {failures['api']} 次、"
                f"判分输出无法解析 {failures['unparsed']} 次；这些 turn 按 0 "
                "分计入 accuracy。若数量可观，请勿把 accuracy 读作模型质量"
                "（判分服务可能限流/超时）。"
            )
        report_speed(rows)
        return summary

    def run(self, run, data):
        self._warmup(run, data)
        summary = self._run_eval(run, data)
        metrics = run.metrics
        if metrics.get("wall_time"):
            metrics["throughput"] = (
                metrics.get("generated_tokens", 0) / metrics["wall_time"]
            )
        else:
            metrics["throughput"] = 0.0
        return summary
