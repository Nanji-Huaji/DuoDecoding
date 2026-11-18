"""CNN/DM 与 XSum 共享的摘要评测运行循环。

两个任务真正的差异只有 prompt 构造、输出后处理，以及 CNN/DM 专属的
摘要 token 预算与解码停串；warmup、多样本种子、异常样本剔除、ROUGE
评分与逐样本落盘完全一致，收拢在本基类。子类在各自入口文件里声明
差异字段并实现 ``preprocess`` / ``postprocess``。
"""

import shortuuid
import torch
import tqdm
from rouge_score import rouge_scorer

from eval.data import load_task_data
from eval.model_ids import determine_model_id


class SummarizationTask:
    """共享运行循环；差异由子类字段与 preprocess/postprocess 决定。"""

    # 子类覆盖：name 同时用作 decoder.task 与输出文件后缀。
    name = "summarization"
    # determine_model_id 的规则链键（cnndm 走标准链，xsum 走自己的链）。
    model_id_task = "summarization"
    doc_field = "article"
    ref_field = "highlights"
    warmup_samples = 5
    warmup_article_tokens = 256
    eval_article_tokens = 1024
    # CNN/DM 专属：摘要 token 预算与解码停串；XSum 两者均为 None。
    summary_max_tokens = None
    stop_sequences = None

    def __init__(self, args, tokenizer):
        self.args = args
        self.tokenizer = tokenizer
        self.model_id = determine_model_id(
            self.model_id_task, args.draft_model, args.target_model
        )

    def load_data(self):
        return load_task_data(self.name, self.args)

    def preprocess(self, input_text):
        raise NotImplementedError

    def postprocess(self, input_text, output_text):
        raise NotImplementedError

    def truncate_article(self, text: str, max_tokens: int) -> str:
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) <= max_tokens:
            return text
        return self.tokenizer.decode(token_ids[:max_tokens], skip_special_tokens=True)

    def _encode(self, prompt):
        # 走 chat template 的模型族：模板自带 BOS，encode 必须关
        # add_special_tokens，否则会双 BOS。
        if self.model_id in ("llama-3.1", "llama-3.2", "llama-3", "qwen"):
            return torch.tensor(
                self.tokenizer([prompt], add_special_tokens=False).input_ids
            )
        return torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)

    def _decode_options(self):
        if self.stop_sequences:
            return {"stop_sequences": self.stop_sequences}
        return {}

    def _warmup(self, run, data):
        count = min(self.warmup_samples, len(data))
        if count <= 0:
            return
        print("Start warm up...")
        for item in tqdm.tqdm(
            data[:count], total=count, disable=not run.is_main_process, ncols=50
        ):
            article = self.truncate_article(
                str(item[self.doc_field]), self.warmup_article_tokens
            )
            input_ids = self._encode(self.preprocess(article)).to(run.device)
            # 旧入口的 partial 对 warmup 同样绑定停串，这里保持一致。
            run.generate(input_ids, record=False, **self._decode_options())

    def run(self, run, data):
        if self.summary_max_tokens is not None:
            # 旧入口在循环前钳制 max_tokens、结束后恢复；run.args 是本次
            # 运行的配置副本（与 decoder.args 为同一对象），钳制即生效，
            # 无需恢复。
            run.args.max_tokens = min(run.args.max_tokens, self.summary_max_tokens)
        self._warmup(run, data)

        scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )
        rouge1_scores, rouge2_scores, rougeL_scores = [], [], []
        total_wall_time = []
        total_num_tokens = []
        num_samples = int(getattr(run.args, "num_samples_per_task", 1) or 1)

        print("Start evaluation...")
        for item in tqdm.tqdm(
            data, total=len(data), disable=not run.is_main_process, ncols=50
        ):
            article = self.truncate_article(
                str(item[self.doc_field]), self.eval_article_tokens
            )
            reference = str(item[self.ref_field])

            for _ in range(num_samples):
                # next_seed 同时负责 seed_everything 与跨样本种子去重
                # （旧实现的 seed_set 循环）。
                run.next_seed()
                input_ids = self._encode(self.preprocess(article)).to(run.device)
                try:
                    output_ids, _, elapsed = run.generate(
                        input_ids, **self._decode_options()
                    )
                except Exception as e:
                    # B31：异常样本整体剔除——不计时、不计 ROUGE，不用
                    # EOS 占位继续（那会把负 token 数混进速度均值）。
                    print(f"Error during decoding: {e}")
                    continue

                # Slice generated part
                if output_ids.shape[1] > input_ids.shape[1]:
                    output_text = self.tokenizer.decode(
                        output_ids[0][input_ids.shape[1] :],
                        skip_special_tokens=True,
                    )
                else:
                    output_text = ""
                output_text = self.postprocess(article, output_text)

                # Calculate ROUGE
                scores = scorer.score(reference, output_text)
                rouge1_scores.append(scores["rouge1"].fmeasure)
                rouge2_scores.append(scores["rouge2"].fmeasure)
                rougeL_scores.append(scores["rougeL"].fmeasure)

                num_tokens = int(output_ids.shape[1] - input_ids.shape[1])
                total_wall_time.append(elapsed)
                total_num_tokens.append(num_tokens)

                run.write(
                    {
                        "id": item.get("id", str(shortuuid.uuid())),
                        "reference": reference,
                        "prediction": output_text,
                        "rouge1": scores["rouge1"].fmeasure,
                        "rouge2": scores["rouge2"].fmeasure,
                        "rougeL": scores["rougeL"].fmeasure,
                        "wall_time": elapsed,
                        "num_tokens": num_tokens,
                    }
                )

        if not run.is_main_process:
            return {}
        elapsed_total = sum(total_wall_time)
        avg_speed = sum(total_num_tokens) / elapsed_total if elapsed_total > 0 else 0
        avg_r1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
        avg_r2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
        avg_rl = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

        print(f"Eval Mode: {run.args.eval_mode}")
        print(f"Average Generation Speed: {avg_speed:.2f} tokens/s")
        print(f"Average ROUGE-1: {avg_r1:.4f}")
        print(f"Average ROUGE-2: {avg_r2:.4f}")
        print(f"Average ROUGE-L: {avg_rl:.4f}")

        run.metrics["accuracy"] = {
            "rouge1": avg_r1,
            "rouge2": avg_r2,
            "rougeL": avg_rl,
        }
        return {}
