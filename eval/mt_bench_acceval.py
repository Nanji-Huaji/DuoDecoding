import argparse
import json
import os
import time
import re
from tqdm import tqdm
import openai
from concurrent.futures import ThreadPoolExecutor

# Standard MT-Bench Prompts
JUDGE_PROMPT_SINGLE = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question}

[The Start of Assistant's Answer]
{answer}
[The End of Assistant's Answer]"""

JUDGE_PROMPT_MULTI = """[Instruction]
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as helpfulness, relevance, accuracy, depth, creativity, and level of detail. Begin your evaluation by providing a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[Question]
{question_1}

[The Start of Assistant's Answer]
{answer_1}
[The End of Assistant's Answer]

[Question]
{question_2}

[The Start of Assistant's Answer]
{answer_2}
[The End of Assistant's Answer]"""

class MTBenchAcceval:
    def __init__(self, eval_model_config):
        self.eval_model_config = eval_model_config
        self.client = openai.Client(
            api_key=eval_model_config.get("api_key"),
            base_url=eval_model_config.get("base_url")
        )
        self.model_name = eval_model_config.get("model_name", "gpt-4")
        self.temperature = eval_model_config.get("temperature", 0.0)
        
        # Load questions
        self.questions = self._load_questions()

    def _load_questions(self):
        questions = {}
        # Assuming data/mt_bench.jsonl exists relative to the project root
        # We try to find the file relative to this script or absolute path
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base_path, "data", "mt_bench.jsonl")
        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    item = json.loads(line)
                    questions[item["question_id"]] = item
        else:
            print(f"Warning: MT-Bench data not found at {path}")
        return questions

    def _extract_answer(self, text):
        # Heuristic to extract answer from full prompt if necessary
        # Assumes Vicuna format or similar where "ASSISTANT:" precedes the answer
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:")[-1].strip()
        return text.strip()

    def _get_score(self, judgment):
        match = re.search(r"\[\[(\d+\.?\d*)\]\]", judgment)
        if match:
            return float(match.group(1))
        return -1

    def evaluate_single_turn(self, question, answer):
        prompt = JUDGE_PROMPT_SINGLE.format(question=question, answer=answer)
        return self._call_judge(prompt)

    def evaluate_multi_turn(self, question1, answer1, question2, answer2):
        prompt = JUDGE_PROMPT_MULTI.format(
            question_1=question1, answer_1=answer1,
            question_2=question2, answer_2=answer2
        )
        return self._call_judge(prompt)

    def _call_judge(self, prompt):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=2048
            )
            judgment = response.choices[0].message.content
            score = self._get_score(judgment)
            return {"judgment": judgment, "score": score}
        except Exception as e:
            print(f"Error calling judge: {e}")
            return {"judgment": str(e), "score": -1}

    def evaluate_json(self, input_file, output_file=None, max_workers=5):
        results = []
        with open(input_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        def process_line(line):
            try:
                item = json.loads(line)
                q_id = item["question_id"]
                if q_id not in self.questions:
                    return None
                
                question_data = self.questions[q_id]
                turns_questions = question_data["turns"]
                
                # Extract answers
                # Assuming choices[0]
                generated_turns = item["choices"][0]["turns"]
                
                item_results = []
                
                # Turn 1
                q1 = turns_questions[0]
                a1 = self._extract_answer(generated_turns[0])
                res1 = self.evaluate_single_turn(q1, a1)
                item_results.append({
                    "turn": 1,
                    "question": q1,
                    "answer": a1,
                    "judgment": res1["judgment"],
                    "score": res1["score"]
                })
                
                # Turn 2 (if exists)
                if len(turns_questions) > 1 and len(generated_turns) > 1:
                    q2 = turns_questions[1]
                    a2 = self._extract_answer(generated_turns[1])
                    res2 = self.evaluate_multi_turn(q1, a1, q2, a2)
                    item_results.append({
                        "turn": 2,
                        "question": q2,
                        "answer": a2,
                        "judgment": res2["judgment"],
                        "score": res2["score"]
                    })
                
                return {
                    "question_id": q_id,
                    "category": item.get("category"),
                    "model_id": item.get("model_id"),
                    "reviews": item_results
                }
            except Exception as e:
                print(f"Error processing line: {e}")
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_line, line) for line in lines]
            for future in tqdm(futures, total=len(lines), desc="Evaluating"):
                res = future.result()
                if res:
                    results.append(res)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                for res in results:
                    f.write(json.dumps(res) + "\n")
        
        return results

def run_mt_bench(input_file, output_file=None, model_name="gpt-4", api_key=os.getenv("mt_bench_key"), max_workers=5):
    """
    Exposed interface to run MT-Bench evaluation.
    
    Args:
        input_file (str): Path to the input jsonl file containing model generations.
        output_file (str, optional): Path to save the evaluation results.
        model_name (str): Name of the judge model (e.g., "gpt-4", "gpt-4o").
        api_key (str, optional): OpenAI API key. If None, reads from OPENAI_API_KEY env var.
        max_workers (int): Number of parallel workers for evaluation.
        
    Returns:
        list: A list of evaluation results.
    """
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        
    if not api_key:
        raise ValueError("API Key is required. Please provide it or set OPENAI_API_KEY environment variable.")
        
    config = {
        "model_name": model_name,
        "api_key": api_key,
        "temperature": 0.0
    }
    
    evaluator = MTBenchAcceval(config)
    return evaluator.evaluate_json(input_file, output_file, max_workers)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True, help="Path to the input jsonl file")
    parser.add_argument("--output-file", type=str, default="mt_bench_reviews.jsonl", help="Path to the output jsonl file")
    parser.add_argument("--model-name", type=str, default="gpt-4", help="Judge model name")
    parser.add_argument("--api-key", type=str, default=os.getenv("OPENAI_API_KEY"), help="OpenAI API Key")
    parser.add_argument("--max-workers", type=int, default=5, help="Max parallel workers")
    
    args = parser.parse_args()
    
    config = {
        "model_name": args.model_name,
        "api_key": args.api_key,
        "temperature": 0.0
    }
    
    evaluator = MTBenchAcceval(config)
    evaluator.evaluate_json(args.input_file, args.output_file, args.max_workers)