import glob
import json
import os
import sys


def get_jsonl_path(exp_name):
    """根据 exp_name 找到对应的 jsonl 文件"""
    search_path = os.path.join("exp", exp_name, "*.jsonl")
    files = glob.glob(search_path)
    if files:
        # 优先返回不含 metrics 的那个
        for f in files:
            if "metrics" not in f:
                return f
        return files[0]
    return None


def calculate_consistency_score(file1, file2):
    """计算两个 jsonl 文件之间的词汇一致度"""
    if not file1 or not file2 or not os.path.exists(file1) or not os.path.exists(file2):
        return None

    def load_data(path):
        data = {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    # 假设是以 question_id 为键
                    qid = obj.get("question_id")
                    if qid is not None:
                        # 提取 turns
                        turns = obj.get("choices", [{}])[0].get("turns", [])
                        data[qid] = turns
        except Exception as e:
            print(f"Error loading {path}: {e}")
        return data

    data1 = load_data(file1)
    data2 = load_data(file2)

    common_ids = set(data1.keys()) & set(data2.keys())
    if not common_ids:
        return 0.0

    total_match_ratio = 0.0
    valid_turns_count = 0

    for qid in common_ids:
        turns1 = data1[qid]
        turns2 = data2[qid]

        for t1, t2 in zip(turns1, turns2):
            if not t1 or not t2:
                continue

            # 我们旨在比较生成的内容。通常 prompt 是相同的。
            # 这里采用简单的前缀匹配/字符匹配比例
            min_len = min(len(t1), len(t2))
            if min_len == 0:
                continue

            matches = 0
            # 逐字符比较，计算一致度
            for i in range(min_len):
                if t1[i] == t2[i]:
                    matches += 1
                else:
                    # 一旦不匹配，后面的可能都对不上了（偏移问题），
                    # 但对于序列生成，通常我们看直到第一个错误的地方的一致性，
                    # 或者整体重合度。这里取整体字符匹配率。
                    pass

            total_match_ratio += matches / min_len
            valid_turns_count += 1

    if valid_turns_count == 0:
        return 0.0

    return total_match_ratio / valid_turns_count


def run_analysis(summary_file):
    if not os.path.exists(summary_file):
        print(f"Summary file not found: {summary_file}")
        return

    with open(summary_file, "r", encoding="utf-8") as f:
        experiments = json.load(f)

    # 按 (数据集, 目标模型) 分组
    groups = {}
    for exp in experiments:
        ds = exp.get("config", {}).get("eval_dataset", "N/A")
        # 处理可能的 result 为 None 的情况
        res = exp.get("result") or {}
        model = res.get("target_model", "N/A")
        key = (ds, model)
        if key not in groups:
            groups[key] = []
        groups[key].append(exp)

    consistency_map = {}  # exp_name -> score

    for key, exps in groups.items():
        # 寻找该组内的 large 实验
        large_exp = next(
            (e for e in exps if (e.get("result") or {}).get("eval_mode") == "large"),
            None,
        )
        if not large_exp:
            continue

        large_file = get_jsonl_path(large_exp["exp_name"])
        if not large_file:
            print(f"Large jsonl not found for {large_exp['exp_name']}")
            continue

        for exp in exps:
            e_name = exp["exp_name"]
            if (exp.get("result") or {}).get("eval_mode") == "large":
                consistency_map[e_name] = 100.0
                continue

            other_file = get_jsonl_path(e_name)
            score = calculate_consistency_score(large_file, other_file)
            if score is not None:
                consistency_map[e_name] = round(score * 100, 2)
            else:
                consistency_map[e_name] = "N/A"

    # 保存结果
    with open("consistency_data.json", "w", encoding="utf-8") as f:
        json.dump(consistency_map, f, ensure_ascii=False, indent=2)
    print(
        f"Consistency data saved to consistency_data.json, total {len(consistency_map)} items."
    )


if __name__ == "__main__":
    summary = (
        sys.argv[1] if len(sys.argv) > 1 else "experiment_summary_20260130_120801.json"
    )
    run_analysis(summary)
