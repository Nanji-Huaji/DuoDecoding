"""chat 模板 model_id 判定单表（D1：原 9 处内联 if-链单点化）。

分叉现状【原样保留，未统一——统一属行为变更，需实验决策后重跑】：

- 标准链（eval / gsm8k / cnndm / humaneval / mt_bench /
  mt_bench_noeval / specbench 共 7 处）：
  Llama-2 仅命中 target → "vicuna"
  其中 eval.py 的兜底是 raise NotImplementedError（硬失败），
  其余 6 个脚本是静默 "vicuna"——同链不同兜底，原样保留
- xsum 链：
  Llama-2 仅命中 target → "llama-2-chat"（与标准链矛盾）
- mixed 链（小写、只看 target）：
  llama-2 区分 base/chat——base 不套对话模板（理由见函数内注释）

即：同一模型对在不同数据集会拿到不同 chat 模板，跨数据集的
质量分不可直接比较。要修需先定统一协议并重跑评测。
"""

from collections.abc import Callable

Rule = tuple[str, Callable[[str, str], bool]]

# ---- 标准链（7 个评测脚本共用；顺序即优先级，else → vicuna）----
_RULES_STANDARD: list[Rule] = [
    ("llama-2-chat", lambda d, t: "Llama-2" in d and "Llama-2" in t),
    ("vicuna", lambda d, t: "Llama-2" in t),
    ("vicuna", lambda d, t: "vicuna" in d and "vicuna" in t),
    ("llama-3.2", lambda d, t: "Llama-3.2" in t or "Llama-3.2" in d),
    ("llama-3.1", lambda d, t: "Llama-3.1" in d and "Llama-3.1" in t),
    ("llama-3", lambda d, t: "Llama-3" in t or "Llama-3" in d),
    ("vicuna", lambda d, t: "llama" in d or "llama" in t),
    ("qwen", lambda d, t: "Qwen" in t or "qwen" in t),
    ("gemma", lambda d, t: "gemma" in t or "gemma" in d),
]

# ---- xsum 链（Llama-2 优先级靠后且映射不同；else → vicuna）----
_RULES_XSUM: list[Rule] = [
    ("llama-3.2", lambda d, t: "Llama-3.2" in t or "Llama-3.2" in d),
    ("llama-3.1", lambda d, t: "Llama-3.1" in t or "Llama-3.1" in d),
    ("llama-3", lambda d, t: "Llama-3" in t or "Llama-3" in d),
    ("qwen", lambda d, t: "Qwen" in t or "qwen" in t),
    ("gemma", lambda d, t: "gemma" in t or "gemma" in d),
    ("llama-2-chat", lambda d, t: "Llama-2" in t),
    ("vicuna", lambda d, t: "llama" in d or "llama" in t),
]

_TABLE: dict[str, tuple[list[Rule], str]] = {
    # eval.py 历史兜底为硬失败（default=None → raise）
    "eval": (_RULES_STANDARD, None),
    "gsm8k": (_RULES_STANDARD, "vicuna"),
    "cnndm": (_RULES_STANDARD, "vicuna"),
    "humaneval": (_RULES_STANDARD, "vicuna"),
    "mt_bench": (_RULES_STANDARD, "vicuna"),
    "mt_bench_noeval": (_RULES_STANDARD, "vicuna"),
    "specbench": (_RULES_STANDARD, "vicuna"),
    "xsum": (_RULES_XSUM, "vicuna"),
}


def determine_model_id(task: str, draft_model, target_model) -> str:
    """按任务对应的规则表判定 chat 模板 model_id。

    Args:
        task: _TABLE 的键之一（各数据集沿用其历史规则，未统一）。
        draft_model / target_model: 模型名或路径（内部转 str 判子串）。
    """
    if task not in _TABLE:
        raise ValueError(
            f"未知任务 {task!r}，model_id 表支持: {sorted(_TABLE)}"
        )
    d, t = str(draft_model), str(target_model)
    rules, default = _TABLE[task]
    for model_id, predicate in rules:
        if predicate(d, t):
            return model_id
    if default is None:
        # 该任务历史上对未知组合硬失败（如 eval.py），保持
        raise NotImplementedError(
            f"Unsupported model combination: draft={draft_model}, target={target_model}"
        )
    return default


def determine_model_id_mixed(draft_model, target_model) -> str:
    """mixed 评测链（小写、只看 target；与其它数据集规则不同，历史如此）。

    llama-2 的 base/chat 区分：只有真正的 chat/instruct 变体才套对话模板。
    `llama-2-13b` 在本项目里映射到 base 权重（llama/Llama-2-13b-hf），套上
    [INST] 会让 base 模型复读提示词——训练/评测协议就与论文的 eval_gsm8k
    路径（纯文本续写）不一致了，RL 会在一个没有可学结构的退化分布上训练。
    """
    target = str(target_model).lower()
    if "llama-3" in target:
        return "llama-3.1"
    if "qwen" in target:
        return "qwen"
    if "gemma" in target:
        return "gemma"
    if "llama-2" in target:
        if "chat" in target or "instruct" in target:
            return "llama-2-chat"
        return "base"
    return "vicuna"
