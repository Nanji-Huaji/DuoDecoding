"""D1：model_id 单表的迁移等价性测试。

锁住三条历史链的判定差异（这些差异是有意保留的现状，见
eval/model_ids.py 模块注释）——防止后续统一协议时无声改变行为。
"""

import unittest

from eval.model_ids import (
    determine_model_id,
    determine_model_id_mixed,
)


class ModelIdTableTests(unittest.TestCase):
    # 标准链（7 脚本）：llama-2 双命中 → chat 模板
    def test_standard_llama2_both(self):
        for task in ("eval", "gsm8k", "cnndm", "humaneval", "mt_bench",
                     "mt_bench_noeval", "specbench"):
            self.assertEqual(
                determine_model_id(task, "Llama-2-7b", "Llama-2-13b"),
                "llama-2-chat",
            )

    # 标准链：仅 target 是 Llama-2 → vicuna
    def test_standard_llama2_target_only_is_vicuna(self):
        self.assertEqual(
            determine_model_id("gsm8k", "llama-68m", "Llama-2-13b"), "vicuna"
        )

    # xsum 矛盾：同一组合 → llama-2-chat（与标准链相反）
    def test_xsum_llama2_target_only_is_chat(self):
        self.assertEqual(
            determine_model_id("xsum", "llama-68m", "Llama-2-13b"),
            "llama-2-chat",
        )

    def test_standard_llama32(self):
        self.assertEqual(
            determine_model_id("gsm8k", "Llama-3.2-1B", "Llama-3.2-3B"),
            "llama-3.2",
        )
        self.assertEqual(
            determine_model_id("xsum", "x", "Llama-3.2-3B"), "llama-3.2"
        )

    def test_standard_qwen_gemma(self):
        self.assertEqual(
            determine_model_id("gsm8k", "qwen-small", "Qwen2.5-7B"), "qwen"
        )
        self.assertEqual(
            determine_model_id("cnndm", "gemma-2b", "gemma-7b"), "gemma"
        )

    # 标准链兜底：静默 vicuna（eval/mt_bench* 为硬失败）
    def test_standard_fallback_vicuna_vs_raise(self):
        self.assertEqual(
            determine_model_id("gsm8k", "phi-2", "mistral-7b"), "vicuna"
        )
        with self.assertRaises(NotImplementedError):
            determine_model_id("eval", "phi-2", "mistral-7b")

    # mixed 链：base 不套模板、chat/instruct 套
    def test_mixed_base_vs_chat(self):
        self.assertEqual(
            determine_model_id_mixed("llama-68m", "llama/Llama-2-13b-hf"),
            "base",
        )
        self.assertEqual(
            determine_model_id_mixed("llama-68m", "Llama-2-13b-chat-hf"),
            "llama-2-chat",
        )
        self.assertEqual(
            determine_model_id_mixed("x", "Llama-3.1-8B"), "llama-3.1"
        )

    def test_unknown_task_raises(self):
        with self.assertRaises(ValueError):
            determine_model_id("no_such_task", "a", "b")


if __name__ == "__main__":
    unittest.main()
