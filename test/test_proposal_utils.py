"""B44：build_draft_probs_override 的 stage_start_len=0 切片回归。

此前 `[:, : stage_start_len - 1, :]` 在 stage_start_len=0 时静默变成
`[:, :-1, :]`（取到倒数第二位的整段前缀），产出与调用意图完全无关的
概率覆盖；负值同样静默走负索引。
"""

import unittest
from types import SimpleNamespace

import torch

from src.proposal_utils import build_draft_probs_override


def _cache(seq_len: int = 6, vocab: int = 8) -> SimpleNamespace:
    # 位置 i 的值全为 i，便于断言取到了哪些位置
    probs = torch.arange(seq_len, dtype=torch.float32).unsqueeze(-1).expand(
        1, seq_len, vocab
    )
    return SimpleNamespace(prob_history=probs)


class BuildDraftProbsOverrideTests(unittest.TestCase):
    def test_zero_start_yields_rebuilt_only(self):
        # stage_start_len=0：前缀必须为空，而非 [:, :-1]
        rebuilt = torch.full((1, 2, 8), -1.0)
        out = build_draft_probs_override(_cache(), 0, rebuilt)
        self.assertEqual(out.shape, (1, 2, 8))
        self.assertTrue(torch.equal(out, rebuilt))

    def test_normal_start_prefix_minus_one(self):
        # stage_start_len=3：前缀取 [0, 3) 去掉最后一个位置 → 位置 0,1
        rebuilt = torch.full((1, 2, 8), -1.0)
        out = build_draft_probs_override(_cache(), 3, rebuilt)
        self.assertEqual(out.shape, (1, 4, 8))
        self.assertTrue(torch.equal(out[:, 0, 0], torch.tensor([0.0])))
        self.assertTrue(torch.equal(out[:, 1, 0], torch.tensor([1.0])))

    def test_negative_start_raises(self):
        with self.assertRaises(ValueError):
            build_draft_probs_override(_cache(), -1, torch.zeros(1, 2, 8))

    def test_none_rebuilt_returns_none(self):
        self.assertIsNone(build_draft_probs_override(_cache(), 3, None))


if __name__ == "__main__":
    unittest.main()
