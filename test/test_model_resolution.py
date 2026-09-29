"""模型解析链回归测试（B22/B23/B24/B26/B33）。

Historical bugs:
- B22: vocab 查表在 zoo 映射之后，别名全部 miss → 每次 parse 读 config.json
  或走网络回退；字典还有重复键 llama-2-70b。
- B23: 未部署模型映射为 "xxx还没部署" 占位路径静默传播。
- B24: 默认 --draft_model codellama-7b / --target_model codellama-70b 不可解析。
- B26: zoo 别名（qwen-3-0.6b）与 CANONICAL_MODEL_ALIASES（qwen3-0.6b）分歧，
  registry 在 zoo 映射前拿到原始别名时静默错过已注册对。
- B33: few-shot 对不支持的 task（如 mt_bench）静默返回空串，--num_shots 无告警。
"""

import warnings
from argparse import Namespace

import pytest

from eval.few_shot_examples import get_few_shot_prompt
from src.acc_head_registry import canonicalize_model_name
from src.utils import model_zoo


def _ns(**overrides) -> Namespace:
    base = dict(
        draft_model="tiny-llama-1.1b",
        target_model="llama-2-13b",
        little_model="llama-68m",
    )
    base.update(overrides)
    return Namespace(**base)


class TestCanonicalAliases:
    def test_zoo_spelling_unifies_to_registered_series(self):
        # B26: 两种拼法必须归一到同一 canonical series
        assert canonicalize_model_name("qwen-3-0.6b") == "qwen3-0.6b"
        assert canonicalize_model_name("qwen3-0.6b") == "qwen3-0.6b"
        assert canonicalize_model_name("qwen-3-1.7b") == "qwen3-1.7b"
        assert canonicalize_model_name("qwen-3-14b") == "qwen3-14b"
        assert canonicalize_model_name("llama-2-chat-7b") == "llama-2-7b-chat"

    def test_mapped_hf_id_still_resolves(self):
        # zoo 映射后的 HF id 也走同一 canonical
        assert canonicalize_model_name("Qwen/Qwen3-0.6B") == "qwen3-0.6b"


class TestModelZoo:
    def test_missing_models_raise(self):
        # B24: 不再有不可解析的默认值，缺失显式报错
        with pytest.raises(ValueError, match="必须显式指定"):
            model_zoo(Namespace(draft_model=None, target_model=None, little_model=None))
        with pytest.raises(ValueError, match="必须显式指定"):
            model_zoo(_ns(draft_model=""))

    def test_undeployed_placeholder_raises(self):
        # B23: 占位路径在解析阶段被拒绝，不再静默传播
        for name in ("deepseek-1.3b", "deepseek-6.7b", "vicuna-7b-v1.5"):
            with pytest.raises(ValueError, match="未部署"):
                model_zoo(_ns(draft_model=name))
        with pytest.raises(ValueError, match="未部署"):
            model_zoo(_ns(target_model="vicuna-7b-v1.3"))

    def test_vocab_hit_by_alias_and_path_mapping(self):
        # B22: vocab 用原始别名命中（不读 config.json/不走网络），路径照常映射
        args = _ns()
        model_zoo(args)
        assert args.draft_model == "llama/tiny-llama-1.1b"
        assert args.target_model == "llama/Llama-2-13b-hf"
        assert args.little_model == "llama/llama-68m"
        assert args.vocab_size == 32000

    def test_qwen_vocab(self):
        args = _ns(draft_model="qwen-3-0.6b", target_model="qwen-3-1.7b")
        model_zoo(args)
        assert args.draft_model == "Qwen/Qwen3-0.6B"
        assert args.vocab_size == 151936


class TestFewShotWarning:
    def test_unsupported_task_warns_and_returns_empty(self):
        with pytest.warns(UserWarning, match="few-shot 不支持"):
            assert get_few_shot_prompt("mt_bench", 3) == ""

    def test_zero_shots_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert get_few_shot_prompt("mt_bench", 0) == ""

    def test_supported_task_returns_prompt(self):
        assert get_few_shot_prompt("cnndm", 1).startswith("Article:")
