"""TK-SLT（Top-K Sparse Logits Transmission）基线的回归测试。

论文 "Communication-Efficient Collaborative LLM Inference via Distributed
Speculative Decoding"（WCSP'25, Zheng & Yang）。锁定三件事：

1. **纯数学**：ODLD/AS²（Theorem 2 的 Lambert-W 闭式 γ*）必须逐格复现
   论文 Table I；S_inf 复现式 (11)；W_{−1} 二分实现解 w·e^w = z。
2. **通信口径**（论文自身，非仓库统一开关）：上行 = γ·K·b_prob bits
   （b_prob=16，FP16，且概率值真实量化——验证判据与计费看到同一个 q̂）；
   草稿 token 索引与下行响应 negligible ⇒ 0 字节报文。字节公式的数值
   断言在 test_unified_comm_accounting.py，这里锁解码循环的实际调用序列。
3. **解码语义**：草稿缓存的采样 top-k = K（softmax 只作用于 top-K
   logits）；验证用 FP16 量化后的稀疏分布；拒绝重采样基于该稀疏 Q̂；
   ODLD/AS² 开关的接线（γ* 生效、standalone 轮无上行）。
"""

import math
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch

from src.baselines import (
    Baselines,
    _lambert_w_minus_one,
    _last_edge_cloud_tx_seconds,
    _quantize_probs_fp16,
    tk_slt_optimal_draft_length,
    tk_slt_select_speculative,
    tk_slt_speedup_ratio,
)
from src.baselines import (
    prepare_verification_inputs as _real_prepare_verification_inputs,
)
from src.communication import CommunicationSimulator, tk_slt_uplink_payload_bytes

VOCAB = 4
K = 2  # top-K 稀疏 logits 的 K（< VOCAB 才有压缩意义）


class _TestBaselines(Baselines):
    def load_data(self):
        return None

    def preprocess(self, input_text):
        return input_text

    def postprocess(self, input_text, output_text):
        return output_text

    def eval(self):
        return None


class _FakeCudaEvent:
    def __init__(self, enable_timing=True):
        self.enable_timing = enable_timing

    def record(self, stream=None):
        return None

    def elapsed_time(self, other):
        return 0.0


class _FakeCommSimulator:
    """记录 transfer/simulate_transfer 调用序列的最小通信模拟器。

    可选 stats 列表模拟真实链路模型的 TransferUnit.tx_time（ODLD 的 b̂
    估计用）；不提供时 _last_edge_cloud_tx_seconds 应回退 None。
    镜像统一往返口径（2026-10-09）用到的接口：coalesce_rounds 默认 False
    （per_transfer 直充），flush_round/set_round 为 no-op。
    """

    instances = []

    # 统一往返口径：默认 per_transfer（直充）；测试 per_round 时改 True
    coalesce_rounds = False

    def flush_round(self):
        """per_round 合并模式的轮末结算；fake 不做字节合并。"""

    def set_round(self, round_idx):
        """per_round 合并模式的轮界标记；fake 不结算。"""

    def __init__(self, *args, **kwargs):
        self.edge_cloud_comm_time = 0.0
        self.edge_end_comm_time = 0.0
        self.edge_cloud_data = 0
        self.edge_end_data = 0
        self.cloud_end_data = 0
        self.total_comm_energy = 0.0
        self.connect_times = {}
        self.edge_cloud_bandwidth_history = []
        self.edge_cloud_topk_history = []
        self.edge_cloud_draft_len_history = []
        self.transfer_calls = []
        self.stats = {"edge_cloud": [], "edge_end": [], "cloud_end": []}
        self.transfer_top_k = None
        self.tx_time_per_uplink = 0.0
        _FakeCommSimulator.instances.append(self)

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "tokens": None if tokens is None else tokens.clone(),
                "probs": None if probs is None else probs.clone(),
                "link_type": link_type,
                "kwargs": dict(kwargs),
            }
        )
        return 0.0

    def simulate_transfer(self, data_size_bytes, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "tokens": None,
                "probs": None,
                "link_type": link_type,
                "kwargs": dict(kwargs),
                "data_size_bytes": int(data_size_bytes),
            }
        )
        if link_type == "edge_cloud" and kwargs.get("draft_len", 0) > 0:
            # 只给"分布上行"这条消息配 tx_time（论文 T_V 的口径）——
            # ODLD 集成测试可用可控值喂 b̂。
            self.stats["edge_cloud"].append(
                {
                    "data_size_bytes": int(data_size_bytes),
                    "tx_time": self.tx_time_per_uplink,
                }
            )
        return 0.0


class _FakeDraftCache:
    """端侧 SLM 缓存：token 序列与稀疏 top-K 概率行都按固定模式循环。

    每次 generate(prefix, γ) 追加 γ+1 行概率（真实缓存在
    generate(prefix, γ) 后有 prefix_len+γ 行），并从 token 循环里
    取 γ 个新 token。概率行是稀疏 top-2：0.6 在 drafted token、
    0.4 在下一个（0 elsewhere）。
    """

    instances = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.init_args = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        self.model = model
        self.device = model.device
        self.vocab_size = VOCAB
        self.rollback_calls = []
        self.generate_calls = []
        self.prob_history = torch.zeros(1, 0, VOCAB)
        self.logits_history = None
        self._token_cycle = [0, 1, 2, 3]
        self._tok_idx = 0
        _FakeDraftCache.instances.append(self)

    def _sparse_row(self, token: int) -> torch.Tensor:
        row = [0.0] * VOCAB
        row[token % VOCAB] = 0.6
        row[(token + 1) % VOCAB] = 0.4
        return torch.tensor([row]).unsqueeze(1)  # (1, 1, VOCAB)

    def generate(self, prefix, gamma):
        self.generate_calls.append((prefix.shape[1], gamma))
        new_tokens = []
        for _ in range(gamma):
            new_tokens.append(self._token_cycle[self._tok_idx % len(self._token_cycle)])
            self._tok_idx += 1
        for tok in new_tokens:
            self.prob_history = torch.cat(
                (self.prob_history, self._sparse_row(tok)), dim=1
            )
        # 最后一行：最后一个 drafted token 之后的分布（真实缓存也有）
        self.prob_history = torch.cat(
            (
                self.prob_history,
                torch.tensor([[[0.25] * VOCAB]]),
            ),
            dim=1,
        )
        toks = torch.tensor(
            [new_tokens], dtype=prefix.dtype, device=prefix.device
        )
        return torch.cat((prefix, toks), dim=1)

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)

    def reset_for_new_sample(self):
        pass


class _FakeTargetCache:
    """BS 侧 LLM 缓存：对 x 的每个位置给"0.9 在该位置 token"的行，

    最后一行（x 之后的位置）用固定分布供 bonus token 采样。
    extra_rows 可覆盖指定位置的行（如构造必拒场景）。
    """

    instances = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.init_args = dict(temperature=temperature, top_k=top_k, top_p=top_p)
        self.model = model
        self.device = model.device
        self.vocab_size = VOCAB
        self.rollback_calls = []
        self.generate_calls = []
        self.prob_history = None
        self.logits_history = None
        self.extra_rows = {}
        _FakeTargetCache.instances.append(self)

    def generate(self, x, gamma):
        """第 pos 行 = P(·|x[:pos+1])：预测位置 pos+1 的 token。

        pos+1 落在 x 内时把 0.9 放在该 token 上（接受友好的可控场景），
        否则用固定 bonus 分布（x 之后的位置）。
        """
        self.generate_calls.append((x.shape[1], gamma))
        rows = []
        for pos in range(x.shape[1]):
            if pos in self.extra_rows:
                rows.append(list(self.extra_rows[pos]))
                continue
            row = [0.1 / (VOCAB - 1)] * VOCAB
            if pos + 1 < x.shape[1]:
                row[int(x[0, pos + 1]) % VOCAB] = 0.9
            else:
                row = [0.4, 0.3, 0.2, 0.1]
            rows.append(row)
        self.prob_history = torch.tensor([rows])
        return x

    def rollback(self, end_pos):
        self.rollback_calls.append(end_pos)

    def reset_for_new_sample(self):
        pass


class TkSltLoopTestBase(unittest.TestCase):
    """解码循环测试的公共脚手架（fakes + patches，仿 test_uncertainty_decoding）。"""

    def setUp(self):
        _FakeDraftCache.instances = []
        _FakeTargetCache.instances = []
        _FakeCommSimulator.instances = []
        torch.manual_seed(1234)
        args = Namespace(
            eval_mode="tk_slt",
            edge_cloud_bandwidth=46,
            max_tokens=2,
            temp=1.0,
            top_k=0,
            top_p=0.0,
            batch_delay=0.0,
            seed=0,
            exp_name="test",
            eval_dataset="unit",
            little_model="little",
            draft_model="draft",
            target_model="target",
            gamma=2,
            gamma1=1,
            gamma2=1,
            use_early_stopping=False,
            dump_network_stats=False,
            protocol_deviations=(),
        )
        self.args = args
        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = VOCAB
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()

    def _run(self, prefix, transfer_top_k=K, **overrides):
        self.args.__dict__.update(overrides)
        with (
            patch("src.baselines.KVCacheModel", _FakeCacheRouter()),
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.PreciseCommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
        ):
            return self.instance.tk_slt(prefix, transfer_top_k=transfer_top_k)

    def _uplinks(self):
        """分布上行：本方法的上行报文总是带 draft_len=γ（topk 可为 0）。"""
        sim = _FakeCommSimulator.instances[0]
        return [
            c
            for c in sim.transfer_calls
            if "data_size_bytes" in c and c["kwargs"].get("draft_len", 0) > 0
        ]

    def _downlinks(self):
        sim = _FakeCommSimulator.instances[0]
        return [
            c
            for c in sim.transfer_calls
            if "data_size_bytes" in c
            and c["data_size_bytes"] == 0
            and c["kwargs"].get("draft_len", 0) == 0
        ]


class _FakeCacheRouter:
    """按 model.kind 分发到 draft/target 假缓存（替代 KVCacheModel 构造）。

    target_extra_rows：注入到 target 缓存的指定位置概率行
    （构造必拒场景用）。
    """

    def __init__(self, target_extra_rows=None):
        self.target_extra_rows = dict(target_extra_rows or {})

    def __call__(self, model, temperature, top_k, top_p, **kwargs):
        if getattr(model, "kind", None) == "draft":
            return _FakeDraftCache(model, temperature, top_k, top_p, **kwargs)
        cache = _FakeTargetCache(model, temperature, top_k, top_p, **kwargs)
        cache.extra_rows = dict(self.target_extra_rows)
        return cache


class TestTkSltDecodeLoop(TkSltLoopTestBase):
    def test_accept_path_bills_paper_payload_and_downlink(self):
        """单轮全接受：上行 = γ·K·2 B（FP16），下行 = 0 B 报文。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        output, metrics = self._run(prefix, max_tokens=3)

        self.assertEqual(tuple(output.shape), (1, 4))
        self.assertTrue(torch.equal(output[:, :1], prefix))
        self.assertEqual(metrics["generated_tokens"], 3)
        self.assertEqual(metrics["draft_generated_tokens"], 2)
        self.assertEqual(metrics["draft_accepted_tokens"], 2)
        self.assertEqual(metrics["avg_top_k"], K)
        self.assertEqual(metrics["avg_draft_len"], 2)
        self.assertEqual(self.instance.num_acc_tokens, [2])

        # 缓存构造：草稿采样 top-k = K（softmax 只作用于 top-K logits），
        # 目标不压缩——TK-SLT 的定义性行为
        draft_cache, target_cache = (
            _FakeDraftCache.instances[0],
            _FakeTargetCache.instances[0],
        )
        self.assertEqual(draft_cache.init_args["top_k"], K)
        self.assertEqual(target_cache.init_args["top_k"], 0)
        self.assertEqual(target_cache.init_args["top_p"], 0)

        sim = _FakeCommSimulator.instances[0]
        # 1) 首轮 prompt 上传（全仓一次性约定）
        self.assertIsNotNone(sim.transfer_calls[0]["tokens"])
        self.assertIsNone(sim.transfer_calls[0]["probs"])
        # 2) 分布上行：γ·K·(16/8) 字节 = 2·2·2 = 8，附 topk/draft_len 历史
        uplink = self._uplinks()[0]
        self.assertEqual(uplink["data_size_bytes"], 2 * K * 2)
        self.assertEqual(uplink["kwargs"].get("topk"), K)
        self.assertEqual(uplink["kwargs"].get("draft_len"), 2)
        # 3) 下行（结果 token + 位置 j）：§II-B negligible ⇒ 0 字节报文
        self.assertEqual(len(self._downlinks()), 1)
        self.assertEqual(self._downlinks()[0]["data_size_bytes"], 0)
        # 回滚：全接受 ⇒ draft n+1 / target n+2
        self.assertEqual(draft_cache.rollback_calls, [3])
        self.assertEqual(target_cache.rollback_calls, [4])

    def test_verification_uses_fp16_quantized_sparse_q(self):
        """验证判据必须用 FP16 量化后的窗口（计费与数据一致）。

        0.6/0.4 这类值 fp16 roundtrip 后会变——覆盖行 ≠ 原始行，
        但与 _quantize_probs_fp16(原始窗口) 逐位相等。
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        overrides = []

        def spy(draft_model_cache, target_model_cache, x, prefix_len, gamma, **kw):
            overrides.append(
                {
                    "prefix_len": prefix_len,
                    "gamma": gamma,
                    "draft_probs_override": kw.get("draft_probs_override"),
                }
            )
            return _real_prepare_verification_inputs(
                draft_model_cache=draft_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=gamma,
                **kw,
            )

        with patch("src.baselines.prepare_verification_inputs", spy):
            output, _ = self._run(prefix, max_tokens=3)

        self.assertEqual(len(overrides), 1)
        ov = overrides[0]
        draft_cache = _FakeDraftCache.instances[0]
        win_lo = ov["prefix_len"] - 1
        win_hi = ov["prefix_len"] + ov["gamma"] - 1
        raw_window = draft_cache.prob_history[:, win_lo:win_hi, :]
        got = ov["draft_probs_override"][:, ov["prefix_len"] - 1 :, :]
        self.assertEqual(tuple(got.shape), tuple(raw_window.shape))
        self.assertFalse(torch.equal(got, raw_window))  # 确实被量化了
        self.assertTrue(torch.equal(got, _quantize_probs_fp16(raw_window)))
        # 量化后仍是合法分布
        self.assertTrue(
            torch.allclose(got.sum(dim=-1), torch.ones(got.shape[:2]), atol=1e-6)
        )

    def test_reject_path_resamples_from_sparse_quantized_q(self):
        """拒绝路径：BS 从 norm(max(0, P−Q̂)) 重采样，Q̂ = 稀疏 FP16 行。

        构造必拒场景：目标在 drafted token 上概率为 0 ⇒ p/q = 0。
        残差 = P−Q̂ 在稀疏支撑之外的位置拿到完整的 P：drafted token 的
        q̂=0.6 被 0 减掉后残差为 0，token 3（q̂=0）拿到完整 P 分量。
        """
        prefix = torch.tensor([[3]], dtype=torch.long)
        # max_tokens=2 ⇒ 首轮 γ_eff = min(2, 1) = 1；token 循环首个 draft = 0
        # 目标在验证行 0（drafted token 0 的位置）的概率：token 0 上为 0
        reject_router = _FakeCacheRouter(target_extra_rows={0: [0.0, 0.9, 0.1, 0.0]})

        with patch("src.baselines.KVCacheModel", reject_router):
            output, metrics = self._run_manual(prefix, transfer_top_k=K)

        self.assertEqual(tuple(output.shape), (1, 3))
        self.assertTrue(torch.equal(output[:, :1], prefix))
        # 拒绝后重采样的 token ∈ {1, 2}：残差只在稀疏 Q̂ 之外/之上的位置
        self.assertIn(int(output[0, 1]), (1, 2))
        self.assertEqual(metrics["draft_accepted_tokens"], 0)
        # 第 2 轮 remaining=1 走 fallback（target 直出）⇒ 再 append 1
        self.assertEqual(self.instance.num_acc_tokens, [0, 1])
        # 上行按 γ_eff=1 计费：1·K·2 = 4 字节
        self.assertEqual(self._uplinks()[0]["data_size_bytes"], K * 2)
        self.assertEqual(self._uplinks()[0]["kwargs"].get("draft_len"), 1)
        # 拒绝轮照常有一次 0 字节下行（结果 token + 位置 j）
        self.assertEqual(len(self._downlinks()), 1)

    def test_multi_round_window_slicing_and_single_prompt_upload(self):
        """两轮 DSD：prompt 只传一次；第二轮的窗口从 prefix_len-1 切起。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        overrides = []

        def spy(draft_model_cache, target_model_cache, x, prefix_len, gamma, **kw):
            overrides.append(
                {
                    "prefix_len": prefix_len,
                    "gamma": gamma,
                    "draft_probs_override": kw.get("draft_probs_override"),
                }
            )
            return _real_prepare_verification_inputs(
                draft_model_cache=draft_model_cache,
                target_model_cache=target_model_cache,
                x=x,
                prefix_len=prefix_len,
                gamma=gamma,
                **kw,
            )

        with patch("src.baselines.prepare_verification_inputs", spy):
            # gamma=1, max_tokens=4 ⇒ 两个完整 DSD 轮
            output, metrics = self._run(prefix, max_tokens=4, gamma=1)

        draft_cache = _FakeDraftCache.instances[0]
        self.assertEqual(draft_cache.generate_calls, [(1, 1), (3, 1)])
        self.assertEqual(len(overrides), 2)
        self.assertEqual([o["prefix_len"] for o in overrides], [1, 3])
        # 第二轮窗口 = draft prob_history 的 [prefix_len-1 : prefix_len+γ-1] = [2:3]
        second = overrides[1]
        raw = draft_cache.prob_history[:, 2:3, :]
        got = second["draft_probs_override"][:, 2:, :]
        self.assertTrue(torch.equal(got, _quantize_probs_fp16(raw)))
        # prompt 只在首轮上传一次
        prompt_uploads = [
            c for c in _FakeCommSimulator.instances[0].transfer_calls
            if c["tokens"] is not None
        ]
        self.assertEqual(len(prompt_uploads), 1)
        self.assertEqual(len(self._uplinks()), 2)
        self.assertEqual(metrics["generated_tokens"], 4)

    def test_k_disabled_degenerates_to_vanilla_dsd_billing(self):
        """K<=0/None ⇒ 全量 softmax 提议 + 整词表载荷（论文自己的 DSD 基线）。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        output, metrics = self._run(prefix, transfer_top_k=0, max_tokens=3)

        draft_cache = _FakeDraftCache.instances[0]
        self.assertEqual(draft_cache.init_args["top_k"], 0)
        # γ=2、|V|=4、FP16 ⇒ 2·4·2 = 16 字节（整词表）
        self.assertEqual(self._uplinks()[0]["data_size_bytes"], 2 * VOCAB * 2)
        self.assertEqual(self._uplinks()[0]["kwargs"].get("topk"), 0)
        self.assertEqual(metrics["avg_top_k"], 0)

    def _run_manual(self, prefix, **dec_kwargs):
        """不走 _run 的 KVCacheModel patch（调用方自带 router）时用这个。"""
        with (
            patch("src.baselines.CommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.PreciseCommunicationSimulator", _FakeCommSimulator),
            patch("src.baselines.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.torch.cuda.synchronize", return_value=None),
        ):
            return self.instance.tk_slt(prefix, **dec_kwargs)


class TestTkSltOdldWiring(TkSltLoopTestBase):
    def test_gamma_star_replaces_cli_gamma_from_round_two(self):
        """ODLD 开启：首轮用 --gamma 兜底，此后逐轮用 γ*。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        with patch(
            "src.baselines.tk_slt_select_speculative",
            return_value=(True, 3, 1.5),
        ) as sel:
            output, metrics = self._run(
                prefix, max_tokens=8, gamma=2, tk_slt_odld=True
            )

        self.assertEqual(sel.call_count, 2)  # 首轮无估计不咨询；第 2/3 轮咨询
        # （第 3 轮 remaining=1，咨询后走 fallback：target 直出）
        draft_cache = _FakeDraftCache.instances[0]
        self.assertEqual(draft_cache.generate_calls, [(1, 2), (4, 3)])
        self.assertEqual(self.instance.num_acc_tokens, [2, 3, 1])
        self.assertTrue(metrics["tk_slt_odld"])
        self.assertEqual(metrics["tk_slt_gamma_star_final"], 3)

    def test_as2_standalone_round_has_no_uplink(self):
        """AS² 判 S*<=1 的轮次：target 直出、无上行分布，只回 0 字节下行。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        with patch(
            "src.baselines.tk_slt_select_speculative",
            return_value=(False, 1, 0.5),
        ):
            output, metrics = self._run(
                prefix, max_tokens=3, gamma=1, tk_slt_odld=True
            )

        # 首轮 DSD（draft 1 次）+ 第 2 轮 standalone（target 直出）
        draft_cache = _FakeDraftCache.instances[0]
        target_cache = _FakeTargetCache.instances[0]
        self.assertEqual(len(draft_cache.generate_calls), 1)
        self.assertEqual(len(target_cache.generate_calls), 2)
        self.assertEqual(metrics["tk_slt_standalone_rounds"], 1)
        # 上行分布只有首轮那一次；两轮各有一次 0 字节下行
        self.assertEqual(len(self._uplinks()), 1)
        self.assertEqual(len(self._downlinks()), 2)
        self.assertEqual(metrics["generated_tokens"], 3)

    def test_odld_off_by_default_keeps_fixed_gamma(self):
        """默认关闭：所有轮都用 --gamma（与其它基线同口径）。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        output, metrics = self._run(prefix, max_tokens=8, gamma=2)
        draft_cache = _FakeDraftCache.instances[0]
        # 第 3 轮 remaining=2 ⇒ γ_eff=1（越界钳位是既有约定）
        self.assertEqual(draft_cache.generate_calls, [(1, 2), (4, 2), (7, 1)])
        self.assertFalse(metrics["tk_slt_odld"])
        self.assertEqual(metrics["tk_slt_standalone_rounds"], 0)


class TestFp16Quantization(unittest.TestCase):
    def test_values_change_and_rows_stay_normalized(self):
        probs = torch.tensor([[[0.1, 0.3, 0.6, 0.0]]])
        out = _quantize_probs_fp16(probs)
        self.assertFalse(torch.equal(out, probs))
        self.assertTrue(
            torch.allclose(out.sum(dim=-1), torch.ones(1, 1), atol=1e-6)
        )

    def test_zeros_and_one_hot_survive(self):
        # temp=0 的 one-hot 行逐位不变（协议 temp=0 下与未量化路径一致）
        one_hot = torch.tensor([[[0.0, 1.0, 0.0, 0.0]]])
        self.assertTrue(torch.equal(_quantize_probs_fp16(one_hot), one_hot))
        # 稀疏支撑（0 值位置）保持为 0
        sparse = torch.tensor([[[0.6, 0.4, 0.0, 0.0]]])
        out = _quantize_probs_fp16(sparse)
        self.assertTrue(torch.equal(out[:, :, 2:], torch.zeros(1, 1, 2)))

    def test_none_and_empty_pass_through(self):
        self.assertIsNone(_quantize_probs_fp16(None))
        empty = torch.zeros(1, 0, VOCAB)
        self.assertTrue(torch.equal(_quantize_probs_fp16(empty), empty))


class TestLambertWMinusOne(unittest.TestCase):
    def test_solves_we_w_equals_z(self):
        for z in (-1e-30, -1e-10, -1e-4, -0.1, -0.3, -0.367):
            w = _lambert_w_minus_one(z)
            self.assertLessEqual(w, -1.0)
            self.assertAlmostEqual(w * math.exp(w), z, delta=abs(z) * 1e-9 + 1e-40)

    def test_boundary_and_domain_errors(self):
        with self.assertRaises(ValueError):
            _lambert_w_minus_one(0.0)
        with self.assertRaises(ValueError):
            _lambert_w_minus_one(-1.0)  # < -1/e
        with self.assertRaises(ValueError):
            _lambert_w_minus_one(0.5)

    def test_minus_one_branch_not_principal(self):
        # W_{-1} 必须取 ≤ -1 的分支（主分支 W_0 ∈ [-1, 0) 解同一方程）
        w = _lambert_w_minus_one(-0.1)
        self.assertLess(w, -3.0)


class TestOdldTableI(unittest.TestCase):
    """论文 Table I 的 γ* 逐格复现（α 行 × L 列）。

    1* = S_inf < 1（AS² 会退回 standalone LLM）。注意两个精确平局格
    （α=0.4/L=0.4 与 α=0.6/L=0.6）：数学上 S_inf(1) = (1+α)/(1+L) = 1
    恰好成立，论文自己的表格因浮点落点不同把前者标成无星、后者标成
    1*——本实现按 Algorithm 2 的字面判据 S* > 1 取 standalone，平局格
    只断言 γ* 与 S*≈1。
    """

    TABLE_I = {
        0.4: [4, 2, 1, "1", "1*"],
        0.6: [7, 3, 2, 1, "1*"],
        0.8: [14, 6, 4, 2, 1],
    }
    L_VALUES = [0.01, 0.1, 0.2, 0.4, 0.6]

    def test_gamma_star_matches_every_cell(self):
        for alpha, row in self.TABLE_I.items():
            for L, expected in zip(self.L_VALUES, row, strict=True):
                with self.subTest(alpha=alpha, L=L):
                    starred = isinstance(expected, str) and expected.endswith("*")
                    gamma_exp = int(str(expected).rstrip("*"))
                    use_dsd, gamma, s = tk_slt_select_speculative(alpha, L, 0.0)
                    self.assertEqual(gamma, gamma_exp)
                    if abs(s - 1.0) < 1e-9:
                        continue  # 精确平局格：论文自身的浮点落点决定分支
                    self.assertEqual(use_dsd, not starred)

    def test_speedup_ratio_formula(self):
        # 式 (11)：L=0 时 S = E[#tokens] = (1-α^{γ+1})/(1-α)（式 12）
        self.assertEqual(tk_slt_speedup_ratio(5, 0.5, 0.0), 1.96875)
        self.assertAlmostEqual(tk_slt_speedup_ratio(3, 0.0, 0.2), 1 / 1.6)
        self.assertAlmostEqual(tk_slt_speedup_ratio(3, 1.0, 0.2), 4 / 1.6)
        # 论文 Fig 2 的量级：α=0.8、L=0.2、γ=4 ⇒ S≈1.868
        self.assertAlmostEqual(
            tk_slt_speedup_ratio(4, 0.8, 0.2), 0.67232 / 0.36, places=6
        )

    def test_degenerate_guards(self):
        # α ≥ 1 / L ≤ 0：γ 取上限
        self.assertEqual(tk_slt_optimal_draft_length(1.0, 0.1, 0.1)[0], 64)
        self.assertEqual(tk_slt_optimal_draft_length(0.5, 0.0, 0.0)[0], 64)
        # α ≤ 0：γ*=1（AS² → standalone）
        g, s = tk_slt_optimal_draft_length(0.0, 0.1, 0.1)
        self.assertEqual((g, s), (1, 1 / 1.2))
        # L ≥ 1：论文约束外，γ*=1 且 S<1 ⇒ standalone
        g, s = tk_slt_optimal_draft_length(0.5, 0.7, 0.5)
        self.assertEqual(g, 1)
        self.assertLess(s, 1.0)


class TestTxTimeHelper(unittest.TestCase):
    def test_reads_last_edge_cloud_tx_time(self):
        sim = CommunicationSimulator(
            bandwidth_edge_cloud=46,
            bandwidth_edge_end=float("inf"),
            bandwidth_cloud_end=float("inf"),
            dimension="Mbps",
            ntt_ms_edge_cloud=200,
            ntt_ms_edge_end=20,
        )
        sim.simulate_transfer(8, "edge_cloud", topk=K, draft_len=1)
        expected = 8 / (46 * 1e6 / 8)
        self.assertAlmostEqual(
            _last_edge_cloud_tx_seconds(sim), expected, places=15
        )

    def test_returns_none_without_stats(self):
        self.assertIsNone(_last_edge_cloud_tx_seconds(_FakeCommSimulator()))


class TestPaperPayloadReachesSimulator(TkSltLoopTestBase):
    def test_uplink_bytes_follow_gamma_k_and_vocab(self):
        """上行字节 = tk_slt_uplink_payload_bytes（γ·K·b_prob/8），单源。"""
        prefix = torch.tensor([[3]], dtype=torch.long)
        self._run(prefix, max_tokens=3)
        uplink = self._uplinks()[0]
        self.assertEqual(
            uplink["data_size_bytes"],
            int(tk_slt_uplink_payload_bytes(K, 2, VOCAB)),
        )


if __name__ == "__main__":
    unittest.main()
