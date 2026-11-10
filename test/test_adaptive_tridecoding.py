import json
import os
import tempfile
import unittest
from argparse import Namespace
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch

from src.baselines import Baselines
from src.utils import rebuild_topk_uniform_probs, sample

#: 三个诊断 JSONL 门控 -> 输出文件名。单开用例按这张表构造 env：
#: 表内其余开关必须**显式置空**（只看"是否设置"不够——外部 shell/CI 可能
#: 已经导出了某个路径），并据此断言"其余两个文件不存在"。
DIAGNOSTIC_TRACE_FILES = {
    "CACHE_CHECK_TRACE": "cache.jsonl",
    "ARP_CALIB_TRACE": "calib.jsonl",
    "TOPK_TRACE": "topk.jsonl",
}


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
        self.bandwidth_edge_cloud = 10
        self.bandwidth_edge_end = 10
        self.ntt_edge_cloud = 0
        self.ntt_edge_end = 0
        self.transfer_calls = []
        self.simulate_transfer_calls = []

    def set_round(self, round_idx):
        # 对齐通信仿真器 per-round 协议（coalesce 计费）；fake 无累积状态
        pass

    def flush_round(self):
        # 对齐通信仿真器 flush_round 协议（轮末结算）；fake 无累积状态
        pass

    def transfer(self, tokens, probs, link_type="edge_cloud", **kwargs):
        self.transfer_calls.append(
            {
                "tokens": None if tokens is None else tokens.clone(),
                "probs": None if probs is None else probs.clone(),
                "link_type": link_type,
                "kwargs": kwargs,
            }
        )
        return 0.0

    def simulate_transfer(self, size, link_type="edge_cloud", **kwargs):
        self.simulate_transfer_calls.append(
            {"size": size, "link_type": link_type, "kwargs": kwargs}
        )
        return 0.0


class _FakeAdapter:
    def __init__(self):
        self.device = torch.device("meta")
        self.threshold = 0.5
        self.step_acc_probs = []

    def reset_step(self):
        self.step_acc_probs = []

    def predict(self, hidden_states):
        self.step_acc_probs.append(1.0)
        return True


class _FakeCache:
    instances = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.device = model.device
        self.constructor_kwargs = kwargs
        self.vocab_size = 4
        self.current_length = 0
        self.prob_history = None
        self.hidden_states = torch.zeros(1, 1, 1)
        self.rollback_calls = []
        _FakeCache.instances.append(self)

    def _set_uniform_history(self, x):
        # A row exists only after its context token was actually forwarded.
        self.current_length = x.shape[1]
        self.prob_history = torch.zeros(
            (1, self.current_length, self.vocab_size), dtype=torch.float
        )
        self.prob_history[:, :, 1] = 1.0

    def _forward_with_kvcache(self, x):
        self._set_uniform_history(x)
        self.hidden_states = torch.zeros(1, 1, 1)
        return self.prob_history[:, -1, :]

    def forward_context(self, prefix):
        assert prefix.shape[1] >= self.current_length
        return self._forward_with_kvcache(prefix)

    def generate(self, prefix, gamma):
        x = prefix.clone()
        for _ in range(gamma):
            self._forward_with_kvcache(x)
            extra = torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
            x = torch.cat((x, extra), dim=1)
        return x

    def generate_with_rebuilt_topk_metadata(self, prefix, gamma, proposal_top_k):
        # 对齐 KVCacheModel 新协议：返回 (x, rebuilt_probs, meta)；
        # fake 不产 top-k 元数据，meta=None（stage_topk_proposal_history 透传）
        x, rebuilt_probs = self.generate_with_rebuilt_topk(
            prefix, gamma, proposal_top_k
        )
        return x, rebuilt_probs, None

    def generate_with_rebuilt_topk(self, prefix, gamma, proposal_top_k):
        if gamma <= 0:
            return prefix.clone(), None
        x = prefix.clone()
        rebuilt_rows = []
        for _ in range(gamma):
            q = self._forward_with_kvcache(x)
            rebuilt_q = rebuild_topk_uniform_probs(q, proposal_top_k)
            rebuilt_rows.append(rebuilt_q.unsqueeze(1))
            next_tok = sample(rebuilt_q)
            x = torch.cat((x, next_tok), dim=1)
        rebuilt = None if not rebuilt_rows else torch.cat(rebuilt_rows, dim=1)
        return x, rebuilt

    def rollback(self, end_pos):
        self.current_length = min(end_pos, self.current_length)
        if self.prob_history is not None:
            self.prob_history = self.prob_history[:, : self.current_length, :]
        self.rollback_calls.append(end_pos)


class AdaptiveTriDecodingTests(unittest.TestCase):
    def setUp(self):
        _FakeCache.instances = []
        args = Namespace(
            eval_mode="adaptive_tridecoding",
            edge_cloud_bandwidth=10,
            edge_end_bandwidth=10,
            cloud_end_bandwidth=10,
            max_tokens=1,
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
            gamma=1,
            gamma1=1,
            gamma2=1,
            use_early_stopping=False,
            dump_network_stats=False,
            disable_rl_update=True,
        )
        self.instance = object.__new__(_TestBaselines)
        self.instance.args = args
        self.instance.accelerator = type("Accel", (), {"is_main_process": True})()
        self.instance.vocab_size = 4
        self.instance.num_acc_tokens = []
        self.instance.draft_forward_times = 0
        self.instance.target_forward_times = 0
        self.instance.little_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "little"}
        )()
        self.instance.draft_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "draft"}
        )()
        self.instance.target_model = type(
            "Model", (), {"device": torch.device("cpu"), "kind": "target"}
        )()
        self.instance.small_draft_adapter = _FakeAdapter()
        self.instance.draft_target_adapter = _FakeAdapter()
        self.instance.rl_adapter = None
        self.instance.little_rl_adapter = None

    def test_adaptive_tridecoding_uses_real_cee_verification_for_both_layers(self):
        prefix = torch.tensor([[0]], dtype=torch.long)
        with (
            patch("src.baselines.runtime.KVCacheModel", _FakeCache),
            patch(
                "src.baselines.simulators.CommunicationSimulator", _FakeCommSimulator
            ),
            patch.object(
                self.instance,
                "_cee_verify_proposal",
                wraps=self.instance._cee_verify_proposal,
            ) as verify,
            patch("src.baselines.core.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.core.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
        ):
            output, metrics = self.instance.adaptive_tridecoding(prefix)
        self.assertEqual(output.tolist(), [[0, 1]])
        self.assertEqual(metrics["generated_tokens"], 1)
        self.assertEqual(metrics["little_accepted_tokens"], 1)
        self.assertEqual(metrics["draft_accepted_tokens"], 3)
        self.assertEqual(len(verify.call_args_list), 2)
        calls = [
            (
                c.args[1].model.kind,
                c.args[2].model.kind,
                c.args[0].prefix_len,
                c.args[0].count,
            )
            for c in verify.call_args_list
        ]
        self.assertEqual(calls, [("little", "draft", 1, 1), ("draft", "target", 1, 3)])

    def test_adaptive_tridecoding_uses_probe_cache_capacity_only_when_configured(self):
        prefix = torch.tensor([[0]], dtype=torch.long)
        with (
            patch("src.baselines.runtime.KVCacheModel", _FakeCache),
            patch(
                "src.baselines.simulators.CommunicationSimulator", _FakeCommSimulator
            ),
            patch("src.baselines.core.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.core.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
        ):
            self.instance.adaptive_tridecoding(prefix)
            default_constructor_kwargs = [
                c.constructor_kwargs for c in _FakeCache.instances
            ]
            _FakeCache.instances = []
            self.instance.args.probe_cache_max_length = 7
            self.instance.adaptive_tridecoding(prefix)
        self.assertEqual(default_constructor_kwargs, [{}, {}, {}])
        self.assertEqual(
            [c.constructor_kwargs for c in _FakeCache.instances],
            [{"max_length": 7}, {"max_length": 7}, {"max_length": 7}],
        )

    def test_adaptive_tridecoding_transfers_vectorized_payloads(self):
        prefix = torch.tensor([[0]], dtype=torch.long)
        fake_comm = _FakeCommSimulator()
        with (
            patch("src.baselines.runtime.KVCacheModel", _FakeCache),
            patch(
                "src.baselines.simulators.CommunicationSimulator",
                return_value=fake_comm,
            ),
            patch("src.baselines.core.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.core.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
        ):
            self.instance.adaptive_tridecoding(prefix)
        payload_calls = [c for c in fake_comm.transfer_calls if c["probs"] is not None]
        self.assertEqual(len(payload_calls), 2)
        for call, link, tokens in zip(
            payload_calls, ("edge_end", "edge_cloud"), (1, 3), strict=True
        ):
            self.assertEqual(call["link_type"], link)
            self.assertTrue(
                torch.equal(call["tokens"], torch.ones((1, tokens), dtype=torch.long))
            )
            self.assertTrue(torch.equal(call["probs"], torch.ones((1, tokens))))

    def test_adaptive_tridecoding_rejects_incomplete_rebuilt_rows_before_payload(self):
        prefix = torch.tensor([[0]], dtype=torch.long)
        fake_comm = _FakeCommSimulator()

        class _ShortSecondStageCache(_FakeCache):
            def generate_with_rebuilt_topk(self, prefix, gamma, proposal_top_k):
                x, rebuilt = super().generate_with_rebuilt_topk(
                    prefix, gamma, proposal_top_k
                )
                if self.model.kind == "draft":
                    rebuilt = rebuilt[:, :1, :]
                return x, rebuilt

        self.instance.args.gamma1 = 2
        with (
            patch("src.baselines.runtime.KVCacheModel", _ShortSecondStageCache),
            patch(
                "src.baselines.simulators.CommunicationSimulator",
                return_value=fake_comm,
            ),
            patch("src.baselines.core.torch.cuda.Event", _FakeCudaEvent),
            patch("src.baselines.core.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
        ):
            with self.assertRaisesRegex(ValueError, "proposal"):
                self.instance.adaptive_tridecoding(prefix, _ablation="static")
        # Incomplete proposals cannot silently pass with truncated payloads.
        self.assertFalse(
            any(c["link_type"] == "edge_cloud" for c in fake_comm.transfer_calls)
        )
        little_payload = [c for c in fake_comm.transfer_calls if c["probs"] is not None]
        self.assertEqual(len(little_payload), 1)
        self.assertEqual(little_payload[0]["link_type"], "edge_end")
        self.assertEqual(little_payload[0]["tokens"].tolist(), [[1]])

    def _run_diagnostic_traces(
        self,
        *,
        rebuilt=False,
        unavailable_path=False,
        enabled=tuple(DIAGNOSTIC_TRACE_FILES),
        directory=None,
    ):
        """跑一轮解码，只打开 ``enabled`` 里的诊断开关（默认三者全开）。

        ``enabled`` 之外的开关被**显式置空**，而不是"不设置"：外部 shell/CI
        若已导出某个 TRACE 路径，不设置会让它继续生效，单开用例就失去隔离性。
        ``patch.dict`` 退出时会恢复外部原值（原本不存在的键会被删除）。

        ``directory`` 由调用方提供时不会被删除，单开用例据此在解码结束后
        断言"其余开关的文件没有生成"。
        """
        enabled = set(enabled)
        unknown = sorted(enabled - set(DIAGNOSTIC_TRACE_FILES))
        if unknown:
            raise ValueError(f"未知诊断开关: {unknown}")

        class _TraceCache(_FakeCache):
            def _set_uniform_history(self, x):
                super()._set_uniform_history(x)
                # Literal hardware rows, not a verifier stub. Adjacent rows and
                # caches differ, making wrong-cache/context diagnostics observable.
                defaults = {
                    "draft": (0.25, 0.125, 0.125, 0.5),
                    "target": (0.5, 0.125, 0.25, 0.125),
                }
                anchors = {
                    "draft": (0.125, 0.5, 0.25, 0.125),
                    "target": (0.125, 0.25, 0.5, 0.125),
                }
                role = self.model.kind
                if role in defaults:
                    for position in range(self.current_length):
                        values = anchors[role] if position % 4 == 0 else defaults[role]
                        self.prob_history[0, position] = torch.tensor(values)
                self.logits_history = None if rebuilt else self.prob_history.log()

        self.instance.args.max_tokens = 8
        with ExitStack() as stack:
            if directory is None:
                directory = stack.enter_context(tempfile.TemporaryDirectory())
            base = Path(directory)
            if unavailable_path:
                base = base / "missing"
            paths = {
                key: base / filename for key, filename in DIAGNOSTIC_TRACE_FILES.items()
            }
            with (
                patch.dict(
                    os.environ,
                    {
                        key: (str(paths[key]) if key in enabled else "")
                        for key in DIAGNOSTIC_TRACE_FILES
                    },
                ),
                patch("src.baselines.runtime.KVCacheModel", _TraceCache),
                patch(
                    "src.baselines.simulators.CommunicationSimulator",
                    _FakeCommSimulator,
                ),
                # Deterministic RNG outcomes; real strict verification and cache
                # rollback remain active, with positive selected target support.
                patch(
                    "torch.rand", side_effect=lambda size, **kw: torch.zeros(size, **kw)
                ),
                patch(
                    "torch.multinomial",
                    side_effect=lambda probs, num_samples=1, **kw: torch.ones(
                        (probs.shape[0], num_samples),
                        dtype=torch.long,
                        device=probs.device,
                    ),
                ),
                patch("src.baselines.core.torch.cuda.Event", _FakeCudaEvent),
                patch(
                    "src.baselines.core.torch.cuda.current_stream", return_value=None
                ),
                patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
            ):
                output, metrics = self.instance.adaptive_tridecoding(
                    torch.tensor([[0]], dtype=torch.long)
                )

            if unavailable_path:
                for key in enabled:
                    self.assertFalse(paths[key].exists(), msg=key)
                return output, metrics, [], [], []
            traces = {
                key: [json.loads(row) for row in paths[key].read_text().splitlines()]
                for key in enabled
            }
        return (
            output,
            metrics,
            traces.get("CACHE_CHECK_TRACE"),
            traces.get("ARP_CALIB_TRACE"),
            traces.get("TOPK_TRACE"),
        )

    def test_diagnostic_traces_append_asymmetric_rows_and_true_logits(self):
        _, _, cache, calib, topk = self._run_diagnostic_traces()
        self.assertEqual(cache, [
            {"where": "pre_verify", "x_len": 4, "prefix_len": 1,
             "target_cache_len": 4, "draft_cache_len": 3, "actual_gamma1": 1},
            {"where": "pre_verify", "x_len": 8, "prefix_len": 5,
             "target_cache_len": 8, "draft_cache_len": 7, "actual_gamma1": 1},
        ])
        self.assertEqual(calib, [
            {"arp": 1.0, "tok": 1, "pd": 0.5, "pt": 0.25,
             "ratio": 0.5, "gmatch": 0},
        ] * 2)
        self.assertEqual(len(topk), 2)
        for row in topk:
            self.assertEqual(row["obj"], "true")
            self.assertEqual(row["tok"], 1)
            self.assertEqual(row["tok_rank"], 1)
            self.assertAlmostEqual(row["tok_pd"], 0.5)
            self.assertEqual(row["t_argmax"], 2)
            self.assertEqual(row["t_argmax_rank"], 2)
            self.assertEqual(row["gmatch"], 0)
            self.assertEqual(row["acc"], 1)
            self.assertEqual(row["k_grid"], [1, 2, 4])
            self.assertEqual(row["cum_mass"], [0.5, 0.75, 1.0])
            self.assertEqual(row["top20"], [0.5, 0.25, 0.125, 0.125])
            self.assertEqual(row["entropy"], 1.213)


    def test_topk_trace_falls_back_to_rebuilt_probabilities(self):
        _, _, _, _, topk = self._run_diagnostic_traces(rebuilt=True)
        self.assertEqual(len(topk), 2)
        for row in topk:
            self.assertEqual(row["obj"], "rebuilt")
            self.assertEqual(row["tok_pd"], 0.5)
            self.assertEqual(row["cum_mass"], [0.5, 0.75, 1.0])

    def test_unavailable_trace_paths_do_not_interrupt_decoding(self):
        output, metrics, _, _, _ = self._run_diagnostic_traces(unavailable_path=True)
        self.assertEqual(output.tolist(), [[0] + [1] * 8])
        self.assertEqual(metrics["generated_tokens"], 8)

    def test_each_diagnostic_switch_enabled_alone_writes_only_its_own_file(self):
        # 非对称行 fake 的完整期望（与三开关全开的用例同源）：单开时每一行
        # 必须仍然是同一份内容，否则"文件存在"可能只是写了个空壳。
        expected_rows = {
            "CACHE_CHECK_TRACE": [
                {
                    "where": "pre_verify",
                    "x_len": 4,
                    "prefix_len": 1,
                    "target_cache_len": 4,
                    "draft_cache_len": 3,
                    "actual_gamma1": 1,
                },
                {
                    "where": "pre_verify",
                    "x_len": 8,
                    "prefix_len": 5,
                    "target_cache_len": 8,
                    "draft_cache_len": 7,
                    "actual_gamma1": 1,
                },
            ],
            "ARP_CALIB_TRACE": [
                {
                    "arp": 1.0,
                    "tok": 1,
                    "pd": 0.5,
                    "pt": 0.25,
                    "ratio": 0.5,
                    "gmatch": 0,
                }
            ]
            * 2,
            "TOPK_TRACE": [
                {
                    "obj": "true",
                    "tok": 1,
                    "tok_rank": 1,
                    "tok_pd": 0.5,
                    "t_argmax": 2,
                    "t_argmax_rank": 2,
                    "gmatch": 0,
                    "acc": 1,
                    "k_grid": [1, 2, 4],
                    "cum_mass": [0.5, 0.75, 1.0],
                    "top20": [0.5, 0.25, 0.125, 0.125],
                    "entropy": 1.213,
                }
            ]
            * 2,
        }

        external_env = {
            key: os.environ.get(key) for key in DIAGNOSTIC_TRACE_FILES
        }
        with tempfile.TemporaryDirectory() as sentinel_dir:
            # 三个开关都预先导出成外部"哨兵"路径：单开某一个时，其余两个必须
            # 被显式置空（而不是"恰好没设置"），否则哨兵路径会被写入。
            sentinel = {
                key: str(Path(sentinel_dir) / f"external-{filename}")
                for key, filename in DIAGNOSTIC_TRACE_FILES.items()
            }
            with patch.dict(os.environ, sentinel):
                for env_name in DIAGNOSTIC_TRACE_FILES:
                    with self.subTest(switch=env_name):
                        with tempfile.TemporaryDirectory() as directory:
                            directory = Path(directory)
                            output, metrics, cache, calib, topk = (
                                self._run_diagnostic_traces(
                                    enabled=(env_name,), directory=directory
                                )
                            )
                            # 解码路径与开关组合无关：仍然 2 轮 / 8 token。
                            self.assertEqual(output.tolist(), [[0] + [1] * 8])
                            self.assertEqual(metrics["generated_tokens"], 8)
                            traces = {
                                "CACHE_CHECK_TRACE": cache,
                                "ARP_CALIB_TRACE": calib,
                                "TOPK_TRACE": topk,
                            }
                            rows = traces[env_name]
                            self.assertIsNotNone(rows, msg=env_name)
                            # 两轮各一条日志（每轮一次 pre_verify / 校准）。
                            self.assertEqual(len(rows), 2, msg=env_name)
                            for row, expected in zip(
                                rows, expected_rows[env_name], strict=True
                            ):
                                for key, value in expected.items():
                                    self.assertEqual(
                                        row[key], value, msg=f"{env_name}.{key}"
                                    )
                            for other_env in DIAGNOSTIC_TRACE_FILES:
                                if other_env == env_name:
                                    continue
                                # 其余两个开关：文件不生成、账本为空、外部导出
                                # 的哨兵路径也没有被写。
                                self.assertFalse(
                                    (
                                        directory
                                        / DIAGNOSTIC_TRACE_FILES[other_env]
                                    ).exists(),
                                    msg=f"{env_name} 单开时 {other_env} 写出了文件",
                                )
                                self.assertFalse(
                                    Path(sentinel[other_env]).exists(),
                                    msg=f"{env_name} 单开时 {other_env} 用了外部值",
                                )
                                self.assertIsNone(traces[other_env], msg=other_env)
                                # patch 退出后，被显式置空的开关恢复成外部值。
                                self.assertEqual(
                                    os.environ.get(other_env), sentinel[other_env]
                                )
                            self.assertEqual(
                                os.environ.get(env_name), sentinel[env_name]
                            )
        # 外层 patch 退出后，进程环境回到测试开始时的原值。
        for key, value in external_env.items():
            self.assertEqual(os.environ.get(key), value, msg=key)


if __name__ == "__main__":
    unittest.main()


class TridecodingStaticAblationTests(unittest.TestCase):
    """tridecoding = adaptive_tridecoding(_ablation="static") 的 wrapper。"""

    def test_wrapper_forwards_static_ablation(self):
        instance = object.__new__(_TestBaselines)
        with patch.object(
            Baselines,
            "adaptive_tridecoding",
            return_value=("out", {"ok": 1}),
        ) as mocked:
            result = instance.tridecoding("prefix", transfer_top_k=1024)
        mocked.assert_called_once_with(
            "prefix",
            transfer_top_k=1024,
            use_precise_comm_sim=False,
            use_stochastic_comm=False,
            ntt_ms_edge_cloud=10,
            ntt_ms_edge_end=1,
            use_early_stopping=False,
            stop_sequences=None,
            _ablation="static",
        )
        self.assertEqual(result, ("out", {"ok": 1}))

    def test_unknown_ablation_is_rejected(self):
        instance = object.__new__(_TestBaselines)
        with self.assertRaises(ValueError):
            instance.adaptive_tridecoding(
                torch.tensor([[1]]), _ablation="bogus"
            )
