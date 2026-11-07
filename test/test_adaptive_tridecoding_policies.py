"""策略门禁：adaptive_tridecoding 的 RL/ARP 动作应用与机会传输分支。

Seam
----
只通过公开入口 Baselines.cee_sd_no_arp / adaptive_tridecoding /
cee_sd_opportunistic 运行生产路径，观察三件事：

1. 返回的 (tokens, metrics)；
2. CommunicationSimulator（真实计费实现，本文件只加有序账本）的逐消息
   (round, kind, link, bytes, topk, draft_len) 账；
3. 假策略头的调用序列 reset_step/predict/threshold/select_config/
   compute_reward/step/save。

允许的轻量替身
--------------
* KVCacheModel：复用 test_adaptive_tridecoding_rounds 的脚本化概率行，
  本地子类额外提供 hidden_states / _forward_with_kvcache /
  logits_history，并记录 proposal 的 (role, gamma, top_k)；
* torch.cuda.Event 等 CPU 缺失对象；
* 机会传输的门控用确定性 CUHLM 子类（只覆盖不确定度数值），
  determine_transfer_strategy 仍走真实阈值比较；
* ARP 头与 RL 策略是记录用假对象（真实头需要 GPU 权重）。

ARP 替身的 ``predict`` 恒返回 False 是刻意的，不是遗漏：生产生成循环每一步
都是「前向 -> 采样 -> adapter.predict」，只有 True 才提前 break。恒 False 让
gamma 步全部发生，于是 (1) 每次 predict 携带的阈值都被逐项记录，(2) gamma
下一轮才生效可由「本轮 predict 次数」与「下一轮生成长度」共同判定。若返回
True，生成会被截断，gamma/topk 的跨轮断言反而失去判别力。阈值与跨轮顺序
用同一份 events 流做跨对象字面值断言，见
``test_arp_predict_sees_old_threshold_until_next_round_and_gamma_waits``。

**不替换** resolve_stage_verification 或 CommunicationSimulator 的
计费实现：所有 token、拒绝/接受、rollback 与字节数都来自生产代码。

测试里的期望值全部是手工推导的独立字面值。
"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

import torch

import test_adaptive_tridecoding_rounds as rounds
from src.communication import CUHLM as _RealCUHLM


PREFIX = torch.tensor([[0]], dtype=torch.long)


class _RoundFixture:
    """借用 rounds 用例的 setUp/tearDown（不把 TestCase 变成顶层 alias）。"""

    def __init__(self):
        self._case = rounds.AdaptiveTriDecodingRoundGateTests(methodName="runTest")
        self._case.setUp()
        self.instance = self._case.instance

    def close(self):
        try:
            self._case.tearDown()
        finally:
            self._case.doCleanups()


class _RecordingPolicyCache(rounds._RowScriptedCache):
    """脚本化缓存 + proposal 记录 + ARP/机会传输所需的最小接口。

    entropy_by_role 是本文件专用的熵注入：state_entropy 会优先读取
    cache.last_entropy（见 src/sampling.py），所以这里在每次 adapter 前向
    （或 no_arp/static 的 rebuilt-topk 生成）时写入按角色配置的固定值。
    未配置的角色保持 None，走原有的 q 回退。
    """

    instances = []
    proposal_calls = []
    entropy_by_role = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hidden_states = torch.zeros(1, 1, 1)

    def _forward_with_kvcache(self, x):
        # 与生产 KVCacheModel 同义：消费到 x 的最后一个位置，返回该位置的
        # （这里脚本化的 one-hot）概率行；ARP 头需要 hidden_states。
        self._consume(x)
        self.hidden_states = torch.zeros(1, 1, 1)
        override = type(self).entropy_by_role.get(self.role)
        self.last_entropy = None if override is None else float(override)
        return self._prob_rows[-1]

    def generate_with_rebuilt_topk(self, prefix, gamma, proposal_top_k):
        # no_arp/static 走 rebuilt-topk 生成而非 adapter 前向；真实 KVCacheModel
        # 在每次前向都会刷新 last_entropy，这里在重建入口按同一角色口径写值。
        type(self).proposal_calls.append((self.role, int(gamma), proposal_top_k))
        override = type(self).entropy_by_role.get(self.role)
        self.last_entropy = None if override is None else float(override)
        return super().generate_with_rebuilt_topk(prefix, gamma, proposal_top_k)

    @property
    def logits_history(self):
        if not self._prob_rows:
            return None
        return torch.stack(
            [torch.log(row.clamp_min(1e-12)) for row in self._prob_rows], dim=1
        )


class _RecordingARPHead:
    """记录 reset/predict/threshold 的 ARP 头替身。

    predict 恒返回 False（见模块 docstring）：每次调用把「调用时刻的阈值」
    写进 calls，因此断言整个 calls 序列既锁住本轮的旧阈值，也锁住跨轮顺序
    （reset -> predict -> select_config 之后才 threshold）。events 与 RL 策略
    共用，用于跨对象的真实执行顺序断言。
    """

    def __init__(self, name, threshold, events=None):
        self.name = name
        self._threshold = float(threshold)
        self.calls = []
        self.events = events
        self.step_acc_probs = []
        # 生产代码在非 static/no_arp 路径断言 adapter.device != cpu。
        self.device = torch.device("meta")

    def _log(self, action):
        if self.events is not None:
            self.events.append((self.name, action))

    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self.calls.append(("threshold", float(value)))
        self._threshold = float(value)
        self._log("threshold")

    def reset_step(self):
        self.calls.append(("reset_step",))
        self._log("reset_step")

    def predict(self, hidden_states):
        self.calls.append(("predict", self._threshold))
        self._log("predict")
        return False


class _RecordingRLPolicy:
    """记录 select_config/compute_reward/step/save 的 RL 策略替身。

    select_config 连同调用时的 training 一起记录；topks 与 gamma 序列按调用
    次数返回（超出后重复最后一个），用来观察本轮结算值与下一轮生成值。
    """

    def __init__(
        self,
        name,
        *,
        topk=None,
        topks=None,
        thresholds,
        gammas=(1,),
        gamma_dim=1,
        events=None,
    ):
        if topks is None and topk is None:
            raise ValueError("_RecordingRLPolicy 需要 topk 或 topks")
        self.name = name
        # topks 给多条动作序列；topk 是单值简写。
        if topks is not None:
            self.topks = tuple(int(v) for v in topks)
        else:
            self.topks = (int(topk),)
        self.topk = self.topks[0]
        self.thresholds = tuple(float(v) for v in thresholds)
        self.gammas = tuple(int(v) for v in gammas)
        self.last_gamma = self.gammas[0]
        self.gamma_dim = gamma_dim
        self.calls = []
        # 每次决策的完整状态特征，按生产调用顺序逐字记录：
        # (bandwidth, latency, tuple(acc_probs), entropy, task_name, training)。
        self.selected_states = []
        self.events = events
        self._select_index = 0

    def _log(self, action):
        if self.events is not None:
            self.events.append((self.name, action))

    def select_config(
        self,
        bandwidth,
        latency,
        acc_probs,
        entropy,
        task_name,
        training,
    ):
        self.calls.append(("select_config", bool(training)))
        self.selected_states.append(
            (
                float(bandwidth),
                float(latency),
                tuple(float(p) for p in acc_probs),
                float(entropy),
                task_name,
                bool(training),
            )
        )
        self._log("select_config")
        index = min(self._select_index, len(self.gammas) - 1)
        self.last_gamma = self.gammas[index]
        threshold = self.thresholds[min(self._select_index, len(self.thresholds) - 1)]
        topk = self.topks[min(self._select_index, len(self.topks) - 1)]
        self._select_index += 1
        return topk, threshold

    def compute_reward(self, **kwargs):
        self.calls.append(("compute_reward",))
        self._log("compute_reward")
        return 1.0

    def step(self, reward):
        self.calls.append(("step", reward))
        self._log("step")

    def save(self, throughput):
        self.calls.append(("save", throughput))
        self._log("save")


class _FallbackGammaPolicy(_RecordingRLPolicy):
    """select 之后把 last_gamma 覆写为给定序列，模拟策略没给出可用 gamma。

    生产写回是 ``int(getattr(adapter, "last_gamma", None) or gamma2_used)``；
    这里让两轮分别返回 None 与 0，锁住「回落到本轮 gamma2_used」这一半逻辑。
    """

    def __init__(self, name, *, last_gammas, **kwargs):
        super().__init__(name, **kwargs)
        self.last_gammas = tuple(last_gammas)
        self._last_gamma_index = 0

    def select_config(self, *args, **kwargs):
        selected = super().select_config(*args, **kwargs)
        index = min(self._last_gamma_index, len(self.last_gammas) - 1)
        self.last_gamma = self.last_gammas[index]
        self._last_gamma_index += 1
        return selected


class _DeterministicCUHLM(_RealCUHLM):
    """只把不确定度数值替换为确定值；阈值比较仍走真实父类实现。"""

    uncertainty_value = 0.0
    gate_calls = []

    def calculate_uncertainty(self, logits, M=20, theta_max=2.0, draft_token=None):
        type(self).gate_calls.append(
            (float(self.uncertainty_threshold), tuple(logits.shape), draft_token)
        )
        return float(type(self).uncertainty_value)


def _cache_by_role():
    return {cache.role: cache for cache in _RecordingPolicyCache.instances}


class AdaptiveTriDecodingPolicyGateTests(unittest.TestCase):
    def setUp(self):
        self._fixture = _RoundFixture()
        self.addCleanup(self._fixture.close)
        self.instance = self._fixture.instance

        # 机会传输门控/统计与缓存实例每次重置。
        _RecordingPolicyCache.instances = []
        _RecordingPolicyCache.proposal_calls = []
        _RecordingPolicyCache.entropy_by_role = {}
        _DeterministicCUHLM.gate_calls = []
        _DeterministicCUHLM.uncertainty_value = 0.0

        self.instance.args.uncertainty_threshold = 0.8
        self.instance.args.small_draft_threshold = 0.5

    def tearDown(self):
        _RecordingPolicyCache.instances = []
        _RecordingPolicyCache.proposal_calls = []
        _RecordingPolicyCache.entropy_by_role = {}
        _DeterministicCUHLM.gate_calls = []
        _DeterministicCUHLM.uncertainty_value = 0.0

    def _run(
        self,
        method_name,
        *,
        max_tokens,
        transfer_top_k=2,
        topk_cap=0,
        disable_rl_update=True,
        opportunistic_uncertainty=None,
        batch_delay=0.0,
        ablation=None,
    ):
        args = self.instance.args
        args.max_tokens = max_tokens
        args.gamma1 = 1
        args.gamma2 = 1
        args.transfer_top_k_cap = topk_cap
        args.disable_rl_update = disable_rl_update
        args.batch_delay = batch_delay
        args.comm_round_trip_mode = "per_transfer"
        args.charge_residual_payload = False

        patches = [
            patch("src.baselines.core.KVCacheModel", _RecordingPolicyCache),
            patch(
                "src.baselines.core.CommunicationSimulator",
                rounds._RecordingCommunicationSimulator,
            ),
            patch("src.baselines.core.torch.cuda.Event", rounds._FakeCudaEvent),
            patch("src.baselines.core.torch.cuda.current_stream", return_value=None),
            patch("src.baselines.core.torch.cuda.synchronize", return_value=None),
        ]
        if opportunistic_uncertainty is not None:
            _DeterministicCUHLM.uncertainty_value = opportunistic_uncertainty
            patches.append(patch("src.baselines.core.CUHLM", _DeterministicCUHLM))

        with ExitStack() as stack:
            for item in patches:
                stack.enter_context(item)
            run_kwargs = {"transfer_top_k": transfer_top_k}
            if ablation is not None:
                run_kwargs["_ablation"] = ablation
            output, metrics = getattr(self.instance, method_name)(
                PREFIX.clone(), **run_kwargs
            )

        simulator = rounds._RecordingCommunicationSimulator.instances[-1]
        return output, metrics, simulator

    def test_no_arp_little_policy_selects_without_step_and_its_topk_does_not_leak(self):
        little = _RecordingRLPolicy(
            "little",
            topk=1,
            thresholds=(0.11, 0.11),
            gammas=(9, 9),
            gamma_dim=1,
        )
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = None
        self.instance.small_draft_adapter = _RecordingARPHead("small", 0.51)
        self.instance.draft_target_adapter = _RecordingARPHead("main", 0.52)

        output, metrics, simulator = self._run(
            "cee_sd_no_arp",
            max_tokens=5,
        )

        # 独立期望：两轮、每轮 little 1 + bonus + draft 1 + bonus = 4 token；
        # 起点 [0]，max_tokens 截断到 6 个位置。全 one-hot token 1。
        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])

        # little 侧 RL 返回的 topk=1 不得进入 transfer：final 计费仍是初始
        # transfer_top_k=2；若被误用，下面账本的 topk 会变成 1 且第二轮
        # proposal 也会按 k=1 重建。
        self.assertEqual(
            _RecordingPolicyCache.proposal_calls,
            [
                ("little", 1, 2),
                ("draft", 1, 2),
                ("little", 1, 2),
                ("draft", 1, 2),
            ],
        )
        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "transfer", "edge_end", 12, 0, 0),
                (1, "simulate", "edge_end", 12.0, 0, 0),
                (1, "transfer", "edge_cloud", 24, 0, 3),
                (1, "transfer", "edge_cloud", 36, 0, 3),
                (1, "simulate", "edge_cloud", 12.0, 2, 3),
                (1, "simulate", "edge_end", 12, 0, 0),
                (2, "transfer", "edge_end", 12, 0, 0),
                (2, "simulate", "edge_end", 12.0, 0, 0),
                (2, "transfer", "edge_cloud", 16, 0, 2),
                (2, "transfer", "edge_cloud", 36, 0, 3),
                (2, "simulate", "edge_cloud", 12.0, 2, 3),
                (2, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        # 关闭 RL 更新：仍 select（training=False）、仍算 reward，但不 step/save。
        self.assertEqual(
            little.calls,
            [
                ("select_config", False),
                ("compute_reward",),
                ("select_config", False),
                ("compute_reward",),
            ],
        )
        self.assertFalse(any(call[0] in ("step", "save") for call in little.calls))

        # no_arp：ARP 头一次也不能碰（reset/predict/threshold 全无）。
        self.assertEqual(self.instance.small_draft_adapter.calls, [])
        self.assertEqual(self.instance.draft_target_adapter.calls, [])

        # gamma_dim=1 时 last_gamma=9 不得覆盖 _next_gamma（两轮 proposal
        # gamma 恒为 args.gamma2=1，已在上面的 proposal_calls 断言）。
        self.assertIsNone(getattr(self.instance, "_next_gamma1", None))
        self.assertIsNone(getattr(self.instance, "_next_gamma2", None))

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 2,
            "little_accepted_tokens": 2,
            "draft_generated_tokens": 6,
            "draft_accepted_tokens": 6,
            "little_forward_times": 2,
            "draft_forward_times": 4,
            "target_forward_times": 2,
            "queuing_time": 0.0,
            "edge_end_data_bytes": 80,
            "edge_cloud_data_bytes": 136,
            "connect_times": {"edge_end": 7, "cloud_end": 0, "edge_cloud": 6},
            "comm_trace_edge_cloud": [
                [24, 1],
                [36, 1],
                [12.0, 1],
                [16, 2],
                [36, 2],
                [12.0, 2],
            ],
            "edge_cloud_topk_history": [0, 0, 2, 0, 0, 2],
            "edge_cloud_draft_len_history": [3, 3, 3, 2, 3, 3],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertAlmostEqual(metrics["communication_time"], 0.0671728, places=12)

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        self.assertEqual(little_cache.rollback_calls, [2, 6])
        self.assertEqual(draft_cache.rollback_calls, [3, 4, 7, 8])
        self.assertEqual(target_cache.rollback_calls, [5, 9])

    def test_no_arp_main_policy_topk_is_capped_for_this_round_and_gamma_waits(self):
        main = _RecordingRLPolicy(
            "main",
            topks=(3, 6),
            thresholds=(0.22, 0.22),
            gammas=(3, 4),
            gamma_dim=2,
        )
        self.instance.little_rl_adapter = None
        self.instance.rl_adapter = main
        self.instance.small_draft_adapter = _RecordingARPHead("small", 0.51)
        self.instance.draft_target_adapter = _RecordingARPHead("main", 0.52)

        output, metrics, simulator = self._run(
            "cee_sd_no_arp",
            max_tokens=5,
            topk_cap=4,
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])

        # 本轮 proposal 在 main 选择之前构造，仍用旧的 transfer_top_k；
        # 第一轮旧值 2，第二轮旧值 3（第一轮 select 的返回值），而不是本轮
        # 刚写回的 4（6 被 cap 到 4）。这样 topk 的「本轮结算 vs 下轮生成」
        # 不会因为两轮取值恰好相同而失去判别力。
        # gamma 同理：第一轮 1，第二轮用第一轮选出的 3，而不是本轮写回的 4。
        self.assertEqual(
            _RecordingPolicyCache.proposal_calls,
            [
                ("little", 1, 2),
                ("draft", 1, 2),
                ("little", 1, 3),
                ("draft", 3, 3),
            ],
        )
        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "transfer", "edge_end", 12, 0, 0),
                (1, "simulate", "edge_end", 12.0, 0, 0),
                (1, "transfer", "edge_cloud", 24, 0, 3),
                (1, "transfer", "edge_cloud", 36, 0, 3),
                (1, "simulate", "edge_cloud", 12.0, 3, 3),
                (1, "simulate", "edge_end", 12, 0, 0),
                (2, "transfer", "edge_end", 12, 0, 0),
                (2, "simulate", "edge_end", 12.0, 0, 0),
                (2, "transfer", "edge_cloud", 16, 0, 2),
                (2, "transfer", "edge_cloud", 60, 0, 5),
                (2, "simulate", "edge_cloud", 12.0, 4, 5),
                (2, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        self.assertEqual(
            main.calls,
            [
                ("select_config", False),
                ("compute_reward",),
                ("select_config", False),
                ("compute_reward",),
            ],
        )
        self.assertFalse(any(call[0] in ("step", "save") for call in main.calls))
        self.assertEqual(self.instance.small_draft_adapter.calls, [])
        self.assertEqual(self.instance.draft_target_adapter.calls, [])

        # 第二轮 select 写回 4，但只可能作用于第三轮；两轮跑完时对外状态是 4。
        self.assertEqual(getattr(self.instance, "_next_gamma1", None), 4)
        self.assertIsNone(getattr(self.instance, "_next_gamma2", None))

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 2,
            "little_accepted_tokens": 2,
            "draft_generated_tokens": 8,
            "draft_accepted_tokens": 8,
            "little_forward_times": 2,
            "draft_forward_times": 6,
            "target_forward_times": 2,
            "queuing_time": 0.0,
            "edge_end_data_bytes": 80,
            "edge_cloud_data_bytes": 160,
            "connect_times": {"edge_end": 7, "cloud_end": 0, "edge_cloud": 6},
            "comm_trace_edge_cloud": [
                [24, 1],
                [36, 1],
                [12.0, 1],
                [16, 2],
                [60, 2],
                [12.0, 2],
            ],
            "edge_cloud_topk_history": [0, 0, 3, 0, 0, 4],
            "edge_cloud_draft_len_history": [3, 3, 3, 2, 5, 5],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertAlmostEqual(metrics["communication_time"], 0.067192, places=12)

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        self.assertEqual(little_cache.rollback_calls, [2, 6])
        self.assertEqual(draft_cache.rollback_calls, [3, 4, 7, 10])
        self.assertEqual(target_cache.rollback_calls, [5, 11])

    def test_no_arp_little_gamma_dim_applies_next_round_and_topk_does_not_leak(self):
        """little gamma_dim>1：本轮生成只认上一轮 last_gamma，写回留给下一轮。

        口径：args.gamma2=1，little 两轮分别选 gamma=2/3；主策略 gamma_dim=1
        （last_gamma=9 不得改写 args.gamma1），两轮 topk 选 3/4。手工推导：
          · 第一轮 little proposal gamma=1（args.gamma2，尚无 _next_gamma2）；
          · 第二轮 gamma=2（第一轮 last_gamma），不是第二轮刚选的 3；
          · 跑完 _next_gamma2=3（第二轮写回），_next_gamma1=None。
        little 的 topk 5/6 与阈值 0.61/0.62 在 no_arp 下都不生效：账本/提案里
        的 topk 只来自主策略，两个 ARP 头零调用（阈值没有落到第一级头上）。
        """
        little = _RecordingRLPolicy(
            "little",
            topks=(5, 6),
            thresholds=(0.61, 0.62),
            gammas=(2, 3),
            gamma_dim=2,
        )
        main = _RecordingRLPolicy(
            "main",
            topks=(3, 4),
            thresholds=(0.71, 0.72),
            gammas=(9, 9),
            gamma_dim=1,
        )
        small_arp = _RecordingARPHead("small", 0.51)
        main_arp = _RecordingARPHead("main", 0.52)
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = main
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp
        self.instance.task = "policy-gate-little-gamma"
        _RecordingPolicyCache.entropy_by_role = {"little": 1.25, "draft": 2.5}

        output, metrics, simulator = self._run(
            "cee_sd_no_arp",
            max_tokens=5,
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])

        # 第三项 topk：第一轮 2（transfer_top_k 初始值），第二轮 3（主策略
        # 第一轮选出的值）；little 的 5/6 从未进入 proposal。
        self.assertEqual(
            _RecordingPolicyCache.proposal_calls,
            [
                ("little", 1, 2),
                ("draft", 1, 2),
                ("little", 2, 3),
                ("draft", 1, 3),
            ],
        )

        # 熵来自各自角色（1.25/2.5），acc_probs 空、training 关闭。
        self.assertEqual(
            little.selected_states,
            [
                (10.0, 1.0, (), 1.25, "policy-gate-little-gamma", False),
                (10.0, 1.0, (), 1.25, "policy-gate-little-gamma", False),
            ],
        )
        self.assertEqual(
            main.selected_states,
            [
                (10.0, 10.0, (), 2.5, "policy-gate-little-gamma", False),
                (10.0, 10.0, (), 2.5, "policy-gate-little-gamma", False),
            ],
        )

        # disable_rl_update=True：select/compute_reward 照常，step/save 不出现。
        for policy in (little, main):
            self.assertEqual(
                policy.calls,
                [
                    ("select_config", False),
                    ("compute_reward",),
                    ("select_config", False),
                    ("compute_reward",),
                ],
            )

        # no_arp 不挂 ARP 头：两个头的 calls 全空，阈值没有被写进第一级。
        self.assertEqual(small_arp.calls, [])
        self.assertEqual(main_arp.calls, [])

        self.assertEqual(getattr(self.instance, "_next_gamma2", None), 3)
        self.assertIsNone(getattr(self.instance, "_next_gamma1", None))

        cloud = [
            (round_idx, kind, topk, draft_len)
            for round_idx, kind, link, _data, topk, draft_len in simulator.ledger
            if link == "edge_cloud"
        ]
        self.assertEqual(
            cloud,
            [
                (1, "transfer", 0, 3),
                (1, "transfer", 0, 3),
                (1, "simulate", 3, 3),
                (2, "transfer", 0, 3),
                (2, "transfer", 0, 4),
                (2, "simulate", 4, 4),
            ],
        )
        # 第一级上行载荷逐 token 计 8 B(token)+4 B(prob)：第二轮 gamma=2
        # 让该笔从 12 涨到 24；初始 prompt 的 8 B 不计入。
        self.assertEqual(
            [
                (round_idx, data)
                for round_idx, kind, link, data, _topk, _draft_len in simulator.ledger
                if link == "edge_end" and kind == "transfer" and data > 8
            ],
            [(1, 12), (2, 24)],
        )

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 3,
            "little_accepted_tokens": 3,
            "draft_generated_tokens": 7,
            "draft_accepted_tokens": 7,
            "little_forward_times": 3,
            "draft_forward_times": 4,
            "target_forward_times": 2,
            "edge_end_data_bytes": 92,
            "edge_cloud_data_bytes": 156,
            "edge_cloud_topk_history": [0, 0, 3, 0, 0, 4],
            "edge_cloud_draft_len_history": [3, 3, 3, 3, 4, 4],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)

    def test_static_ablation_keeps_rl_gate_closed_for_little_select(self):
        """static 消融下即使挂了策略/ARP 头，caller 的门控也必须关死。

        抽取 little select/apply 后，RL/static 门控仍留在调用方：本用例的
        little/main 策略 select、两个 ARP 头 reset/predict/threshold 都不得
        发生，proposal 恒用 args.gamma1/args.gamma2=1 与初始 topk=2。
        """
        little = _RecordingRLPolicy(
            "little",
            topk=1,
            thresholds=(0.61, 0.62),
            gammas=(2, 3),
            gamma_dim=2,
        )
        main = _RecordingRLPolicy(
            "main",
            topk=3,
            thresholds=(0.71, 0.72),
            gammas=(2, 3),
            gamma_dim=2,
        )
        small_arp = _RecordingARPHead("small", 0.51)
        main_arp = _RecordingARPHead("main", 0.52)
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = main
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp

        output, metrics, _ = self._run(
            "adaptive_tridecoding",
            max_tokens=5,
            ablation="static",
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])
        self.assertEqual(
            _RecordingPolicyCache.proposal_calls,
            [
                ("little", 1, 2),
                ("draft", 1, 2),
                ("little", 1, 2),
                ("draft", 1, 2),
            ],
        )
        self.assertEqual(little.calls, [])
        self.assertEqual(main.calls, [])
        self.assertEqual(small_arp.calls, [])
        self.assertEqual(main_arp.calls, [])
        self.assertIsNone(getattr(self.instance, "_next_gamma1", None))
        self.assertIsNone(getattr(self.instance, "_next_gamma2", None))

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 2,
            "draft_generated_tokens": 6,
            "edge_end_data_bytes": 80,
            "edge_cloud_data_bytes": 136,
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)

    def test_little_gamma_falls_back_to_args_gamma2_when_last_gamma_missing_or_zero(
        self,
    ):
        """last_gamma=None/0 时，_next_gamma2 必须回落到本轮 gamma2_used。

        两轮分别返回 None 与 0；写回值是 ``last_gamma or gamma2_used``，所以
        两轮生成都保持 args.gamma2=1，跑完 _next_gamma2=1（不是 0/None）。
        若抽取时把回落写丢（例如直接 int(last_gamma)），这里会抛错或把
        gamma 写成 0 而让 proposal/账本立刻变形。
        """
        little = _FallbackGammaPolicy(
            "little",
            topk=1,
            thresholds=(0.11, 0.11),
            gammas=(2, 3),
            gamma_dim=2,
            last_gammas=(None, 0),
        )
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = None
        self.instance.small_draft_adapter = _RecordingARPHead("small", 0.51)
        self.instance.draft_target_adapter = _RecordingARPHead("main", 0.52)

        output, metrics, _ = self._run(
            "cee_sd_no_arp",
            max_tokens=5,
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])
        self.assertEqual(
            _RecordingPolicyCache.proposal_calls,
            [
                ("little", 1, 2),
                ("draft", 1, 2),
                ("little", 1, 2),
                ("draft", 1, 2),
            ],
        )
        self.assertEqual(getattr(self.instance, "_next_gamma2", None), 1)
        self.assertIsNone(getattr(self.instance, "_next_gamma1", None))

        # 回落路径与现有 no_arp 两轮全接受用例同形：gamma 始终 1。
        self.assertEqual(
            metrics["edge_cloud_draft_len_history"],
            [3, 3, 3, 2, 3, 3],
        )
        self.assertEqual(metrics["little_generated_tokens"], 2)
        self.assertEqual(metrics["draft_generated_tokens"], 6)
        self.assertEqual(metrics["edge_end_data_bytes"], 80)
        self.assertEqual(metrics["edge_cloud_data_bytes"], 136)

    def test_normal_arp_threshold_updates_after_select_and_applies_next_round(self):
        order = []
        little = _RecordingRLPolicy(
            "little_rl",
            topk=2,
            thresholds=(0.61, 0.62),
            gammas=(9, 9),
            gamma_dim=1,
            events=order,
        )
        main = _RecordingRLPolicy(
            "main_rl",
            topk=2,
            thresholds=(0.71, 0.72),
            gammas=(9, 9),
            gamma_dim=1,
            events=order,
        )
        small_arp = _RecordingARPHead("small_arp", 0.51, events=order)
        main_arp = _RecordingARPHead("main_arp", 0.52, events=order)
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = main
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp

        # 状态特征必须逐项可判别：两条链路的 bandwidth/latency 互不相同
        # （23/7 Mbps 与 10/1 ms），little/draft 的 entropy 与 acc_probs
        # 也各自不同且非空，task 用显式哨兵。
        self.instance.args.edge_cloud_bandwidth = 23.0
        self.instance.args.edge_end_bandwidth = 7.0
        self.instance.task = "policy-gate-normal-arp"
        _RecordingPolicyCache.entropy_by_role = {"little": 1.25, "draft": 2.5}
        small_arp.step_acc_probs = [0.125, 0.75]
        main_arp.step_acc_probs = [0.25, 0.5, 0.875]

        output, metrics, simulator = self._run(
            "adaptive_tridecoding",
            max_tokens=5,
            topk_cap=0,
            disable_rl_update=False,
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])

        # select_config 收到的状态字面值（不是 capability 或派生近似值）：
        # bandwidth 与 latency 不同，熵来自 cache.last_entropy（state_entropy
        # 优先读它），acc_probs 来自各自的 ARP 头。两轮逐值相同，因此
        # 把熵误写成 gamma(=1) 的变异也会在这里失败。
        self.assertEqual(
            little.selected_states,
            [
                (7.0, 1.0, (0.125, 0.75), 1.25, "policy-gate-normal-arp", True),
                (7.0, 1.0, (0.125, 0.75), 1.25, "policy-gate-normal-arp", True),
            ],
        )
        self.assertEqual(
            main.selected_states,
            [
                (
                    23.0,
                    10.0,
                    (0.25, 0.5, 0.875),
                    2.5,
                    "policy-gate-normal-arp",
                    True,
                ),
                (
                    23.0,
                    10.0,
                    (0.25, 0.5, 0.875),
                    2.5,
                    "policy-gate-normal-arp",
                    True,
                ),
            ],
        )

        # 正常（非 no_arp）路径：select 写出阈值后立刻落到对应 ARP 头；
        # 本轮 proposal 的 predict 用的是上一轮阈值，所以第二轮的 predict
        # 观察到 0.61 / 0.71，而不是本轮刚写的 0.62 / 0.72。
        self.assertEqual(
            small_arp.calls,
            [
                ("reset_step",),
                ("predict", 0.51),
                ("threshold", 0.61),
                ("reset_step",),
                ("predict", 0.61),
                ("threshold", 0.62),
            ],
        )
        self.assertEqual(
            main_arp.calls,
            [
                ("reset_step",),
                ("predict", 0.52),
                ("threshold", 0.71),
                ("reset_step",),
                ("predict", 0.71),
                ("threshold", 0.72),
            ],
        )

        # 跨对象的真实顺序：第一级 select -> 落阈值 -> reward/step，然后才
        # 进入第二级的 reset/predict/select/threshold/reward/step。
        round_order = [
            ("small_arp", "reset_step"),
            ("small_arp", "predict"),
            ("little_rl", "select_config"),
            ("small_arp", "threshold"),
            ("little_rl", "compute_reward"),
            ("little_rl", "step"),
            ("main_arp", "reset_step"),
            ("main_arp", "predict"),
            ("main_rl", "select_config"),
            ("main_arp", "threshold"),
            ("main_rl", "compute_reward"),
            ("main_rl", "step"),
        ]
        self.assertEqual(
            order,
            round_order
            + round_order
            + [("main_rl", "save"), ("little_rl", "save")],
        )

        # RL 更新开启：select(training=True) -> reward -> step，末尾 save 一次。
        for policy in (little, main):
            self.assertEqual(
                [call[0] for call in policy.calls],
                [
                    "select_config",
                    "compute_reward",
                    "step",
                    "select_config",
                    "compute_reward",
                    "step",
                    "save",
                ],
            )
            self.assertTrue(
                all(
                    call[1] is True
                    for call in policy.calls
                    if call[0] == "select_config"
                )
            )
            self.assertTrue(
                all(
                    call[1] == 1.0
                    for call in policy.calls
                    if call[0] == "step"
                )
            )

        # gamma_dim=1：last_gamma=9 不覆盖 args.gamma1/gamma2，两轮 draft_len
        # 恒为 3，proposal 也不会出现 gamma=9。
        self.assertIsNone(getattr(self.instance, "_next_gamma1", None))
        self.assertIsNone(getattr(self.instance, "_next_gamma2", None))

        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "transfer", "edge_end", 12, 0, 0),
                (1, "simulate", "edge_end", 12.0, 0, 0),
                (1, "transfer", "edge_cloud", 24, 0, 3),
                (1, "transfer", "edge_cloud", 36, 0, 3),
                (1, "simulate", "edge_cloud", 12.0, 2, 3),
                (1, "simulate", "edge_end", 12, 0, 0),
                (2, "transfer", "edge_end", 12, 0, 0),
                (2, "simulate", "edge_end", 12.0, 0, 0),
                (2, "transfer", "edge_cloud", 16, 0, 2),
                (2, "transfer", "edge_cloud", 36, 0, 3),
                (2, "simulate", "edge_cloud", 12.0, 2, 3),
                (2, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 2,
            "little_accepted_tokens": 2,
            "draft_generated_tokens": 6,
            "draft_accepted_tokens": 6,
            "little_forward_times": 2,
            "draft_forward_times": 4,
            "target_forward_times": 2,
            "queuing_time": 0.0,
            "edge_end_data_bytes": 80,
            "edge_cloud_data_bytes": 136,
            "connect_times": {"edge_end": 7, "cloud_end": 0, "edge_cloud": 6},
            "comm_trace_edge_cloud": [
                [24, 1],
                [36, 1],
                [12.0, 1],
                [16, 2],
                [36, 2],
                [12.0, 2],
            ],
            "edge_cloud_topk_history": [0, 0, 2, 0, 0, 2],
            "edge_cloud_draft_len_history": [3, 3, 3, 2, 3, 3],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        # edge_end: 7*1 ms + 80 B / 7 Mbps = 0.007091428571428571
        # edge_cloud: 6*10 ms + 136 B / 23 Mbps = 0.060047304347826096
        # 合计 0.06713873291925465（place=12 下与浮点求和一致）。
        self.assertAlmostEqual(
            metrics["communication_time"], 0.06713873291925465, places=12
        )

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        self.assertEqual(little_cache.rollback_calls, [2, 6])
        self.assertEqual(draft_cache.rollback_calls, [3, 4, 7, 8])
        self.assertEqual(target_cache.rollback_calls, [5, 9])

    def test_arp_predict_sees_old_threshold_until_next_round_and_gamma_waits(self):
        """gamma>1 时，本轮生成的每一步 predict 都只用上一轮阈值/长度。"""
        order = []
        little = _RecordingRLPolicy(
            "little_rl",
            topk=2,
            thresholds=(0.61, 0.62),
            gammas=(9, 9),
            gamma_dim=1,
            events=order,
        )
        main = _RecordingRLPolicy(
            "main_rl",
            topk=2,
            thresholds=(0.71, 0.72),
            gammas=(3, 4),
            gamma_dim=2,
            events=order,
        )
        small_arp = _RecordingARPHead("small_arp", 0.51, events=order)
        main_arp = _RecordingARPHead("main_arp", 0.52, events=order)
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = main
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp

        output, metrics, simulator = self._run(
            "adaptive_tridecoding",
            max_tokens=5,
            topk_cap=0,
            disable_rl_update=False,
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])

        # 第一轮生成 gamma1=1：一次 predict，阈值是初始 0.52；select 之后
        # 才写 0.71。第二轮生成用第一轮选出的 gamma=3，三步 predict 都仍
        # 看到 0.71；第二轮 select 的 0.72 没有中途插进生成循环。
        self.assertEqual(
            small_arp.calls,
            [
                ("reset_step",),
                ("predict", 0.51),
                ("threshold", 0.61),
                ("reset_step",),
                ("predict", 0.61),
                ("threshold", 0.62),
            ],
        )
        self.assertEqual(
            main_arp.calls,
            [
                ("reset_step",),
                ("predict", 0.52),
                ("threshold", 0.71),
                ("reset_step",),
                ("predict", 0.71),
                ("predict", 0.71),
                ("predict", 0.71),
                ("threshold", 0.72),
            ],
        )

        # 跨对象真实顺序：第二轮的三个 predict 全部位于本轮 select 之前。
        round_one = [
            ("small_arp", "reset_step"),
            ("small_arp", "predict"),
            ("little_rl", "select_config"),
            ("small_arp", "threshold"),
            ("little_rl", "compute_reward"),
            ("little_rl", "step"),
            ("main_arp", "reset_step"),
            ("main_arp", "predict"),
            ("main_rl", "select_config"),
            ("main_arp", "threshold"),
            ("main_rl", "compute_reward"),
            ("main_rl", "step"),
        ]
        round_two = [
            ("small_arp", "reset_step"),
            ("small_arp", "predict"),
            ("little_rl", "select_config"),
            ("small_arp", "threshold"),
            ("little_rl", "compute_reward"),
            ("little_rl", "step"),
            ("main_arp", "reset_step"),
            ("main_arp", "predict"),
            ("main_arp", "predict"),
            ("main_arp", "predict"),
            ("main_rl", "select_config"),
            ("main_arp", "threshold"),
            ("main_rl", "compute_reward"),
            ("main_rl", "step"),
        ]
        self.assertEqual(
            order,
            round_one
            + round_two
            + [("main_rl", "save"), ("little_rl", "save")],
        )

        # ARP 路径由 adapter 驱动生成，不走 rebuilt-topk 入口；gamma 从
        # predict 次数（1 -> 3）直接可读。
        self.assertEqual(_RecordingPolicyCache.proposal_calls, [])

        # RL 更新开启：两轮都是 select(training=True) -> reward -> step，
        # 末尾各 save 一次；ARP 路径不影响这套顺序。
        for policy in (little, main):
            self.assertEqual(
                [call[0] for call in policy.calls],
                [
                    "select_config",
                    "compute_reward",
                    "step",
                    "select_config",
                    "compute_reward",
                    "step",
                    "save",
                ],
            )
            self.assertTrue(
                all(
                    call[1] is True
                    for call in policy.calls
                    if call[0] == "select_config"
                )
            )
            self.assertTrue(
                all(
                    call[1] == 1.0
                    for call in policy.calls
                    if call[0] == "step"
                )
            )

        # 第二轮 select 写回 gamma=4，只可能作用于第三轮。
        self.assertEqual(getattr(self.instance, "_next_gamma1", None), 4)
        # little 的 gamma_dim=1：last_gamma=9 不得覆盖 args.gamma2。
        self.assertIsNone(getattr(self.instance, "_next_gamma2", None))

        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "transfer", "edge_end", 12, 0, 0),
                (1, "simulate", "edge_end", 12, 0, 0),
                (1, "transfer", "edge_cloud", 24, 0, 3),
                (1, "transfer", "edge_cloud", 36, 0, 3),
                (1, "simulate", "edge_cloud", 12.0, 2, 3),
                (1, "simulate", "edge_end", 12, 0, 0),
                (2, "transfer", "edge_end", 12, 0, 0),
                (2, "simulate", "edge_end", 12.0, 0, 0),
                (2, "transfer", "edge_cloud", 16, 0, 2),
                (2, "transfer", "edge_cloud", 60, 0, 5),
                (2, "simulate", "edge_cloud", 12.0, 2, 5),
                (2, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 2,
            "little_accepted_tokens": 2,
            "draft_generated_tokens": 8,
            "draft_accepted_tokens": 8,
            "little_forward_times": 2,
            # 每轮 stage-1 验证 +1；stage-2 生成 1 + 3。
            "draft_forward_times": 6,
            "target_forward_times": 2,
            "queuing_time": 0.0,
            "edge_end_data_bytes": 80,
            "edge_cloud_data_bytes": 160,
            "connect_times": {"edge_end": 7, "cloud_end": 0, "edge_cloud": 6},
            "comm_trace_edge_cloud": [
                [24, 1],
                [36, 1],
                [12.0, 1],
                [16, 2],
                [60, 2],
                [12.0, 2],
            ],
            "edge_cloud_topk_history": [0, 0, 2, 0, 0, 2],
            "edge_cloud_draft_len_history": [3, 3, 3, 2, 5, 5],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertAlmostEqual(metrics["communication_time"], 0.067192, places=12)

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        self.assertEqual(little_cache.rollback_calls, [2, 6])
        self.assertEqual(draft_cache.rollback_calls, [3, 4, 7, 10])
        self.assertEqual(target_cache.rollback_calls, [5, 11])

    def test_opportunistic_skip_keeps_stage_two_verification_real(self):
        small_arp = _RecordingARPHead("small", 0.5)
        main_arp = _RecordingARPHead("main", 0.5)
        self.instance.little_rl_adapter = None
        self.instance.rl_adapter = None
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp

        output, metrics, simulator = self._run(
            "cee_sd_opportunistic",
            max_tokens=1,
            transfer_top_k=2,
            opportunistic_uncertainty=0.0,
        )

        # 门控用真实小模型 logits 行（shape 1x4）和 args.small_draft_threshold；
        # 0.0 < 0.5 走 skip：第一级不跑 draft 验证，直接接受 little token。
        self.assertEqual(_DeterministicCUHLM.gate_calls, [(0.5, (1, 4), 1)])
        self.assertEqual(small_arp.calls, [("reset_step",), ("predict", 0.5)])
        self.assertEqual(main_arp.calls, [("reset_step",), ("predict", 0.5)])

        self.assertEqual(output.tolist(), [[0, 1]])
        # skip：edge_end 只付 8 B 的 little token 与 6 B accept 报文，没有
        # 第一级 proposal 载荷/回传 index；第二级仍走真实 verify。
        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "simulate", "edge_end", 8, 0, 0),
                (1, "simulate", "edge_end", 6, 0, 0),
                (1, "transfer", "edge_cloud", 16, 0, 2),
                (1, "transfer", "edge_cloud", 24, 0, 2),
                (1, "simulate", "edge_cloud", 12.0, 2, 2),
                (1, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        expected_metrics = {
            "generated_tokens": 1,
            "little_generated_tokens": 1,
            "little_accepted_tokens": 1,
            "draft_generated_tokens": 2,
            "draft_accepted_tokens": 2,
            "little_forward_times": 1,
            "draft_forward_times": 1,
            "target_forward_times": 1,
            "edge_end_data_bytes": 34,
            "edge_cloud_data_bytes": 52,
            "connect_times": {"edge_end": 4, "cloud_end": 0, "edge_cloud": 3},
            "comm_trace_edge_cloud": [[16, 1], [24, 1], [12.0, 1]],
            "edge_cloud_topk_history": [0, 0, 2],
            "edge_cloud_draft_len_history": [2, 2, 2],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertAlmostEqual(metrics["communication_time"], 0.0340688, places=12)

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        # 第一级没有 resolve_stage_verification -> little 无 rollback；
        # 第二级真实 verify 仍会回滚 draft/target。
        self.assertEqual(little_cache.rollback_calls, [])
        self.assertEqual(draft_cache.rollback_calls, [3])
        self.assertEqual(target_cache.rollback_calls, [4])

    def test_opportunistic_transfer_runs_both_real_verification_stages(self):
        small_arp = _RecordingARPHead("small", 0.5)
        main_arp = _RecordingARPHead("main", 0.5)
        self.instance.little_rl_adapter = None
        self.instance.rl_adapter = None
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp

        output, metrics, simulator = self._run(
            "cee_sd_opportunistic",
            max_tokens=1,
            transfer_top_k=2,
            opportunistic_uncertainty=0.6,
        )

        self.assertEqual(_DeterministicCUHLM.gate_calls, [(0.5, (1, 4), 1)])
        self.assertEqual(output.tolist(), [[0, 1]])

        # 0.6 >= 0.5 走 transfer：第一级付真实 proposal 载荷并真实 verify；
        # 第二级同样真实 verify（账本与 skip 路径不同）。
        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "transfer", "edge_end", 12, 0, 0),
                (1, "simulate", "edge_end", 12.0, 0, 0),
                (1, "transfer", "edge_cloud", 24, 0, 3),
                (1, "transfer", "edge_cloud", 36, 0, 3),
                (1, "simulate", "edge_cloud", 12.0, 2, 3),
                (1, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        expected_metrics = {
            "generated_tokens": 1,
            "little_generated_tokens": 1,
            "little_accepted_tokens": 1,
            "draft_generated_tokens": 3,
            "draft_accepted_tokens": 3,
            "little_forward_times": 1,
            "draft_forward_times": 2,
            "target_forward_times": 1,
            "edge_end_data_bytes": 44,
            "edge_cloud_data_bytes": 72,
            "connect_times": {"edge_end": 4, "cloud_end": 0, "edge_cloud": 3},
            "comm_trace_edge_cloud": [[24, 1], [36, 1], [12.0, 1]],
            "edge_cloud_topk_history": [0, 0, 2],
            "edge_cloud_draft_len_history": [3, 3, 3],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertAlmostEqual(metrics["communication_time"], 0.0340928, places=12)

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        self.assertEqual(little_cache.rollback_calls, [2])
        self.assertEqual(draft_cache.rollback_calls, [3, 4])
        self.assertEqual(target_cache.rollback_calls, [5])

    def test_opportunistic_gate_reuses_selected_arp_threshold_for_same_round(self):
        """第一级 RL 写回的 ARP 阈值必须在本轮机会门控里立刻复用。

        初始 CUHLM 阈值是 args.small_draft_threshold=0.5；little RL 两轮
        分别选 0.61/0.62。不确定度固定 0.55：
          · 复用本轮 0.61/0.62 -> 0.55 < 阈值 -> skip 第一级验证；
          · 若仍用初始 0.5（没有复用）-> 0.55 >= 0.5 -> 走真实 transfer。
        两条路径的账本完全不同，因此同时用 gate_calls 与逐消息账本锁定。
        """
        order = []
        little = _RecordingRLPolicy(
            "little_rl",
            topk=2,
            thresholds=(0.61, 0.62),
            gammas=(9, 9),
            gamma_dim=1,
            events=order,
        )
        small_arp = _RecordingARPHead("small_arp", 0.51, events=order)
        main_arp = _RecordingARPHead("main_arp", 0.52, events=order)
        self.instance.little_rl_adapter = little
        self.instance.rl_adapter = None
        self.instance.small_draft_adapter = small_arp
        self.instance.draft_target_adapter = main_arp

        output, metrics, simulator = self._run(
            "cee_sd_opportunistic",
            max_tokens=5,
            transfer_top_k=2,
            opportunistic_uncertainty=0.55,
        )

        self.assertEqual(output.tolist(), [[0, 1, 1, 1, 1, 1]])

        # 生成仍用旧阈值 0.51/0.61；gate 复用的是本轮 select 刚写回的
        # 0.61/0.62（既不是初始 0.5，也不是生成时的旧 0.51/0.61）。
        self.assertEqual(
            _DeterministicCUHLM.gate_calls,
            [(0.61, (1, 4), 1), (0.62, (1, 4), 1)],
        )
        self.assertEqual(
            small_arp.calls,
            [
                ("reset_step",),
                ("predict", 0.51),
                ("threshold", 0.61),
                ("reset_step",),
                ("predict", 0.61),
                ("threshold", 0.62),
            ],
        )
        # 主适配器没有 RL：阈值保持 0.52 不变。
        self.assertEqual(
            main_arp.calls,
            [
                ("reset_step",),
                ("predict", 0.52),
                ("reset_step",),
                ("predict", 0.52),
            ],
        )

        # 复用发生在同一轮 select 之后、机会门控之前；生成永远在其前面。
        self.assertEqual(
            order,
            [
                ("small_arp", "reset_step"),
                ("small_arp", "predict"),
                ("little_rl", "select_config"),
                ("small_arp", "threshold"),
                ("little_rl", "compute_reward"),
                ("main_arp", "reset_step"),
                ("main_arp", "predict"),
                ("small_arp", "reset_step"),
                ("small_arp", "predict"),
                ("little_rl", "select_config"),
                ("small_arp", "threshold"),
                ("little_rl", "compute_reward"),
                ("main_arp", "reset_step"),
                ("main_arp", "predict"),
            ],
        )

        # disable_rl_update=True 只抑制 step/save；select 和 reward 照常发生。
        self.assertEqual(
            little.calls,
            [
                ("select_config", False),
                ("compute_reward",),
                ("select_config", False),
                ("compute_reward",),
            ],
        )
        self.assertFalse(any(call[0] in ("step", "save") for call in little.calls))

        # 第一级被 opportunistic 接管，gamma 恒为 1；little 的 gamma_dim=1
        # 也不得把 last_gamma=9 写进 _next_gamma2。
        self.assertIsNone(getattr(self.instance, "_next_gamma1", None))
        self.assertIsNone(getattr(self.instance, "_next_gamma2", None))
        self.assertEqual(_RecordingPolicyCache.proposal_calls, [])

        # 两轮都 skip：edge_end 只有 8 B little token、6 B accept 报文与
        # 12 B 下行；第二级仍真实验证并计费。
        self.assertEqual(
            simulator.ledger,
            [
                (None, "transfer", "edge_end", 8, 0, 0),
                (1, "simulate", "edge_end", 8, 0, 0),
                (1, "simulate", "edge_end", 6, 0, 0),
                (1, "transfer", "edge_cloud", 16, 0, 2),
                (1, "transfer", "edge_cloud", 24, 0, 2),
                (1, "simulate", "edge_cloud", 12.0, 2, 2),
                (1, "simulate", "edge_end", 12, 0, 0),
                (2, "simulate", "edge_end", 8, 0, 0),
                (2, "simulate", "edge_end", 6, 0, 0),
                (2, "transfer", "edge_cloud", 8, 0, 1),
                (2, "transfer", "edge_cloud", 24, 0, 2),
                (2, "simulate", "edge_cloud", 12.0, 2, 2),
                (2, "simulate", "edge_end", 12, 0, 0),
            ],
        )

        expected_metrics = {
            "generated_tokens": 5,
            "little_generated_tokens": 2,
            "little_accepted_tokens": 2,
            "draft_generated_tokens": 4,
            "draft_accepted_tokens": 4,
            "little_forward_times": 2,
            # 两轮都 skip 第一级验证，只有第二级生成各 1 次。
            "draft_forward_times": 2,
            "target_forward_times": 2,
            "queuing_time": 0.0,
            "edge_end_data_bytes": 60,
            "edge_cloud_data_bytes": 96,
            "connect_times": {"edge_end": 7, "cloud_end": 0, "edge_cloud": 6},
            "comm_trace_edge_cloud": [
                [16, 1],
                [24, 1],
                [12.0, 1],
                [8, 2],
                [24, 2],
                [12.0, 2],
            ],
            "edge_cloud_topk_history": [0, 0, 2, 0, 0, 2],
            "edge_cloud_draft_len_history": [2, 2, 2, 1, 2, 2],
        }
        for name, expected in expected_metrics.items():
            self.assertEqual(metrics[name], expected, msg=name)
        self.assertAlmostEqual(metrics["communication_time"], 0.0671248, places=12)

        caches = _cache_by_role()
        little_cache = caches["little"]
        draft_cache = caches["draft"]
        target_cache = caches["target"]
        # skip 让第一级不跑验证 -> little 无 rollback；第二级每轮各一次。
        self.assertEqual(little_cache.rollback_calls, [])
        self.assertEqual(draft_cache.rollback_calls, [3, 6])
        self.assertEqual(target_cache.rollback_calls, [4, 7])
