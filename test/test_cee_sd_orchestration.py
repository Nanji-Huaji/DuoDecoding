"""Public CEE-SD orchestration gates, with an auxiliary portable old oracle.

Only the decorated b72a854 method is compiled, never the old module or Git history.
Normal-path replay is auxiliary: independent contracts own intentionally corrected
sampling/cache cases. Hardware boundaries reuse pure scripted fixtures, not another
TestCase's lifecycle. Real verification, quantization and communication billing run.
The clock is constant; no real timing/RNG-trajectory claim is made. Rollback events
record actual before/after cache lengths, not an ineffective overlarge request.
"""

import ast
import copy
import hashlib
import textwrap
from argparse import Namespace
from contextlib import ExitStack
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import test_adaptive_tridecoding_policies as policies
import test_adaptive_tridecoding_rounds as rounds
import torch

from src.baselines import Baselines, cee, core, runtime, simulators

ORACLE_PATH = Path(__file__).parent / "golden/cee_sd/baseline_b72a854.py.txt"
ORACLE_SHA256 = "4dad0c0dc8583cb58ce19bb3096b5120cb143746c165e55fa0fd82a2122c62fe"
NEW_HELPERS = (
    "_cee_propose",
    "_cee_first_stage",
    "_cee_target_proposal",
    "_cee_verify_proposal",
    "_cee_finalize",
)
MODES = ("adaptive_tridecoding", "tridecoding", "cee_sd_no_arp", "cee_sd_opportunistic")


@lru_cache(maxsize=1)
def _oracle_code():
    frozen = ORACLE_PATH.read_bytes()
    assert hashlib.sha256(frozen).hexdigest() == ORACLE_SHA256
    module = ast.parse(textwrap.dedent(frozen.decode("utf-8")))
    assert len(module.body) == 1 and isinstance(module.body[0], ast.FunctionDef)
    method = copy.deepcopy(module.body[0])
    assert method.name == "adaptive_tridecoding" and len(method.decorator_list) == 3
    assert [ast.unparse(d) for d in method.decorator_list] == [
        "Register.register_decoding('adaptive_tridecoding')",
        "Register.register_decoding('cee_sd')",
        "torch.no_grad()",
    ]
    method.decorator_list = method.decorator_list[-1:]
    return compile(
        ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])),
        "<b72a854:adaptive_tridecoding>",
        "exec",
    )


def _oracle():
    namespace = {}
    exec(_oracle_code(), dict(core.__dict__), namespace)
    return namespace["adaptive_tridecoding"]


def _plain(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return tuple(_plain(item) for item in value)
    return value


@dataclass(frozen=True)
class _Case:
    mode: str = "tridecoding"
    budget: int = 5
    gamma1: int = 1
    gamma2: int = 1
    topk: int | None = 2
    cap: int = 0
    bits: int = 16
    residual: bool = False
    round_mode: str = "per_transfer"
    trace_mode: str = "static"
    live_policy: bool = False
    disable_updates: bool = True
    uncertainty: float = 0.0
    graph: bool = False
    samples: int = 1
    arp_limit: int | None = None
    eos: int = -1
    early_stop: bool = False


class _Decoder(rounds._TestBaselines):
    def _stop_at_eos(self, prefix, prompt_len):
        result, hit = super()._stop_at_eos(prefix, prompt_len)
        if hit:
            self.events.append(("eos", True, _plain(result)))
        return result, hit

    def _check_stopping_criteria(self, prefix, stop_sequences=None, prompt_len=None):
        result = super()._check_stopping_criteria(prefix, stop_sequences, prompt_len)
        self.events.append(("stop", result, _plain(prefix)))
        return result


def _run(case, *, oracle=False):
    """Fresh, function-scoped fixture; no real CUDA/model/NVML or test lifecycle."""
    events = []

    class Cache(policies._RecordingPolicyCache):
        instances = []
        scripts = {}
        default_token = 1
        metadata_calls = 0
        proposal_calls = []
        entropy_by_role = {}

        def __init__(self, model, temperature, top_k, top_p, **kwargs):
            super().__init__(model, temperature, top_k, top_p, **kwargs)
            events.append(("cache.init", self.role, top_k, top_p, dict(kwargs)))

        def _consume(self, x):
            assert not torch.is_grad_enabled()
            result = super()._consume(x)
            events.append(("consume", self.role, _plain(x), self.current_length))
            return result

        def forward_context(self, prefix):
            events.append(("forward_context", self.role, int(prefix.shape[1])))
            return super().forward_context(prefix)

        def generate(self, prefix, gamma):
            events.append(("generate", self.role, int(gamma)))
            return super().generate(prefix, gamma)

        def generate_with_rebuilt_topk_metadata(self, prefix, gamma, proposal_top_k):
            # One semantic proposal event, regardless of the private cache API.
            type(self).proposal_calls.append((self.role, int(gamma), proposal_top_k))
            return super().generate_with_rebuilt_topk_metadata(
                prefix, gamma, proposal_top_k
            )

        def rollback(self, end_pos):
            before = self.current_length
            super().rollback(end_pos)
            events.append(("rollback", self.role, before, self.current_length))

        def reset_for_new_sample(self):
            events.append(("cache.reset", self.role, self.current_length))
            self._prob_rows = []
            self.hidden_states = None
            self.last_entropy = None

    class Simulator(rounds._RecordingCommunicationSimulator):
        instances = []

        def __init__(self, *args, **kwargs):
            self.constructor_mode = kwargs.get("mode")
            events.append(("sim.init.mode", self.constructor_mode))
            super().__init__(*args, **kwargs)

        def set_round(self, round_idx):
            events.append(("round", int(round_idx)))
            super().set_round(round_idx)

        def flush_round(self):
            events.append(("flush", self._round_idx))
            super().flush_round()

        def simulate_transfer(self, size, link, *args, **kwargs):
            events.append(("comm", self._round_idx, link, float(size)))
            return super().simulate_transfer(size, link, *args, **kwargs)

    class Head(policies._RecordingARPHead):
        def reset_step(self):
            self.count = 0
            super().reset_step()

        def predict(self, hidden_states):
            super().predict(hidden_states)
            self.count += 1
            return case.arp_limit is not None and self.count >= case.arp_limit

    class Policy(policies._RecordingRLPolicy):
        def compute_reward(self, **kwargs):
            events.append((self.name, "reward.inputs", _plain(kwargs)))
            return super().compute_reward(**kwargs)

    class Gate(policies._DeterministicCUHLM):
        uncertainty_value = case.uncertainty
        gate_calls = []

    args = Namespace(
        eval_mode=case.mode,
        edge_cloud_bandwidth=10.0,
        edge_end_bandwidth=10.0,
        cloud_end_bandwidth=10.0,
        min_bandwidth_mbps=5.0,
        comm_bw_model="instant",
        stochastic_ntt=False,
        ntt_trace_file="",
        ntt_trace_scale=1.0,
        comm_trace_mode=case.trace_mode,
        comm_round_trip_mode=case.round_mode,
        charge_residual_payload=case.residual,
        transfer_top_k_cap=case.cap,
        force_full_vocab_transfer=False,
        prob_payload_bits=case.bits,
        max_tokens=case.budget,
        temp=1.0,
        top_k=0,
        top_p=0.0,
        batch_delay=0.125,
        seed=0,
        exp_name="unit",
        eval_dataset="unit",
        little_model="little",
        draft_model="draft",
        target_model="target",
        gamma=1,
        gamma1=case.gamma1,
        gamma2=case.gamma2,
        use_cuda_graph=case.graph,
        graph_verify_sizes="4,8",
        probe_cache_max_length=32,
        disable_rl_update=case.disable_updates,
        disable_eos_stop=case.eos < 0,
        dump_network_stats=False,
        uncertainty_threshold=0.8,
        small_draft_threshold=0.5,
        rl_comm_cost_mode="nominal",
    )
    decoder = object.__new__(_Decoder)
    decoder.args = args
    decoder.events = events
    decoder.accelerator = SimpleNamespace(is_main_process=True)
    decoder.vocab_size = 4
    decoder.num_acc_tokens = []
    decoder.draft_forward_times = decoder.target_forward_times = 0
    for role in ("little", "draft", "target"):
        setattr(
            decoder,
            f"{role}_model",
            SimpleNamespace(
                device=torch.device("cpu"),
                kind=role,
            ),
        )
    decoder.small_draft_adapter = Head("small", 0.51, events=events)
    decoder.draft_target_adapter = Head("main", 0.52, events=events)
    decoder.little_rl_adapter = decoder.rl_adapter = None
    if case.live_policy:
        decoder.little_rl_adapter = Policy(
            "little.rl",
            topk=1,
            thresholds=(0.11, 0.12),
            gammas=(2, 1),
            gamma_dim=2,
            events=events,
        )
        decoder.rl_adapter = Policy(
            "main.rl",
            topks=(3, 1),
            thresholds=(0.21, 0.22),
            gammas=(2, 1),
            gamma_dim=2,
            events=events,
        )
    decoder.tokenizer = SimpleNamespace(
        eos_token_id=case.eos,
        decode=lambda *a, **kw: "STOP",
    )
    outputs = []
    with torch.random.fork_rng(devices=[]), ExitStack() as stack:
        torch.manual_seed(1234)
        for key in (*rounds.DIAGNOSTIC_ENV_KEYS, "COMM_TRACE"):
            stack.enter_context(patch.dict("os.environ", {key: ""}))
        stack.enter_context(patch.object(core, "KVCacheModel", Cache))
        stack.enter_context(patch.object(runtime, "KVCacheModel", Cache))
        # Shared factory relocation must retain real accounting under both seams.
        stack.enter_context(patch.object(core, "CommunicationSimulator", Simulator))
        stack.enter_context(
            patch.object(simulators, "CommunicationSimulator", Simulator)
        )
        stack.enter_context(patch.object(core, "CUHLM", Gate))
        stack.enter_context(patch.object(cee, "CUHLM", Gate))
        stack.enter_context(patch.object(torch.cuda, "Event", rounds._FakeCudaEvent))
        stack.enter_context(
            patch.object(torch.cuda, "current_stream", return_value=None)
        )
        stack.enter_context(patch.object(torch.cuda, "synchronize", return_value=None))
        stack.enter_context(patch.object(core.time, "time", return_value=1.0))
        if oracle:
            stack.enter_context(
                patch.object(_Decoder, "adaptive_tridecoding", _oracle())
            )
        for _ in range(case.samples):
            tokens, metrics = getattr(decoder, case.mode)(
                torch.tensor([[0]], dtype=torch.int32),
                transfer_top_k=case.topk,
                use_early_stopping=case.early_stop,
                stop_sequences=["STOP"],
            )
            outputs.append((tokens.tolist(), copy.deepcopy(metrics)))
    return SimpleNamespace(
        outputs=outputs,
        events=events,
        proposals=Cache.proposal_calls,
        caches=Cache.instances,
        simulators=Simulator.instances,
        decoder=decoder,
        gates=Gate.gate_calls,
    )


def _snapshot(result):
    # Replay cases use top-k proposals; every legacy generate marker is exactly
    # one discarded draft/target verification sample, never a real proposal.
    assert all(
        event[1] in ("draft", "target") and event[2] == 1
        for event in result.events if event[0] == "generate"
    )
    return {
        "outputs": result.outputs,
        # Only API entry markers differ: both still record actual consume events
        # at the same stage boundary. Delta-law replay is orchestration coverage,
        # not a claim that the obsolete discarded samples consume the same RNG.
        "events": [
            event for event in result.events
            if event[0] not in ("generate", "forward_context")
        ],
        "proposals": result.proposals,
        "caches": [
            (c.role, c.current_length, _plain(c.prob_history))
            for c in result.caches
            if c.current_length
        ],
        "ledgers": [
            (s.ledger, s.charge_ledger, s._pending_bytes) for s in result.simulators
        ],
        "heads": (
            result.decoder.small_draft_adapter.calls,
            result.decoder.draft_target_adapter.calls,
        ),
        "policies": [
            (p.calls, p.selected_states)
            for p in (
                result.decoder.little_rl_adapter,
                result.decoder.rl_adapter,
            )
            if p is not None
        ],
    }


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("live", [False, True])
def test_normal_paths_auxiliary_frozen_replay(mode, live):
    case = _Case(mode=mode, live_policy=live)
    assert _snapshot(_run(case)) == _snapshot(_run(case, oracle=True))


@pytest.mark.parametrize(
    "case",
    [
        _Case(budget=0),
        _Case(budget=1),
        _Case(bits=8, residual=True),
        _Case(mode="cee_sd_opportunistic", uncertainty=1.0),
        _Case(mode="adaptive_tridecoding", gamma1=4, gamma2=3, arp_limit=1),
        _Case(graph=True, samples=2),
        _Case(samples=2),
        _Case(mode="cee_sd_no_arp", live_policy=True, disable_updates=False),
    ],
)
def test_normal_wiring_auxiliary_frozen_replay(case):
    assert _snapshot(_run(case)) == _snapshot(_run(case, oracle=True))


def test_live_policy_uses_old_proposal_topk_then_current_billing_and_next_gamma():
    result = _run(_Case(mode="cee_sd_no_arp", live_policy=True, cap=3))
    assert result.outputs[0][0] == [[0, 1, 1, 1, 1, 1]]
    assert result.proposals == [
        ("little", 1, 2),
        ("draft", 1, 2),
        ("little", 2, 3),
        ("draft", 2, 3),
    ]
    bills = [
        e for e in result.simulators[0].ledger if e[1:3] == ("simulate", "edge_cloud")
    ]
    assert [(e[4], e[5]) for e in bills] == [(3, 3), (1, 5)]
    assert result.decoder._next_gamma1 == result.decoder._next_gamma2 == 1
    assert result.decoder.small_draft_adapter.calls == []
    assert result.decoder.draft_target_adapter.calls == []


@pytest.mark.parametrize(
    "bits,payloads,totals",
    [
        (8, (9, 27), (41, 63)),
        (16, (12, 36), (44, 72)),
    ],
)
@pytest.mark.parametrize("residual", [False, True])
def test_quantized_wire_bits_price_actual_scalar_payloads(
    bits, payloads, totals, residual
):
    result = _run(_Case(budget=1, bits=bits, residual=residual))
    simulator = result.simulators[0]
    # One-hot rows make wire/sample q identical; test only the transport contract.
    uplinks = [
        event[3]
        for event in simulator.ledger
        if event[0] == 1 and event[1] == "transfer"
    ]
    assert uplinks == [payloads[0], 24, payloads[1]]
    _, metrics = result.outputs[0]
    assert (metrics["edge_end_data_bytes"], metrics["edge_cloud_data_bytes"]) == totals
    assert result.outputs[0][0] == [[0, 1]]


@pytest.mark.parametrize(
    "mode", ["tridecoding", "cee_sd_no_arp", "adaptive_tridecoding"]
)
def test_static_and_frozen_policy_gates_do_not_update_or_save(mode):
    result = _run(_Case(mode=mode, budget=1, live_policy=True))
    for policy in (result.decoder.little_rl_adapter, result.decoder.rl_adapter):
        assert policy.calls == (
            []
            if mode == "tridecoding"
            else [
                ("select_config", False),
                ("compute_reward",),
            ]
        )
    if mode in ("tridecoding", "cee_sd_no_arp"):
        assert result.decoder.small_draft_adapter.calls == []
        assert result.decoder.draft_target_adapter.calls == []
    else:
        assert result.decoder.small_draft_adapter.calls == [
            ("reset_step",),
            ("predict", 0.51),
            ("threshold", 0.11),
        ]
        assert result.decoder.draft_target_adapter.calls == [
            ("reset_step",),
            ("predict", 0.52),
            ("threshold", 0.21),
        ]


def test_trace_mode_reaches_simulator_constructor_not_posthoc_attribute():
    result = _run(_Case(trace_mode="driving", budget=1))
    assert result.simulators[0].constructor_mode == "driving"
    assert ("sim.init.mode", "driving") in result.events


@pytest.mark.parametrize("graph", [False, True])
def test_cache_construction_and_cross_sample_graph_reuse(graph):
    result = _run(_Case(graph=graph, samples=2))
    assert len(result.caches) == (3 if graph else 6)
    resets = [e[1] for e in result.events if e[0] == "cache.reset"]
    assert resets == (["little", "draft", "target"] if graph else [])
    constructors = [e for e in result.events if e[0] == "cache.init"]
    assert [e[2] for e in constructors[:3]] == [2, 2, 0]
    for event in constructors:
        expected = {"max_length": 32}
        if graph:
            expected.update(
                use_cuda_graph=True, verify_graph_sizes=[4, 8], graph_len_budget=261
            )
        assert event[4] == expected
    assert all(tokens == [[0, 1, 1, 1, 1, 1]] for tokens, _ in result.outputs)


@pytest.mark.parametrize("uncertainty,verify_little", [(0.0, False), (1.0, True)])
def test_opportunistic_gate_controls_draft_verify_not_final_target(
    uncertainty,
    verify_little,
):
    result = _run(_Case(mode="cee_sd_opportunistic", budget=1, uncertainty=uncertainty))
    draft_contexts = [
        e for e in result.events if e[:2] == ("forward_context", "draft")
    ]
    # gamma1=1 always needs one ARP proposal forward; only the optional inner
    # verification contributes a second public context call.
    assert len(draft_contexts) == 1 + int(verify_little)
    assert [e for e in result.events if e[:2] == ("forward_context", "target")]
    assert len(result.gates) == 1
    assert result.outputs[0][0] == [[0, 1]]


@pytest.mark.parametrize("mode", ["per_transfer", "per_round"])
def test_reward_precedes_downlink_and_final_flush_prices_complete_wall(mode):
    result = _run(
        _Case(
            mode="cee_sd_no_arp",
            budget=1,
            live_policy=True,
            disable_updates=False,
            round_mode=mode,
        )
    )
    events = result.events
    main_reward = next(
        i for i, e in enumerate(events) if e[:2] == ("main.rl", "reward.inputs")
    )
    reward = events[main_reward][2]
    downlink = next(
        i
        for i, e in enumerate(events[main_reward + 1 :], main_reward + 1)
        if e[:3] == ("comm", 1, "edge_end")
    )
    assert main_reward < downlink
    assert reward["forward_counts"] == {"little": 1.0, "draft": 1.0, "target": 1.0}
    assert reward["transferred_bytes"] == 60.0  # 3 token prefix + 3 scalar-q payload
    assert reward["queue_s"] == 0.125 and reward["compute_s"] == 0.0
    assert ("main.rl", "step") in events and ("little.rl", "step") in events
    _, metrics = result.outputs[0]
    assert metrics["wall_time"] == pytest.approx(
        metrics["communication_time"] + metrics["queuing_time"],
        abs=1e-12,
    )
    assert metrics["throughput"] == pytest.approx(1 / metrics["wall_time"])
    assert result.simulators[0]._pending_bytes == {}
    assert metrics["edge_cloud_data_bytes"] == sum(
        row[0] for row in metrics["comm_trace_edge_cloud"]
    )


def test_zero_little_stage_has_no_uplink_but_keeps_draft_bonus_and_target():
    result = _run(_Case(gamma2=0, budget=1))
    _, metrics = result.outputs[0]
    assert result.outputs[0][0] == [[0, 1]]
    assert metrics["little_generated_tokens"] == metrics["little_accepted_tokens"] == 0
    assert [
        metrics[key]
        for key in (
            "little_forward_times",
            "draft_forward_times",
            "target_forward_times",
        )
    ] == [0, 2, 1]
    assert not any(
        event[:3] == (1, "transfer", "edge_end")
        for event in result.simulators[0].ledger
    )
    assert (metrics["edge_end_data_bytes"], metrics["edge_cloud_data_bytes"]) == (
        32,
        52,
    )


def test_arp_short_proposals_preserve_actual_forward_counts():
    result = _run(
        _Case(mode="adaptive_tridecoding", budget=4, gamma1=4, gamma2=3, arp_limit=1)
    )
    _, metrics = result.outputs[0]
    assert result.outputs[0][0] == [[0, 1, 1, 1, 1]]
    assert [
        metrics[key]
        for key in (
            "little_forward_times",
            "draft_forward_times",
            "target_forward_times",
        )
    ] == [1, 2, 1]
    assert result.decoder.small_draft_adapter.calls == [
        ("reset_step",),
        ("predict", 0.51),
    ]
    assert result.decoder.draft_target_adapter.calls == [
        ("reset_step",),
        ("predict", 0.52),
    ]


@pytest.mark.parametrize("eos,early", [(1, False), (-1, True)])
def test_eos_and_stop_sequence_exit_before_next_round_but_still_finalize(eos, early):
    result = _run(_Case(budget=20, eos=eos, early_stop=early, round_mode="per_round"))
    assert [e for e in result.events if e[0] == "round"] == [("round", 1)]
    assert result.simulators[0]._pending_bytes == {}
    if eos == 1:
        assert result.outputs[0][0] == [[0, 1]]
        assert any(e[:2] == ("forward_context", "target") for e in result.events)
        eos_event = next(
            i for i, e in enumerate(result.events) if e[:2] == ("eos", True)
        )
        replies = [e for e in result.events[:eos_event] if e[0] == "comm"]
        assert replies[-2:] == [
            ("comm", 1, "edge_cloud", 12.0),
            ("comm", 1, "edge_end", 12.0),
        ]
    else:
        assert result.outputs[0][0] == [[0, 1, 1, 1, 1]]
        stop = next(i for i, e in enumerate(result.events) if e[:2] == ("stop", True))
        assert any(e[:3] == ("comm", 1, "edge_end") for e in result.events[:stop])
        assert any(e[0] == "flush" for e in result.events[stop + 1 :])


@pytest.mark.parametrize("mode", MODES)
def test_frozen_oracle_cannot_call_new_cee_helpers(mode):
    available = [name for name in NEW_HELPERS if hasattr(Baselines, name)]
    assert tuple(available) == NEW_HELPERS, "CEE extraction must install all five seams"

    def poisoned(*args, **kwargs):
        raise AssertionError("new CEE helper called")

    case = _Case(mode=mode)
    with ExitStack() as stack:
        for name in available:
            stack.enter_context(patch.object(Baselines, name, poisoned))
        assert _run(case, oracle=True).outputs[0][0] == [[0, 1, 1, 1, 1, 1]]
        with pytest.raises(AssertionError, match="new CEE helper called"):
            _run(case)
