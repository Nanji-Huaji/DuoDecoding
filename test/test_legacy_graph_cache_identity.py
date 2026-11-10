"""Retained protocol entrypoints must invalidate stale graph construction state.

CPU tests observe cache construction/reset at the resource seam. They do not
capture CUDA graphs or claim that a graph flag proves replay.
"""

from types import SimpleNamespace

import pytest
import torch

from src.baselines import Baselines


class _Runner(Baselines):
    def load_data(self):
        pass

    def preprocess(self, text):
        return text

    def postprocess(self, text, output):
        return output

    def eval(self):
        pass


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 2))

    @property
    def device(self):
        return self.weight.device

    @property
    def dtype(self):
        return self.weight.dtype


class _Cache:
    created = []

    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.configuration = (temperature, top_k, top_p, kwargs)
        self.resets = 0
        self.created.append(self)

    def reset_for_new_sample(self):
        self.resets += 1


class _Clock:
    def __init__(self, **kwargs):
        pass

    def record(self, **kwargs):
        pass

    def elapsed_time(self, other):
        return 1.0


MODES = ("uncertainty_decoding", "tk_slt", "cee_cuhlm")


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr("src.baselines.core.KVCacheModel", _Cache)
    monkeypatch.setattr(torch.cuda, "Event", _Clock)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    _Cache.created = []
    obj = object.__new__(_Runner)
    obj.args = SimpleNamespace(
        temp=0.7,
        top_k=0,
        top_p=0.9,
        max_tokens=0,
        gamma=2,
        gamma1=1,
        gamma2=2,
        use_cuda_graph=True,
        graph_verify_sizes=None,
        use_rl_adapter=False,
        disable_rl_update=True,
        disable_eos_stop=True,
        batch_delay=0,
        edge_cloud_bandwidth=100,
        edge_end_bandwidth=100,
        cloud_end_bandwidth=100,
        min_bandwidth_mbps=5,
        comm_bw_model="instant",
        comm_round_trip_mode="per_transfer",
        dump_network_stats=False,
    )
    obj.draft_model, obj.target_model, obj.little_model = _Model(), _Model(), _Model()
    obj.vocab_size = 4
    obj.tokenizer = SimpleNamespace(eos_token_id=None)
    obj.color_print = lambda *args, **kwargs: None
    obj.num_acc_tokens, obj.resample_count, obj.target_sample_count = [], [], []
    return obj


def _call(runner, mode, *, top_k=2):
    runner.args.eval_mode = mode
    output, _ = getattr(runner, mode)(
        torch.tensor([[0, 1]]),
        transfer_top_k=top_k,
        use_precise_comm_sim=False,
    )
    assert output.tolist() == [[0, 1]]


@pytest.mark.parametrize("mode", MODES)
def test_changed_temperature_rebuilds_retained_protocol_caches(runner, mode):
    _call(runner, mode)
    initial = list(_Cache.created)
    assert len(initial) == (3 if mode == "cee_cuhlm" else 2)
    runner.args.temp = 0.4
    _call(runner, mode)
    assert len(_Cache.created) == 2 * len(initial)
    assert all(cache.resets == 0 for cache in initial)
    assert all(
        cache.configuration[0] == 0.4 for cache in _Cache.created[len(initial) :]
    )


@pytest.mark.parametrize("mode", MODES)
def test_equivalent_calls_reset_and_preserve_effective_graph_policy(runner, mode):
    _call(runner, mode)
    initial = list(_Cache.created)
    _call(runner, mode)
    assert _Cache.created == initial
    assert all(cache.resets == 1 for cache in initial)
    graph_flags = [
        bool(cache.configuration[3].get("use_cuda_graph")) for cache in initial
    ]
    assert graph_flags == ([True, True, True] if mode == "cee_cuhlm" else [True, False])


@pytest.mark.parametrize("mode", MODES)
def test_eager_calls_are_fresh_without_model_identity_introspection(
    runner, mode, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager must not compute graph construction identity")

    monkeypatch.setattr("src.baselines.core._kv_cache_construction_key", forbidden)
    runner.args.use_cuda_graph = False
    _call(runner, mode)
    count = len(_Cache.created)
    _call(runner, mode)
    assert len(_Cache.created) == 2 * count
    assert all(
        cache.resets == 0 and not cache.configuration[3] for cache in _Cache.created
    )


ROLES = [
    (mode, role)
    for mode in MODES
    for role in (
        ("little_model", "draft_model", "target_model")
        if mode == "cee_cuhlm"
        else ("draft_model", "target_model")
    )
]


@pytest.mark.parametrize("mode,role", ROLES)
@pytest.mark.parametrize("change", ("model", "storage", "dtype"))
def test_each_role_model_storage_and_dtype_changes_invalidate(
    runner, mode, role, change
):
    _call(runner, mode)
    initial = list(_Cache.created)
    model = getattr(runner, role)
    if change == "model":
        setattr(runner, role, _Model())
    elif change == "storage":
        model.weight.data = model.weight.data.clone()
    else:
        model.double()
    _call(runner, mode)
    assert len(_Cache.created) == 2 * len(initial)
    assert all(cache.resets == 0 for cache in initial)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("change", ("vocab", "ladder"))
def test_vocabulary_and_actual_graph_configuration_changes_invalidate(
    runner, mode, change
):
    _call(runner, mode)
    count = len(_Cache.created)
    if change == "vocab":
        runner.vocab_size = 3
    else:
        runner.args.graph_verify_sizes = "3,6"
    _call(runner, mode)
    assert len(_Cache.created) == 2 * count


@pytest.mark.parametrize("mode", MODES)
def test_only_effective_sampling_settings_invalidate(runner, mode):
    _call(runner, mode, top_k=2)
    initial = list(_Cache.created)
    runner.args.top_p = 0.5
    _call(runner, mode, top_k=3)
    if mode == "cee_cuhlm":  # All three cache constructors intentionally use k=p=0.
        assert _Cache.created == initial
        assert all(cache.resets == 1 for cache in initial)
    else:
        assert len(_Cache.created) == 2 * len(initial)
        assert _Cache.created[len(initial)].configuration[1:3] == (3, 0.5)
        assert _Cache.created[-1].configuration[1:3] == (0, 0)


@pytest.mark.parametrize("mode", MODES)
def test_in_place_values_do_not_invalidate_storage_identity(runner, mode):
    _call(runner, mode)
    initial = list(_Cache.created)
    runner.draft_model.weight.data.add_(1)
    _call(runner, mode)
    assert _Cache.created == initial
    assert all(cache.resets == 1 for cache in initial)
