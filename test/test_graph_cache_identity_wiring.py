"""CPU-only effective identity tests through all four production cache factories.

The fake cache records construction/reset, not graph execution. Models have real
CPU parameters and buffers so a same-object dtype/storage change is observable.
No weights are downloaded; the metadata key never inspects their contents.
"""

from types import SimpleNamespace

import pytest
import torch

from src.baselines import runtime
from src.engine import Decoding
from src.graph_decode import kv_cache_construction_key


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 4)
        self.output = torch.nn.Linear(4, 8)
        self.register_buffer("scale", torch.ones(1))
        self.name_or_path = "same/path/is/not/identity"

    @property
    def device(self):
        return self.embedding.weight.device

    @property
    def dtype(self):
        return self.embedding.weight.dtype

    def get_input_embeddings(self):
        return self.embedding


class _Cache:
    def __init__(self, model, temperature, top_k, top_p, **kwargs):
        self.model = model
        self.sampling = (temperature, top_k, top_p)
        self.kwargs = kwargs
        self.resets = 0

    def reset_for_new_sample(self):
        self.resets += 1


@pytest.fixture(params=["three", "pair", "ar", "sd"])
def factory(request, monkeypatch):
    monkeypatch.setattr(runtime, "KVCacheModel", _Cache)
    monkeypatch.setattr("src.engine.KVCacheModel", _Cache)
    holder = SimpleNamespace(
        args=SimpleNamespace(temp=0.7, top_k=3, top_p=0.8),
        little_model=_Model(),
        draft_model=_Model(),
        target_model=_Model(),
        vocab_size=8,
    )
    graph_kw = {
        "use_cuda_graph": True,
        "verify_graph_sizes": [4, 8],
        "graph_len_budget": 300,
    }
    extra_kw = {}
    settings = SimpleNamespace(draft_top_k=3, target_top_k=0, target_top_p=0.0)

    def acquire():
        if request.param == "three":
            return runtime.RuntimeResources._acquire_three_layer_caches(
                holder, "_caches", graph_kw, settings.draft_top_k,
                cache_kwargs=extra_kw,
            )
        if request.param == "pair":
            draft, target = runtime.RuntimeResources._acquire_draft_target_caches(
                holder, "_caches", graph_kw, settings.draft_top_k,
                settings.target_top_k, settings.target_top_p,
            )
            return {"draft": draft, "target": target}
        if request.param == "ar":
            cache = Decoding._acquire_autoregressive_cache(
                holder, "_caches", graph_kw, holder.draft_model,
            )
            return {"draft": cache}
        return Decoding._acquire_speculative_caches(
            holder, "_caches", graph_kw, settings.draft_top_k,
        )

    return SimpleNamespace(
        name=request.param, holder=holder, graph_kw=graph_kw, extra_kw=extra_kw,
        settings=settings, acquire=acquire,
    )


def _assert_rebuilt(before, after):
    assert before.keys() == after.keys()
    assert all(before[role] is not after[role] for role in before)
    assert all(cache.resets == 0 for cache in after.values())


def test_equivalent_constructor_reuses_and_resets_each_cache(factory):
    first = factory.acquire()
    factory.graph_kw["verify_graph_sizes"] = (4, 8)
    second = factory.acquire()
    assert all(first[role] is second[role] for role in first)
    assert all(cache.resets == 1 for cache in second.values())
    if factory.name in ("pair", "sd"):
        assert second["target"].kwargs == {}  # key must describe actual eager target


def test_eager_always_fresh_and_never_introspects_model(factory, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("eager must not calculate graph identities")

    monkeypatch.setattr(runtime, "kv_cache_construction_key", forbidden)
    monkeypatch.setattr("src.engine.kv_cache_construction_key", forbidden)
    factory.graph_kw.clear()
    first = factory.acquire()
    _assert_rebuilt(first, factory.acquire())
    assert not hasattr(factory.holder, "_caches")


@pytest.mark.parametrize("field", ["temperature", "top_k", "top_p", "vocab_size"])
def test_effective_sampling_and_vocabulary_changes_rebuild(factory, field):
    first = factory.acquire()
    if field == "temperature":
        factory.holder.args.temp = 0.5
    elif field == "top_k":
        factory.holder.args.top_k = 5
        factory.settings.draft_top_k = 5
    elif field == "top_p":
        factory.holder.args.top_p = 0.6
    else:
        factory.holder.vocab_size = 7
    _assert_rebuilt(first, factory.acquire())


@pytest.mark.parametrize("field", ["target_top_k", "target_top_p"])
def test_pair_target_sampling_is_part_of_group_identity(monkeypatch, field):
    monkeypatch.setattr(runtime, "KVCacheModel", _Cache)
    holder = SimpleNamespace(
        args=SimpleNamespace(temp=1.0, top_p=0.9),
        draft_model=_Model(), target_model=_Model(), vocab_size=8,
    )
    settings = {"target_top_k": 0, "target_top_p": 0.0}

    def acquire():
        return runtime.RuntimeResources._acquire_draft_target_caches(
            holder, "_pair", {"use_cuda_graph": True}, 3, **settings,
        )

    first = acquire()
    settings[field] = 2 if field == "target_top_k" else 0.7
    second = acquire()
    assert all(a is not b for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("field,value", [
    ("max_length", 128), ("graph_len_budget", 400),
    ("verify_graph_sizes", [4, 8, 16]), ("return_hidden_states", True),
])
def test_actual_capacity_and_graph_constructor_kwargs_rebuild(factory, field, value):
    first = factory.acquire()
    factory.graph_kw[field] = value
    _assert_rebuilt(first, factory.acquire())


def test_three_layer_extra_capacity_and_graph_precedence(monkeypatch):
    monkeypatch.setattr(runtime, "KVCacheModel", _Cache)
    holder = SimpleNamespace(
        args=SimpleNamespace(temp=1.0, top_p=0.9), little_model=_Model(),
        draft_model=_Model(), target_model=_Model(), vocab_size=8,
    )
    extra = {"max_length": 64, "graph_len_budget": 999}

    def acquire():
        return runtime.RuntimeResources._acquire_three_layer_caches(
            holder, "_three", {"use_cuda_graph": True, "graph_len_budget": 300},
            3, cache_kwargs=extra,
        )

    first = acquire()
    extra["graph_len_budget"] = 888  # shadowed, not an effective setting
    assert acquire() is first
    extra["max_length"] = 128
    _assert_rebuilt(first, acquire())


@pytest.mark.parametrize(
    "mutation", ["model", "storage", "parameter", "buffer", "dtype"]
)
def test_every_model_role_and_weight_storage_participates(factory, mutation):
    first = factory.acquire()
    for role in first:
        model = getattr(factory.holder, f"{role}_model")
        if mutation == "model":
            setattr(factory.holder, f"{role}_model", _Model())
        elif mutation == "storage":
            # Last parameter, not the embedding/first parameter; same object.
            parameter = model.output.bias
            old_id = id(parameter)
            parameter.data = parameter.detach().clone()
            assert id(parameter) == old_id
        elif mutation == "parameter":
            model.output.bias = torch.nn.Parameter(model.output.bias.detach().clone())
        elif mutation == "buffer":
            model.scale = model.scale.clone()
        else:
            model.to(dtype=torch.float64)
        second = factory.acquire()
        _assert_rebuilt(first, second)
        first = second


def test_in_place_values_keep_addresses_and_reuse(factory):
    first = factory.acquire()
    with torch.no_grad():
        factory.holder.draft_model.output.weight.add_(1)
    second = factory.acquire()
    assert all(first[role] is second[role] for role in first)


@pytest.mark.parametrize("field,value", [("device", "cpu:1"), ("dtype", torch.float64)])
def test_parameterless_fake_still_has_explicit_identity(factory, field, value):
    fake = SimpleNamespace(device="cpu", dtype=torch.float32)
    factory.holder.draft_model = fake
    first = factory.acquire()
    assert factory.acquire()["draft"] is first["draft"]
    setattr(fake, field, value)  # metadata only; no CUDA or device allocation
    _assert_rebuilt(first, factory.acquire())


def test_parameterless_embedding_storage_is_not_ignored():
    embedding = SimpleNamespace(weight=torch.zeros(8, 4))
    model = SimpleNamespace(get_input_embeddings=lambda: embedding)

    def key():
        return kv_cache_construction_key(
            model, temperature=1.0, top_k=0, top_p=0.0, vocab_size=8,
        )

    first = key()
    assert first["weights"] == {"parameters": None, "buffers": None}
    embedding.weight = embedding.weight.clone()
    assert key() != first


def test_meta_weights_are_explicitly_rejected_without_materializing():
    model = torch.nn.Linear(2, 2, device="meta")
    with pytest.raises(ValueError, match="materialized strided weights"):
        kv_cache_construction_key(
            model, temperature=1.0, top_k=0, top_p=0.0, vocab_size=8,
        )
    assert model.weight.device.type == "meta"


def test_non_strided_weights_are_explicitly_rejected_without_densifying():
    model = torch.nn.Linear(2, 2)
    model.weight = torch.nn.Parameter(torch.eye(2).to_sparse())
    with pytest.raises(ValueError, match="materialized strided weights"):
        kv_cache_construction_key(
            model, temperature=1.0, top_k=0, top_p=0.0, vocab_size=8,
        )
    assert model.weight.layout == torch.sparse_coo
