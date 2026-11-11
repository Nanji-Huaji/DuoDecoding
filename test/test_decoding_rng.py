"""Contract tests for the private per-decode RNG seam.

All behaviour is exercised through DecodingRng.sample/rand. Torch generator
and rand factories are mocked only where a CUDA device cannot exist on this
CPU-only host; no CUDA call is made.
"""

import pytest
import torch

from src.decoding_rng import DecodingRng
from src.sampling import sampling_weights

CPU = torch.device("cpu")


def _record_multinomial(monkeypatch):
    real_multinomial = torch.multinomial
    calls = []

    def spy(weights, num_samples=1, generator=None):
        calls.append(
            {
                "weights": weights,
                "num_samples": num_samples,
                "generator": generator,
            }
        )
        return real_multinomial(weights, num_samples=num_samples, generator=generator)

    monkeypatch.setattr(torch, "multinomial", spy)
    return calls


def _record_rand(monkeypatch):
    real_rand = torch.rand
    calls = []

    def spy(shape, *, device=None, generator=None):
        calls.append(
            {"shape": shape, "device": torch.device(device), "generator": generator}
        )
        return real_rand(shape, device=device, generator=generator)

    monkeypatch.setattr(torch, "rand", spy)
    return calls


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32, torch.float64])
def test_sample_passes_exact_sampling_weights(monkeypatch, dtype):
    probs = torch.tensor([[0.05, 0.15, 0.30, 0.50], [1.0, 0.0, 0.0, 0.0]], dtype=dtype)
    calls = _record_multinomial(monkeypatch)

    token = DecodingRng().sample(probs)

    assert len(calls) == 1
    assert calls[0]["weights"].dtype == torch.float32
    assert torch.equal(calls[0]["weights"], sampling_weights(probs))
    assert calls[0]["num_samples"] == 1
    assert token.shape == (2, 1)


def test_sample_returns_one_token_for_one_and_many_rows():
    rng = DecodingRng()

    one_row = rng.sample(torch.tensor([0.2, 0.3, 0.5]))
    many_rows = rng.sample(torch.tensor([[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]]))

    assert one_row.shape == (1,)
    assert many_rows.shape == (2, 1)
    assert one_row.dtype == many_rows.dtype == torch.int64
    assert 0 <= one_row.item() < 3
    assert torch.all((many_rows >= 0) & (many_rows < 3))


def test_sample_reuses_and_advances_one_private_generator(monkeypatch):
    calls = _record_multinomial(monkeypatch)
    probs = torch.tensor([[0.25, 0.75]])
    rng = DecodingRng()

    rng.sample(probs)
    generator = calls[0]["generator"]
    state_after_first = generator.get_state().clone()
    rng.sample(probs)

    assert calls[1]["generator"] is generator
    assert not torch.equal(generator.get_state(), state_after_first)


def test_rand_reuses_cpu_generator_across_cpu_alias(monkeypatch):
    multinomial_calls = _record_multinomial(monkeypatch)
    rand_calls = _record_rand(monkeypatch)
    probs = torch.tensor([[0.25, 0.75]])
    rng = DecodingRng()

    rng.sample(probs)
    alias_draw = rng.rand((5,), device=torch.device("cpu:0"))
    plain_draw = rng.rand((64,), device=CPU)

    assert rand_calls[0]["device"] == CPU
    assert rand_calls[0]["generator"] is multinomial_calls[0]["generator"]
    assert rand_calls[1]["generator"] is rand_calls[0]["generator"]
    assert alias_draw.shape == (5,)
    assert plain_draw.shape == (64,)
    assert plain_draw.dtype == torch.get_default_dtype()
    assert torch.all((plain_draw >= 0) & (plain_draw < 1))


def test_every_draw_receives_an_explicit_generator(monkeypatch):
    multinomial_calls = _record_multinomial(monkeypatch)
    rand_calls = _record_rand(monkeypatch)
    rng = DecodingRng()

    rng.sample(torch.tensor([0.1, 0.9]))
    rng.rand((3,), device=CPU)
    rng.sample(torch.tensor([0.4, 0.6]))

    generators = [call["generator"] for call in multinomial_calls + rand_calls]
    assert len(generators) == 3
    assert all(isinstance(generator, torch.Generator) for generator in generators)


def test_distinct_requested_cuda_devices_get_distinct_generators(monkeypatch):
    real_generator = torch.Generator
    real_rand = torch.rand
    created = []
    drawn = []

    def fake_generator(device="cpu", **kwargs):
        created.append(torch.device(device))
        return real_generator(device="cpu")

    def fake_rand(shape, *, device=None, generator=None):
        drawn.append((torch.device(device), generator))
        return real_rand(shape, generator=generator)

    def fail_on_cuda_query(*args, **kwargs):
        raise AssertionError("explicit cuda:N must not query torch.cuda")

    monkeypatch.setattr(torch, "Generator", fake_generator)
    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(torch.cuda, "current_device", fail_on_cuda_query)

    rng = DecodingRng()
    rng.rand((4,), device=torch.device("cuda:0"))
    rng.rand((4,), device=torch.device("cuda:1"))
    rng.rand((4,), device=torch.device("cuda:0"))

    cuda0 = torch.device("cuda:0")
    cuda1 = torch.device("cuda:1")
    assert created == [cuda0, cuda1]
    assert [device for device, _ in drawn] == [cuda0, cuda1, cuda0]
    assert drawn[0][1] is drawn[2][1]
    assert drawn[0][1].initial_seed() != drawn[1][1].initial_seed()
    assert drawn[0][1] is not drawn[1][1]


def test_unresolved_cuda_device_uses_current_index(monkeypatch):
    real_generator = torch.Generator
    real_rand = torch.rand
    created = []
    drawn = []

    def fake_generator(device="cpu", **kwargs):
        created.append(torch.device(device))
        return real_generator(device="cpu")

    def fake_rand(shape, *, device=None, generator=None):
        drawn.append((torch.device(device), generator))
        return real_rand(shape, generator=generator)

    monkeypatch.setattr(torch, "Generator", fake_generator)
    monkeypatch.setattr(torch, "rand", fake_rand)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)

    DecodingRng().rand((2,), device=torch.device("cuda"))

    assert created == [torch.device("cuda:3")]
    assert drawn[0][0] == torch.device("cuda:3")


def test_reseed_reproduces_but_fresh_contexts_differ(monkeypatch):
    rand_calls = _record_rand(monkeypatch)

    def draw():
        return DecodingRng().rand((16,), device=CPU)

    torch.manual_seed(20261111)
    first = draw()
    first_seed = rand_calls[-1]["generator"].initial_seed()

    torch.manual_seed(20261111)
    second = draw()
    second_seed = rand_calls[-1]["generator"].initial_seed()

    third = draw()
    third_seed = rand_calls[-1]["generator"].initial_seed()

    assert first_seed == second_seed
    assert torch.equal(first, second)
    assert third_seed != second_seed
    assert not torch.equal(third, second)


def test_default_cpu_rng_is_only_used_for_lazy_seed_allocation():
    probs = torch.tensor([[0.25, 0.75]])
    before = torch.get_rng_state()

    rng = DecodingRng()
    assert torch.equal(torch.get_rng_state(), before)

    rng.sample(probs)
    after_seed = torch.get_rng_state()
    assert not torch.equal(after_seed, before)

    rng.sample(probs)
    assert torch.equal(torch.get_rng_state(), after_seed)

    rng.rand((8,), device=CPU)
    assert torch.equal(torch.get_rng_state(), after_seed)


def test_empty_weights_fail_before_generator_or_seed_allocation(monkeypatch):
    def fail_generator(*args, **kwargs):
        raise AssertionError("generator must not be created before weights")

    monkeypatch.setattr(torch, "Generator", fail_generator)
    before = torch.get_rng_state()

    with pytest.raises(IndexError):
        DecodingRng().sample(torch.empty((2, 0)))

    assert torch.equal(torch.get_rng_state(), before)
