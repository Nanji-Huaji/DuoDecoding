"""Public CEE sequence-law regression; CPU dependency substitute, not CUDA.

A context-independent uniform target has an iid three-token output law. With
cloned logical-role default categorical streams, little's second proposal and
draft's second categorical draw (after a discarded first draw) coincide. This
has correct one-position marginals but a wrong sequence law.

The categorical tape enumerates independent inverse-CDF coordinates. Acceptance
uniforms are fixed strictly inside [0,1): P=Q here, so every proposal is accepted.
This does not emulate the CUDA multinomial kernel, its random-word consumption,
or CPU/CUDA exponential-vs-uniform alignment. Explicit CPU generators share one
advancing categorical tape; default-role streams are identical clones. Only the
public baseline and Torch/cache dependency seams are exercised.
"""
import importlib.util
import sys
from collections import Counter
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_cee_rng_entry_fixture", ROOT / "test/test_cee_refactor.py"
)
_fixture = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _fixture
_spec.loader.exec_module(_fixture)


class _UniformCache(_fixture._CeeForwardedCache):
    phase = None

    def _set_uniform_history(self, prefix):
        type(self).phase = self.model.kind
        shape = (1, prefix.shape[1], self.vocab_size)
        self.prob_history = torch.full(shape, 0.25)
        self.logits_history = torch.zeros(shape)


class _CategoricalTape:
    def __init__(self, coordinates):
        # Four independent categorical coordinates cover the first three output
        # tokens, including the old discarded verifier draw. Thus fixing the
        # cloned streams can pass even before removing that redundant draw.
        self.values = tuple(coordinates) + (0.125,) * 8
        self.offsets = {}

    def multinomial(self, weights, num_samples=1, *, generator=None, **kwargs):
        assert num_samples == 1 and weights.shape == (1, 4)
        torch.testing.assert_close(weights, torch.full((1, 4), 0.25), rtol=0, atol=0)
        key = generator if generator is not None else ("default", _UniformCache.phase)
        offset = self.offsets.get(key, 0)
        self.offsets[key] = offset + 1
        threshold = weights.new_full((1, 1), self.values[offset])
        return torch.searchsorted(weights.cumsum(-1), threshold, right=True)

    @staticmethod
    def rand(*args, device=None, **kwargs):
        shape = args[0] if len(args) == 1 else args
        if isinstance(shape, int):
            shape = (shape,)
        return torch.full(shape, 0.5, device=device)


def _run_uniform(mode, coordinates):
    runtime = _fixture.CeeRefactorTests()._make_instance(mode)
    runtime.args.max_tokens = 3
    runtime.args.gamma1 = 1
    runtime.args.gamma2 = 2
    runtime.color_print = lambda *args, **kwargs: None
    _UniformCache.instances = []
    _UniformCache.phase = None
    tape = _CategoricalTape(coordinates)
    with ExitStack() as stack:
        for target, value in (
            ("src.baselines.core.KVCacheModel", _UniformCache),
            ("src.baselines.runtime.KVCacheModel", _UniformCache),
            (
                "src.baselines.simulators.CommunicationSimulator",
                _fixture._FakeCommSimulator,
            ),
            ("torch.cuda.Event", _fixture._FakeCudaEvent),
            ("torch.cuda.current_stream", lambda *args, **kwargs: None),
            ("torch.cuda.synchronize", lambda *args, **kwargs: None),
            ("torch.multinomial", tape.multinomial),
            ("torch.rand", tape.rand),
        ):
            stack.enter_context(patch(target, value))
        stack.enter_context(torch.random.fork_rng(devices=[]))
        output, metrics = getattr(runtime, mode)(
            torch.tensor([[0]], dtype=torch.long), transfer_top_k=None,
        )
    assert output.shape == (1, 4) and metrics["generated_tokens"] == 3
    return tuple(output[0, 1:].tolist())


@pytest.mark.parametrize("mode", ("cee_dsd", "cee_dssd"))
def test_uniform_target_three_token_joint_law_does_not_clone_role_defaults(mode):
    observed = Counter(
        _run_uniform(mode, tuple((i + 0.5) / 4 for i in indexes))
        for indexes in product(range(4), repeat=4)
    )
    # Independent target specification: each of 4^3 triplets has mass 1/64.
    # The 4^4 grid includes one nuisance coordinate, so each triplet occurs4 times.
    expected = Counter({triplet: 4 for triplet in product(range(4), repeat=3)})
    assert observed == expected, (
        "CEE uniform target must have the full product law, not cloned "
        f"little/draft categorical pairs: {dict(observed)}"
    )
