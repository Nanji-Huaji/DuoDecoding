"""Device-local capture contracts and opt-in native two-GPU regression.

CPU tests exercise the real runner's capture path with CUDA scheduling replaced.
Native tests require DUODECODING_RUN_CUDA_TESTS=1 and two *authorized* visible GPUs;
run them under the operator's idle-GPU guard, never automatically on shared GPUs.
"""

import os
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

import src.graph_decode as graph_decode
from src.graph_decode import GraphDecodeRunner


@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("unindexed_device", [False, True])
def test_capture_binds_device_stream_and_restores_caller(monkeypatch, order, unindexed_device):
    current = [0]
    expected = [None]
    streams = []
    captures = []
    synchronizations = []

    def index(device):
        if device is None:
            return current[0]
        if isinstance(device, int):
            return device
        return torch.device(device).index

    @contextmanager
    def device_context(device):
        previous = current[0]
        current[0] = index(device)
        try:
            yield
        finally:
            current[0] = previous

    def stream_factory(device=None, **kwargs):
        stream = SimpleNamespace(device=torch.device("cuda", index(device)))
        streams.append(stream)
        return stream

    @contextmanager
    def stream_context(stream):
        with device_context(stream.device):
            yield

    @contextmanager
    def capture(graph, *, stream=None, **kwargs):
        assert current[0] == expected[0], "capture uses the caller's device"
        assert stream is not None, "shared default capture stream is unsafe"
        assert stream.device == torch.device("cuda", expected[0])
        captures.append(stream)
        yield

    def synchronize(device=None):
        assert device is not None, "synchronize must name the runner's device"
        assert index(device) == expected[0]
        synchronizations.append(index(device))

    # Only storage is redirected to CPU; the actual prefill/capture control flow
    # and its two distinct torch.cuda.graph call sites remain under test.
    for name in ("zeros", "empty", "arange"):
        original = getattr(torch, name)

        def on_cpu(*args, _original=original, **kwargs):
            kwargs["device"] = "cpu"
            return _original(*args, **kwargs)

        monkeypatch.setattr(torch, name, on_cpu)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(torch.cuda, "Stream", stream_factory)
    monkeypatch.setattr(torch.cuda, "stream", stream_context)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", SimpleNamespace)
    monkeypatch.setattr(torch.cuda, "graph", capture)
    monkeypatch.setattr(
        graph_decode, "StaticCache", lambda **kwargs: SimpleNamespace(reset=lambda: None)
    )

    class Model:
        config = SimpleNamespace(hidden_size=0)

        def __call__(self, tokens, **kwargs):
            return SimpleNamespace(logits=tokens.float().unsqueeze(-1).expand(-1, -1, 7))

    for device in order:
        expected[0] = device
        # An unindexed "cuda" must be pinned at construction, not resolved later
        # against whichever device happens to be current at prefill time.
        with device_context(device):
            runner = GraphDecodeRunner(
                Model(), 32, device="cuda" if unindexed_device else f"cuda:{device}",
                verify_sizes=[2, 4],
            )
        assert runner.device == torch.device("cuda", device)
        before = len(captures)
        runner.prefill(torch.tensor([[1, 2, 3]]))
        assert current[0] == 0
        assert runner.capture_count == 3
        assert len(captures) == before + 3
        assert all(stream is captures[before] for stream in captures[before:])
        runner.prefill(torch.tensor([[3, 1]]))
        assert runner.capture_count == 3
        assert len(captures) == before + 3
        assert current[0] == 0
    assert len(streams) == 2
    assert set(synchronizations) == {0, 1}


_RUN_NATIVE = os.environ.get("DUODECODING_RUN_CUDA_TESTS") == "1"


@pytest.mark.skipif(not _RUN_NATIVE, reason="opt-in native CUDA regression")
@pytest.mark.parametrize("order", [(0, 1), (1, 0)])
@pytest.mark.parametrize("capture_post", [False, True])
@pytest.mark.parametrize("unindexed_device", [False, True])
@torch.inference_mode()
def test_native_multidevice_changed_input_replay(order, capture_post, unindexed_device):
    from transformers import LlamaConfig, LlamaForCausalLM

    assert torch.cuda.device_count() >= 2, "two authorized visible GPUs required"
    original_device = torch.cuda.current_device()
    original_streams = {i: torch.cuda.current_stream(i) for i in (0, 1)}
    caller_streams = {i: torch.cuda.Stream(device=i) for i in (0, 1)}
    runners = []
    try:
        for i in order:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(47)
                config = LlamaConfig(
                    vocab_size=7, hidden_size=32, intermediate_size=64,
                    num_hidden_layers=2, num_attention_heads=4,
                    num_key_value_heads=2, max_position_embeddings=128,
                    output_hidden_states=True, attention_dropout=0.0,
                )
                config._attn_implementation = "sdpa"
                model = LlamaForCausalLM(config).eval()
                model.lm_head.weight.mul_(8)
            with torch.cuda.device(i):
                model = model.to(device=i, dtype=torch.float32)
                runner = GraphDecodeRunner(
                    model, 32, device="cuda" if unindexed_device else f"cuda:{i}",
                    dtype=torch.float32, verify_sizes=[2, 4],
                )
                if capture_post:
                    runner.attach_history_buffers(
                        torch.empty((1, 32, 7), device=i),
                        torch.empty((1, 32, 7), device=i),
                        vocab_size=7,
                    )
                caller_streams[i].wait_stream(torch.cuda.current_stream(i))
                torch.cuda.set_stream(caller_streams[i])
            runners.append(runner)
        torch.cuda.set_device(0)
        rng_before = [state.clone() for state in torch.cuda.get_rng_state_all()]

        def check(runner, tokens, logits, hidden=None, start=None):
            assert torch.cuda.current_device() == 0
            assert torch.cuda.current_stream(runner.device) == caller_streams[runner.device.index]
            reference = runner.model(tokens, use_cache=False)
            count = 1 if logits.ndim == 2 else logits.shape[1]
            expected_logits = reference.logits[:, -count:]
            if logits.ndim == 2:
                expected_logits = expected_logits[:, -1]
            torch.testing.assert_close(logits, expected_logits, atol=4e-5, rtol=5e-4)
            if hidden is not None:
                torch.testing.assert_close(
                    hidden, reference.hidden_states[-1][:, -1:], atol=4e-5, rtol=5e-4
                )
            if capture_post and start is not None:
                end = tokens.shape[1]
                expected = reference.logits[:, start:end]
                torch.testing.assert_close(
                    runner.hist_logits_buf[:, start:end], expected, atol=4e-5, rtol=5e-4
                )
                assert torch.equal(
                    runner.hist_prob_buf[:, start:end].argmax(-1), expected.argmax(-1)
                )
            return tokens

        def poison(runner, start):
            runner.logits_buf.fill_(float("nan"))
            runner.hidden_buf.fill_(float("nan"))
            for buffer in runner.verify_logits.values():
                buffer.fill_(float("nan"))
            if capture_post:
                runner.hist_logits_buf[:, start:].fill_(float("nan"))
                runner.hist_prob_buf[:, start:].fill_(float("nan"))

        for runner in runners:
            prefix = torch.tensor([[1, 2, 3]], device=runner.device)
            check(runner, prefix, runner.prefill(prefix))
        for runner in reversed(runners):
            identity = (runner.graph, dict(runner.verify_graphs), runner.capture_count)
            prefix = torch.tensor([[1, 2, 3]], device=runner.device)
            for token in (1, 5):
                runner.rollback(3)
                suffix = torch.tensor([[token]], device=runner.device)
                poison(runner, 3)
                logits, hidden = runner.step(suffix)
                check(runner, torch.cat((prefix, suffix), 1), logits, hidden, start=3)
            for suffix in ([[6, 4, 2]], [[2, 6]]):
                runner.rollback(3)
                suffix = torch.tensor(suffix, device=runner.device)
                poison(runner, 3)
                logits, hidden = runner.verify(suffix)
                check(runner, torch.cat((prefix, suffix), 1), logits, hidden, start=3)
            alternate = torch.tensor([[3, 1]], device=runner.device)
            check(runner, alternate, runner.prefill(alternate))
            suffix = torch.tensor([[4]], device=runner.device)
            poison(runner, 2)
            logits, hidden = runner.step(suffix)
            check(runner, torch.cat((alternate, suffix), 1), logits, hidden, start=2)
            assert (runner.graph, runner.verify_graphs, runner.capture_count) == identity
            assert runner.capture_count == 3
        assert all(
            torch.equal(before, after)
            for before, after in zip(rng_before, torch.cuda.get_rng_state_all())
        )
    finally:
        for i in (0, 1):
            torch.cuda.synchronize(i)
            with torch.cuda.device(i):
                torch.cuda.set_stream(original_streams[i])
        torch.cuda.set_device(original_device)
