"""CEETriPrefix 的 buffer/cat 两种模式等价性测试。"""

import torch

from src.cee_prefix import CEETriPrefix


def _run(mode):
    prefix = torch.tensor([[1, 2, 3]])
    ops = CEETriPrefix(prefix, capacity=16) if mode == "buffer" else CEETriPrefix(prefix)
    out = [ops.tensor.clone()]
    out.append(ops.assign(torch.tensor([[1, 2, 3, 4, 5]]), 5).clone())
    out.append(ops.append(torch.tensor([[6, 7]])).clone())
    out.append(ops.assign(torch.tensor([[9, 9, 9]]), 3).clone())
    out.append(ops.append(torch.tensor([[8]])).clone())
    return out


def test_buffer_and_cat_modes_match():
    buf, cat = _run("buffer"), _run("cat")
    assert len(buf) == len(cat)
    for a, b in zip(buf, cat):
        assert torch.equal(a, b)


def test_initial_tensor_is_prefix():
    p = torch.tensor([[4, 5]])
    assert torch.equal(CEETriPrefix(p).tensor, p)
    assert torch.equal(CEETriPrefix(p, capacity=8).tensor, p)
