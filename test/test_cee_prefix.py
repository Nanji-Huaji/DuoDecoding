"""CEETriPrefix：buffer/cat 两种模式的逐步值、容量边界与别名契约。"""

import pytest
import torch

from src.cee_prefix import CEETriPrefix


def _run(mode, capacity=8):
    prefix = torch.tensor([[1, 2, 3]])
    ops = CEETriPrefix(prefix, capacity=capacity) if mode == "buffer" else CEETriPrefix(prefix)
    steps = []
    steps.append(ops.tensor.clone())
    steps.append(ops.assign(torch.tensor([[1, 2, 3, 4, 5]]), 5).clone())
    steps.append(ops.append(torch.tensor([[6, 7]])).clone())
    steps.append(ops.assign(torch.tensor([[9, 9, 9]]), 3).clone())
    steps.append(ops.append(torch.tensor([[8]])).clone())
    return steps


def test_buffer_and_cat_modes_match_stepwise():
    buf, cat = _run("buffer"), _run("cat")
    assert len(buf) == len(cat)
    for a, b in zip(buf, cat):
        assert torch.equal(a, b)
        assert a.shape == b.shape


def test_capacity_boundary_is_enforced():
    ops = CEETriPrefix(torch.tensor([[1, 2, 3]]), capacity=5)
    ops.append(torch.tensor([[4, 5]]))  # 正好到容量
    with pytest.raises(ValueError, match="buffer overflow"):
        ops.append(torch.tensor([[6]]))
    with pytest.raises(ValueError, match="buffer overflow"):
        ops.assign(torch.tensor([[1, 2, 3, 4, 5, 6]]), 6)


def test_aliasing_contract_differs_by_mode():
    prefix = torch.tensor([[1, 2, 3]])
    buf = CEETriPrefix(prefix, capacity=8)
    cat = CEETriPrefix(prefix)
    buf_handle = buf.tensor
    cat_handle = cat.tensor
    buf.assign(torch.tensor([[7, 7, 7, 7]]), 4)
    cat.assign(torch.tensor([[7, 7, 7, 7]]), 4)
    # buffer 的旧句柄是 _buffer 的视图：前 3 列被写成了 7；
    # cat 的旧句柄是独立张量：保持原值。
    assert torch.equal(buf_handle, torch.tensor([[7, 7, 7]]))
    assert torch.equal(cat_handle, torch.tensor([[1, 2, 3]]))


def test_initial_tensor_is_prefix():
    p = torch.tensor([[4, 5]])
    assert torch.equal(CEETriPrefix(p).tensor, p)
    assert torch.equal(CEETriPrefix(p, capacity=8).tensor, p)


def test_dtype_device_and_assign_guard():
    p = torch.tensor([[1, 2, 3]], dtype=torch.long)
    ops = CEETriPrefix(p, capacity=8)
    assert ops.tensor.dtype == p.dtype
    assert ops.tensor.device == p.device
    with pytest.raises(ValueError, match="num_tokens larger"):
        ops.assign(torch.tensor([[1, 2]]), 3)
    with pytest.raises(ValueError, match="batch=1"):
        CEETriPrefix(torch.tensor([[1, 2], [3, 4]]), capacity=8)
