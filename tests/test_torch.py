"""Tests for the PyTorch backend path.

v0.0.1 routes torch tensors through the NumPy reference kernels (move
to CPU, run reference, wrap result fields as torch tensors on the
original device). Native FP8 acceleration lands in M17 (Triton).
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from breccia import (
    cast,
    dequantize,
    matmul,
    requantize,
    ScaledTensor,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)


def _cos_sim(a, b):
    a = a.detach().to(torch.float64).flatten()
    b = b.detach().to(torch.float64).flatten()
    return float(torch.dot(a, b) / (torch.linalg.norm(a) * torch.linalg.norm(b) + 1e-12))


@pytest.fixture
def rng():
    return torch.Generator().manual_seed(0)


# ---------- cast accepts torch input, returns torch tensors ----------


@pytest.mark.parametrize(
    "recipe",
    [
        DelayedScaling(),
        Float8CurrentScaling(),
        Float8BlockScaling(block_k=128),
        MXFP8BlockScaling(),
        NVFP4BlockScaling(),
        INT4Scaling(group_size=128),
    ],
)
def test_cast_returns_torch_fields(rng, recipe):
    x = torch.randn(4, 128, generator=rng, dtype=torch.float32)
    st = cast(x, recipe)
    assert isinstance(st, ScaledTensor)
    assert isinstance(st.data, torch.Tensor)
    assert isinstance(st.scale, torch.Tensor)


def test_cast_preserves_device_cpu():
    x = torch.zeros(4, 128, dtype=torch.float32)
    st = cast(x, Float8CurrentScaling())
    assert st.data.device.type == "cpu"
    assert st.scale.device.type == "cpu"


def test_dequantize_returns_torch(rng):
    x = torch.randn(4, 128, generator=rng, dtype=torch.float32)
    st = cast(x, Float8CurrentScaling())
    out = dequantize(st)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


# ---------- Round-trip quality ----------


def test_torch_round_trip_per_tensor_fp8(rng):
    x = torch.randn(4, 128, generator=rng, dtype=torch.float32)
    out = dequantize(cast(x, Float8CurrentScaling()))
    assert _cos_sim(x, out) > 0.99


def test_torch_round_trip_block_fp8(rng):
    x = torch.randn(4, 256, generator=rng, dtype=torch.float32)
    out = dequantize(cast(x, Float8BlockScaling(block_k=128)))
    assert _cos_sim(x, out) > 0.99


def test_torch_round_trip_nvfp4(rng):
    x = torch.randn(4, 32, generator=rng, dtype=torch.float32)
    out = dequantize(cast(x, NVFP4BlockScaling()))
    assert _cos_sim(x, out) > 0.95


# ---------- matmul on torch tensors ----------


def test_torch_matmul_per_tensor(rng):
    A = torch.randn(8, 128, generator=rng, dtype=torch.float32)
    B = torch.randn(128, 16, generator=rng, dtype=torch.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (8, 16)
    assert _cos_sim(out, A @ B) > 0.99


def test_torch_matmul_block(rng):
    A = torch.randn(8, 256, generator=rng, dtype=torch.float32)
    B = torch.randn(256, 128, generator=rng, dtype=torch.float32)
    sa = cast(A, Float8BlockScaling(block_k=128))
    sb = cast(B, Float8BlockScaling(block_k=128))
    out = matmul(sa, sb)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (8, 128)
    assert _cos_sim(out, A @ B) > 0.99


def test_torch_matmul_with_raw_tensor(rng):
    A = torch.randn(8, 64, generator=rng, dtype=torch.float32)
    B = torch.randn(64, 16, generator=rng, dtype=torch.float32)
    sa = cast(A, Float8CurrentScaling())
    out = matmul(sa, B)  # ScaledTensor × raw torch tensor
    assert isinstance(out, torch.Tensor)
    assert _cos_sim(out, A @ B) > 0.99


def test_torch_matmul_dtype_override(rng):
    A = torch.randn(4, 64, generator=rng, dtype=torch.float32)
    B = torch.randn(64, 16, generator=rng, dtype=torch.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb, out_dtype=np.float16)
    assert out.dtype == torch.float16


# ---------- requantize ----------


def test_torch_requantize(rng):
    x = torch.randn(4, 128, generator=rng, dtype=torch.float32)
    st1 = cast(x, Float8BlockScaling(block_k=128))
    st2 = requantize(st1, Float8CurrentScaling())
    assert isinstance(st2.data, torch.Tensor)
    assert isinstance(st2.recipe, Float8CurrentScaling)


# ---------- Cross-backend agreement ----------


def test_torch_vs_numpy_dequantize_agrees(rng):
    """Same input through numpy and torch paths produces equivalent output."""
    x_np = np.array(torch.randn(4, 128, generator=rng).numpy())
    x_t = torch.from_numpy(x_np.copy())

    out_np = dequantize(cast(x_np, Float8CurrentScaling()))
    out_t = dequantize(cast(x_t, Float8CurrentScaling())).numpy()

    np.testing.assert_allclose(out_np, out_t, atol=1e-5)
