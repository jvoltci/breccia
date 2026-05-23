"""Tests for the MLX backend path (Apple Silicon).

v0.0.1 routes MLX arrays through the NumPy reference kernels — MLX has
no native FP8 / FP4 / INT4 dtype as of late 2025, so the data buffer
is uint8 either way. This path is correctness-only; hardware-accelerated
quantized matmul on Metal arrives in a later release.
"""

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

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


def _cos_sim_np(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _to_np(x):
    if hasattr(x, "__array__"):
        return np.array(x)
    return np.asarray(x)


# ---------- cast accepts MLX input, returns MLX fields ----------


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
def test_cast_returns_mlx_fields(recipe):
    np.random.seed(0)
    x = mx.array(np.random.standard_normal((4, 128)).astype(np.float32))
    st = cast(x, recipe)
    assert isinstance(st, ScaledTensor)
    # MLX arrays don't subclass np.ndarray.
    assert type(st.data).__module__.startswith("mlx")
    assert type(st.scale).__module__.startswith("mlx")


def test_dequantize_returns_mlx():
    np.random.seed(0)
    x = mx.array(np.random.standard_normal((4, 128)).astype(np.float32))
    st = cast(x, Float8CurrentScaling())
    out = dequantize(st)
    assert type(out).__module__.startswith("mlx")
    assert tuple(out.shape) == (4, 128)


# ---------- Round-trip quality ----------


def test_mlx_round_trip_per_tensor():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 128)).astype(np.float32)
    x = mx.array(x_np)
    out = dequantize(cast(x, Float8CurrentScaling()))
    assert _cos_sim_np(_to_np(out), x_np) > 0.99


def test_mlx_round_trip_block():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 256)).astype(np.float32)
    x = mx.array(x_np)
    out = dequantize(cast(x, Float8BlockScaling(block_k=128)))
    assert _cos_sim_np(_to_np(out), x_np) > 0.99


def test_mlx_round_trip_nvfp4():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 32)).astype(np.float32)
    x = mx.array(x_np)
    out = dequantize(cast(x, NVFP4BlockScaling()))
    assert _cos_sim_np(_to_np(out), x_np) > 0.95


# ---------- matmul on MLX ----------


def test_mlx_matmul_per_tensor():
    np.random.seed(0)
    A_np = np.random.standard_normal((8, 128)).astype(np.float32)
    B_np = np.random.standard_normal((128, 16)).astype(np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb)
    assert type(out).__module__.startswith("mlx")
    assert tuple(out.shape) == (8, 16)
    assert _cos_sim_np(_to_np(out), A_np @ B_np) > 0.99


def test_mlx_matmul_block():
    np.random.seed(0)
    A_np = np.random.standard_normal((8, 256)).astype(np.float32)
    B_np = np.random.standard_normal((256, 128)).astype(np.float32)
    A = mx.array(A_np)
    B = mx.array(B_np)
    sa = cast(A, Float8BlockScaling(block_k=128))
    sb = cast(B, Float8BlockScaling(block_k=128))
    out = matmul(sa, sb)
    assert _cos_sim_np(_to_np(out), A_np @ B_np) > 0.99


# ---------- Cross-backend agreement ----------


def test_mlx_vs_numpy_dequantize_agrees():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 128)).astype(np.float32)
    x_mx = mx.array(x_np)
    out_np = dequantize(cast(x_np, Float8CurrentScaling()))
    out_mx = _to_np(dequantize(cast(x_mx, Float8CurrentScaling())))
    np.testing.assert_allclose(out_np, out_mx, atol=1e-5)
