"""Tests for the JAX backend path.

v0.1 routes JAX arrays through the NumPy reference kernels (immutable
JAX arrays make in-place mutation a non-starter, so the round-trip is
the natural path). Hardware-accelerated JAX paths (Pallas / XLA scaled
GEMM) land in v0.2.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

from breccia import (
    cast,
    dequantize,
    matmul,
    requantize,
    ScaledTensor,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
    DelayedScaling,
)


def _cos_sim_np(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------- cast/dequantize ----------


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
def test_cast_returns_jax_fields(recipe):
    np.random.seed(0)
    x = jnp.asarray(np.random.standard_normal((4, 128)).astype(np.float32))
    st = cast(x, recipe)
    assert isinstance(st, ScaledTensor)
    assert type(st.data).__module__.startswith("jax") or type(
        st.data
    ).__module__.startswith("jaxlib")
    assert type(st.scale).__module__.startswith("jax") or type(
        st.scale
    ).__module__.startswith("jaxlib")


def test_dequantize_returns_jax():
    np.random.seed(0)
    x = jnp.asarray(np.random.standard_normal((4, 128)).astype(np.float32))
    st = cast(x, Float8CurrentScaling())
    out = dequantize(st)
    assert type(out).__module__.startswith("jax") or type(
        out
    ).__module__.startswith("jaxlib")
    assert tuple(out.shape) == (4, 128)


# ---------- Round-trip quality ----------


def test_jax_round_trip_per_tensor():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 128)).astype(np.float32)
    x = jnp.asarray(x_np)
    out = dequantize(cast(x, Float8CurrentScaling()))
    assert _cos_sim_np(np.asarray(out), x_np) > 0.99


def test_jax_round_trip_block():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 256)).astype(np.float32)
    x = jnp.asarray(x_np)
    out = dequantize(cast(x, Float8BlockScaling(block_k=128)))
    assert _cos_sim_np(np.asarray(out), x_np) > 0.99


def test_jax_round_trip_nvfp4():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 32)).astype(np.float32)
    x = jnp.asarray(x_np)
    out = dequantize(cast(x, NVFP4BlockScaling()))
    assert _cos_sim_np(np.asarray(out), x_np) > 0.95


# ---------- matmul ----------


def test_jax_matmul_per_tensor():
    np.random.seed(0)
    A_np = np.random.standard_normal((8, 128)).astype(np.float32)
    B_np = np.random.standard_normal((128, 16)).astype(np.float32)
    A = jnp.asarray(A_np)
    B = jnp.asarray(B_np)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb)
    assert type(out).__module__.startswith("jax") or type(
        out
    ).__module__.startswith("jaxlib")
    assert tuple(out.shape) == (8, 16)
    assert _cos_sim_np(np.asarray(out), A_np @ B_np) > 0.99


def test_jax_matmul_block():
    np.random.seed(0)
    A_np = np.random.standard_normal((8, 256)).astype(np.float32)
    B_np = np.random.standard_normal((256, 128)).astype(np.float32)
    A = jnp.asarray(A_np)
    B = jnp.asarray(B_np)
    sa = cast(A, Float8BlockScaling(block_k=128))
    sb = cast(B, Float8BlockScaling(block_k=128))
    out = matmul(sa, sb)
    assert _cos_sim_np(np.asarray(out), A_np @ B_np) > 0.99


def test_jax_matmul_with_raw_array():
    np.random.seed(0)
    A_np = np.random.standard_normal((8, 64)).astype(np.float32)
    B_np = np.random.standard_normal((64, 16)).astype(np.float32)
    A = jnp.asarray(A_np)
    B = jnp.asarray(B_np)
    sa = cast(A, Float8CurrentScaling())
    out = matmul(sa, B)
    assert type(out).__module__.startswith("jax") or type(
        out
    ).__module__.startswith("jaxlib")
    assert _cos_sim_np(np.asarray(out), A_np @ B_np) > 0.99


# ---------- requantize ----------


def test_jax_requantize():
    np.random.seed(0)
    x = jnp.asarray(np.random.standard_normal((4, 128)).astype(np.float32))
    st1 = cast(x, Float8BlockScaling(block_k=128))
    st2 = requantize(st1, Float8CurrentScaling())
    assert isinstance(st2.recipe, Float8CurrentScaling)
    assert type(st2.data).__module__.startswith("jax") or type(
        st2.data
    ).__module__.startswith("jaxlib")


# ---------- Cross-backend agreement ----------


def test_jax_vs_numpy_dequantize_agrees():
    np.random.seed(0)
    x_np = np.random.standard_normal((4, 128)).astype(np.float32)
    x_jax = jnp.asarray(x_np)
    out_np = dequantize(cast(x_np, Float8CurrentScaling()))
    out_jax = np.asarray(dequantize(cast(x_jax, Float8CurrentScaling())))
    np.testing.assert_allclose(out_np, out_jax, atol=1e-5)
