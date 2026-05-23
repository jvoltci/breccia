"""Tests for breccia.kernels.reference.matmul."""

import numpy as np
import pytest

from breccia import (
    cast,
    matmul,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)


def _cos_sim(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ---------- Basic correctness ----------


def test_matmul_per_tensor_fp8(rng):
    M, K, N = 8, 128, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out_scaled = matmul(sa, sb)
    out_ref = A @ B
    assert out_scaled.shape == (M, N)
    assert _cos_sim(out_scaled, out_ref) > 0.99


def test_matmul_block_fp8(rng):
    """Block-scaled recipes: both sides need last-dim divisible by block_size."""
    M, K, N = 8, 256, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, Float8BlockScaling(block_k=128))
    sb = cast(B, Float8BlockScaling(block_k=128))
    out_scaled = matmul(sa, sb)
    out_ref = A @ B
    assert _cos_sim(out_scaled, out_ref) > 0.99


def test_matmul_mxfp8(rng):
    M, K, N = 4, 64, 64
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, MXFP8BlockScaling())
    sb = cast(B, MXFP8BlockScaling())
    out_scaled = matmul(sa, sb)
    out_ref = A @ B
    # MXFP8 has power-of-two scale → slightly worse than per-block fp32 scale.
    assert _cos_sim(out_scaled, out_ref) > 0.98


def test_matmul_nvfp4(rng):
    M, K, N = 8, 64, 32
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, NVFP4BlockScaling())
    sb = cast(B, NVFP4BlockScaling())
    out_scaled = matmul(sa, sb)
    out_ref = A @ B
    # FP4 weights × FP4 activations + FP8 scale: cos sim still > 0.95.
    assert _cos_sim(out_scaled, out_ref) > 0.95


def test_matmul_int4(rng):
    M, K, N = 8, 256, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, INT4Scaling(group_size=128))
    sb = cast(B, INT4Scaling(group_size=128))
    out_scaled = matmul(sa, sb)
    out_ref = A @ B
    assert _cos_sim(out_scaled, out_ref) > 0.95


# ---------- Mixed-recipe matmul ----------


def test_matmul_mixed_recipes(rng):
    """ScaledTensor with one recipe × ScaledTensor with another recipe."""
    M, K, N = 8, 128, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())  # per-tensor FP8
    sb = cast(B, Float8BlockScaling(block_k=128))  # per-block FP8
    out = matmul(sa, sb)
    out_ref = A @ B
    assert _cos_sim(out, out_ref) > 0.99


def test_matmul_scaled_times_raw(rng):
    """ScaledTensor @ raw ndarray (one side high-precision)."""
    M, K, N = 8, 64, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())
    out = matmul(sa, B)
    out_ref = A @ B
    assert _cos_sim(out, out_ref) > 0.99


def test_matmul_raw_times_scaled(rng):
    M, K, N = 8, 64, 16
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sb = cast(B, Float8CurrentScaling())
    out = matmul(A, sb)
    out_ref = A @ B
    assert _cos_sim(out, out_ref) > 0.99


# ---------- Batched ----------


def test_matmul_batched(rng):
    Bsz, M, K, N = 3, 4, 64, 16
    A = rng.standard_normal((Bsz, M, K)).astype(np.float32)
    B = rng.standard_normal((Bsz, K, N)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb)
    assert out.shape == (Bsz, M, N)
    assert _cos_sim(out, A @ B) > 0.99


# ---------- Shape validation ----------


def test_matmul_shape_mismatch_raises(rng):
    A = rng.standard_normal((4, 128)).astype(np.float32)
    B = rng.standard_normal((64, 16)).astype(np.float32)  # K=64 != A's K=128
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    with pytest.raises(ValueError, match="shape mismatch"):
        matmul(sa, sb)


def test_matmul_output_dtype_default(rng):
    A = rng.standard_normal((4, 64)).astype(np.float32)
    B = rng.standard_normal((64, 8)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb)
    assert out.dtype == np.float32


def test_matmul_output_dtype_override(rng):
    A = rng.standard_normal((4, 64)).astype(np.float32)
    B = rng.standard_normal((64, 8)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb, out_dtype=np.float16)
    assert out.dtype == np.float16


# ---------- Recipe comparison ----------


def test_matmul_quality_ordering(rng):
    """Higher-precision recipes should give higher matmul quality (in expectation)."""
    M, K, N = 16, 256, 128
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    out_ref = A @ B

    out_fp8 = matmul(cast(A, Float8CurrentScaling()), cast(B, Float8CurrentScaling()))
    out_int4 = matmul(cast(A, INT4Scaling(group_size=128)), cast(B, INT4Scaling(group_size=128)))

    cos_fp8 = _cos_sim(out_fp8, out_ref)
    cos_int4 = _cos_sim(out_int4, out_ref)
    # FP8 should beat INT4 on a well-behaved Gaussian input.
    assert cos_fp8 >= cos_int4
