"""Hypothesis property tests for breccia.

Tests generative invariants: shape preservation, round-trip quality,
recipe / layout properties, and matmul shape rules. These complement
the example-based tests in the other test files.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, assume, given, settings, strategies as st

from breccia import (
    ScaledTensor,
    cast,
    dequantize,
    matmul,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)
from breccia.layouts import PerBlockK, PerBlockMN


_SLOW = settings(
    max_examples=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)


def _cos_sim(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------- Strategies ----------


def _gaussian_2d(rows_max=32, cols_choices=(64, 128, 256)):
    """A 2-D fp32 tensor of standard-normal values."""
    return st.builds(
        lambda r, c, seed: np.random.default_rng(seed).standard_normal((r, c)).astype(np.float32),
        st.integers(min_value=1, max_value=rows_max),
        st.sampled_from(cols_choices),
        st.integers(min_value=0, max_value=10_000),
    )


def _gaussian_2d_block128(rows_max=16):
    """2-D fp32 tensor with K-axis divisible by 128 (for Float8BlockScaling)."""
    return st.builds(
        lambda r, c, seed: np.random.default_rng(seed).standard_normal((r, c)).astype(np.float32),
        st.integers(min_value=1, max_value=rows_max),
        st.sampled_from((128, 256, 384)),
        st.integers(min_value=0, max_value=10_000),
    )


def _gaussian_2d_mx32(rows_max=8):
    """2-D fp32 tensor with K-axis divisible by 32 (for MXFP8)."""
    return st.builds(
        lambda r, c, seed: np.random.default_rng(seed).standard_normal((r, c)).astype(np.float32),
        st.integers(min_value=1, max_value=rows_max),
        st.sampled_from((32, 64, 96, 128)),
        st.integers(min_value=0, max_value=10_000),
    )


def _gaussian_2d_nvfp4(rows_max=8):
    """2-D fp32 tensor with K-axis divisible by 16 (for NVFP4)."""
    return st.builds(
        lambda r, c, seed: np.random.default_rng(seed).standard_normal((r, c)).astype(np.float32),
        st.integers(min_value=1, max_value=rows_max),
        st.sampled_from((16, 32, 48, 64)),
        st.integers(min_value=0, max_value=10_000),
    )


# ---------- Property: shape preserved across cast → dequantize ----------


@_SLOW
@given(_gaussian_2d())
def test_property_shape_preserved_per_tensor(x):
    out = dequantize(cast(x, Float8CurrentScaling()))
    assert out.shape == x.shape


@_SLOW
@given(_gaussian_2d_block128())
def test_property_shape_preserved_block_fp8(x):
    out = dequantize(cast(x, Float8BlockScaling(block_k=128)))
    assert out.shape == x.shape


@_SLOW
@given(_gaussian_2d_mx32())
def test_property_shape_preserved_mxfp8(x):
    out = dequantize(cast(x, MXFP8BlockScaling()))
    assert out.shape == x.shape


@_SLOW
@given(_gaussian_2d_nvfp4())
def test_property_shape_preserved_nvfp4(x):
    out = dequantize(cast(x, NVFP4BlockScaling()))
    assert out.shape == x.shape


# ---------- Property: round-trip cosine similarity ----------


@_SLOW
@given(_gaussian_2d())
def test_property_round_trip_cosine_fp8(x):
    out = dequantize(cast(x, Float8CurrentScaling()))
    assert _cos_sim(x, np.asarray(out)) > 0.99


@_SLOW
@given(_gaussian_2d_block128())
def test_property_round_trip_cosine_block_fp8(x):
    out = dequantize(cast(x, Float8BlockScaling(block_k=128)))
    assert _cos_sim(x, np.asarray(out)) > 0.99


@_SLOW
@given(_gaussian_2d_mx32())
def test_property_round_trip_cosine_mxfp8(x):
    out = dequantize(cast(x, MXFP8BlockScaling()))
    assert _cos_sim(x, np.asarray(out)) > 0.97


@_SLOW
@given(_gaussian_2d_nvfp4())
def test_property_round_trip_cosine_nvfp4(x):
    out = dequantize(cast(x, NVFP4BlockScaling()))
    # NVFP4 has only 16 representable values, but cos sim stays > 0.93 on Gaussian.
    assert _cos_sim(x, np.asarray(out)) > 0.93


# ---------- Property: scale shape obeys layout ----------


@_SLOW
@given(_gaussian_2d_block128())
def test_property_block_fp8_scale_shape(x):
    st_obj = cast(x, Float8BlockScaling(block_k=128))
    assert isinstance(st_obj.layout, PerBlockK)
    expected_scale = x.shape[:-1] + (x.shape[-1] // 128,)
    assert tuple(st_obj.scale.shape) == expected_scale


@_SLOW
@given(_gaussian_2d_mx32())
def test_property_mxfp8_scale_shape(x):
    st_obj = cast(x, MXFP8BlockScaling())
    assert isinstance(st_obj.layout, PerBlockMN)
    expected_scale = x.shape[:-1] + (x.shape[-1] // 32,)
    assert tuple(st_obj.scale.shape) == expected_scale


# ---------- Property: recipes are hashable and equal-when-fields-equal ----------


@given(
    st.sampled_from(["E4M3", "E5M2"]),
    st.integers(min_value=1, max_value=32),
)
def test_property_delayed_equal_same_fields(fmt, hist):
    r1 = DelayedScaling(fp8_format=fmt, amax_history_len=hist)
    r2 = DelayedScaling(fp8_format=fmt, amax_history_len=hist)
    assert r1 == r2
    assert hash(r1) == hash(r2)


@given(
    st.integers(min_value=64, max_value=512).filter(lambda x: x > 0),
)
def test_property_block_recipes_distinct_when_block_k_differs(k):
    assume(k != 128)
    assert Float8BlockScaling(block_k=k) != Float8BlockScaling(block_k=128)


# ---------- Property: matmul shape rule ----------


@_SLOW
@given(
    st.integers(min_value=1, max_value=8),
    st.sampled_from((64, 128, 256)),
    st.integers(min_value=1, max_value=8),
    st.integers(min_value=0, max_value=10_000),
)
def test_property_matmul_shape(M, K, N, seed):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((M, K)).astype(np.float32)
    B = rng.standard_normal((K, N)).astype(np.float32)
    sa = cast(A, Float8CurrentScaling())
    sb = cast(B, Float8CurrentScaling())
    out = matmul(sa, sb)
    assert out.shape == (M, N)


# ---------- Property: dequantize is bounded by recipe's max range ----------


@_SLOW
@given(_gaussian_2d())
def test_property_dequantize_finite(x):
    """The recovered tensor never contains NaN or Inf for finite input."""
    out = np.asarray(dequantize(cast(x, Float8CurrentScaling())))
    assert np.all(np.isfinite(out))


@_SLOW
@given(_gaussian_2d_nvfp4())
def test_property_dequantize_nvfp4_finite(x):
    out = np.asarray(dequantize(cast(x, NVFP4BlockScaling())))
    assert np.all(np.isfinite(out))


# ---------- Property: zero input round-trips to near-zero ----------


@given(
    st.integers(min_value=1, max_value=8),
    st.sampled_from((32, 64, 128, 256)),
)
def test_property_zero_input_dequantizes_to_zero_fp8(M, K):
    x = np.zeros((M, K), dtype=np.float32)
    out = np.asarray(dequantize(cast(x, Float8CurrentScaling())))
    # Recovered must be within a tiny epsilon of zero.
    assert np.max(np.abs(out)) < 1e-3


# ---------- Property: data buffer size matches logical shape ----------


@_SLOW
@given(_gaussian_2d())
def test_property_data_shape_matches_input_shape(x):
    """ScaledTensor.data has the same shape as the input for v0.0.1 (no nibble packing)."""
    st_obj = cast(x, Float8CurrentScaling())
    assert tuple(st_obj.data.shape) == x.shape
