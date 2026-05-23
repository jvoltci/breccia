"""Tests for breccia.kernels.reference.cast (NumPy reference path).

For each of the 6 recipes:
- ``cast(x, recipe)`` produces a valid ScaledTensor with the right layout
- ``dequantize(cast(x))`` is close to ``x`` within the recipe's precision
- Block-scaled recipes produce a scale tensor of the expected shape

Quality is measured via cosine similarity (robust to small-magnitude
inputs where relative error explodes). Format-specific bounds:

- Per-tensor FP8 E4M3: cos sim > 0.99
- Per-block FP8: cos sim > 0.99
- MXFP8 (power-of-two scale): cos sim > 0.98
- NVFP4 (FP4 data, FP8 scale): cos sim > 0.95
- INT4 group-quantized: cos sim > 0.95

Tighter accuracy bounds live in ``benchmarks/bench_accuracy.py``.
"""

import numpy as np
import pytest

from breccia._core import ScaledTensor
from breccia.kernels.reference.cast import cast, dequantize, requantize
from breccia.layouts import PerTensor, PerBlockK, PerBlockMN
from breccia.recipes import (
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)


def _cos_sim(a, b):
    a_flat = a.astype(np.float64).ravel()
    b_flat = b.astype(np.float64).ravel()
    na = np.linalg.norm(a_flat)
    nb = np.linalg.norm(b_flat)
    return float(a_flat @ b_flat / (na * nb + 1e-12))


@pytest.fixture
def rng():
    return np.random.default_rng(42)


# ---------- DelayedScaling / Float8CurrentScaling (per-tensor) ----------


@pytest.mark.parametrize(
    "recipe",
    [DelayedScaling(fp8_format="E4M3"), Float8CurrentScaling(fp8_format="E4M3")],
)
def test_per_tensor_e4m3_round_trip(rng, recipe):
    x = rng.standard_normal((4, 64)).astype(np.float32)
    st = cast(x, recipe)
    assert isinstance(st, ScaledTensor)
    assert isinstance(st.layout, PerTensor)
    assert st.scale.ndim == 0
    assert st.shape == (4, 64)
    x_recovered = dequantize(st)
    assert _cos_sim(x, x_recovered) > 0.99


def test_per_tensor_e5m2_wider_range(rng):
    """E5M2 should handle 1000s where E4M3 saturates."""
    x = rng.standard_normal((4, 16)).astype(np.float32) * 1000
    st = cast(x, DelayedScaling(fp8_format="E5M2"))
    x_recovered = dequantize(st)
    assert _cos_sim(x, x_recovered) > 0.98


def test_per_tensor_recipe_carried_through(rng):
    x = rng.standard_normal((4, 16)).astype(np.float32)
    recipe = DelayedScaling(fp8_format="E5M2", amax_history_len=8)
    st = cast(x, recipe)
    assert st.recipe is recipe


# ---------- Float8BlockScaling (per-block-K) ----------


def test_block_fp8_scale_shape(rng):
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, Float8BlockScaling(block_k=128))
    assert st.scale.shape == (4, 2)  # K=256, B=128 → 2 blocks
    assert isinstance(st.layout, PerBlockK)
    assert st.layout.block_size == 128


def test_block_fp8_round_trip(rng):
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, Float8BlockScaling(block_k=128))
    x_recovered = dequantize(st)
    assert x_recovered.shape == x.shape
    assert _cos_sim(x, x_recovered) > 0.99


def test_block_fp8_handles_varying_block_amax(rng):
    """One block with huge magnitude shouldn't bleed precision into others."""
    x = rng.standard_normal((1, 256)).astype(np.float32) * 0.1
    x[0, :128] *= 100  # block 0 has much larger magnitude than block 1
    st = cast(x, Float8BlockScaling(block_k=128))
    x_recovered = dequantize(st)
    # Block 1 must retain its small-magnitude content as well as block 0.
    assert _cos_sim(x[0, 128:], x_recovered[0, 128:]) > 0.99


# ---------- MXFP8 ----------


def test_mxfp8_scale_is_uint8(rng):
    x = rng.standard_normal((4, 64)).astype(np.float32)
    st = cast(x, MXFP8BlockScaling())
    assert st.scale.dtype == np.uint8  # E8M0 stored as uint8
    assert st.scale.shape == (4, 2)  # 64 // 32 = 2 blocks
    assert isinstance(st.layout, PerBlockMN)
    assert st.layout.block_n == 32


def test_mxfp8_round_trip(rng):
    x = rng.standard_normal((4, 64)).astype(np.float32)
    st = cast(x, MXFP8BlockScaling())
    x_recovered = dequantize(st)
    assert x_recovered.shape == x.shape
    # MXFP8 has FP8 mantissa + power-of-two scale, so cos sim stays high.
    assert _cos_sim(x, x_recovered) > 0.98


# ---------- NVFP4 ----------


def test_nvfp4_scale_is_fp8_e4m3(rng):
    x = rng.standard_normal((4, 32)).astype(np.float32)
    st = cast(x, NVFP4BlockScaling())
    assert st.scale.dtype == np.uint8  # FP8 E4M3 stored as uint8
    assert st.scale.shape == (4, 2)  # 32 // 16 = 2 blocks
    assert isinstance(st.layout, PerBlockMN)
    assert st.layout.block_n == 16


def test_nvfp4_round_trip(rng):
    x = rng.standard_normal((4, 32)).astype(np.float32)
    st = cast(x, NVFP4BlockScaling())
    x_recovered = dequantize(st)
    assert x_recovered.shape == x.shape
    # NVFP4 has only 16 representable values per element; cos sim still > 0.95.
    assert _cos_sim(x, x_recovered) > 0.95


# ---------- INT4 ----------


def test_int4_signed_scale_shape(rng):
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128))
    assert st.scale.dtype == np.float16
    assert st.scale.shape == (4, 2)
    assert isinstance(st.layout, PerBlockK)
    assert st.layout.block_size == 128


def test_int4_unsigned(rng):
    x = rng.uniform(0, 1, size=(4, 128)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128, signed=False))
    x_recovered = dequantize(st)
    assert x_recovered.shape == x.shape
    assert _cos_sim(x, x_recovered) > 0.99


def test_int4_round_trip_signed(rng):
    x = rng.standard_normal((4, 128)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128))
    x_recovered = dequantize(st)
    assert _cos_sim(x, x_recovered) > 0.95


def test_int4_scale_dtype_options(rng):
    x = rng.standard_normal((4, 128)).astype(np.float32)
    for dt_name, expected_dtype in [
        ("fp16", np.float16),
        ("bf16", np.float32),
        ("fp32", np.float32),
    ]:
        st = cast(x, INT4Scaling(group_size=128, scale_dtype=dt_name))
        assert st.scale.dtype == expected_dtype


# ---------- Cross-recipe: requantize ----------


def test_requantize_changes_recipe(rng):
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st1 = cast(x, Float8BlockScaling(block_k=128))
    st2 = requantize(st1, Float8CurrentScaling())
    assert isinstance(st2.recipe, Float8CurrentScaling)
    assert isinstance(st2.layout, PerTensor)
    x_recovered = dequantize(st2)
    assert x_recovered.shape == x.shape


def test_requantize_fp8_to_int4(rng):
    """Converting FP8 to INT4 loses precision but should not crash or NaN."""
    x = rng.standard_normal((4, 128)).astype(np.float32)
    st_fp8 = cast(x, Float8CurrentScaling())
    st_int4 = requantize(st_fp8, INT4Scaling(group_size=128))
    x_recovered = dequantize(st_int4)
    assert not np.any(np.isnan(x_recovered))
    assert x_recovered.shape == x.shape


# ---------- Shape preservation ----------


@pytest.mark.parametrize(
    "recipe",
    [
        DelayedScaling(),
        Float8CurrentScaling(),
        Float8BlockScaling(block_k=64),
        MXFP8BlockScaling(),
        NVFP4BlockScaling(),
        INT4Scaling(group_size=64),
    ],
)
def test_shape_preserved(rng, recipe):
    x = rng.standard_normal((2, 4, 64)).astype(np.float32)
    st = cast(x, recipe)
    x_recovered = dequantize(st)
    assert x_recovered.shape == x.shape


# ---------- Edge case: all zeros ----------


@pytest.mark.parametrize(
    "recipe",
    [
        DelayedScaling(),
        Float8CurrentScaling(),
        Float8BlockScaling(block_k=64),
        MXFP8BlockScaling(),
        NVFP4BlockScaling(),
        INT4Scaling(group_size=64),
    ],
)
def test_zero_input_does_not_crash(recipe):
    x = np.zeros((2, 64), dtype=np.float32)
    st = cast(x, recipe)
    x_recovered = dequantize(st)
    # Allow small noise from quantization grid.
    assert np.max(np.abs(x_recovered)) < 1e-3
