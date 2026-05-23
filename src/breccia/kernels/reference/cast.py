"""Reference cast / dequantize / requantize for every ScalingRecipe.

The cast forward path is, for each recipe:

1. Compute the per-{tensor, block, channel, group} ``amax``.
2. Derive the dequantization scale: ``scale = amax / fmt_max``.
3. Quantize ``x / scale`` to the recipe's data format.

Dequantize reverses this: ``x_recovered = decode(data) * scale``.

By convention, ``ScaledTensor.scale`` stores the *dequantization* scale —
the value you multiply the decoded low-precision data by to recover the
high-precision value. This is the OCP MX convention and matches how
hardware scaled-matmul kernels consume the scale tensor.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from breccia._core import ScaledTensor
from breccia._formats import (
    E4M3_MAX,
    E5M2_MAX,
    E2M1_MAX,
    encode_e4m3,
    decode_e4m3,
    encode_e5m2,
    decode_e5m2,
    encode_e2m1,
    decode_e2m1,
    encode_int4,
    decode_int4,
)
from breccia.layouts import (
    PerTensor,
    PerBlockK,
    PerChannel,
    PerBlockMN,
)
from breccia.recipes import (
    ScalingRecipe,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)

from ._utils import (
    block_amax,
    tensor_amax,
    quantize_e8m0_scale,
    dequantize_e8m0_scale,
)


# ---------- Public entry points ----------


def cast(x: Any, recipe: ScalingRecipe) -> ScaledTensor:
    """Quantize a high-precision tensor to a ScaledTensor using ``recipe``."""
    x_np = np.asarray(x, dtype=np.float32)
    if isinstance(recipe, (DelayedScaling, Float8CurrentScaling)):
        return _cast_per_tensor_fp8(x_np, recipe)
    if isinstance(recipe, Float8BlockScaling):
        return _cast_block_fp8(x_np, recipe)
    if isinstance(recipe, MXFP8BlockScaling):
        return _cast_mxfp8(x_np, recipe)
    if isinstance(recipe, NVFP4BlockScaling):
        return _cast_nvfp4(x_np, recipe)
    if isinstance(recipe, INT4Scaling):
        return _cast_int4(x_np, recipe)
    raise TypeError(f"unsupported recipe: {type(recipe).__name__}")


def dequantize(scaled: ScaledTensor) -> np.ndarray:
    """Reverse :func:`cast`: produce a float32 tensor from a ScaledTensor."""
    r = scaled.recipe
    if isinstance(r, (DelayedScaling, Float8CurrentScaling)):
        return _dequantize_per_tensor_fp8(scaled)
    if isinstance(r, Float8BlockScaling):
        return _dequantize_block_fp8(scaled)
    if isinstance(r, MXFP8BlockScaling):
        return _dequantize_mxfp8(scaled)
    if isinstance(r, NVFP4BlockScaling):
        return _dequantize_nvfp4(scaled)
    if isinstance(r, INT4Scaling):
        return _dequantize_int4(scaled)
    raise TypeError(f"unsupported recipe: {type(r).__name__}")


def requantize(scaled: ScaledTensor, recipe: ScalingRecipe) -> ScaledTensor:
    """Convert a ScaledTensor from one recipe to another.

    Implemented as ``cast(dequantize(scaled), recipe)``. Direct cross-recipe
    paths that avoid the round-trip to float32 are an optimization
    deferred to v0.1.
    """
    return cast(dequantize(scaled), recipe)


# ---------- Per-tensor FP8 (DelayedScaling, Float8CurrentScaling) ----------


def _fp8_max(fmt: str) -> float:
    return E4M3_MAX if fmt == "E4M3" else E5M2_MAX


def _encode_fp8(x: np.ndarray, fmt: str) -> np.ndarray:
    return encode_e4m3(x) if fmt == "E4M3" else encode_e5m2(x)


def _decode_fp8(b: np.ndarray, fmt: str) -> np.ndarray:
    return decode_e4m3(b) if fmt == "E4M3" else decode_e5m2(b)


def _cast_per_tensor_fp8(
    x: np.ndarray,
    recipe: DelayedScaling | Float8CurrentScaling,
) -> ScaledTensor:
    fmt = recipe.fp8_format
    amax = tensor_amax(x)
    scale_dequant = np.float32(amax / _fp8_max(fmt))
    data = _encode_fp8(x / scale_dequant, fmt)
    scale_tensor = np.asarray(scale_dequant, dtype=np.float32)
    return ScaledTensor(data=data, scale=scale_tensor, recipe=recipe, layout=PerTensor())


def _dequantize_per_tensor_fp8(scaled: ScaledTensor) -> np.ndarray:
    decoded = _decode_fp8(scaled.data, scaled.recipe.fp8_format)
    return decoded * np.float32(scaled.scale)


# ---------- Per-block-K FP8 (Float8BlockScaling) ----------


def _cast_block_fp8(x: np.ndarray, recipe: Float8BlockScaling) -> ScaledTensor:
    if x.ndim < 2:
        raise ValueError(
            f"Float8BlockScaling requires x.ndim >= 2, got {x.ndim}"
        )
    B = recipe.block_k
    fmt = recipe.fp8_format
    fmt_max = _fp8_max(fmt)
    amax = block_amax(x, B)  # shape (..., M, K//B)
    scale_dequant = (amax / fmt_max).astype(np.float32)

    # Apply per-block scaling.
    K = x.shape[-1]
    blocks = x.reshape(x.shape[:-1] + (K // B, B))
    scaled_blocks = blocks / scale_dequant[..., None]
    scaled_x = scaled_blocks.reshape(x.shape)
    data = _encode_fp8(scaled_x, fmt)

    return ScaledTensor(
        data=data, scale=scale_dequant, recipe=recipe, layout=PerBlockK(block_size=B)
    )


def _dequantize_block_fp8(scaled: ScaledTensor) -> np.ndarray:
    fmt = scaled.recipe.fp8_format
    B = scaled.recipe.block_k
    decoded = _decode_fp8(scaled.data, fmt)
    K = decoded.shape[-1]
    blocks = decoded.reshape(decoded.shape[:-1] + (K // B, B))
    out = blocks * scaled.scale[..., None]
    return out.reshape(decoded.shape).astype(np.float32)


# ---------- MXFP8 (32-element blocks, E8M0 scale) ----------


def _cast_mxfp8(x: np.ndarray, recipe: MXFP8BlockScaling) -> ScaledTensor:
    if x.ndim < 2:
        raise ValueError(f"MXFP8BlockScaling requires x.ndim >= 2, got {x.ndim}")
    B = recipe.block_size  # 32
    fmt = recipe.fp8_format
    fmt_max = _fp8_max(fmt)

    # Per-32-element block amax along last dim.
    amax = block_amax(x, B)  # (..., M, K // 32)
    scale_dequant_fp32 = (amax / fmt_max).astype(np.float32)

    # MX scale is a *power of two* — round to the nearest representable E8M0.
    scale_e8m0 = quantize_e8m0_scale(scale_dequant_fp32)
    # Reconstruct the actual power-of-two scale that was stored.
    actual_scale = dequantize_e8m0_scale(scale_e8m0)

    K = x.shape[-1]
    blocks = x.reshape(x.shape[:-1] + (K // B, B))
    scaled_blocks = blocks / actual_scale[..., None]
    data = _encode_fp8(scaled_blocks.reshape(x.shape), fmt)

    return ScaledTensor(
        data=data,
        scale=scale_e8m0,
        recipe=recipe,
        layout=PerBlockMN(block_m=1, block_n=B),
    )


def _dequantize_mxfp8(scaled: ScaledTensor) -> np.ndarray:
    fmt = scaled.recipe.fp8_format
    B = scaled.recipe.block_size
    decoded = _decode_fp8(scaled.data, fmt)
    scale_fp32 = dequantize_e8m0_scale(scaled.scale)
    K = decoded.shape[-1]
    blocks = decoded.reshape(decoded.shape[:-1] + (K // B, B))
    # MXFP8 layout is PerBlockMN(1, B), so scale shape is (..., M, K // B);
    # broadcast the inner block dim.
    out = blocks * scale_fp32[..., None]
    return out.reshape(decoded.shape).astype(np.float32)


# ---------- NVFP4 (16-element blocks, FP8 E4M3 scale) ----------


_E4M3_MIN_SUBNORMAL = np.float32(2.0 ** -9)  # smallest non-zero E4M3 value


def _cast_nvfp4(x: np.ndarray, recipe: NVFP4BlockScaling) -> ScaledTensor:
    if x.ndim < 2:
        raise ValueError(f"NVFP4BlockScaling requires x.ndim >= 2, got {x.ndim}")
    B = recipe.block_size  # 16
    fp4_max = E2M1_MAX

    amax = block_amax(x, B)  # (..., M, K // 16)
    scale_dequant_fp32 = (amax / fp4_max).astype(np.float32)

    # NVFP4 stores the scale as FP8 E4M3. If a block's scale underflows
    # the smallest E4M3 value (≈ 2^-9), encoding round-trips it to 0;
    # we'd then divide by zero. Floor to the smallest representable
    # E4M3 subnormal — matches what NVIDIA Blackwell hardware does for
    # near-zero blocks.
    scale_e4m3 = encode_e4m3(scale_dequant_fp32)
    actual_scale = decode_e4m3(scale_e4m3)
    actual_scale = np.where(actual_scale > 0, actual_scale, _E4M3_MIN_SUBNORMAL)

    K = x.shape[-1]
    blocks = x.reshape(x.shape[:-1] + (K // B, B))
    scaled_blocks = blocks / actual_scale[..., None]
    data = encode_e2m1(scaled_blocks.reshape(x.shape))

    return ScaledTensor(
        data=data,
        scale=scale_e4m3,
        recipe=recipe,
        layout=PerBlockMN(block_m=1, block_n=B),
    )


def _dequantize_nvfp4(scaled: ScaledTensor) -> np.ndarray:
    B = scaled.recipe.block_size
    decoded = decode_e2m1(scaled.data)
    scale_fp32 = decode_e4m3(scaled.scale)
    K = decoded.shape[-1]
    blocks = decoded.reshape(decoded.shape[:-1] + (K // B, B))
    out = blocks * scale_fp32[..., None]
    return out.reshape(decoded.shape).astype(np.float32)


# ---------- INT4 (group_size, fp16/bf16/fp32 scale) ----------


def _int4_dtype(name: str) -> np.dtype:
    return {"fp16": np.float16, "bf16": np.float32, "fp32": np.float32}[name]


def _cast_int4(x: np.ndarray, recipe: INT4Scaling) -> ScaledTensor:
    if x.ndim < 2:
        raise ValueError(f"INT4Scaling requires x.ndim >= 2, got {x.ndim}")
    G = recipe.group_size
    max_int = 7 if recipe.signed else 15
    amax = block_amax(x, G)  # (..., M, K // G)
    scale_dequant_fp32 = (amax / max_int).astype(np.float32)

    K = x.shape[-1]
    blocks = x.reshape(x.shape[:-1] + (K // G, G))
    scaled_blocks = blocks / scale_dequant_fp32[..., None]
    data = encode_int4(scaled_blocks.reshape(x.shape), signed=recipe.signed)

    # Down-cast scale to the recipe's declared dtype.
    scale_dtype = _int4_dtype(recipe.scale_dtype)
    scale_stored = scale_dequant_fp32.astype(scale_dtype)

    return ScaledTensor(
        data=data,
        scale=scale_stored,
        recipe=recipe,
        layout=PerBlockK(block_size=G),
    )


def _dequantize_int4(scaled: ScaledTensor) -> np.ndarray:
    G = scaled.recipe.group_size
    decoded = decode_int4(scaled.data, signed=scaled.recipe.signed)
    scale_fp32 = np.asarray(scaled.scale, dtype=np.float32)
    K = decoded.shape[-1]
    blocks = decoded.reshape(decoded.shape[:-1] + (K // G, G))
    out = blocks * scale_fp32[..., None]
    return out.reshape(decoded.shape).astype(np.float32)
