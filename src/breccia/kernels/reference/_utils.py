"""Internal helpers shared across reference kernels."""

from __future__ import annotations

import numpy as np


_AMAX_EPS = 1e-10


def block_amax(x: np.ndarray, block_size: int, axis: int = -1) -> np.ndarray:
    """Per-block absolute maximum along ``axis``.

    Reshapes the requested axis to ``(num_blocks, block_size)``, takes the
    abs-max over the inner dim, and returns an array of shape
    ``x.shape[:axis] + (num_blocks,) + x.shape[axis+1:]`` (with ``axis``
    replaced by the per-block axis). A small epsilon is mixed in to keep
    the result away from zero so downstream divisions never blow up.
    """
    if axis != -1 and axis != x.ndim - 1:
        raise NotImplementedError(
            "block_amax v0.0.1 supports last-axis blocking only"
        )
    K = x.shape[-1]
    if K % block_size != 0:
        raise ValueError(
            f"last-axis size {K} must be divisible by block_size {block_size}"
        )
    reshaped = x.reshape(x.shape[:-1] + (K // block_size, block_size))
    amax = np.max(np.abs(reshaped), axis=-1)
    return np.maximum(amax, _AMAX_EPS).astype(np.float32)


def tensor_amax(x: np.ndarray) -> float:
    """Whole-tensor abs-max, floored at a small epsilon."""
    v = float(np.max(np.abs(x))) if x.size else 0.0
    return max(v, _AMAX_EPS)


def quantize_e8m0_scale(scale_fp32: np.ndarray) -> np.ndarray:
    """Encode a positive fp32 scale to OCP MX's E8M0 (uint8) format.

    E8M0 stores ``e`` as an 8-bit unsigned integer where the represented
    value is ``2 ** (e - 127)``. We round ``log2(scale)`` to the nearest
    integer, offset by 127, and clip to ``[0, 255]``.
    """
    s = np.maximum(scale_fp32, _AMAX_EPS)
    e_raw = np.round(np.log2(s)).astype(np.int32) + 127
    e_clipped = np.clip(e_raw, 0, 255)
    return e_clipped.astype(np.uint8)


def dequantize_e8m0_scale(scale_uint8: np.ndarray) -> np.ndarray:
    """Reverse :func:`quantize_e8m0_scale`."""
    e = scale_uint8.astype(np.int32) - 127
    return np.float32(2.0) ** e.astype(np.float32)
