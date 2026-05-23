"""Reference scaled matmul: dequantize both sides, FP32 matmul.

This is the correctness baseline. Optimized scaled-matmul kernels (Triton,
TransformerEngine, cuBLAS FP8 GEMM) fuse dequantization into the kernel
and operate on the low-precision data directly, accumulating in FP32.
Mathematically, the two are equivalent up to FP32 accumulation order.

For per-tensor scaling, this equivalence is exact in infinite precision:

    A_dequant @ B_dequant = (A_data * a_scale) @ (B_data * b_scale)
                         = (a_scale * b_scale) * (A_data @ B_data)

For per-block-K scaling, the block scale factors fold inside the K-sum:

    out[m, n] = sum_k A_data[m, k] * a_scale[m, k//B] *
                      B_data[k, n] * b_scale[k//B, n]
"""

from __future__ import annotations

from typing import Any

import numpy as np

from breccia._core import ScaledTensor

from .cast import dequantize


def matmul(a: Any, b: Any, out_dtype: Any = np.float32) -> np.ndarray:
    """Scaled matmul. Accepts ScaledTensor or raw array on either side.

    Parameters
    ----------
    a : ScaledTensor | ndarray
        Left operand, shape ``(..., M, K)``.
    b : ScaledTensor | ndarray
        Right operand, shape ``(..., K, N)``.
    out_dtype
        Output dtype. Defaults to float32.

    Returns
    -------
    ndarray of shape ``(..., M, N)``.

    Notes
    -----
    For a ScaledTensor whose recipe stores the scale per-block-K (e.g.,
    ``Float8BlockScaling``, ``INT4Scaling``), the K dim of the data must
    be the **last** axis. Weight matrices that store per-output-channel
    scales should be passed as ``(N, K)`` (last-axis = K) and the matmul
    re-orients them internally; v0.0.1 only supports ``(..., M, K) @
    (..., K, N)``, so callers must lay out weights in K-last form.
    """
    a_fp = dequantize(a) if isinstance(a, ScaledTensor) else np.asarray(a, dtype=np.float32)
    b_fp = dequantize(b) if isinstance(b, ScaledTensor) else np.asarray(b, dtype=np.float32)

    if a_fp.shape[-1] != b_fp.shape[-2]:
        raise ValueError(
            f"matmul shape mismatch: a.shape[-1]={a_fp.shape[-1]} != "
            f"b.shape[-2]={b_fp.shape[-2]}"
        )

    out = a_fp @ b_fp
    return out.astype(out_dtype)
