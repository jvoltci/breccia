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

from breccia._core import ScaledTensor, _is_torch, _is_mlx

from .cast import dequantize


def _to_numpy(t: Any) -> np.ndarray:
    if _is_torch(t):
        import torch

        return t.detach().to(torch.float32).cpu().numpy()
    if _is_mlx(t):
        return np.asarray(np.array(t), dtype=np.float32)
    return np.asarray(t, dtype=np.float32)


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
    # If either operand carries torch tensors, route the result back to torch
    # (lets users do scaled-matmul end-to-end without leaving torch).
    a_data = a.data if isinstance(a, ScaledTensor) else a
    b_data = b.data if isinstance(b, ScaledTensor) else b
    is_torch_path = _is_torch(a_data) or _is_torch(b_data)
    is_mlx_path = _is_mlx(a_data) or _is_mlx(b_data)

    if is_torch_path:
        import torch

        a_t = dequantize(a) if isinstance(a, ScaledTensor) else a
        b_t = dequantize(b) if isinstance(b, ScaledTensor) else b
        if not _is_torch(a_t):
            a_t = torch.from_numpy(np.asarray(a_t, dtype=np.float32))
        if not _is_torch(b_t):
            b_t = torch.from_numpy(np.asarray(b_t, dtype=np.float32))
        if a_t.shape[-1] != b_t.shape[-2]:
            raise ValueError(
                f"matmul shape mismatch: a.shape[-1]={a_t.shape[-1]} != "
                f"b.shape[-2]={b_t.shape[-2]}"
            )
        return (a_t.float() @ b_t.float()).to(_torch_dtype(out_dtype))

    if is_mlx_path:
        import mlx.core as mx

        a_x = dequantize(a) if isinstance(a, ScaledTensor) else a
        b_x = dequantize(b) if isinstance(b, ScaledTensor) else b
        if not _is_mlx(a_x):
            a_x = mx.array(np.asarray(a_x, dtype=np.float32))
        if not _is_mlx(b_x):
            b_x = mx.array(np.asarray(b_x, dtype=np.float32))
        if a_x.shape[-1] != b_x.shape[-2]:
            raise ValueError(
                f"matmul shape mismatch: a.shape[-1]={a_x.shape[-1]} != "
                f"b.shape[-2]={b_x.shape[-2]}"
            )
        return a_x @ b_x

    a_fp = dequantize(a) if isinstance(a, ScaledTensor) else np.asarray(a, dtype=np.float32)
    b_fp = dequantize(b) if isinstance(b, ScaledTensor) else np.asarray(b, dtype=np.float32)

    if a_fp.shape[-1] != b_fp.shape[-2]:
        raise ValueError(
            f"matmul shape mismatch: a.shape[-1]={a_fp.shape[-1]} != "
            f"b.shape[-2]={b_fp.shape[-2]}"
        )

    out = a_fp @ b_fp
    return out.astype(out_dtype)


def _torch_dtype(np_dtype: Any) -> Any:
    """Map a NumPy dtype to the equivalent torch dtype."""
    import torch

    mapping = {
        np.float32: torch.float32,
        np.float64: torch.float64,
        np.float16: torch.float16,
    }
    # Convert dtype-like to concrete dtype.
    key = np.dtype(np_dtype).type
    return mapping.get(key, torch.float32)
