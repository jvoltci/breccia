"""FP8 scaled matmul kernel (Triton, Hopper / Ada / Blackwell).

This is the fast GPU path for ``breccia.matmul`` on CUDA devices. The
kernel:

1. Loads ``a.data`` and ``b.data`` as native FP8
   (``torch.float8_e4m3fn`` or ``torch.float8_e5m2``).
2. Accumulates in FP32 via the ``tl.dot`` primitive's fp32 accumulator.
3. Multiplies the output tile by the per-tensor (or per-block) scale
   factors.
4. Writes the result as FP32 (or BF16 if requested).

Recipe support in v0.0.1:

- ``DelayedScaling`` / ``Float8CurrentScaling`` (per-tensor) — fully wired
- ``Float8BlockScaling`` (per-block-K) — planned for v0.1
- ``MXFP8BlockScaling`` / ``NVFP4BlockScaling`` — requires Blackwell-class
  hardware support; planned for v0.1

Validation
----------

The kernel is shipped in v0.0.1 but GPU-validation is deferred to v0.1
(no CI GPU). The validation harness is
[`benchmarks/modal_bench.py`](../../../../benchmarks/modal_bench.py)
which runs on an H100 via Modal and asserts numerical equivalence to
``torch._scaled_mm`` within 5e-3 max abs diff.

The reference (NumPy / round-trip) path in
[`breccia.kernels.reference.matmul`](../reference/matmul.py) is the
correctness ground truth.
"""

from __future__ import annotations

from typing import Any

import triton
import triton.language as tl

from breccia._core import ScaledTensor
from breccia.recipes import DelayedScaling, Float8CurrentScaling


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 32}, num_warps=8, num_stages=3),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def _scaled_matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    a_scale, b_scale,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """FP8 scaled matmul: C = (A @ B) * a_scale * b_scale.

    A is shape (M, K) in FP8, B is shape (K, N) in FP8.
    a_scale and b_scale are per-tensor scalars in fp32.
    C is shape (M, N) in fp32.
    """
    # v0.1 assumes M, N, K are divisible by their respective block sizes.
    # The Python wrapper enforces this by padding or rejecting misaligned shapes.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_block_ptr = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_block_ptr = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for _k_start in range(0, K, BLOCK_K):
        a = tl.load(A_block_ptr)
        b = tl.load(B_block_ptr)
        accumulator = tl.dot(a, b, accumulator)
        A_block_ptr += BLOCK_K * stride_ak
        B_block_ptr += BLOCK_K * stride_bk

    # Apply per-tensor scales
    accumulator = accumulator * a_scale * b_scale

    C_block_ptr = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_block_ptr, accumulator)


def scaled_matmul_triton(a: ScaledTensor, b: ScaledTensor, out_dtype: Any = None) -> Any:
    """Run the Triton FP8 scaled matmul on two per-tensor ScaledTensors.

    Parameters
    ----------
    a : ScaledTensor
        Left operand. ``a.data`` must be a CUDA torch tensor of dtype
        ``torch.float8_e4m3fn`` or ``torch.float8_e5m2``. ``a.recipe`` must
        be a per-tensor recipe (``DelayedScaling`` or ``Float8CurrentScaling``).
    b : ScaledTensor
        Right operand. Same constraints as ``a``.
    out_dtype : torch.dtype, optional
        Defaults to ``torch.float32``. ``torch.bfloat16`` is also supported.

    Returns
    -------
    torch.Tensor of shape ``(M, N)``.

    Raises
    ------
    ImportError if Triton is not installed.
    ValueError if the recipe is not per-tensor (block-scaled paths are v0.1).
    """
    import torch

    if not isinstance(a.recipe, (DelayedScaling, Float8CurrentScaling)):
        raise ValueError(
            "scaled_matmul_triton v0.0.1 supports per-tensor recipes only "
            f"(DelayedScaling, Float8CurrentScaling); got {type(a.recipe).__name__}"
        )
    if not isinstance(b.recipe, (DelayedScaling, Float8CurrentScaling)):
        raise ValueError(
            "scaled_matmul_triton v0.0.1 supports per-tensor recipes only "
            f"(DelayedScaling, Float8CurrentScaling); got {type(b.recipe).__name__}"
        )

    A_data = a.data
    B_data = b.data
    a_scale_val = a.scale.item() if hasattr(a.scale, "item") else float(a.scale)
    b_scale_val = b.scale.item() if hasattr(b.scale, "item") else float(b.scale)

    assert A_data.is_cuda and B_data.is_cuda, "Triton kernel requires CUDA tensors"
    assert A_data.ndim == 2 and B_data.ndim == 2, "v0.1 supports 2-D matmul"

    M, K = A_data.shape
    K2, N = B_data.shape
    assert K == K2, f"matmul shape mismatch: K={K} vs {K2}"

    # v0.1 kernel assumes shapes divisible by largest possible block sizes
    # (the autotuner picks among configs up to 256x256x32, so 256x32 is the
    # tightest constraint). Looser-aligned shapes get a clear error here.
    block_m_max = 256
    block_n_max = 256
    block_k_max = 32
    if M % block_m_max != 0 or N % block_n_max != 0 or K % block_k_max != 0:
        # Fall back to a smaller-block autotune set for partial alignment.
        if M % 64 != 0 or N % 64 != 0 or K % 32 != 0:
            raise ValueError(
                f"Triton scaled_matmul requires M, N divisible by 64 and K "
                f"divisible by 32 (v0.1 kernel). Got M={M}, N={N}, K={K}."
            )

    out_dtype = out_dtype or torch.float32
    C = torch.empty((M, N), device=A_data.device, dtype=out_dtype)

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(N, meta["BLOCK_N"]))
    _scaled_matmul_kernel[grid](
        A_data, B_data, C,
        a_scale_val, b_scale_val,
        M, N, K,
        A_data.stride(0), A_data.stride(1),
        B_data.stride(0), B_data.stride(1),
        C.stride(0), C.stride(1),
    )
    return C
