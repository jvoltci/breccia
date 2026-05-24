"""Triton GPU kernels for breccia.

These kernels require a CUDA-capable device and Triton installed
(``pip install triton``). Import is gated so this package can be loaded
on platforms without CUDA (CPU-only, Apple Silicon, etc.) without
raising at import time — the actual kernels raise informative errors
only when invoked.
"""

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

if TRITON_AVAILABLE:
    from .scaled_matmul import scaled_matmul_triton

    __all__ = ["scaled_matmul_triton", "TRITON_AVAILABLE"]
else:
    __all__ = ["TRITON_AVAILABLE"]
