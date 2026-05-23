"""Reference and optimized kernels for ScaledTensor operations.

- ``breccia.kernels.reference`` — slow but correct, used as CI ground truth.
- ``breccia.kernels.triton`` — fast GPU kernels (import-gated on Triton
  availability; safe to import on macOS / CPU-only environments).
"""
