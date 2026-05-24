"""Quickstart: cast a tensor to FP8, matmul, dequantize.

Run with: python examples/01_quickstart.py
"""

import numpy as np

import breccia

rng = np.random.default_rng(0)

# Activations: shape (M, K) = (8, 128).
A = rng.standard_normal((8, 128)).astype(np.float32)

# Weights: shape (K, N) = (128, 64).
W = rng.standard_normal((128, 64)).astype(np.float32)

# Quantize both to FP8 (per-tensor current scaling).
A_q = breccia.cast(A, breccia.Float8CurrentScaling())
W_q = breccia.cast(W, breccia.Float8CurrentScaling())

print(f"A: {A_q}")
print(f"W: {W_q}")

# Scaled matmul. Output is float32.
Y = breccia.matmul(A_q, W_q)

# Compare to the unquantized result.
Y_ref = A @ W

cos = np.dot(Y.ravel(), Y_ref.ravel()) / (np.linalg.norm(Y) * np.linalg.norm(Y_ref))
max_abs_err = np.max(np.abs(Y - Y_ref))
print(f"\nOutput shape: {Y.shape}")
print(f"Cosine similarity to FP32 reference: {cos:.4f}")
print(f"Max abs error: {max_abs_err:.4f}")
