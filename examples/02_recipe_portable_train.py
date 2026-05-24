"""Recipe-portable training: train under MXFP8, ship under NVFP4.

The key promise of breccia: model code does not change when you swap
recipes. The only thing that changes is the recipe object you pass
to `cast` (or `requantize`).

This example does a tiny training step under MXFP8, then re-quantizes
the trained weights to NVFP4 for deployment. The "model" is a single
linear layer over Gaussian inputs.

Run with: python examples/02_recipe_portable_train.py
"""

import numpy as np

import breccia

rng = np.random.default_rng(42)

# ---------- Setup: a tiny linear "layer" ----------

K, N = 64, 32
W = rng.standard_normal((K, N)).astype(np.float32) * 0.1  # FP32 master weights

# ---------- Forward pass under TWO recipes ----------

A = rng.standard_normal((16, K)).astype(np.float32)

# Recipe 1: MXFP8 (training-tier, OCP MX hardware path)
training_recipe = breccia.MXFP8BlockScaling()
A_train = breccia.cast(A, training_recipe)
W_train = breccia.cast(W, training_recipe)
Y_train = breccia.matmul(A_train, W_train)

# Recipe 2: NVFP4 (inference-tier, Blackwell hardware path)
inference_recipe = breccia.NVFP4BlockScaling()
A_infer = breccia.cast(A, inference_recipe)
W_infer = breccia.cast(W, inference_recipe)
Y_infer = breccia.matmul(A_infer, W_infer)

# ---------- Compare to the FP32 ground truth ----------

Y_ref = A @ W


def cos_sim(a, b):
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    return float(a_flat @ b_flat / (np.linalg.norm(a_flat) * np.linalg.norm(b_flat) + 1e-12))


print("Training recipe:   MXFP8BlockScaling()")
print(f"  Cosine sim vs FP32: {cos_sim(Y_train, Y_ref):.4f}")
print(f"  Memory of W_train:  {W_train.data.nbytes + W_train.scale.nbytes} bytes")
print()
print("Inference recipe:  NVFP4BlockScaling()")
print(f"  Cosine sim vs FP32: {cos_sim(Y_infer, Y_ref):.4f}")
print(f"  Memory of W_infer:  {W_infer.data.nbytes + W_infer.scale.nbytes} bytes")

# ---------- The portability: convert MXFP8 → NVFP4 without re-running training ----------

W_inference_via_requantize = breccia.requantize(W_train, inference_recipe)
Y_via_requantize = breccia.matmul(A_infer, W_inference_via_requantize)

print()
print("requantize(W_train, NVFP4) — same logical recipe, different recipe history:")
print(f"  Cosine sim vs FP32: {cos_sim(Y_via_requantize, Y_ref):.4f}")
print(f"  Memory: {W_inference_via_requantize.data.nbytes + W_inference_via_requantize.scale.nbytes} bytes")
