"""Migrating from NVIDIA TransformerEngine to breccia.

If you have TE's ``Float8Tensor`` instances (e.g., from a fine-tuned
TE training run), the breccia bridge wraps them as ``ScaledTensor`` so
the rest of your stack can be framework-neutral.

This example shows the bridge call patterns. TransformerEngine installs
only on Linux + CUDA, so the bridge calls are wrapped in try/except for
demonstration on other platforms; on a real CUDA Linux box the calls
work zero-copy in both directions.

Run with: python examples/04_te_migration.py
"""

from dataclasses import dataclass

import torch

import breccia
from breccia.bridges import from_transformer_engine, to_transformer_engine


# ---------- Mock TE Float8Tensor with the same attributes the real one has ----------


@dataclass
class _MockTEFloat8Tensor:
    _data: torch.Tensor       # uint8 FP8 bytes
    _scale_inv: torch.Tensor  # fp32 dequantization scale
    _fp8_dtype: str = "E4M3"


# ---------- TE → breccia ----------

print("Step 1: bridge a TE Float8Tensor into breccia.ScaledTensor")
print("-" * 60)

fake_te = _MockTEFloat8Tensor(
    _data=torch.randint(0, 256, (4, 128), dtype=torch.uint8),
    _scale_inv=torch.tensor(0.05, dtype=torch.float32),
)

try:
    st = from_transformer_engine(fake_te)
    print(f"  → {st}")
    print("  (zero-copy: st.data is the TE tensor's _data)")
except ImportError as e:
    print(f"  TransformerEngine not installed (expected on non-CUDA Linux):")
    print(f"    {e}")
    print("  On a CUDA Linux box, from_transformer_engine() would return:")
    print("    breccia.ScaledTensor(shape=(4, 128), recipe=DelayedScaling, layout=PerTensor)")

# ---------- breccia: do work that doesn't need TE ----------

print("\nStep 2: use breccia primitives independent of TE")
print("-" * 60)

W_fp32 = torch.randn(4, 8, dtype=torch.float32)
W_st = breccia.cast(W_fp32, breccia.Float8CurrentScaling())

A_fp32 = torch.randn(16, 4, dtype=torch.float32)
A_st = breccia.cast(A_fp32, breccia.Float8CurrentScaling())

Y = breccia.matmul(A_st, W_st)
Y_ref = A_fp32 @ W_fp32

cos = torch.dot(Y.flatten(), Y_ref.flatten()) / (
    torch.linalg.norm(Y) * torch.linalg.norm(Y_ref) + 1e-12
)
print(f"  matmul shape: {tuple(Y.shape)}")
print(f"  cosine sim vs FP32: {float(cos):.4f}")

# ---------- breccia → TE ----------

print("\nStep 3: bridge back to TE-compatible Float8Tensor")
print("-" * 60)
try:
    te_tensor = to_transformer_engine(W_st)
    print(f"  → {type(te_tensor).__name__}")
except ImportError as e:
    print(f"  TransformerEngine not installed (expected on non-CUDA Linux):")
    print(f"    {e}")

print(
    "\nDone. On Linux + CUDA + TE, both bridge directions are zero-copy and "
    "you can mix TE / breccia / torchao at will."
)
