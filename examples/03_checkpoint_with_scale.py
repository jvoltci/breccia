"""Save and load a quantized model checkpoint with scale metadata.

Demonstrates the HuggingFace safetensors bridge: quantize several
tensors, save them in one file, load them back, and verify the round-
trip preserves the recipe and layout (so dequantize produces identical
output).

Run with: python examples/03_checkpoint_with_scale.py
"""

import tempfile
from pathlib import Path

import numpy as np

import breccia
from breccia.bridges import save_safetensors, load_safetensors

rng = np.random.default_rng(123)

# ---------- Build a "model": three quantized weight tensors ----------

w_attn = breccia.cast(
    rng.standard_normal((64, 256)).astype(np.float32),
    breccia.Float8BlockScaling(block_k=128),
)
w_ffn = breccia.cast(
    rng.standard_normal((64, 128)).astype(np.float32),
    breccia.Float8CurrentScaling(),
)
w_gate = breccia.cast(
    rng.standard_normal((64, 256)).astype(np.float32),
    breccia.INT4Scaling(group_size=128),
)

model = {
    "attn.W": w_attn,
    "ffn.W": w_ffn,
    "gate.W": w_gate,
}

# ---------- Save ----------

with tempfile.TemporaryDirectory() as tmp:
    path = str(Path(tmp) / "model.safetensors")
    save_safetensors(
        model,
        path,
        extra_metadata={
            "model_version": "v0.0.1",
            "framework": "breccia",
        },
    )
    print(f"Saved {len(model)} tensors to {path}")
    print(f"  File size: {Path(path).stat().st_size} bytes")

    # ---------- Load ----------
    loaded = load_safetensors(path)

print(f"\nLoaded keys: {sorted(loaded)}")

# ---------- Verify round-trip ----------

print("\nRound-trip check (deqaunt before vs after save+load):")
for name in sorted(model):
    out_before = np.asarray(breccia.dequantize(model[name]))
    out_after = np.asarray(breccia.dequantize(loaded[name]))
    max_abs = float(np.max(np.abs(out_before - out_after)))
    same_recipe = type(model[name].recipe).__name__ == type(loaded[name].recipe).__name__
    print(
        f"  {name:10s}  recipe={type(loaded[name].recipe).__name__:25s}"
        f"  max_abs_diff={max_abs:.2e}  recipe_preserved={same_recipe}"
    )
