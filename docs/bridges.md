# Bridges & migration

breccia ships five bridges, one per major external convention. Each one
handles a specific direction of interop and degrades gracefully when its
optional dep is missing.

| Bridge | Direction | External dep | Status |
| --- | --- | --- | --- |
| TransformerEngine | TE Float8Tensor ↔ ScaledTensor | `transformer-engine` (Linux + CUDA) | ✅ v0.0.1 |
| torchao | AffineQuantizedTensor ↔ ScaledTensor | `torchao` | ✅ v0.0.1 (symmetric only) |
| HuggingFace safetensors | safetensors file ↔ ScaledTensor dict | `safetensors` | ✅ v0.0.1 |
| DLPack | zero-copy across NumPy / PyTorch / MLX / JAX | none (built-in) | ✅ v0.0.1 |
| DeepSeek-v3 | (data, scale) buffers ↔ ScaledTensor | none | ✅ v0.0.1 |

---

## TransformerEngine

```python
from breccia.bridges import from_transformer_engine, to_transformer_engine

# TE → breccia
st = from_transformer_engine(te_float8tensor)

# breccia → TE (per-tensor recipes only in v0.0.1)
te_t = to_transformer_engine(st)
```

The mapping is essentially a buffer pass-through: TE's `_data` becomes
`st.data`, and TE's `_scale_inv` becomes `st.scale` (same dequantization
convention).

**Recipe support**: bridge defaults to `DelayedScaling` for `from_*`. The
`to_transformer_engine` direction supports `DelayedScaling` and
`Float8CurrentScaling` in v0.0.1.

**Installation**: TransformerEngine installs on Linux + CUDA only. On
other platforms, calling the bridge functions raises a clear
`ImportError` with install instructions.

---

## torchao

```python
from breccia.bridges import from_torchao, to_torchao

# torchao AffineQuantizedTensor → breccia
st = from_torchao(aqt)

# breccia → torchao (INT4 symmetric only)
aqt = to_torchao(st)
```

v0.0.1 supports **symmetric** quantization only (zero_point = 0).
Asymmetric INT4/INT8 lands in v0.1.

`from_torchao` infers the group size from `int_data.shape[-1] // scale.shape[-1]`
when scale is 2-D. Override with the `recipe=` argument if your layout
differs.

---

## HuggingFace safetensors

```python
from breccia.bridges import save_safetensors, load_safetensors

# Save a dict of ScaledTensors to a single file
save_safetensors(
    {"w_q": w_quantized, "w_k": k_quantized},
    "model.safetensors",
    extra_metadata={"model_version": "v0.0.1"},
)

# Load back, recipes + layouts reconstructed from metadata
loaded = load_safetensors("model.safetensors")
# loaded["w_q"] is a ScaledTensor with the original recipe/layout
```

### Format convention

For each name in the input dict:

- `f"{name}.data"` — the data buffer (torch tensor in the safetensors file)
- `f"{name}.scale"` — the scale buffer
- `f"{name}.config"` (in metadata) — JSON of recipe + layout

The breccia metadata lives in safetensors' `metadata` dict, which other
safetensors readers silently ignore. So a breccia safetensors file is
backwards-compatible with any safetensors loader (it just gets raw
data/scale tensors, no recipe info).

### Multiple tensors per file

The function packs as many ScaledTensors as you pass. Each name gets its
own `.data` / `.scale` / `.config` triple.

### Skipping tensors without config

`load_safetensors` only returns tensors that have all three of
`.data`, `.scale`, and `.config`. Plain tensors in the same file are
silently ignored.

---

## DLPack

```python
from breccia.bridges import to_dlpack, from_dlpack

# Move a ScaledTensor's buffers to another framework (zero-copy when possible)
st_torch = from_dlpack(st_numpy, framework="torch")
st_mlx   = from_dlpack(st_torch, framework="mlx")

# Raw capsules (for advanced use)
data_capsule, scale_capsule = to_dlpack(st_torch)
```

`framework` accepts: `"numpy"`, `"torch"`, `"mlx"`, `"jax"`. Recipe and
layout are unchanged; only `data` and `scale` are moved.

DLPack is the standard cross-framework zero-copy protocol. Most
framework `from_dlpack` implementations want the source *tensor* (with a
`__dlpack__` method) rather than the raw capsule; the `from_dlpack`
helper above passes the tensor directly.

---

## DeepSeek-v3

```python
from breccia.bridges import from_deepseek_v3, to_deepseek_v3

# Raw DeepSeek-v3 FP8 buffers (block_k=128) → ScaledTensor
st = from_deepseek_v3(data, scale, block_k=128, fp8_format="E4M3")

# Inverse
data, scale = to_deepseek_v3(st)
```

DeepSeek-v3 ships FP8 E4M3 weights with per-128-element block scaling.
That's exactly `Float8BlockScaling(block_k=128)` + `PerBlockK(128)`.
The bridge is a thin wrapper around `from_buffer` that picks the right
recipe and layout.

---

## When to write a new bridge

If you're integrating breccia with a library that has its own quantized
tensor type, the recipe for a new bridge is:

1. Identify which breccia recipe matches (or extend the recipe set if
   yours doesn't fit any). See [recipes.md](recipes.md).
2. Identify which layout matches (or add a new one to
   [`breccia/layouts.py`](../src/breccia/layouts.py)).
3. Write a `_yourlib.py` in [`breccia/bridges/`](../src/breccia/bridges/)
   with `from_yourlib(...)` and `to_yourlib(...)` functions. Use lazy
   imports — `breccia.bridges` should be importable even if your library
   isn't installed.
4. Add a row to the bridges/__init__.py exports.
5. Add tests in `tests/test_bridges.py`. Skip when the external dep is
   absent (`pytest.importorskip`).
6. Document in this file.

See [`_deepseek.py`](../src/breccia/bridges/_deepseek.py) for the
simplest example (no external dep) and
[`_huggingface.py`](../src/breccia/bridges/_huggingface.py) for the most
complex (custom file format + metadata).
