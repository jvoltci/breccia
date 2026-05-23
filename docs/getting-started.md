# Getting started

## Install

```bash
pip install breccia                    # NumPy backend
pip install "breccia[torch]"           # + PyTorch
pip install "breccia[mlx]"             # + MLX (Apple Silicon, Metal)
pip install "breccia[torch,mlx]"       # + both
pip install "breccia[bridges]"         # + safetensors for HF bridge
```

breccia supports Python 3.10+. NumPy is the only required dependency.
PyTorch, MLX, JAX, Triton, and safetensors are all optional and
detected at runtime.

## Your first breccia program

```python
import numpy as np
import breccia

# Quantize a tensor to FP8 with per-tensor current scaling.
x = np.random.randn(8, 128).astype(np.float32)
st = breccia.cast(x, breccia.Float8CurrentScaling())

print(st)
# breccia.ScaledTensor(shape=(8, 128), data_dtype=uint8, scale_shape=(),
#                     recipe=Float8CurrentScaling, layout=PerTensor)

# Recover the high-precision tensor.
x_recovered = breccia.dequantize(st)
print(np.allclose(x, x_recovered, atol=0.1))  # True within FP8 precision
```

That's the core idea: one type (`ScaledTensor`), six recipes that say *how*
the data was quantized, four layouts that say *how the scale maps to the data*.

## Three patterns you'll use constantly

### 1. Quantize for compute, dequantize when you need precision back

```python
import breccia

# Block-scaled FP8 weights — better dynamic range than per-tensor.
w_q = breccia.cast(weight, breccia.Float8BlockScaling(block_k=128))

# Use it in matmul (output is float32):
y = breccia.matmul(activations, w_q)
```

### 2. Pick a recipe by use case

```python
# Training with TE-style amax history
breccia.DelayedScaling(fp8_format="E4M3")

# Inference with synchronous amax
breccia.Float8CurrentScaling(fp8_format="E4M3")

# DeepSeek-v3-style block-scaled weights
breccia.Float8BlockScaling(block_k=128)

# OCP MX microscaling (32-element blocks, E8M0 scale)
breccia.MXFP8BlockScaling()

# NVIDIA Blackwell NVFP4 (16-element blocks, FP4 + FP8 scale)
breccia.NVFP4BlockScaling()

# INT4 weight-only (GPTQ / AWQ family)
breccia.INT4Scaling(group_size=128)
```

See [recipes.md](recipes.md) for the full guide.

### 3. Bridge to/from your existing pipeline

```python
import breccia.bridges as bridges

# DeepSeek-v3 style: raw (data, scale) buffers → ScaledTensor
st = bridges.from_deepseek_v3(data, scale, block_k=128)

# Save to safetensors with scale metadata
bridges.save_safetensors({"w0": st, "w1": st2}, "weights.safetensors")
loaded = bridges.load_safetensors("weights.safetensors")

# Zero-copy cross-framework via DLPack
st_torch = bridges.from_dlpack(st_numpy, framework="torch")

# TransformerEngine round-trip (Linux + CUDA)
st = bridges.from_transformer_engine(te_float8tensor)
te_t = bridges.to_transformer_engine(st)
```

See [bridges.md](bridges.md) for every bridge and its constraints.

## Five-line round-trip demo

```python
import numpy as np, breccia
x = np.random.randn(4, 256).astype(np.float32)
st = breccia.cast(x, breccia.Float8BlockScaling(block_k=128))
y = breccia.dequantize(st)
print(f"max abs err: {np.max(np.abs(x - y)):.4f}")
# max abs err: 0.0421   (≈ E4M3 precision)
```

## What to read next

- If you want to understand the design: [concepts.md](concepts.md)
- If you need to pick a recipe: [recipes.md](recipes.md)
- If you're integrating with TE/torchao/HF: [bridges.md](bridges.md)
- If you're looking up a specific function: [api.md](api.md)
- If you want to contribute: [architecture.md](architecture.md) +
  [CONTRIBUTING](../CONTRIBUTING.md)
