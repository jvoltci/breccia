# API reference

Every public function and class in `breccia`, with signatures and behavior.

The library is intentionally small: one data class, four operations,
six recipes, four layouts, five bridges, and the reference + Triton
kernel modules.

---

## `breccia.ScaledTensor`

```python
@dataclass(frozen=True)
class ScaledTensor:
    data: Any        # low-precision bytes
    scale: Any       # scale tensor; shape depends on layout
    recipe: Any      # ScalingRecipe instance
    layout: Any      # Layout instance
```

### Construction

`ScaledTensor` is `@dataclass(frozen=True)`. You can construct directly:

```python
import numpy as np
from breccia import ScaledTensor, Float8CurrentScaling
from breccia.layouts import PerTensor

st = ScaledTensor(
    data=np.zeros((4, 8), dtype=np.uint8),
    scale=np.float32(1.0),
    recipe=Float8CurrentScaling(),
    layout=PerTensor(),
)
```

…but in practice you go through `breccia.cast` or `breccia.from_buffer`.

### Invariants (enforced in `__post_init__`)

| Invariant | Raised |
| --- | --- |
| `data` has `.shape` and `.dtype` | `TypeError("data must be array-like ...")` |
| `scale` has `.shape` and `.dtype` | `TypeError("scale must be array-like ...")` |
| `data.ndim >= 1` | `ValueError("data must be at least 1-D ...")` |
| `recipe is not None` | `ValueError("recipe is required ...")` |
| `layout is not None` | `ValueError("layout is required ...")` |
| `layout.validate(data, scale)` succeeds | `ValueError` from the layout |

### Properties

| Property | Type | Returns |
| --- | --- | --- |
| `shape` | `tuple` | `data.shape` as a tuple |
| `ndim` | `int` | `data.ndim` |
| `data_dtype` | dtype | `data.dtype` |
| `scale_dtype` | dtype | `scale.dtype` |

### Dunder methods

- `repr(st)` → `"breccia.ScaledTensor(shape=..., data_dtype=..., scale_shape=..., recipe=..., layout=...)"`

---

## Core operations

### `breccia.cast(x, recipe) -> ScaledTensor`

Quantize a high-precision tensor.

| Parameter | Type | Notes |
| --- | --- | --- |
| `x` | NumPy / PyTorch / MLX array | Any backend; the result's `data` and `scale` match the input backend. |
| `recipe` | ScalingRecipe | One of the 6 recipes. |

Block-scaled recipes (`Float8BlockScaling`, `MXFP8BlockScaling`,
`NVFP4BlockScaling`, `INT4Scaling`) require `x.ndim >= 2` and the last
dim divisible by the block size.

### `breccia.dequantize(scaled) -> array`

Recover a high-precision tensor from a ScaledTensor. The output backend
matches the input (`ScaledTensor.data`'s backend).

### `breccia.matmul(a, b, out_dtype=np.float32) -> array`

Scaled matmul. Each operand can be a `ScaledTensor` or a raw array.

| Parameter | Type |
| --- | --- |
| `a` | `ScaledTensor` or array of shape `(..., M, K)` |
| `b` | `ScaledTensor` or array of shape `(..., K, N)` |
| `out_dtype` | NumPy dtype for the output (default `np.float32`) |

Returns an array of shape `(..., M, N)`.

The reference implementation dequantizes both operands and runs an FP32
matmul. The Triton kernel (M17) fuses dequantization for FP8.

### `breccia.requantize(scaled, new_recipe) -> ScaledTensor`

Convert a ScaledTensor between recipes. Implemented in v0.0.1 as
`cast(dequantize(scaled), new_recipe)`.

### `breccia.from_buffer(data, scale, recipe, layout) -> ScaledTensor`

Zero-copy constructor for when you already have quantized buffers (e.g.,
from a checkpoint or vendor library). No reductions, no quantization —
just wraps the buffers in a typed primitive.

---

## Recipes (`breccia.*`)

All recipes are frozen dataclasses, hashable, and JSON-serializable.

### `DelayedScaling(fp8_format="E4M3", amax_history_len=16, margin=0)`

TE-style delayed scaling. See [recipes.md → DelayedScaling](recipes.md#delayedscaling).

### `Float8CurrentScaling(fp8_format="E4M3")`

Per-tensor amax computed each step. See [recipes.md → Float8CurrentScaling](recipes.md#float8currentscaling).

### `Float8BlockScaling(fp8_format="E4M3", block_k=128)`

Per-K-block FP8 scaling. See [recipes.md → Float8BlockScaling](recipes.md#float8blockscaling).

### `MXFP8BlockScaling(fp8_format="E4M3", block_size=32)`

OCP MX microscaling FP8. `block_size` is fixed at 32 by the spec.
See [recipes.md → MXFP8BlockScaling](recipes.md#mxfp8blockscaling).

### `NVFP4BlockScaling(fp4_format="E2M1", block_size=16, scale_format="E4M3")`

NVIDIA Blackwell NVFP4. All three fields are fixed by the hardware spec.
See [recipes.md → NVFP4BlockScaling](recipes.md#nvfp4blockscaling).

### `INT4Scaling(group_size=128, signed=True, scale_dtype="fp16")`

GPTQ / AWQ family INT4. See [recipes.md → INT4Scaling](recipes.md#int4scaling).

---

## Layouts (`breccia.*`)

All layouts implement `.validate(data, scale) -> None`, called from
`ScaledTensor.__post_init__`.

### `PerTensor()`

Single scalar scale. Used by `DelayedScaling` and `Float8CurrentScaling`.

### `PerBlockK(block_size=128)`

One scale per `block_size`-element block along the last axis. Used by
`Float8BlockScaling` and `INT4Scaling`.

### `PerChannel()`

One scale per output row. Accepts `scale.shape == (M,)` or
`scale.shape == (..., M, 1)`. Used by INT4 row-wise quantization.

### `PerBlockMN(block_m=1, block_n=32)`

2-D grid of scales. Used by `MXFP8BlockScaling` (1, 32) and
`NVFP4BlockScaling` (1, 16).

---

## Bridges (`breccia.bridges`)

See [bridges.md](bridges.md) for full docs and per-bridge constraints.

| Function | Direction |
| --- | --- |
| `from_transformer_engine(te_t, recipe=None)` | TE → breccia |
| `to_transformer_engine(scaled)` | breccia → TE |
| `from_torchao(aqt, recipe=None)` | torchao → breccia |
| `to_torchao(scaled)` | breccia → torchao |
| `save_safetensors(tensors, path, extra_metadata=None)` | breccia → safetensors file |
| `load_safetensors(path)` | safetensors file → breccia |
| `to_dlpack(scaled)` | breccia → `(data_capsule, scale_capsule)` |
| `from_dlpack(scaled, framework)` | move buffers to a new framework |
| `from_deepseek_v3(data, scale, block_k=128, fp8_format="E4M3")` | DeepSeek-v3 → breccia |
| `to_deepseek_v3(scaled)` | breccia → `(data, scale)` |

---

## Kernels

### `breccia.kernels.reference`

| Function | What |
| --- | --- |
| `cast(x, recipe)` | reference (NumPy) quantize per recipe |
| `dequantize(scaled)` | reference dequantize |
| `requantize(scaled, recipe)` | `cast(dequantize(scaled), recipe)` |
| `matmul(a, b, out_dtype=np.float32)` | reference scaled matmul |

These are also exposed at the package root: `breccia.cast`, etc.

### `breccia.kernels.triton`

Import-gated. On platforms without Triton (macOS, CPU-only), this module
sets `TRITON_AVAILABLE = False` and exports nothing.

| Function | What |
| --- | --- |
| `scaled_matmul_triton(a, b)` | FP8 scaled matmul on Hopper / Ada (M17; GPU validation pending) |

See [kernels.md](kernels.md) for kernel design and validation.

---

## `breccia.__version__`

The version string. v0.0.x is pre-alpha. The public API stabilizes at v0.1.

---

## What's NOT in the public API

These exist in `breccia._core` but are not exported:

- `_is_torch(x)`, `_is_mlx(x)`, `_is_jax(x)` — backend dispatch predicates

Anything starting with `_` in any module is private and subject to change
between any two commits in the v0.0 line.
