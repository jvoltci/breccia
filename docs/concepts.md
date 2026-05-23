# Concepts

The mental model behind `breccia.ScaledTensor`. Read this once and the rest
of the library stops surprising you.

## The data structure

A `ScaledTensor` is **four things**:

```
data    — low-precision bytes (FP8 native dtype, or uint8 for FP4/INT4)
scale   — scale tensor that gives the data its high-precision meaning
recipe  — a ScalingRecipe describing HOW the data was quantized
layout  — a Layout describing HOW the scale maps to data blocks
```

For a 2-D weight matrix of shape `(M, K) = (8, 128)` quantized with
`Float8CurrentScaling`:

```
data:   shape (8, 128), dtype uint8       # one byte per FP8 value
scale:  shape (), dtype float32           # single scalar
recipe: Float8CurrentScaling(fp8_format="E4M3")
layout: PerTensor()
```

That's it. Every operation in breccia is defined on top of this quadruple.

## Why this representation

Every framework today reinvents block-scaled low-precision in incompatible
ways. The fragmentation is real:

| Approach | Pros | Cons |
| --- | --- | --- |
| **NVIDIA TransformerEngine** | hardware-tuned for Hopper / Blackwell | NVIDIA-only, 4 non-composable recipe classes |
| **PyTorch torchao** | autograd-friendly | PyTorch-only |
| **DeepSeek-v3 format** | proven recipe for FP8 training | private to one repo |
| **HuggingFace + custom**| huge ecosystem | every model rolls its own scale convention |
| **FP8-Flow-MoE, COAT**| each solves a specific gap | each is incompatible |

`ScaledTensor` is what they all converge on, exposed as one neutral type
that round-trips with each of them via [`breccia.bridges`](bridges.md).

## Why `scale` is the *dequantization* scale

Convention varies across libraries. breccia picks the **dequantization scale**:

```
data_decoded     = decode(data)                 # decode the low-precision bytes
high_precision_x = data_decoded * scale         # multiply by stored scale to recover
```

So if `x` had `amax = 10` and we're encoding to FP8 E4M3 (whose max is 448):

```
scale = amax / fp8_max  = 10 / 448  = 0.0223
data  = encode(x / scale) = encode(x * 44.8)    # values now in FP8 range
```

This convention matches the OCP MX standard and is exactly how hardware
scaled-matmul kernels (FP8 GEMM, NVFP4 GEMM) consume the scale tensor.
NVIDIA TransformerEngine calls the same thing `_scale_inv`.

## The invariants

`ScaledTensor` enforces at construction:

| Invariant | Raised |
| --- | --- |
| `data` is array-like (has `.shape`, `.dtype`) | `TypeError("data must be array-like ...")` |
| `scale` is array-like | `TypeError("scale must be array-like ...")` |
| `data.ndim >= 1` | `ValueError("data must be at least 1-D ...")` |
| `recipe is not None` | `ValueError("recipe is required ...")` |
| `layout is not None` | `ValueError("layout is required ...")` |
| `layout.validate(data, scale)` succeeds | `ValueError` from the layout |

The Layout's `.validate(data, scale)` method is the single source of truth
for the data-vs-scale shape relationship. See [recipes.md](recipes.md) and
[api.md](api.md) for the layouts.

## Recipes are pure metadata

A recipe is **declarative configuration** — it carries the format identifier
and any recipe-specific parameters (block size, amax history length,
zero-point semantics) but contains *no quantization behavior itself*. All
behavior lives in [`breccia.cast`](api.md) and dispatches on recipe type.

This means recipes are:

- **Frozen dataclasses** — immutable and hashable
- **Comparable** — two recipes with the same fields are equal
- **JSON-serializable** — used by the HuggingFace bridge to round-trip
  through safetensors metadata
- **Hardware-portable** — the same recipe can be implemented by any backend

## Layouts are how the scale maps to data blocks

Four layouts cover today's recipe fragmentation:

| Layout | Scale shape for data `(M, K)` | Used by |
| --- | --- | --- |
| `PerTensor` | `()` — single scalar | DelayedScaling, Float8CurrentScaling |
| `PerBlockK(B)` | `(M, K // B)` | Float8BlockScaling, INT4Scaling |
| `PerChannel` | `(M,)` or `(M, 1)` | INT4 row-wise quantization |
| `PerBlockMN(Bm, Bn)` | `(M // Bm, K // Bn)` | MXFP8 (1, 32), NVFP4 (1, 16) |

A layout's `.validate(data, scale)` enforces this contract. The validator
is called from `ScaledTensor.__post_init__`, so an inconsistent
`(data, scale)` pair fails at construction time — not at the next matmul.

## The relationship to NVIDIA TransformerEngine

TransformerEngine's `Float8Tensor` carries `_data` (uint8 bytes) and
`_scale_inv` (the dequantization scale). The mapping to breccia is
direct:

```python
from breccia.bridges import from_transformer_engine, to_transformer_engine

# TE → breccia
st = from_transformer_engine(te_tensor)
# st.data is te_tensor._data, st.scale is te_tensor._scale_inv

# breccia → TE
te_t = to_transformer_engine(st)
```

See [bridges.md](bridges.md).

## The relationship to OCP MX (Microscaling)

OCP MX is the open standard for block-scaled low-precision (FP8, FP6, FP4)
with an E8M0 (uint8 exponent-only) scale. breccia's `MXFP8BlockScaling`
recipe with `PerBlockMN(1, 32)` layout is the OCP MX MXFP8 format.

The scale stored is an E8M0 byte; multiplying by `2^(byte - 127)` gives the
floating-point dequantization scale.

## Dispatch and backend selection

`ScaledTensor` is backend-agnostic: `data` and `scale` can be NumPy arrays,
PyTorch tensors, or MLX arrays. The functions in `breccia.*` detect the
backend at runtime and dispatch.

```python
type(st.data).__module__   # 'numpy' | 'torch' | 'mlx.core'
```

This is implemented with three small helper predicates inside `breccia._core`:

```python
def _is_torch(x): return type(x).__module__.startswith("torch")
def _is_mlx(x):   return type(x).__module__.startswith("mlx")
def _is_jax(x):   mod = type(x).__module__; return mod.startswith("jax") or mod.startswith("jaxlib")
```

Anything not torch / MLX / JAX falls into the NumPy code path. No plugin
registry; the dispatch table is small enough to inline. See
[architecture.md](architecture.md) for the rationale.

## When NOT to use breccia

- **You have an FP16 / BF16 workload that fits in memory.** breccia trades
  precision for memory and bandwidth. If you don't need the trade, don't pay
  the cost.
- **You need autograd-tracked low-precision tensors that subclass
  `torch.Tensor`.** breccia's `ScaledTensor` is a plain dataclass, not a
  Tensor subclass. The autograd "lives in" `data` (which can be a Tensor)
  but the wrapper itself doesn't participate in autograd graphs.
- **You only target one vendor's hardware.** If you're NVIDIA-only,
  TransformerEngine is more tuned today. If you're PyTorch-only, torchao
  ships native autograd integration. breccia is the substrate for when you
  need *both* (and AMD, Trainium, TPU…).

## Reading further

- [api.md](api.md) — exact signatures for every public function and class
- [recipes.md](recipes.md) — when to use each of the 6 recipes
- [bridges.md](bridges.md) — migration paths from TE / torchao / HF
- [architecture.md](architecture.md) — how the package is laid out internally
