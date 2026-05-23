# Architecture

How the breccia package is laid out, what each module is responsible for,
and why the design choices were made.

This document is for people who want to contribute to breccia or
understand the codebase deeply. For just using the library, start with
[getting-started.md](getting-started.md) and [concepts.md](concepts.md).

## Repository layout

```
breccia/
├── pyproject.toml                 # package metadata, optional deps
├── README.md                      # one-screen pitch, status table
├── LICENSE                        # Apache-2.0
├── CONTRIBUTING.md                # how to propose changes
├── CHANGELOG.md                   # version history
├── CLAUDE.md                      # behavioral guidelines for code review
├── .github/workflows/ci.yml       # CI on Python 3.10/3.11/3.12 × Ubuntu + macOS
├── docs/                          # this directory
├── src/breccia/                   # the package source (src-layout)
│   ├── __init__.py                # public API surface
│   ├── _core.py                   # ScaledTensor + backend predicates + from_buffer
│   ├── recipes.py                 # 6 ScalingRecipe variants
│   ├── layouts.py                 # 4 Layout variants
│   ├── _formats.py                # FP8/FP4/INT4 LUTs + encode/decode + nibble pack
│   ├── bridges/                   # interop with external libs
│   │   ├── __init__.py
│   │   ├── _transformer_engine.py
│   │   ├── _torchao.py
│   │   ├── _huggingface.py
│   │   ├── _dlpack.py
│   │   └── _deepseek.py
│   └── kernels/
│       ├── __init__.py
│       ├── reference/             # slow but correct, CI ground truth
│       │   ├── __init__.py
│       │   ├── cast.py            # cast/dequantize/requantize per recipe
│       │   ├── matmul.py          # scaled matmul
│       │   └── _utils.py
│       └── triton/                # fast GPU kernels (import-gated)
│           ├── __init__.py
│           └── scaled_matmul.py
├── tests/                         # pytest suite
│   ├── test_core.py               # ScaledTensor invariants + construction
│   ├── test_recipes.py            # 6 recipes
│   ├── test_layouts.py            # 4 layouts
│   ├── test_formats.py            # FP8/FP4/INT4 round-trip
│   ├── test_cast.py               # cast quality per recipe
│   ├── test_matmul.py             # scaled matmul correctness
│   ├── test_bridges.py            # all 5 bridges
│   ├── test_properties.py         # hypothesis property tests
│   ├── test_torch.py              # PyTorch backend
│   └── test_mlx.py                # MLX backend
├── benchmarks/
│   ├── README.md
│   ├── bench_memory.py            # memory savings per recipe
│   ├── bench_accuracy.py          # accuracy degradation
│   └── modal_bench.py             # H100 scaled-matmul vs cuBLAS FP8 GEMM
└── examples/
    ├── 01_quickstart.py
    ├── 02_recipe_portable_train.py
    ├── 03_checkpoint_with_scale.py
    └── 04_te_migration.py
```

## Module responsibilities

### `breccia._core`

The center of the library. Defines:

- The `ScaledTensor` dataclass and its invariants
- Construction helper `from_buffer`
- Backend predicates: `_is_torch(x)`, `_is_mlx(x)`, `_is_jax(x)`

Everything else in the package imports from `_core`. There is exactly
one source of truth for the data structure.

### `breccia.recipes`

Six frozen dataclasses, each representing one quantization recipe.
Recipes carry configuration only — no quantization behavior. The
quantization algorithm lives in `kernels/reference/cast.py` and
dispatches on recipe type.

### `breccia.layouts`

Four frozen dataclasses, each representing one (data, scale) shape
relationship. Each implements `validate(data, scale) -> None`, called
from `ScaledTensor.__post_init__`.

### `breccia._formats`

Bit-level FP8 / FP4 / INT4 encode and decode. Two 256-entry uint8→float32
lookup tables for FP8 (E4M3 and E5M2), one 16-entry table for FP4 E2M1,
plus direct INT4 encode/decode. Nibble packing helpers for compact 4-bit
storage in checkpoint bridges.

### `breccia.bridges`

One file per external convention; each one imports the external dep
lazily so `import breccia.bridges` does not require any optional dep.

- `_transformer_engine.py` — TE Float8Tensor round-trip
- `_torchao.py` — AffineQuantizedTensor round-trip
- `_huggingface.py` — safetensors save/load with scale metadata
- `_dlpack.py` — cross-framework zero-copy
- `_deepseek.py` — DeepSeek-v3 FP8 block-scaled weight format

### `breccia.kernels.reference`

Pure-Python (NumPy / PyTorch via round-trip / MLX via round-trip)
implementations of `cast`, `dequantize`, `matmul`, `requantize`. These
are intentionally slow and obviously correct. They serve as ground
truth in the test suite — every optimized kernel must produce
numerically equivalent output.

The dispatch chain in each entry point is consistent:

```python
def cast(x, recipe):
    if _is_torch(x):  return _cast_torch(x, recipe)
    if _is_mlx(x):    return _cast_mlx(x, recipe)
    return _cast_numpy(np.asarray(x, dtype=np.float32), recipe)
```

### `breccia.kernels.triton`

The fast GPU kernels.

`__init__.py` is **import-gated**: it tries `import triton` and sets
`TRITON_AVAILABLE = True/False`. On macOS / CPU-only environments the
import is silent and no kernel names are exported. This lets
`import breccia` and `import breccia.kernels.triton` work everywhere.

## Design choices and rationale

### Why a dataclass, not a Tensor subclass

`ScaledTensor` is `@dataclass(frozen=True)`. The frozenness gives:

- **Hashability** — the same ScaledTensor value hashes the same way
  (useful for memoization across kernel calls)
- **Safety** — code that receives a ScaledTensor can't accidentally
  mutate scale and break invariants
- **Backend-portability** — no inheritance from `torch.Tensor` /
  `jax.numpy.ndarray` / `mx.array` means we don't pick a side

The dataclass shape (four fields, no methods beyond properties) signals
that the type is a value object. All operations live in the surrounding
module as free functions.

### Why backend dispatch instead of a unified namespace

Considered using
[`array-api-compat`](https://github.com/data-apis/array-api-compat) and
writing one set of operations against the Array API. Rejected for v0.0:

1. **Bit-level encode/decode is not in the Array API.** Mapping float32
   to FP8 / FP4 / INT4 nibbles is intrinsically per-element bit
   manipulation; the Array API has no facility for it.
2. **Per-backend idiomatic paths exist.** PyTorch has `torch.float8_e4m3fn`
   native; MLX doesn't. Writing one path that pretends both have the
   same dtype set hides real perf differences.
3. **The dispatch is small.** Three backends × a handful of operations
   is ~300 lines of branching. Maintaining that is cheaper than the
   indirection through array-api-compat.

We can revisit when the Array API spec covers FP8 / FP4 dtypes (no
movement on this as of late 2025) or when a fourth backend lands.

### Why no autograd subclass

`ScaledTensor` is not a `torch.Tensor` subclass and does not register
with PyTorch's autograd. The `data` field IS a torch tensor that
participates in autograd normally; the wrapper is just a typed view.

This is intentional. Tying ScaledTensor to PyTorch's autograd would:

- Make the type PyTorch-specific, breaking the cross-framework story
- Force users to choose between breccia and other tensor-wrapping types

For training (v0.1), the cast and matmul operations will provide a
straight-through estimator wrapper. The wrapper is opt-in; the bare
ScaledTensor is autograd-neutral.

### Why import-gate the Triton module

Triton requires a CUDA-capable GPU and PTX. On macOS / Apple Silicon /
CPU-only servers, `import triton` raises `ImportError`. If breccia
eagerly imported triton, the library would be unimportable on those
platforms — including the developer's M5 dev machine.

The gate (`try: import triton`) lets the same package serve both
groups. `TRITON_AVAILABLE` is the contract.

### Why the dequantization scale convention

OCP MX and NVIDIA TransformerEngine both store the dequantization scale
(multiply data by scale to recover the high-precision value). Hardware
scaled-matmul kernels (cuBLAS FP8 GEMM, Blackwell NVFP4 GEMM) consume
the dequantization scale directly.

Storing the forward scale would force every kernel boundary to invert
it. The dequant convention matches the hardware's data flow.

## Test infrastructure

The test suite uses pytest and Hypothesis. Test files:

- `tests/test_core.py` — ScaledTensor invariants + construction
- `tests/test_recipes.py` — 6 recipes
- `tests/test_layouts.py` — 4 layouts + integration with ScaledTensor
- `tests/test_formats.py` — FP8 / FP4 / INT4 + nibble packing
- `tests/test_cast.py` — cast quality per recipe (cosine similarity)
- `tests/test_matmul.py` — scaled matmul correctness
- `tests/test_bridges.py` — TE / torchao / HF / DLPack / DeepSeek
- `tests/test_properties.py` — Hypothesis generative invariants
- `tests/test_torch.py` — PyTorch backend
- `tests/test_mlx.py` — MLX backend

Total: 192 tests as of v0.0.1. CI runs them on Python 3.10/3.11/3.12
(Ubuntu) and 3.11 (macOS), plus runs the examples and memory benchmark
to catch bitrot.

The Triton kernel is not tested in CI (CI doesn't have a CUDA GPU). Its
correctness is verified by `benchmarks/modal_bench.py`, which runs on
an H100 via Modal on demand.

## Performance characteristics (v0.0.1)

- Memory: a ScaledTensor of logical shape `(M, K)` quantized to FP8
  uses `M * K + scale_overhead` bytes, where `scale_overhead` depends on
  the layout (1 byte for PerTensor, `(M, K // B) * scale_dtype_size`
  for block scaling).
- Throughput: the reference kernels are Python-loop-based and intentionally
  slow. They're correctness-only. Production speed comes from the Triton
  kernel (v0.1).

See [benchmarks.md](benchmarks.md) for the methodology and numbers.

## Adding a new backend

To add JAX (the most likely next backend after v0.0.1):

1. Add `_is_jax(x)` to `breccia._core` (already done — it's a no-op
   gate for v0.1 to fill in).
2. Add JAX branches in `kernels/reference/cast.py`'s dispatch:
   `_cast_jax` and `_dequantize_jax`. The pattern: convert to NumPy,
   run the reference, wrap result back as jnp arrays.
3. Update `kernels/reference/matmul.py` with a JAX path.
4. Add `tests/test_jax.py` mirroring `tests/test_torch.py`.
5. Add `jax` extra to `pyproject.toml` (already declared).
6. Update README's status table and docs.

The torch integration in `kernels/reference/cast.py:_cast_torch` is the
template. JAX should be near-identical (with `jnp.asarray` instead of
`torch.as_tensor`).

## Adding a new recipe

When a new low-precision format becomes important (e.g., FP6 OCP MX):

1. Add the format's bit-level encode/decode to `_formats.py`.
2. Add the recipe dataclass to `recipes.py` (frozen, hashable, with
   `name` class attribute).
3. Add the new recipe to the dispatch in
   `kernels/reference/cast.py:_cast_numpy`.
4. Add tests in `test_recipes.py` and `test_cast.py`.
5. Update [recipes.md](recipes.md) and [api.md](api.md).

## Versioning

breccia follows semver:

- **v0.0.x** — pre-alpha; the API may break between any two commits
- **v0.1.0** — first beta; the public API in `breccia.*` becomes stable
- **v1.0.0** — first stable release; backward-incompatible changes
  require a major bump

The v0.0 → v0.1 milestone gate is: the Triton scaled-matmul kernel
must hit ≤ 1.2× of cuBLAS FP8 GEMM with PASS correctness on H100.

## Where decisions get made

- **Public API surface** — `src/breccia/__init__.py` is the contract.
  Anything not re-exported there is private.
- **Behavior changes** — should be discussed in a GitHub Discussion
  before being implemented; PRs that change behavior without prior
  discussion will be asked to retroactively open one.
- **New backends or kernels** — a short Discussion thread (1-2
  paragraphs) is enough.

See [../CONTRIBUTING.md](../CONTRIBUTING.md) for the contribution flow.
