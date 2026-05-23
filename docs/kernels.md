# Kernels

How breccia implements `cast`, `dequantize`, `matmul`, and `requantize`
across backends, and how to add a new optimized kernel.

## The hierarchy

```
breccia.cast        ─┐
breccia.dequantize  ─┤  public dispatch entry points (in breccia._core / __init__)
breccia.matmul      ─┤
breccia.requantize  ─┘
                  ↓
breccia.kernels.reference.{cast, matmul}   slow but correct (NumPy/Torch/MLX)
breccia.kernels.triton.{scaled_matmul}     fast (CUDA + Triton)
```

The reference impls are the **contract**. Optimized kernels (Triton,
hand-written CUDA, etc.) must produce numerically equivalent output
within the recipe's declared tolerance — verified in CI.

## Reference kernels

Each reference kernel lives in
[`src/breccia/kernels/reference/`](../src/breccia/kernels/reference/):

| File | Contents |
| --- | --- |
| `cast.py` | `cast(x, recipe)`, `dequantize(scaled)`, `requantize(scaled, recipe)` |
| `matmul.py` | `matmul(a, b, out_dtype=np.float32)` |
| `_utils.py` | `block_amax`, `tensor_amax`, `quantize_e8m0_scale`, `dequantize_e8m0_scale` |

### Dispatch by recipe

`cast.py`'s public `cast` function dispatches on backend *first*, then on
recipe type:

```python
def cast(x, recipe):
    if _is_torch(x):  return _cast_torch(x, recipe)
    if _is_mlx(x):    return _cast_mlx(x, recipe)
    x_np = np.asarray(x, dtype=np.float32)
    return _cast_numpy(x_np, recipe)

def _cast_numpy(x_np, recipe):
    if isinstance(recipe, (DelayedScaling, Float8CurrentScaling)):
        return _cast_per_tensor_fp8(x_np, recipe)
    if isinstance(recipe, Float8BlockScaling):
        return _cast_block_fp8(x_np, recipe)
    # ... one branch per recipe
```

The torch and MLX paths round-trip through NumPy for v0.0.1 — they're
correctness-only, not performance paths. (Performance lives in the
Triton kernel.)

### The cast forward path

For each recipe, the algorithm is:

1. Compute the per-{tensor, block, channel, group} `amax`.
2. Derive the dequantization scale: `scale = amax / fmt_max`.
3. Quantize `x / scale` to the recipe's data format.

Dequantize reverses this: `x_recovered = decode(data) * scale`.

### The cast backward path (autograd)

`ScaledTensor` is **not** a `torch.Tensor` subclass — it's a plain
dataclass. The autograd "lives in" `data` (which can be a torch tensor
that participates in PyTorch's autograd normally).

For training, the cast operation in v0.0.1 is **non-differentiable**
through the quantization step (round-to-nearest has zero gradient).
Straight-through estimator (STE) wrappers for autograd land in v0.1.

For inference (no gradients needed), the v0.0.1 cast / matmul path works
end-to-end via the round-trip.

## Triton kernels (`breccia.kernels.triton`)

Import-gated. On platforms without Triton (macOS, CPU-only Linux), the
module sets `TRITON_AVAILABLE = False` and exports nothing, so
`import breccia.kernels.triton` is always safe.

### `scaled_matmul_triton(a, b)` (v0.0.1, ships untested on GPU)

FP8 scaled matmul for Hopper / Ada / Blackwell. The kernel:

1. Loads `a.data` and `b.data` as native FP8 (`torch.float8_e4m3fn` or
   `torch.float8_e5m2`).
2. Accumulates in FP32 via the `tl.dot` primitive's fp32 accumulator.
3. Multiplies the output tile by the per-tensor or per-block scale
   factors (depending on the recipe).
4. Returns an FP32 (or BF16 if requested) output tile.

The kernel is written but **not GPU-validated** in v0.0.1 — the user
needs CUDA hardware to exercise it. The Modal benchmark script
([`benchmarks/modal_bench.py`](../benchmarks/modal_bench.py)) handles
validation when run.

### Why import-gating matters

Triton requires CUDA + PTX. On macOS / Apple Silicon / CPU-only servers,
`import triton` raises `ImportError`. If breccia eagerly imported
triton, the library would be unimportable on those platforms — including
the developer's M5 / dev machine.

The gate (`try: import triton`) lets the same package serve both groups.
`TRITON_AVAILABLE` is the contract.

## Adding a new kernel

Suppose you want to add a Metal-native scaled matmul for Apple Silicon.

1. **Reference impl** — make sure `matmul` produces correct output for
   the recipe combinations you'll accelerate. The reference is the
   ground truth.
2. **New kernel directory** — `src/breccia/kernels/metal/` with its own
   import-gate (`try: import mlx.fast`).
3. **Implementation** — write the kernel and a Python wrapper that takes
   ScaledTensor arguments.
4. **Test** — add `tests/test_metal.py` that asserts numerical
   equivalence to the reference within the recipe's tolerance.
5. **Benchmark** — `benchmarks/bench_metal.py` measuring throughput vs
   the reference baseline.
6. **Docs** — a section in this file describing the kernel's shape
   support, expected hardware, and known limitations.

## Performance characteristics (v0.0.1)

| Path | Where it runs | Speed |
| --- | --- | --- |
| NumPy reference | any CPU | Correctness-only |
| PyTorch reference | any device (CPU after round-trip) | Correctness-only |
| MLX reference | Apple Silicon | Correctness-only |
| Triton kernel (M17) | Hopper / Ada / Blackwell | Targets cuBLAS FP8 GEMM parity |

The v0.0.1 release ships the *primitive and bridges*. Hardware
acceleration is the v0.1 work — see the
[CHANGELOG](../CHANGELOG.md) and [benchmarks](benchmarks.md) for the
roadmap.
