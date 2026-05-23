# Recipes

breccia ships six `ScalingRecipe` variants in v0.0.1. Each is declarative
metadata — the *what*, not the *how*. The quantization algorithm dispatches
on recipe type inside [`breccia.cast`](api.md).

## At a glance

| Recipe | Format | Block / scope | Scale dtype | When to reach for it |
| --- | --- | --- | --- | --- |
| `DelayedScaling` | FP8 E4M3 or E5M2 | per-tensor | fp32 | FP8 training with TE-style amax history |
| `Float8CurrentScaling` | FP8 E4M3 or E5M2 | per-tensor (synchronous) | fp32 | FP8 inference; FP8 forward in simple training loops |
| `Float8BlockScaling` | FP8 E4M3 or E5M2 | per-128-block along K | fp32 | FP8 weights with varying row magnitudes (DeepSeek-v3) |
| `MXFP8BlockScaling` | FP8 (E4M3/E5M2) + E8M0 scale | per-32-block along K | E8M0 (uint8) | Hardware microscaling on Blackwell / future MX-capable chips |
| `NVFP4BlockScaling` | FP4 E2M1 + FP8 E4M3 scale | per-16-block along K | E4M3 (uint8) | Blackwell-class FP4 inference; FP4 training |
| `INT4Scaling` | INT4 (signed or unsigned) | per-group along K | fp16 / bf16 / fp32 | INT4 weight-only inference (GPTQ / AWQ family) |

## DelayedScaling

```python
from breccia import DelayedScaling

DelayedScaling(
    fp8_format="E4M3",       # "E4M3" or "E5M2"
    amax_history_len=16,     # how many previous amax values the training loop retains
    margin=0,                # power-of-two exponent margin (TE default = 0)
)
```

The classic TransformerEngine recipe. Uses a rolling history of recent
`amax` values to compute the next scale, avoiding a synchronous reduction
on every cast.

The history itself **lives outside the recipe** (in your training loop's
optimizer state). The recipe is portable metadata.

**Use when**: you're training in FP8 and want TE-comparable throughput.

## Float8CurrentScaling

```python
Float8CurrentScaling(fp8_format="E4M3")
```

The simplest FP8 recipe: compute `amax(x)` each step, scale by
`fp8_max / amax`. Synchronous (every cast is a reduction over the tensor),
so slightly slower than delayed scaling in training, but trivial to reason
about and exact for the current tensor.

**Use when**: FP8 inference; quick experiments; small-batch training where
the reduction is cheap.

## Float8BlockScaling

```python
Float8BlockScaling(fp8_format="E4M3", block_k=128)
```

One scale per `block_k`-element block along the K (contraction) dim.
Better dynamic range than per-tensor scaling when row magnitudes vary
substantially — common in attention weights and FFN gates.

DeepSeek-v3 ships its FP8 weights with `block_k=128` and `E4M3` format;
[`breccia.bridges.from_deepseek_v3`](bridges.md) is a thin wrapper.

**Use when**: weight matrices where some rows are systematically larger
than others (post-norm features, gated linear units).

## MXFP8BlockScaling

```python
MXFP8BlockScaling(fp8_format="E4M3", block_size=32)  # block_size fixed at 32 by spec
```

The OCP MX microscaling standard for FP8. Two key properties fixed by
the standard:

- `block_size == 32` (hardware-locked)
- scale is **E8M0** — an 8-bit unsigned exponent encoding a power-of-two
  scale `2^(byte - 127)`

This is the format Blackwell-class hardware and future MX-aware silicon
accelerate natively. The trade-off vs `Float8BlockScaling`: the scale
is power-of-two (less precise) but takes only 1 byte per block (smaller
overhead).

**Use when**: targeting MX-capable hardware where the power-of-two scale
is a hardware-native operation.

## NVFP4BlockScaling

```python
NVFP4BlockScaling(
    fp4_format="E2M1",    # fixed
    block_size=16,        # fixed by NVIDIA Blackwell spec
    scale_format="E4M3",  # fixed: FP8 E4M3 scale
)
```

NVIDIA Blackwell's NVFP4 format. FP4 data (only 16 representable values
per element) with a per-16-block FP8 E4M3 scale (256 representable values
per block).

This is the densest format breccia ships — 4 bits per value. Cosine
similarity vs FP32 stays above 0.95 on Gaussian-distributed inputs;
real workloads have shown convergence-quality FP4 training (see the
"Pretraining LLMs with NVFP4" arXiv paper).

**Use when**: targeting Blackwell hardware for inference; experimental
FP4 training.

## INT4Scaling

```python
INT4Scaling(
    group_size=128,       # values per shared scale
    signed=True,          # -8..7 (signed) or 0..15 (unsigned)
    scale_dtype="fp16",   # "fp16" | "bf16" | "fp32"
)
```

INT4 quantization with one scale per group along the K dim. This is the
GPTQ / AWQ family of recipes used by ~every open-weight model that ships
4-bit quantized weights.

v0.0.1 supports symmetric quantization only (zero-point = 0). Asymmetric
quantization (with a non-zero zero-point) is on the v0.1 list.

**Use when**: INT4 weight-only inference; loading GPTQ/AWQ-quantized
checkpoints; deploying to memory-constrained inference hardware.

## Choosing a recipe

A rough decision tree:

```
Am I doing FP4? ──── NVFP4BlockScaling (Blackwell)

Am I doing INT4? ─── INT4Scaling

Am I doing FP8?
├─ training, want TE-style amax history → DelayedScaling
├─ training, simpler / inference / experimenting → Float8CurrentScaling
├─ weights with varying row magnitudes (DeepSeek-v3) → Float8BlockScaling
└─ targeting MX-capable hardware → MXFP8BlockScaling
```

## Switching recipes (`requantize`)

`breccia.requantize(scaled, new_recipe)` converts a ScaledTensor between
recipes:

```python
import breccia

# Train in MXFP8…
st_mx = breccia.cast(x, breccia.MXFP8BlockScaling())

# …ship as NVFP4 (no model rewrite required):
st_nv = breccia.requantize(st_mx, breccia.NVFP4BlockScaling())
```

In v0.0.1 `requantize` is implemented as `cast(dequantize(scaled),
new_recipe)`. Direct cross-recipe paths that avoid the FP32 round-trip
are an optimization for v0.1.

## Comparing recipes

See [numerics.md](numerics.md) for measured accuracy / range trade-offs
and [benchmarks.md](benchmarks.md) for memory savings tables.
