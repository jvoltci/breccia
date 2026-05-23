# Numerics

Accuracy, range, and trade-offs for each format breccia handles. Read
this before picking a recipe for a sensitive workload.

For the bit-level format definitions, see [formats.md](formats.md). For
how recipes compose with formats, see [recipes.md](recipes.md).

## Precision summary

The relative precision of a format is roughly `2^(-mantissa_bits)`. For
each format breccia ships:

| Format | Mantissa bits | Worst-case relative error | Distinct values per sign |
| --- | --- | --- | --- |
| FP8 E4M3 | 3 | ~6.25% (1/16) | 127 |
| FP8 E5M2 | 2 | ~12.5% (1/8) | 127 |
| FP4 E2M1 | 1 | ~25% (1/4) | 7 |
| E8M0 (scale only) | 0 | exact 2^N | 256 |
| INT4 | — | 1 unit | 7 (signed) or 15 (unsigned) |

These are *theoretical* bounds for a single value, **before** the scale
factor is applied. The scale factor changes the absolute precision but
not the relative precision in the format's representable range.

## Range summary

| Format | Min normal | Max finite |
| --- | --- | --- |
| FP8 E4M3 | 2^-6 ≈ 0.0156 | 448 |
| FP8 E5M2 | 2^-14 ≈ 6.1e-5 | 57344 |
| FP4 E2M1 | 0.5 (or 0 subnormal) | 6 |
| E8M0 | 2^-127 ≈ 5.9e-39 | 2^128 |
| INT4 (signed) | -8 | 7 |
| INT4 (unsigned) | 0 | 15 |

For real workloads, **scaling brings the data into format range**.
That's the whole point. So the absolute range constraint matters only
when:

1. The block / tensor `amax` is extremely small (near machine epsilon),
   in which case the format may underflow.
2. The block contains a single huge outlier that monopolizes the scale,
   leaving the rest of the block at the format's quantization grid bottom.

Both cases are handled in [`breccia.kernels.reference.cast`](kernels.md)
with explicit fallbacks (an `_AMAX_EPS` floor of 1e-10 on `amax`, and
an E4M3-subnormal floor on the NVFP4 scale).

## Cosine similarity vs relative error

breccia's test suite uses cosine similarity to validate cast quality —
not relative error — because relative error explodes on small-magnitude
inputs (1e-6 / 1e-6 = 1.0 if both have similar small absolute error).

Cosine similarity is robust: it measures the *direction* of the recovered
vector vs the original, ignoring scaling. For Gaussian-distributed
inputs:

| Recipe | Expected cos sim vs FP32 (typical) |
| --- | --- |
| `DelayedScaling` / `Float8CurrentScaling` (E4M3) | > 0.997 |
| `Float8BlockScaling(block_k=128)` | > 0.998 |
| `MXFP8BlockScaling` (E8M0 scale) | > 0.997 |
| `NVFP4BlockScaling` | > 0.95 |
| `INT4Scaling(group_size=128)` | > 0.98 |

Real workloads (e.g., LLM activations) often have heavier tails than
Gaussian. The format quality is still acceptable but the margin shrinks;
this is why production deployments use block / group scaling rather
than per-tensor.

## Saturation behavior

All breccia encoders **saturate** on overflow — values larger than the
format's max collapse to the max, not Inf:

```python
encode_e4m3(1e10) → 448      # not Inf
encode_e2m1(100)  → 6        # not Inf
encode_int4(100)  → 7        # signed; 15 for unsigned
```

This matches NVIDIA TransformerEngine and the OCP MX spec — saturation
is more predictable than Inf propagation in mixed-precision pipelines.

The cast functions also clip the input to `[-fmt_max, fmt_max]` before
the round-to-nearest argmin, so the saturation is consistent across
implementations.

## NaN handling

| Format | NaN representation |
| --- | --- |
| FP8 E4M3 | Single NaN code: `0b01111111` (sign bit ignored) |
| FP8 E5M2 | IEEE-style: `E = 31, M != 0` |
| FP4 E2M1 | No NaN — input NaN maps to 0 |
| E8M0 | No NaN — scales are always positive |
| INT4 | No NaN — input NaN maps to 0 |

A NaN in `breccia.cast` input propagates correctly only for FP8
formats. For FP4 / INT4, a NaN becomes a zero — same behavior as
torchao and NVIDIA Quark.

## Block-scaling accuracy advantages

Per-block (or per-group) scaling is more robust than per-tensor scaling
when row magnitudes vary widely. Quantifying with a thought experiment:

Suppose a `(1, 256)` tensor has block-0 (indices 0–127) at magnitude
0.1 and block-1 (indices 128–255) at magnitude 100. With per-tensor
scaling, the scale is set by the max (100), so block-0 quantizes with
a step size of `100 / 448 ≈ 0.22` — much coarser than the block-0
values themselves. Block-0 effectively loses most of its information.

With per-block scaling, each block gets its own scale. Block-0 gets a
fine grid; block-1 gets a coarse grid. No information loss in either.

This is why DeepSeek-v3 (which has weight rows with extremely varied
magnitudes) ships FP8 with `block_k=128` rather than per-tensor.

## E8M0 power-of-two scale

`MXFP8BlockScaling` uses an E8M0 scale: an 8-bit unsigned int encoding
a power of two. This means the scale is restricted to `{2^k : k ∈ [-127, 128]}`.

Compared to a free fp32 scale, this introduces extra quantization noise
in the *scale factor itself*. In practice:

- The mantissa of the data already has plenty of precision relative to
  the scale (3 mantissa bits in E4M3 vs the power-of-two step).
- Real distributions have block-amax values that cluster near powers of
  two (within ~12% on either side).
- The 1-byte scale vs 4-byte float scale is a 4× scale-buffer reduction.

So the trade is: ~0.5% extra cosine-similarity loss for a 4× reduction in
scale-tensor memory. For most workloads this is the right trade — and
it's why the OCP MX standard fixes the scale at E8M0.

## NVFP4: only 16 values per element

FP4 E2M1 has only 8 positive + 8 negative representable values. After
scaling, the per-element grid still has only 16 points. This is
substantially coarser than FP8.

The thing that makes NVFP4 work is the **small block size (16)**: each
group of 16 elements gets its own FP8 E4M3 scale, giving the effective
representable range a per-block adjustment of 256 possible scale values.

For Gaussian-distributed inputs, cos sim stays above 0.95 — surprisingly
good given how few raw values FP4 has. For non-Gaussian inputs with
heavy tails or large outliers, NVFP4 quality degrades faster than FP8.

## When precision matters most

| Concern | Use |
| --- | --- |
| Maximum accuracy, FP8 budget | `Float8BlockScaling(block_k=128)` |
| Hardware-accelerated FP8 (Hopper, Blackwell) | `DelayedScaling` or `MXFP8BlockScaling` |
| Smallest memory, FP4 budget | `NVFP4BlockScaling` |
| INT-friendly inference hardware | `INT4Scaling(group_size=128)` |
| Range tolerance > precision | `Float8CurrentScaling(fp8_format="E5M2")` |

For training, the standard recipe is `DelayedScaling(E4M3)` for forward
activations and `DelayedScaling(E5M2)` for backward gradients.

## Limitations

- breccia v0.0.1 **does not** model the absolute-error guarantees that
  some FP8 verification tools require (e.g., NVIDIA's TE has an
  `amax_history` check that asserts no NaN in the history). Those
  belong in the training loop, not the primitive.
- breccia **does not** track straight-through-estimator gradients for
  the cast step. If you want autograd-aware quantized training, use TE
  or torchao for now; the breccia STE wrapper lands in v0.1.
- breccia **does not** check for catastrophic cancellation in the matmul
  accumulator. The reference matmul uses FP32 accumulation, which is
  what every production scaled-matmul kernel uses; if your workload
  needs FP64 accumulation, breccia is not the right tool.
