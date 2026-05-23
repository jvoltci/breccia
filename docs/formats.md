# Formats

Bit-level reference for every low-precision format breccia handles. This
is the document to consult when implementing a new kernel, debugging a
checkpoint, or wiring breccia into a vendor library.

For *recipes* (which formats compose with which scale layouts), see
[recipes.md](recipes.md). For *kernels* (what runs at compute time),
see [kernels.md](kernels.md).

## FP8 E4M3 (Open Compute "E4M3FN")

```
bit:    7 | 6 5 4 3 | 2 1 0
        S |    E    |   M
```

| Field | Bits | Bias | Notes |
| --- | --- | --- | --- |
| Sign | 1 | — | Signed |
| Exponent | 4 | 7 | |
| Mantissa | 3 | — | |

- **Normal**: `value = (-1)^S * 2^(E - 7) * (1 + M / 8)` for `E ∈ [1, 14]`
- **Subnormal**: `value = (-1)^S * 2^-6 * (M / 8)` for `E = 0`
- **NaN**: byte `0b01111111` (`E = 15`, `M = 7`). No Infinity.
- **Max normal**: `448.0` (S=0, E=14, M=6)
- **Min subnormal**: `2^-9 ≈ 1.95 × 10^-3`

256 possible bytes → 254 distinct floats + 2 NaN codes (one for each sign).

Helpers in [`breccia._formats`](../src/breccia/_formats.py):

```python
encode_e4m3(x: np.ndarray) -> uint8 ndarray   # round-to-nearest, saturating
decode_e4m3(b: np.ndarray) -> float32 ndarray # via 256-entry LUT
```

## FP8 E5M2 (IEEE 754 compatible)

```
bit:    7 | 6 5 4 3 2 | 1 0
        S |     E     |  M
```

| Field | Bits | Bias | Notes |
| --- | --- | --- | --- |
| Sign | 1 | — | |
| Exponent | 5 | 15 | |
| Mantissa | 2 | — | |

- **Normal**: `value = (-1)^S * 2^(E - 15) * (1 + M / 4)` for `E ∈ [1, 30]`
- **Subnormal**: `value = (-1)^S * 2^-14 * (M / 4)` for `E = 0`
- **±Infinity**: `E = 31, M = 0`
- **NaN**: `E = 31, M ∈ {1, 2, 3}`
- **Max normal**: `57344.0` (S=0, E=30, M=3)
- **Min subnormal**: `2^-16 ≈ 1.53 × 10^-5`

Wider range than E4M3, fewer mantissa bits → less precision. Typical use:
gradient tensors in backward pass, weights with very wide dynamic range.

## FP4 E2M1 (NVFP4 data format)

```
bit:    3 | 2 1 | 0
        S |  E  | M
```

| Field | Bits | Bias |
| --- | --- | --- |
| Sign | 1 | — |
| Exponent | 2 | 1 |
| Mantissa | 1 | — |

- **Normal**: `value = (-1)^S * 2^(E - 1) * (1 + M / 2)` for `E ∈ [1, 3]`
- **Subnormal**: `value = (-1)^S * (M / 2)` for `E = 0`
- **No Infinity, no NaN**
- **Max**: `6.0`

The full 16-value table (8 positive + 8 negative):

```
0, 0.5, 1, 1.5, 2, 3, 4, 6,
-0, -0.5, -1, -1.5, -2, -3, -4, -6
```

In breccia's `ScaledTensor.data` field, FP4 values are stored one-per-byte
(low 4 bits used) for simplicity. For compact storage in checkpoints,
`breccia.bridges` packs two nibbles per byte via `pack_nibbles` —
documented in [bridges.md](bridges.md).

## E8M0 (OCP MX scale format)

```
bit:    7 6 5 4 3 2 1 0
            E (all 8 bits)
```

No sign, no mantissa. Pure exponent encoding:

```
value = 2 ^ (byte - 127)
```

| Byte | Value |
| --- | --- |
| 0 | `2^-127` (= 0 in practice, since fp32 can't represent it as a normal) |
| 127 | `2^0 = 1.0` |
| 128 | `2^1 = 2.0` |
| 255 | `2^128` (overflow; treated as Inf) |

Used by `MXFP8BlockScaling` as the per-block scale. The trade-off:
power-of-two scales are less precise than fp32 but use 1 byte per block
(vs 4 for fp32) — a 4× scale-buffer reduction.

## INT4

Two interpretations, both stored in the low 4 bits of a uint8 byte:

| Mode | Range | Bit pattern |
| --- | --- | --- |
| Signed (two's complement) | `[-8, 7]` | `0b1000 → -8`, `0b0111 → 7` |
| Unsigned | `[0, 15]` | `0b0000 → 0`, `0b1111 → 15` |

```python
encode_int4(x: ndarray, signed: bool = True) -> uint8 ndarray
decode_int4(b: ndarray, signed: bool = True) -> float32 ndarray
```

Compact storage (2 nibbles per byte) via `pack_nibbles` / `unpack_nibbles`.

## Scale dtypes summary

Different recipes use different scale dtypes. Breccia stores each scale
in its declared native dtype:

| Recipe | Scale dtype on disk |
| --- | --- |
| DelayedScaling, Float8CurrentScaling, Float8BlockScaling | `float32` |
| MXFP8BlockScaling | `uint8` (E8M0 encoded) |
| NVFP4BlockScaling | `uint8` (FP8 E4M3 encoded) |
| INT4Scaling | `float16` / `bfloat16` / `float32` (configurable) |

The dequantization function for each recipe knows how to decode its
scale dtype back to float32 before multiplying with the decoded data.

## Where this is implemented

[`src/breccia/_formats.py`](../src/breccia/_formats.py) contains:

- `_build_e4m3_lut()`, `_build_e5m2_lut()`, `_build_e2m1_lut()` — build the
  decode tables at import time
- `encode_*` and `decode_*` for each format
- `pack_nibbles` / `unpack_nibbles` for 4-bit compact storage
- Module constants: `E4M3_MAX = 448.0`, `E5M2_MAX = 57344.0`, `E2M1_MAX = 6.0`

## Why these specific formats (and not others)

| Format | Why included |
| --- | --- |
| FP8 E4M3 / E5M2 | Used by TransformerEngine (NVIDIA), torchao (PyTorch), CUDA cuBLAS FP8 GEMM |
| FP4 E2M1 | NVIDIA Blackwell NVFP4; also OCP MX FP4 |
| E8M0 | The block-scale encoding for OCP MX FP8 / FP6 / FP4 |
| INT4 | GPTQ / AWQ / smoothquant family; deployed in nearly every open inference engine |

Formats *not* in v0.0.1: FP6 (OCP MX FP6 E3M2 / E2M3) — added when there's
a stable kernel target. INT8 — covered well by torchao; bridge support is
sufficient. INT2 / binary — niche.
