"""Bit-level format helpers for FP8 E4M3 / E5M2, FP4 E2M1, and INT4.

NumPy has no native FP8 / FP4 / INT4 dtypes, so breccia represents these
formats as ``uint8`` byte buffers in its reference path. This module
provides the encode / decode primitives between ``float32`` values and
their FP8 / FP4 / INT4 ``uint8`` codes.

Representation conventions (v0.0.1):

- **FP8 E4M3 / E5M2**: one byte per value. The byte's bit pattern is the
  IEEE-style sign/exponent/mantissa encoding.
- **FP4 E2M1**: one byte per value, with only the low 4 bits used (high
  4 bits are zero). Two-per-byte storage packing is a separate concern
  handled by ``pack_nibbles`` / ``unpack_nibbles`` — invoked by bridges
  for compact checkpoint storage but not used inside ScaledTensor's
  ``data`` field.
- **INT4**: one byte per value, low 4 bits hold the value (signed or
  unsigned). Same packing rule as FP4.

Encoding uses round-to-nearest based on lookup tables built at import
time. The tables are exact for the format definitions:

- **E4M3** (Open Compute "E4M3FN"): 1 sign + 4 exp + 3 mantissa bits,
  bias 7, no Inf, NaN at ``0b01111111`` (and ``0b11111111``). Max
  normal: ``448``. Range: ``[-448, 448]``.
- **E5M2**: 1 sign + 5 exp + 2 mantissa bits, bias 15, IEEE 754
  compatible (with Inf and NaN). Max normal: ``57344``.
- **E2M1**: 1 sign + 2 exp + 1 mantissa bit, bias 1, no Inf, no NaN.
  Representable values: ``{0, 0.5, 1, 1.5, 2, 3, 4, 6}`` and their
  negations.
"""

from __future__ import annotations

import numpy as np


# ---------- Lookup table builders ----------


def _build_e4m3_lut() -> np.ndarray:
    """256-entry uint8 → float32 lookup for FP8 E4M3 (FN variant).

    Special values:
    - 0b01111111 (127): NaN (positive)
    - 0b11111111 (255): NaN (negative-coded; treated as NaN)
    """
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        s = (byte >> 7) & 1
        e = (byte >> 3) & 0b1111
        m = byte & 0b111
        if e == 15 and m == 7:
            table[byte] = np.nan
        elif e == 0:
            val = (m / 8.0) * (2.0 ** -6)
            table[byte] = -val if s else val
        else:
            val = (1.0 + m / 8.0) * (2.0 ** (e - 7))
            table[byte] = -val if s else val
    return table


def _build_e5m2_lut() -> np.ndarray:
    """256-entry uint8 → float32 lookup for FP8 E5M2 (IEEE-compatible)."""
    table = np.zeros(256, dtype=np.float32)
    for byte in range(256):
        s = (byte >> 7) & 1
        e = (byte >> 2) & 0b11111
        m = byte & 0b11
        if e == 31:
            table[byte] = np.nan if m else (-np.inf if s else np.inf)
        elif e == 0:
            val = (m / 4.0) * (2.0 ** -14)
            table[byte] = -val if s else val
        else:
            val = (1.0 + m / 4.0) * (2.0 ** (e - 15))
            table[byte] = -val if s else val
    return table


def _build_e2m1_lut() -> np.ndarray:
    """16-entry uint8 → float32 lookup for FP4 E2M1.

    Only the low 4 bits of the byte index are meaningful; values 16-255
    are unused.
    """
    table = np.zeros(16, dtype=np.float32)
    for nibble in range(16):
        s = (nibble >> 3) & 1
        e = (nibble >> 1) & 0b11
        m = nibble & 1
        if e == 0:
            val = (m / 2.0) * (2.0 ** 0)
            table[nibble] = -val if s else val
        else:
            val = (1.0 + m / 2.0) * (2.0 ** (e - 1))
            table[nibble] = -val if s else val
    return table


# Built once at import.
_E4M3_LUT = _build_e4m3_lut()
_E5M2_LUT = _build_e5m2_lut()
_E2M1_LUT = _build_e2m1_lut()


# Format limits (the largest finite magnitude each format can hold).
E4M3_MAX = 448.0
E5M2_MAX = 57344.0
E2M1_MAX = 6.0


# ---------- Encode (float32 → uint8 code) ----------


def _encode_via_lut(x: np.ndarray, lut: np.ndarray, lut_size: int) -> np.ndarray:
    """Round-to-nearest encode using ``argmin(|lut - x|)``.

    Brute-force vectorized: O(n * lut_size). Correct for any finite x;
    NaNs are mapped to the LUT's NaN code by the caller.
    """
    flat = np.asarray(x, dtype=np.float32).ravel()
    # |flat[:, None] - lut[None, :]| → (n, lut_size); argmin gives code.
    diffs = np.abs(flat[:, None] - lut[None, :lut_size])
    # NaNs in diffs propagate; mask them so they don't win argmin.
    finite_lut = np.isfinite(lut[:lut_size])
    diffs = np.where(finite_lut[None, :], diffs, np.inf)
    codes = np.argmin(diffs, axis=1).astype(np.uint8)
    return codes


def encode_e4m3(x: np.ndarray) -> np.ndarray:
    """Quantize float32 → FP8 E4M3 (round-to-nearest), returned as uint8.

    Saturates to ``±448`` on overflow. NaN inputs map to the E4M3 NaN code
    (``0b01111111``). The output has the same shape as the input.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    is_nan = np.isnan(x_arr)
    # Saturate to representable range before the argmin search to keep
    # behavior consistent across implementations.
    saturated = np.clip(x_arr, -E4M3_MAX, E4M3_MAX)
    codes = _encode_via_lut(saturated, _E4M3_LUT, 256).reshape(x_arr.shape)
    codes[is_nan] = 0b01111111
    return codes


def encode_e5m2(x: np.ndarray) -> np.ndarray:
    """Quantize float32 → FP8 E5M2 (round-to-nearest), returned as uint8.

    Saturates to ``±57344`` on overflow. NaN inputs map to E5M2 NaN
    (``0b01111101``: e=31, m=1).
    """
    x_arr = np.asarray(x, dtype=np.float32)
    is_nan = np.isnan(x_arr)
    saturated = np.clip(x_arr, -E5M2_MAX, E5M2_MAX)
    codes = _encode_via_lut(saturated, _E5M2_LUT, 256).reshape(x_arr.shape)
    codes[is_nan] = 0b01111101
    return codes


def encode_e2m1(x: np.ndarray) -> np.ndarray:
    """Quantize float32 → FP4 E2M1 (round-to-nearest), returned as uint8.

    Only the low 4 bits of each output byte are used. Saturates to
    ``±6``. NaN inputs map to ``0`` (E2M1 has no NaN representation).
    """
    x_arr = np.asarray(x, dtype=np.float32)
    is_nan = np.isnan(x_arr)
    saturated = np.clip(x_arr, -E2M1_MAX, E2M1_MAX)
    codes = _encode_via_lut(saturated, _E2M1_LUT, 16).reshape(x_arr.shape)
    codes[is_nan] = 0
    return codes


def encode_int4(x: np.ndarray, signed: bool = True) -> np.ndarray:
    """Round float32 → INT4, returned as uint8 (low 4 bits used).

    Signed range: ``[-8, 7]`` stored in two's-complement low nibble.
    Unsigned range: ``[0, 15]``.
    """
    x_arr = np.asarray(x, dtype=np.float32)
    if signed:
        rounded = np.round(np.clip(x_arr, -8, 7)).astype(np.int8)
        return (rounded.astype(np.uint8) & 0x0F)
    else:
        rounded = np.round(np.clip(x_arr, 0, 15)).astype(np.uint8)
        return rounded & 0x0F


# ---------- Decode (uint8 code → float32) ----------


def decode_e4m3(b: np.ndarray) -> np.ndarray:
    """Decode FP8 E4M3 bytes to float32 via lookup."""
    return _E4M3_LUT[np.asarray(b, dtype=np.uint8)]


def decode_e5m2(b: np.ndarray) -> np.ndarray:
    """Decode FP8 E5M2 bytes to float32 via lookup."""
    return _E5M2_LUT[np.asarray(b, dtype=np.uint8)]


def decode_e2m1(b: np.ndarray) -> np.ndarray:
    """Decode FP4 E2M1 nibbles (low 4 bits) to float32 via lookup."""
    return _E2M1_LUT[np.asarray(b, dtype=np.uint8) & 0x0F]


def decode_int4(b: np.ndarray, signed: bool = True) -> np.ndarray:
    """Decode INT4 nibbles (low 4 bits) to float32.

    Signed values are interpreted as two's-complement 4-bit ints,
    so 0b1000 → -8 and 0b0111 → 7.
    """
    low = np.asarray(b, dtype=np.uint8) & 0x0F
    if signed:
        # Two's complement: sign bit is bit 3.
        sign_extended = np.where(low >= 8, low.astype(np.int16) - 16, low.astype(np.int16))
        return sign_extended.astype(np.float32)
    return low.astype(np.float32)


# ---------- Packing (2 nibbles per byte) ----------


def pack_nibbles(codes: np.ndarray) -> np.ndarray:
    """Pack a uint8 array of 4-bit codes (low nibbles) into one-byte-per-pair.

    The flattened input must have an even number of elements; the high
    nibble of byte ``i`` of the output is ``codes[2*i]`` and the low
    nibble is ``codes[2*i + 1]``.

    Returns an output buffer of half the input's size (number of bytes).
    """
    flat = np.asarray(codes, dtype=np.uint8).ravel()
    if flat.size % 2 != 0:
        raise ValueError(
            f"pack_nibbles requires an even number of elements, got {flat.size}"
        )
    high = (flat[0::2] & 0x0F) << 4
    low = flat[1::2] & 0x0F
    return (high | low).astype(np.uint8)


def unpack_nibbles(packed: np.ndarray, n_values: int) -> np.ndarray:
    """Unpack a packed nibble buffer into ``n_values`` low-nibble bytes.

    Inverse of :func:`pack_nibbles`. The output's high nibble is always 0.
    """
    flat = np.asarray(packed, dtype=np.uint8).ravel()
    if n_values > 2 * flat.size:
        raise ValueError(
            f"unpack_nibbles: requested {n_values} values but buffer has "
            f"capacity for at most {2 * flat.size}"
        )
    out = np.empty(2 * flat.size, dtype=np.uint8)
    out[0::2] = (flat >> 4) & 0x0F
    out[1::2] = flat & 0x0F
    return out[:n_values]
