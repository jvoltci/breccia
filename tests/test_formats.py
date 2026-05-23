"""Tests for breccia._formats: FP8 / FP4 / INT4 encode / decode + nibble packing."""

import numpy as np
import pytest

from breccia._formats import (
    E4M3_MAX,
    E5M2_MAX,
    E2M1_MAX,
    encode_e4m3,
    decode_e4m3,
    encode_e5m2,
    decode_e5m2,
    encode_e2m1,
    decode_e2m1,
    encode_int4,
    decode_int4,
    pack_nibbles,
    unpack_nibbles,
    _E4M3_LUT,
    _E5M2_LUT,
    _E2M1_LUT,
)


# ---------- Lookup table shape + special values ----------


def test_e4m3_lut_has_256_entries():
    assert _E4M3_LUT.shape == (256,)


def test_e4m3_lut_max_is_448():
    finite = _E4M3_LUT[np.isfinite(_E4M3_LUT)]
    assert finite.max() == pytest.approx(448.0)
    assert finite.min() == pytest.approx(-448.0)


def test_e4m3_lut_nan_code():
    """Byte 0b01111111 is E4M3 NaN."""
    assert np.isnan(_E4M3_LUT[0b01111111])


def test_e4m3_lut_zero_codes():
    """Bytes 0 and 128 are both zero (positive and negative)."""
    assert _E4M3_LUT[0] == 0.0
    assert _E4M3_LUT[0b10000000] == 0.0


def test_e5m2_lut_has_256_entries():
    assert _E5M2_LUT.shape == (256,)


def test_e5m2_lut_has_infinity():
    """E5M2 is IEEE-compatible and includes ±Inf."""
    # Byte e=31 m=0 with sign=0 → +Inf; with sign=1 → -Inf.
    assert _E5M2_LUT[0b01111100] == np.inf
    assert _E5M2_LUT[0b11111100] == -np.inf


def test_e5m2_lut_has_nan():
    """E5M2 NaN: e=31, m != 0."""
    assert np.isnan(_E5M2_LUT[0b01111101])
    assert np.isnan(_E5M2_LUT[0b01111110])


def test_e5m2_lut_max():
    finite = _E5M2_LUT[np.isfinite(_E5M2_LUT)]
    assert finite.max() == pytest.approx(57344.0)


def test_e2m1_lut_has_16_entries():
    assert _E2M1_LUT.shape == (16,)


def test_e2m1_lut_exact_values():
    """E2M1 has exactly these 16 values."""
    expected = sorted([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ])
    actual = sorted(_E2M1_LUT.tolist())
    assert actual == expected


# ---------- Round-trip: decode(encode(x)) ≈ x ----------


def test_e4m3_round_trip_zero():
    x = np.zeros(8, dtype=np.float32)
    assert np.array_equal(decode_e4m3(encode_e4m3(x)), x)


def test_e4m3_round_trip_representable_values():
    """Values that are exactly representable in E4M3 should round-trip exactly."""
    # All E4M3 representable values are in the LUT.
    representable = _E4M3_LUT[np.isfinite(_E4M3_LUT)]
    decoded = decode_e4m3(encode_e4m3(representable))
    np.testing.assert_array_equal(decoded, representable)


def test_e4m3_round_trip_within_precision():
    """For arbitrary values, |decode(encode(x)) - x| <= local precision."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 100, size=64).astype(np.float32)
    decoded = decode_e4m3(encode_e4m3(x))
    # E4M3 precision is at worst about 12.5% relative (3-bit mantissa).
    # For values near the max we accept the round-to-nearest error.
    relative_err = np.abs(decoded - x) / np.maximum(np.abs(x), 1e-3)
    assert np.max(relative_err) < 0.15


def test_e4m3_saturation():
    """Out-of-range values clamp to ±448."""
    x = np.array([1e10, -1e10], dtype=np.float32)
    decoded = decode_e4m3(encode_e4m3(x))
    assert decoded[0] == E4M3_MAX
    assert decoded[1] == -E4M3_MAX


def test_e4m3_nan_round_trip():
    x = np.array([np.nan, 1.0, np.nan], dtype=np.float32)
    decoded = decode_e4m3(encode_e4m3(x))
    assert np.isnan(decoded[0])
    assert decoded[1] == pytest.approx(1.0)
    assert np.isnan(decoded[2])


def test_e5m2_round_trip_representable():
    finite = _E5M2_LUT[np.isfinite(_E5M2_LUT)]
    decoded = decode_e5m2(encode_e5m2(finite))
    np.testing.assert_array_equal(decoded, finite)


def test_e5m2_wider_range_than_e4m3():
    """E5M2 represents 1000 without saturating (E4M3 saturates above 448)."""
    x = np.array([1000.0], dtype=np.float32)
    decoded_e5m2 = decode_e5m2(encode_e5m2(x))
    decoded_e4m3 = decode_e4m3(encode_e4m3(x))
    # E5M2 keeps 1000 (within precision); E4M3 saturates to 448.
    assert decoded_e5m2[0] > 600
    assert decoded_e4m3[0] == E4M3_MAX


def test_e2m1_round_trip_representable_values():
    """All 16 E2M1 values must round-trip exactly."""
    decoded = decode_e2m1(encode_e2m1(_E2M1_LUT))
    np.testing.assert_array_equal(decoded, _E2M1_LUT)


def test_e2m1_round_trip_arbitrary():
    """For arbitrary x in [-6, 6], the result is one of the 16 representable values."""
    rng = np.random.default_rng(1)
    x = rng.uniform(-6, 6, size=32).astype(np.float32)
    decoded = decode_e2m1(encode_e2m1(x))
    for v in decoded:
        assert v in _E2M1_LUT


def test_e2m1_saturation():
    x = np.array([100.0, -100.0], dtype=np.float32)
    decoded = decode_e2m1(encode_e2m1(x))
    assert decoded[0] == E2M1_MAX
    assert decoded[1] == -E2M1_MAX


# ---------- INT4 ----------


def test_int4_signed_round_trip():
    """Signed INT4 represents -8..7 exactly."""
    x = np.arange(-8, 8, dtype=np.float32)
    decoded = decode_int4(encode_int4(x, signed=True), signed=True)
    np.testing.assert_array_equal(decoded, x)


def test_int4_unsigned_round_trip():
    """Unsigned INT4 represents 0..15 exactly."""
    x = np.arange(16, dtype=np.float32)
    decoded = decode_int4(encode_int4(x, signed=False), signed=False)
    np.testing.assert_array_equal(decoded, x)


def test_int4_signed_saturation():
    x = np.array([100.0, -100.0], dtype=np.float32)
    decoded = decode_int4(encode_int4(x, signed=True), signed=True)
    assert decoded[0] == 7
    assert decoded[1] == -8


def test_int4_rounding():
    """Round-to-nearest-int."""
    x = np.array([0.4, 0.6, -0.4, -0.6, 2.5], dtype=np.float32)
    decoded = decode_int4(encode_int4(x, signed=True), signed=True)
    # NumPy round-half-to-even: 0.5 → 0, 2.5 → 2, etc.
    expected = np.round(x).astype(np.float32)
    np.testing.assert_array_equal(decoded, expected)


def test_int4_shape_preserved():
    x = np.zeros((4, 8), dtype=np.float32)
    assert encode_int4(x).shape == (4, 8)
    assert decode_int4(encode_int4(x)).shape == (4, 8)


# ---------- Nibble packing ----------


def test_pack_nibbles_round_trip():
    codes = np.array([0x0, 0x1, 0x2, 0x3, 0xF, 0xA, 0x5, 0x7], dtype=np.uint8)
    packed = pack_nibbles(codes)
    assert packed.shape == (4,)
    # First byte: high=0, low=1 → 0x01.
    assert packed[0] == 0x01
    assert packed[1] == 0x23
    assert packed[2] == 0xFA
    assert packed[3] == 0x57
    unpacked = unpack_nibbles(packed, n_values=8)
    np.testing.assert_array_equal(unpacked, codes)


def test_pack_nibbles_rejects_odd_length():
    with pytest.raises(ValueError, match="even"):
        pack_nibbles(np.array([0x0, 0x1, 0x2], dtype=np.uint8))


def test_pack_nibbles_strips_high_bits():
    """Only the low 4 bits of each input are honored."""
    codes = np.array([0xF0, 0xF1], dtype=np.uint8)  # high bits should be ignored
    packed = pack_nibbles(codes)
    assert packed[0] == 0x01


def test_unpack_nibbles_with_smaller_n_values():
    codes = np.array([0x01, 0x02, 0x03, 0x04], dtype=np.uint8)
    packed = pack_nibbles(codes)
    unpacked = unpack_nibbles(packed, n_values=3)
    np.testing.assert_array_equal(unpacked, codes[:3])


def test_unpack_nibbles_too_many_values():
    packed = np.array([0x12], dtype=np.uint8)
    with pytest.raises(ValueError, match="capacity"):
        unpack_nibbles(packed, n_values=10)


# ---------- Shape preservation ----------


def test_e4m3_preserves_shape_2d():
    x = np.zeros((4, 8), dtype=np.float32)
    assert encode_e4m3(x).shape == (4, 8)
    assert decode_e4m3(encode_e4m3(x)).shape == (4, 8)


def test_e5m2_preserves_shape_3d():
    x = np.zeros((2, 4, 8), dtype=np.float32)
    assert encode_e5m2(x).shape == (2, 4, 8)


def test_e2m1_preserves_shape():
    x = np.zeros((3, 5), dtype=np.float32)
    assert encode_e2m1(x).shape == (3, 5)
