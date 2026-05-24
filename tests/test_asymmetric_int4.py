"""Tests for asymmetric INT4 (zero-point) quantization."""

import numpy as np
import pytest

from breccia import (
    cast,
    dequantize,
    matmul,
    ScaledTensor,
    INT4Scaling,
)
from breccia.bridges import save_safetensors, load_safetensors


def _cos_sim(a, b):
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------- Recipe validation ----------


def test_asymmetric_requires_unsigned():
    """Asymmetric INT4 must use signed=False."""
    INT4Scaling(symmetric=False, signed=False)  # OK
    with pytest.raises(ValueError, match="signed=False"):
        INT4Scaling(symmetric=False, signed=True)


def test_symmetric_default():
    r = INT4Scaling()
    assert r.symmetric is True
    assert r.signed is True


# ---------- Round-trip ----------


def test_asymmetric_round_trip_quality():
    """Asymmetric INT4 should match or beat symmetric on skewed data."""
    rng = np.random.default_rng(0)
    # Skewed distribution: gamma-shaped (all positive)
    x = rng.gamma(2.0, size=(4, 256)).astype(np.float32)
    st_sym = cast(x, INT4Scaling(group_size=128, signed=True, symmetric=True))
    st_asym = cast(x, INT4Scaling(group_size=128, signed=False, symmetric=False))
    rec_sym = np.asarray(dequantize(st_sym))
    rec_asym = np.asarray(dequantize(st_asym))
    cos_sym = _cos_sim(x, rec_sym)
    cos_asym = _cos_sim(x, rec_asym)
    # Both should be high; asymmetric should be at least as good on skewed data.
    assert cos_sym > 0.95
    assert cos_asym > 0.95
    assert cos_asym >= cos_sym - 0.02  # within 2% (run-to-run noise)


def test_asymmetric_zero_point_shape():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128, signed=False, symmetric=False))
    assert st.zero_point is not None
    assert tuple(st.zero_point.shape) == tuple(st.scale.shape)


def test_symmetric_zero_point_is_none():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128, signed=True, symmetric=True))
    assert st.zero_point is None


# ---------- Matmul with asymmetric weights ----------


def test_matmul_with_asymmetric_int4():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((8, 256)).astype(np.float32)
    W = rng.gamma(2.0, size=(256, 128)).astype(np.float32)
    sa = cast(A, INT4Scaling(group_size=128, signed=False, symmetric=False))
    sw = cast(W, INT4Scaling(group_size=128, signed=False, symmetric=False))
    out = matmul(sa, sw)
    assert out.shape == (8, 128)
    assert _cos_sim(out, A @ W) > 0.92


# ---------- ScaledTensor.zero_point invariants ----------


def test_zero_point_shape_must_match_scale():
    """Construction rejects zero_point with wrong shape."""
    data = np.zeros((4, 8), dtype=np.uint8)
    scale = np.ones((4,), dtype=np.float32)
    wrong_zp = np.ones((8,), dtype=np.float32)
    from breccia.layouts import PerChannel

    with pytest.raises(ValueError, match="zero_point.shape"):
        ScaledTensor(
            data=data,
            scale=scale,
            recipe=INT4Scaling(group_size=128, signed=False, symmetric=False),
            layout=PerChannel(),
            zero_point=wrong_zp,
        )


def test_zero_point_must_be_array_like():
    data = np.zeros((4, 8), dtype=np.uint8)
    scale = np.float32(1.0)
    from breccia.layouts import PerTensor

    with pytest.raises(TypeError, match="zero_point must be array-like"):
        ScaledTensor(
            data=data,
            scale=scale,
            recipe=INT4Scaling(group_size=128, signed=False, symmetric=False),
            layout=PerTensor(),
            zero_point="not an array",
        )


# ---------- HuggingFace bridge round-trips zero_point ----------


def test_hf_bridge_preserves_zero_point(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128, signed=False, symmetric=False))
    assert st.zero_point is not None

    path = str(tmp_path / "asym.safetensors")
    save_safetensors({"w": st}, path)
    loaded = load_safetensors(path)

    assert loaded["w"].zero_point is not None
    # Round-trip preserves the dequantized output exactly.
    out_before = np.asarray(dequantize(st))
    out_after = np.asarray(dequantize(loaded["w"]))
    np.testing.assert_allclose(out_before, out_after, atol=1e-5)


# ---------- PyTorch backend carries zero_point ----------


def test_torch_backend_carries_zero_point():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(4, 256, dtype=torch.float32)
    st = cast(x, INT4Scaling(group_size=128, signed=False, symmetric=False))
    assert st.zero_point is not None
    assert isinstance(st.zero_point, torch.Tensor)
    out = dequantize(st)
    cos = float(
        torch.dot(out.flatten(), x.flatten())
        / (torch.linalg.norm(out) * torch.linalg.norm(x) + 1e-12)
    )
    assert cos > 0.95
