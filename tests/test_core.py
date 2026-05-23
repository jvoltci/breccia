"""Tests for breccia._core: ScaledTensor invariants + construction.

M1 — these tests use stub recipe/layout classes so they're independent of
the concrete types added in M2 (recipes.py) and M3 (layouts.py).
"""

import dataclasses
import numpy as np
import pytest

from breccia._core import (
    ScaledTensor,
    from_buffer,
    _is_torch,
    _is_mlx,
    _is_jax,
)


class _StubRecipe:
    """Minimal stand-in until M2 ships the real recipes."""


class _StubLayout:
    """Minimal stand-in until M3 ships the real layouts."""


# ---------- Construction ----------


def test_construction_minimal():
    data = np.zeros((4, 8), dtype=np.uint8)
    scale = np.float32(1.0)
    st = ScaledTensor(data=data, scale=scale, recipe=_StubRecipe(), layout=_StubLayout())
    assert st.shape == (4, 8)
    assert st.ndim == 2
    assert st.data_dtype == np.uint8
    assert st.scale_dtype == np.float32


def test_from_buffer_does_not_copy():
    data = np.arange(16, dtype=np.uint8).reshape(4, 4)
    scale = np.float32(2.0)
    st = from_buffer(data, scale, _StubRecipe(), _StubLayout())
    assert st.data is data
    assert st.scale is scale


def test_frozen_dataclass():
    st = ScaledTensor(
        data=np.zeros((4,), dtype=np.uint8),
        scale=np.float32(1.0),
        recipe=_StubRecipe(),
        layout=_StubLayout(),
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        st.data = np.ones((4,), dtype=np.uint8)


# ---------- Invariants ----------


def test_invariant_data_must_be_array_like():
    with pytest.raises(TypeError, match="data must be array-like"):
        ScaledTensor(
            data=42,
            scale=np.float32(1.0),
            recipe=_StubRecipe(),
            layout=_StubLayout(),
        )


def test_invariant_scale_must_be_array_like():
    with pytest.raises(TypeError, match="scale must be array-like"):
        ScaledTensor(
            data=np.zeros((4,), dtype=np.uint8),
            scale=2.0,
            recipe=_StubRecipe(),
            layout=_StubLayout(),
        )


def test_invariant_data_min_1d():
    with pytest.raises(ValueError, match="at least 1-D"):
        ScaledTensor(
            data=np.array(0, dtype=np.uint8),
            scale=np.float32(1.0),
            recipe=_StubRecipe(),
            layout=_StubLayout(),
        )


def test_invariant_recipe_required():
    with pytest.raises(ValueError, match="recipe is required"):
        ScaledTensor(
            data=np.zeros((4,), dtype=np.uint8),
            scale=np.float32(1.0),
            recipe=None,
            layout=_StubLayout(),
        )


def test_invariant_layout_required():
    with pytest.raises(ValueError, match="layout is required"):
        ScaledTensor(
            data=np.zeros((4,), dtype=np.uint8),
            scale=np.float32(1.0),
            recipe=_StubRecipe(),
            layout=None,
        )


# ---------- Layout-driven cross-field validation ----------


def test_layout_validate_is_called():
    calls = []

    class _ValidateLayout:
        def validate(self, data, scale):
            calls.append((tuple(data.shape), tuple(scale.shape)))

    data = np.zeros((4,), dtype=np.uint8)
    scale = np.float32(1.0)
    ScaledTensor(data=data, scale=scale, recipe=_StubRecipe(), layout=_ValidateLayout())
    assert calls == [((4,), ())]


def test_layout_validate_can_reject():
    class _RejectingLayout:
        def validate(self, data, scale):
            raise ValueError("scale.shape does not match layout")

    with pytest.raises(ValueError, match="does not match layout"):
        ScaledTensor(
            data=np.zeros((4,), dtype=np.uint8),
            scale=np.float32(1.0),
            recipe=_StubRecipe(),
            layout=_RejectingLayout(),
        )


def test_layout_without_validate_is_accepted():
    class _NoValidateLayout:
        pass

    ScaledTensor(
        data=np.zeros((4,), dtype=np.uint8),
        scale=np.float32(1.0),
        recipe=_StubRecipe(),
        layout=_NoValidateLayout(),
    )


# ---------- Repr ----------


def test_repr_format():
    st = ScaledTensor(
        data=np.zeros((4, 8), dtype=np.uint8),
        scale=np.ones((), dtype=np.float32),
        recipe=_StubRecipe(),
        layout=_StubLayout(),
    )
    r = repr(st)
    assert "ScaledTensor" in r
    assert "shape=(4, 8)" in r
    assert "uint8" in r
    assert "_StubRecipe" in r
    assert "_StubLayout" in r


# ---------- Backend predicates ----------


def test_backend_predicates_numpy():
    arr = np.zeros((4,))
    assert not _is_torch(arr)
    assert not _is_mlx(arr)
    assert not _is_jax(arr)


def test_backend_predicates_torch():
    try:
        import torch
    except ImportError:
        pytest.skip("torch not installed")
    arr = torch.zeros((4,))
    assert _is_torch(arr)
    assert not _is_mlx(arr)
    assert not _is_jax(arr)


def test_backend_predicates_mlx():
    try:
        import mlx.core as mx
    except ImportError:
        pytest.skip("mlx not installed")
    arr = mx.zeros((4,))
    assert not _is_torch(arr)
    assert _is_mlx(arr)
    assert not _is_jax(arr)
