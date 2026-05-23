"""Tests for breccia.layouts: 4 Layout types + shape rules."""

import dataclasses

import numpy as np
import pytest

from breccia.layouts import (
    Layout,
    PerTensor,
    PerBlockK,
    PerChannel,
    PerBlockMN,
)


ALL_LAYOUTS = [PerTensor, PerBlockK, PerChannel, PerBlockMN]


# ---------- Base + meta ----------


def test_all_layouts_subclass_base():
    for cls in ALL_LAYOUTS:
        assert issubclass(cls, Layout)


def test_all_layouts_have_distinct_names():
    names = {cls.name for cls in ALL_LAYOUTS}
    assert names == {"per_tensor", "per_block_k", "per_channel", "per_block_mn"}


def test_all_layouts_have_validate():
    for cls in ALL_LAYOUTS:
        assert callable(getattr(cls, "validate"))


def test_layouts_are_frozen():
    layout = PerBlockK(block_size=128)
    with pytest.raises(dataclasses.FrozenInstanceError):
        layout.block_size = 64


# ---------- PerTensor ----------


def test_per_tensor_accepts_scalar():
    PerTensor().validate(np.zeros((4, 8), dtype=np.uint8), np.float32(1.0))


def test_per_tensor_rejects_nonscalar():
    with pytest.raises(ValueError, match="scalar"):
        PerTensor().validate(
            np.zeros((4, 8), dtype=np.uint8),
            np.array([1.0], dtype=np.float32),
        )


# ---------- PerBlockK ----------


def test_per_block_k_valid_2d():
    layout = PerBlockK(block_size=128)
    data = np.zeros((4, 256), dtype=np.uint8)  # K=256, K//B=2
    scale = np.zeros((4, 2), dtype=np.float32)
    layout.validate(data, scale)


def test_per_block_k_valid_3d():
    """Leading axes pass through."""
    layout = PerBlockK(block_size=32)
    data = np.zeros((2, 4, 64), dtype=np.uint8)  # leading=(2,), M=4, K=64
    scale = np.zeros((2, 4, 2), dtype=np.float32)
    layout.validate(data, scale)


def test_per_block_k_rejects_misaligned_K():
    layout = PerBlockK(block_size=128)
    with pytest.raises(ValueError, match="divisible"):
        layout.validate(
            np.zeros((4, 200), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
        )


def test_per_block_k_rejects_wrong_scale_shape():
    layout = PerBlockK(block_size=128)
    with pytest.raises(ValueError, match="scale.shape"):
        layout.validate(
            np.zeros((4, 256), dtype=np.uint8),  # expects (4, 2)
            np.zeros((4, 1), dtype=np.float32),
        )


def test_per_block_k_rejects_1d_data():
    with pytest.raises(ValueError, match="ndim"):
        PerBlockK(block_size=128).validate(
            np.zeros((128,), dtype=np.uint8),
            np.zeros((1,), dtype=np.float32),
        )


def test_per_block_k_block_size_validation():
    with pytest.raises(ValueError, match="block_size"):
        PerBlockK(block_size=0)
    with pytest.raises(ValueError, match="block_size"):
        PerBlockK(block_size=-1)


# ---------- PerChannel ----------


def test_per_channel_accepts_1d_scale():
    PerChannel().validate(
        np.zeros((4, 128), dtype=np.uint8),
        np.zeros((4,), dtype=np.float32),
    )


def test_per_channel_accepts_column_scale():
    PerChannel().validate(
        np.zeros((4, 128), dtype=np.uint8),
        np.zeros((4, 1), dtype=np.float32),
    )


def test_per_channel_rejects_wrong_M():
    with pytest.raises(ValueError, match="scale.shape"):
        PerChannel().validate(
            np.zeros((4, 128), dtype=np.uint8),
            np.zeros((8,), dtype=np.float32),
        )


def test_per_channel_rejects_full_2d_when_not_column():
    with pytest.raises(ValueError, match="scale.shape"):
        PerChannel().validate(
            np.zeros((4, 128), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
        )


def test_per_channel_rejects_1d_data():
    with pytest.raises(ValueError, match="ndim"):
        PerChannel().validate(
            np.zeros((4,), dtype=np.uint8),
            np.zeros((4,), dtype=np.float32),
        )


# ---------- PerBlockMN ----------


def test_per_block_mn_valid_2d():
    layout = PerBlockMN(block_m=1, block_n=32)
    data = np.zeros((4, 64), dtype=np.uint8)  # M//Bm=4, N//Bn=2
    scale = np.zeros((4, 2), dtype=np.float32)
    layout.validate(data, scale)


def test_per_block_mn_valid_with_block_m():
    layout = PerBlockMN(block_m=2, block_n=8)
    data = np.zeros((8, 64), dtype=np.uint8)  # M//Bm=4, N//Bn=8
    scale = np.zeros((4, 8), dtype=np.float32)
    layout.validate(data, scale)


def test_per_block_mn_rejects_misaligned_M():
    layout = PerBlockMN(block_m=4, block_n=32)
    with pytest.raises(ValueError, match="block_m"):
        layout.validate(
            np.zeros((5, 64), dtype=np.uint8),
            np.zeros((1, 2), dtype=np.float32),
        )


def test_per_block_mn_rejects_misaligned_N():
    layout = PerBlockMN(block_m=1, block_n=32)
    with pytest.raises(ValueError, match="block_n"):
        layout.validate(
            np.zeros((4, 50), dtype=np.uint8),
            np.zeros((4, 2), dtype=np.float32),
        )


def test_per_block_mn_rejects_wrong_scale_shape():
    layout = PerBlockMN(block_m=1, block_n=32)
    with pytest.raises(ValueError, match="scale.shape"):
        layout.validate(
            np.zeros((4, 64), dtype=np.uint8),  # expects (4, 2)
            np.zeros((4, 3), dtype=np.float32),
        )


def test_per_block_mn_block_size_validation():
    with pytest.raises(ValueError, match="block_"):
        PerBlockMN(block_m=0)
    with pytest.raises(ValueError, match="block_"):
        PerBlockMN(block_n=-1)


# ---------- Integration: ScaledTensor invokes layout.validate ----------


def test_scaledtensor_respects_layout_validation_valid():
    from breccia._core import ScaledTensor
    from breccia.recipes import Float8BlockScaling

    layout = PerBlockK(block_size=128)
    ScaledTensor(
        data=np.zeros((4, 256), dtype=np.uint8),
        scale=np.zeros((4, 2), dtype=np.float32),
        recipe=Float8BlockScaling(),
        layout=layout,
    )


def test_scaledtensor_respects_layout_validation_invalid():
    from breccia._core import ScaledTensor
    from breccia.recipes import Float8BlockScaling

    layout = PerBlockK(block_size=128)
    with pytest.raises(ValueError, match="scale.shape"):
        ScaledTensor(
            data=np.zeros((4, 256), dtype=np.uint8),
            scale=np.zeros((4, 1), dtype=np.float32),
            recipe=Float8BlockScaling(),
            layout=layout,
        )
