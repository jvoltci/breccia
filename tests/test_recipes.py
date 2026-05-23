"""Tests for breccia.recipes: 6 ScalingRecipe variants."""

import dataclasses

import pytest

from breccia.recipes import (
    ScalingRecipe,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)


ALL_RECIPES = [
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
]


# ---------- Base ----------


def test_all_recipes_subclass_base():
    for cls in ALL_RECIPES:
        assert issubclass(cls, ScalingRecipe)


def test_all_recipes_have_distinct_names():
    names = [cls.name for cls in ALL_RECIPES]
    assert len(names) == len(set(names)), f"duplicate names: {names}"
    expected = {"delayed", "current", "block", "mxfp8", "nvfp4", "int4"}
    assert set(names) == expected


def test_all_recipes_construct_with_defaults():
    for cls in ALL_RECIPES:
        cls()


def test_recipes_are_frozen():
    r = DelayedScaling()
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.fp8_format = "E5M2"


def test_recipes_are_hashable():
    r1 = DelayedScaling()
    r2 = DelayedScaling()
    assert hash(r1) == hash(r2)
    assert r1 == r2
    assert {r1, r2} == {r1}


def test_recipes_distinguishable():
    assert DelayedScaling() != Float8CurrentScaling()
    assert DelayedScaling(fp8_format="E4M3") != DelayedScaling(fp8_format="E5M2")


# ---------- DelayedScaling ----------


def test_delayed_fp8_format_validation():
    DelayedScaling(fp8_format="E4M3")
    DelayedScaling(fp8_format="E5M2")
    with pytest.raises(ValueError, match="E4M3.*E5M2"):
        DelayedScaling(fp8_format="BF16")


def test_delayed_history_validation():
    DelayedScaling(amax_history_len=1)
    DelayedScaling(amax_history_len=128)
    with pytest.raises(ValueError, match="amax_history_len"):
        DelayedScaling(amax_history_len=0)
    with pytest.raises(ValueError, match="amax_history_len"):
        DelayedScaling(amax_history_len=-5)


def test_delayed_margin_default():
    r = DelayedScaling()
    assert r.margin == 0


# ---------- Float8CurrentScaling ----------


def test_current_fp8_format_validation():
    Float8CurrentScaling(fp8_format="E4M3")
    Float8CurrentScaling(fp8_format="E5M2")
    with pytest.raises(ValueError):
        Float8CurrentScaling(fp8_format="FP16")


# ---------- Float8BlockScaling ----------


def test_block_block_k_default():
    r = Float8BlockScaling()
    assert r.block_k == 128


def test_block_block_k_validation():
    Float8BlockScaling(block_k=64)
    Float8BlockScaling(block_k=128)
    with pytest.raises(ValueError, match="block_k"):
        Float8BlockScaling(block_k=0)
    with pytest.raises(ValueError, match="block_k"):
        Float8BlockScaling(block_k=-1)


# ---------- MXFP8BlockScaling ----------


def test_mxfp8_block_size_is_fixed():
    MXFP8BlockScaling()
    MXFP8BlockScaling(block_size=32)
    with pytest.raises(ValueError, match="32"):
        MXFP8BlockScaling(block_size=64)
    with pytest.raises(ValueError, match="32"):
        MXFP8BlockScaling(block_size=16)


# ---------- NVFP4BlockScaling ----------


def test_nvfp4_defaults():
    r = NVFP4BlockScaling()
    assert r.fp4_format == "E2M1"
    assert r.block_size == 16
    assert r.scale_format == "E4M3"


def test_nvfp4_all_fields_are_fixed():
    with pytest.raises(ValueError, match="E2M1"):
        NVFP4BlockScaling(fp4_format="E1M2")
    with pytest.raises(ValueError, match="block_size"):
        NVFP4BlockScaling(block_size=32)
    with pytest.raises(ValueError, match="E4M3"):
        NVFP4BlockScaling(scale_format="E5M2")


# ---------- INT4Scaling ----------


def test_int4_defaults():
    r = INT4Scaling()
    assert r.group_size == 128
    assert r.signed is True
    assert r.scale_dtype == "fp16"


def test_int4_group_size_validation():
    INT4Scaling(group_size=64)
    INT4Scaling(group_size=256)
    with pytest.raises(ValueError, match="group_size"):
        INT4Scaling(group_size=0)


def test_int4_scale_dtype_validation():
    INT4Scaling(scale_dtype="fp16")
    INT4Scaling(scale_dtype="bf16")
    INT4Scaling(scale_dtype="fp32")
    with pytest.raises(ValueError, match="scale_dtype"):
        INT4Scaling(scale_dtype="fp8")


def test_int4_unsigned_variant():
    r = INT4Scaling(signed=False)
    assert r.signed is False


# ---------- Name attribute mapping ----------


def test_name_mapping():
    assert DelayedScaling.name == "delayed"
    assert Float8CurrentScaling.name == "current"
    assert Float8BlockScaling.name == "block"
    assert MXFP8BlockScaling.name == "mxfp8"
    assert NVFP4BlockScaling.name == "nvfp4"
    assert INT4Scaling.name == "int4"
