"""Tests for breccia.bridges: TE, torchao, HF safetensors, DLPack, DeepSeek-v3.

Bridges to external deps (TE, torchao) are skipped if the dep is not
installed. HF / DLPack / DeepSeek are tested directly.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from breccia import (
    cast,
    dequantize,
    ScaledTensor,
    Float8CurrentScaling,
    Float8BlockScaling,
    INT4Scaling,
    NVFP4BlockScaling,
)
from breccia.bridges import (
    from_deepseek_v3,
    to_deepseek_v3,
    save_safetensors,
    load_safetensors,
    from_transformer_engine,
    to_transformer_engine,
    from_torchao,
    to_torchao,
    to_dlpack,
    from_dlpack,
)


def _cos_sim(a, b):
    a = np.asarray(a).astype(np.float64).ravel()
    b = np.asarray(b).astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# ---------- DeepSeek-v3 (no external dep) ----------


def test_deepseek_round_trip():
    np.random.seed(0)
    x = np.random.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, Float8BlockScaling(block_k=128))
    data, scale = to_deepseek_v3(st)
    st2 = from_deepseek_v3(data, scale, block_k=128)
    out = dequantize(st2)
    assert out.shape == x.shape
    assert _cos_sim(out, x) > 0.99


def test_deepseek_rejects_wrong_recipe():
    np.random.seed(0)
    x = np.random.standard_normal((4, 128)).astype(np.float32)
    st = cast(x, Float8CurrentScaling())  # not a block recipe
    with pytest.raises(NotImplementedError, match="Float8BlockScaling"):
        to_deepseek_v3(st)


# ---------- HuggingFace safetensors ----------


def test_hf_save_load_round_trip(tmp_path):
    np.random.seed(0)
    weight = cast(np.random.standard_normal((4, 64)).astype(np.float32), Float8CurrentScaling())
    activations = cast(
        np.random.standard_normal((8, 128)).astype(np.float32),
        Float8BlockScaling(block_k=128),
    )

    path = str(tmp_path / "checkpoint.safetensors")
    save_safetensors({"weight": weight, "act": activations}, path)
    loaded = load_safetensors(path)

    assert set(loaded) == {"weight", "act"}
    assert type(loaded["weight"].recipe).__name__ == "Float8CurrentScaling"
    assert type(loaded["act"].recipe).__name__ == "Float8BlockScaling"
    assert loaded["act"].recipe.block_k == 128

    # Round-trip preserves data: dequantize before and after agree.
    out_before = dequantize(weight)
    out_after = dequantize(loaded["weight"])
    np.testing.assert_allclose(
        np.asarray(out_before),
        np.asarray(out_after),
        atol=1e-5,
    )


def test_hf_save_with_extra_metadata(tmp_path):
    np.random.seed(0)
    st = cast(np.random.standard_normal((4, 64)).astype(np.float32), Float8CurrentScaling())
    path = str(tmp_path / "ckpt.safetensors")
    save_safetensors({"w": st}, path, extra_metadata={"model_version": "v0.0.1"})

    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        md = f.metadata()
    assert md["model_version"] == "v0.0.1"
    assert "w.config" in md


def test_hf_load_skips_tensors_without_config(tmp_path):
    """Tensors saved without breccia's `.config` metadata are silently ignored."""
    import torch
    from safetensors.torch import save_file

    path = str(tmp_path / "mixed.safetensors")
    save_file({"foo.data": torch.zeros(4, dtype=torch.uint8)}, path, metadata=None)
    loaded = load_safetensors(path)
    assert loaded == {}


def test_hf_round_trip_int4(tmp_path):
    np.random.seed(0)
    x = np.random.standard_normal((4, 256)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128))
    path = str(tmp_path / "int4.safetensors")
    save_safetensors({"w": st}, path)
    loaded = load_safetensors(path)
    assert type(loaded["w"].recipe).__name__ == "INT4Scaling"
    assert loaded["w"].recipe.group_size == 128


# ---------- DLPack ----------


def test_dlpack_torch_to_numpy():
    pytest.importorskip("torch")
    import torch

    np.random.seed(0)
    x = torch.randn(4, 64, dtype=torch.float32)
    st = cast(x, Float8CurrentScaling())
    # Round-trip via DLPack: torch → numpy → ScaledTensor (numpy-backed).
    st_np = from_dlpack(st, framework="numpy")
    assert type(st_np.data).__module__.startswith("numpy")
    out = dequantize(st_np)
    assert isinstance(out, np.ndarray)


def test_dlpack_returns_capsules():
    """Plain to_dlpack returns the raw capsules (consumed by any framework)."""
    pytest.importorskip("torch")
    import torch

    x = torch.randn(4, 64, dtype=torch.float32)
    st = cast(x, Float8CurrentScaling())
    data_cap, scale_cap = to_dlpack(st)
    # DLPack capsules are opaque Python objects of type PyCapsule.
    assert "PyCapsule" in type(data_cap).__name__ or hasattr(data_cap, "__dlpack__")


# ---------- TransformerEngine (skipped on macOS / non-CUDA) ----------


te_skip = pytest.mark.skipif(
    True,  # noqa: B011
    reason="TransformerEngine bridge requires Linux + CUDA + TE install; not testable in CI on macOS.",
)


@te_skip
def test_te_round_trip():
    pytest.importorskip("transformer_engine")
    import transformer_engine.pytorch as te
    import torch

    x = torch.randn(4, 64, dtype=torch.float32, device="cuda")
    # Pseudocode: construct a TE Float8Tensor with default DelayedScaling
    # then bridge to breccia and back.
    raise NotImplementedError("TE bridge runtime test requires a CUDA box")


def test_te_raises_clear_error_without_install():
    """Calling the bridge without TE installed must raise ImportError."""
    np.random.seed(0)
    st = cast(np.random.standard_normal((4, 64)).astype(np.float32), Float8CurrentScaling())
    # If TE happens to be installed in CI, skip rather than test the error path.
    try:
        import transformer_engine  # noqa: F401

        pytest.skip("TE is installed; error-path test only runs when absent")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="TransformerEngine"):
        to_transformer_engine(st)


# ---------- torchao (skipped if not installed) ----------


def test_torchao_raises_clear_error_without_install():
    np.random.seed(0)
    x = np.random.standard_normal((4, 128)).astype(np.float32)
    st = cast(x, INT4Scaling(group_size=128))
    try:
        import torchao  # noqa: F401

        pytest.skip("torchao is installed; error-path test only runs when absent")
    except ImportError:
        pass
    with pytest.raises(ImportError, match="torchao"):
        to_torchao(st)
