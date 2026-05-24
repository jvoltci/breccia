"""Bridge to/from NVIDIA TransformerEngine's ``Float8Tensor``.

TransformerEngine (``transformer_engine.pytorch``) stores quantized
tensors as a ``Float8Tensor`` whose public attributes are:

- ``_data``         : ``torch.Tensor`` of dtype ``torch.uint8`` containing the FP8 bytes
- ``_scale_inv``    : ``torch.Tensor`` of dtype ``torch.float32``, the *dequantization*
                       scale (multiply data by this to recover the high-precision value)
- ``_fp8_dtype``    : the TE dtype enum (``E4M3`` or ``E5M2``)

This convention matches breccia's: ``ScaledTensor.scale`` is the
dequantization scale. So the bridge is essentially a copy of the buffers
plus recipe + layout selection.

Supported recipes (TE → breccia mapping):

- TE ``DelayedScaling``         → breccia ``DelayedScaling``
- TE ``Float8CurrentScaling``   → breccia ``Float8CurrentScaling``
- TE ``Float8BlockScaling``     → breccia ``Float8BlockScaling``
- TE ``MXFP8BlockScaling``      → breccia ``MXFP8BlockScaling``

The bridge accepts a ``recipe`` argument; if omitted, we default to
``DelayedScaling`` (TE's most common case).

TransformerEngine installs only on Linux + CUDA. On other platforms,
calling these functions raises ``ImportError`` with a helpful message.
"""

from __future__ import annotations

from typing import Any, Optional

from breccia._core import ScaledTensor
from breccia.layouts import PerTensor
from breccia.recipes import (
    ScalingRecipe,
    DelayedScaling,
    Float8CurrentScaling,
)


def _require_te():
    try:
        import transformer_engine.pytorch as te  # noqa: F401

        return te
    except ImportError as e:
        raise ImportError(
            "TransformerEngine is required for the TE bridge. "
            "Install with: pip install transformer-engine "
            "(Linux + CUDA only)"
        ) from e


def _te_dtype_to_fp8_format(te_dtype: Any) -> str:
    """Map a TE FP8 dtype enum to breccia's ``"E4M3"`` / ``"E5M2"`` string."""
    name = repr(te_dtype).lower()
    if "e4m3" in name:
        return "E4M3"
    if "e5m2" in name:
        return "E5M2"
    raise ValueError(f"unknown TE FP8 dtype: {te_dtype!r}")


def from_transformer_engine(
    te_tensor: Any,
    recipe: Optional[ScalingRecipe] = None,
) -> ScaledTensor:
    """Wrap a TransformerEngine ``Float8Tensor`` as a breccia ``ScaledTensor``.

    No buffer copy: the resulting ``ScaledTensor`` shares ``_data`` and
    ``_scale_inv`` with the TE tensor. Recipe defaults to
    ``DelayedScaling`` with ``fp8_format`` inferred from the TE dtype.

    TE stores ``_scale_inv`` as a shape-``(1,)`` tensor (a 1-D
    one-element vector, not a scalar). breccia's ``PerTensor`` layout
    expects a 0-D scalar, so we squeeze to match.
    """
    _require_te()
    data = te_tensor._data
    scale = te_tensor._scale_inv
    # Coerce TE's (1,)-shape scale to 0-D scalar for PerTensor compatibility.
    if hasattr(scale, "squeeze") and scale.ndim == 1 and scale.shape[0] == 1:
        scale = scale.squeeze(0)
    if recipe is None:
        fp8_format = _te_dtype_to_fp8_format(getattr(te_tensor, "_fp8_dtype", "E4M3"))
        recipe = DelayedScaling(fp8_format=fp8_format)
    return ScaledTensor(data=data, scale=scale, recipe=recipe, layout=PerTensor())


def to_transformer_engine(scaled: ScaledTensor) -> Any:
    """Convert a per-tensor FP8 ``ScaledTensor`` to a TE ``Float8Tensor``.

    Restricted to per-tensor recipes (``DelayedScaling`` /
    ``Float8CurrentScaling``) in v0.0.1 — block-scaled / NVFP4 / INT4
    don't have stable TE counterparts yet.
    """
    te = _require_te()
    if not isinstance(scaled.recipe, (DelayedScaling, Float8CurrentScaling)):
        raise NotImplementedError(
            "to_transformer_engine v0.0.1 supports DelayedScaling / "
            f"Float8CurrentScaling only, got {type(scaled.recipe).__name__}"
        )

    fp8_dtype = getattr(te, "DType", None)
    if fp8_dtype is None:
        raise RuntimeError(
            "TransformerEngine layout has changed; this bridge needs an update"
        )
    # TE constructor location varies by version; try both common paths.
    try:
        from transformer_engine.pytorch.tensor import Float8Tensor
    except ImportError:
        from transformer_engine.pytorch import Float8Tensor  # type: ignore[attr-defined]

    fp8_enum = (
        fp8_dtype.kFloat8E4M3
        if scaled.recipe.fp8_format == "E4M3"
        else fp8_dtype.kFloat8E5M2
    )
    return Float8Tensor(
        data=scaled.data,
        fp8_scale_inv=scaled.scale,
        fp8_dtype=fp8_enum,
    )
