"""Bridge to/from PyTorch ``torchao.dtypes.AffineQuantizedTensor``.

torchao is PyTorch's official quantization library
(``pip install torchao``). Its ``AffineQuantizedTensor`` carries:

- ``int_data``    : low-precision integer values (typically int8 or packed int4)
- ``scale``       : dequantization scale (per-tensor / per-channel / per-group)
- ``zero_point``  : optional zero-point for asymmetric quantization
- ``layout_type`` : ``PlainLayout`` / ``MarlinSparseLayout`` / etc.

For symmetric INT4 / INT8 (the most common case), this maps cleanly to
breccia's ``INT4Scaling`` / per-channel layout. Asymmetric quantization
(non-zero zero-point) is not modeled in breccia v0.0.1.

torchao installs on most platforms (Linux / macOS / Windows). The bridge
raises a clear error if torchao is missing.
"""

from __future__ import annotations

from typing import Any, Optional

from breccia._core import ScaledTensor
from breccia.layouts import PerChannel, PerBlockK
from breccia.recipes import ScalingRecipe, INT4Scaling


def _require_torchao():
    try:
        import torchao  # noqa: F401

        return torchao
    except ImportError as e:
        raise ImportError(
            "torchao is required for the torchao bridge. "
            "Install with: pip install torchao"
        ) from e


def from_torchao(
    aqt: Any,
    recipe: Optional[ScalingRecipe] = None,
) -> ScaledTensor:
    """Wrap a torchao ``AffineQuantizedTensor`` as a breccia ``ScaledTensor``.

    Only symmetric (zero_point=0) INT4 / INT8 is supported in v0.0.1.
    Asymmetric quantization will be added in v0.1.
    """
    _require_torchao()
    zero_point = getattr(aqt, "zero_point", None)
    if zero_point is not None and bool(
        (zero_point != 0).any() if hasattr(zero_point, "any") else zero_point != 0
    ):
        raise NotImplementedError(
            "torchao bridge v0.0.1 supports symmetric quantization only "
            "(zero_point=0); asymmetric quantization arrives in v0.1"
        )

    data = aqt.int_data
    scale = aqt.scale
    if recipe is None:
        # Default to INT4 group quantization with group size inferred from scale shape.
        group_size = data.shape[-1] // scale.shape[-1] if scale.ndim >= 2 else 128
        recipe = INT4Scaling(group_size=group_size)

    if isinstance(recipe, INT4Scaling):
        layout = PerBlockK(block_size=recipe.group_size)
    else:
        layout = PerChannel()

    return ScaledTensor(data=data, scale=scale, recipe=recipe, layout=layout)


def to_torchao(scaled: ScaledTensor) -> Any:
    """Convert an INT4/INT8 ``ScaledTensor`` to a torchao ``AffineQuantizedTensor``.

    Restricted to symmetric quantization (zero_point=0) in v0.0.1.
    """
    _require_torchao()
    if not isinstance(scaled.recipe, INT4Scaling):
        raise NotImplementedError(
            "to_torchao v0.0.1 supports INT4Scaling only, got "
            f"{type(scaled.recipe).__name__}"
        )

    try:
        from torchao.dtypes import AffineQuantizedTensor, PlainLayout
    except ImportError:
        # torchao restructured its dtypes module across versions; fall back gracefully.
        raise RuntimeError(
            "Could not import torchao.dtypes.AffineQuantizedTensor. "
            "The torchao API may have changed; this bridge needs an update."
        )

    return AffineQuantizedTensor(
        int_data=scaled.data,
        scale=scaled.scale,
        zero_point=None,
        layout_type=PlainLayout(),
    )
