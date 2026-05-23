"""Bridge to/from the DeepSeek-v3 FP8 block-scaled weight format.

DeepSeek-v3's released weights use FP8 E4M3 with per-128-element block
scaling along the K (contraction) dim. That's exactly breccia's
``Float8BlockScaling(block_k=128, fp8_format="E4M3")`` with
``PerBlockK(128)`` layout.

The bridge is therefore a thin wrapper around ``from_buffer`` that
asserts the format and validates the scale shape.
"""

from __future__ import annotations

from typing import Any

from breccia._core import ScaledTensor, from_buffer
from breccia.layouts import PerBlockK
from breccia.recipes import Float8BlockScaling


def from_deepseek_v3(
    data: Any,
    scale: Any,
    *,
    block_k: int = 128,
    fp8_format: str = "E4M3",
) -> ScaledTensor:
    """Wrap raw DeepSeek-v3-style FP8 buffers as a ``ScaledTensor``.

    Parameters
    ----------
    data
        FP8 byte buffer of shape ``(M, K)`` (or higher rank with K last).
    scale
        Per-block dequantization scales of shape ``(M, K // block_k)``,
        dtype float32.
    block_k
        Block size along K. DeepSeek-v3 uses 128.
    fp8_format
        ``"E4M3"`` (default) or ``"E5M2"``.

    Returns
    -------
    ScaledTensor
        Recipe is ``Float8BlockScaling(block_k, fp8_format)``;
        layout is ``PerBlockK(block_k)``.
    """
    recipe = Float8BlockScaling(block_k=block_k, fp8_format=fp8_format)
    layout = PerBlockK(block_size=block_k)
    return from_buffer(data=data, scale=scale, recipe=recipe, layout=layout)


def to_deepseek_v3(scaled: ScaledTensor) -> tuple:
    """Extract raw DeepSeek-v3-style ``(data, scale)`` buffers.

    Raises if the recipe is not a per-block-K FP8 recipe.
    """
    if not isinstance(scaled.recipe, Float8BlockScaling):
        raise NotImplementedError(
            "to_deepseek_v3 requires a Float8BlockScaling ScaledTensor, "
            f"got {type(scaled.recipe).__name__}"
        )
    if not isinstance(scaled.layout, PerBlockK):
        raise NotImplementedError(
            "to_deepseek_v3 requires PerBlockK layout, "
            f"got {type(scaled.layout).__name__}"
        )
    return scaled.data, scaled.scale
