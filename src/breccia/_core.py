"""Core ScaledTensor type and construction operations.

ScaledTensor is breccia's only data structure. It packs:

- ``data``: the low-precision bytes (FP8 native dtype, or uint8 for packed
  FP4/INT4/MX formats)
- ``scale``: the scale tensor that gives the data its high-precision meaning
- ``recipe``: an instance of a ScalingRecipe (see ``breccia.recipes``)
  describing how the data was quantized
- ``layout``: an instance of a Layout (see ``breccia.layouts``) describing
  how the scale maps to data blocks

The recipe and layout are first-class state. They travel with the tensor
through sharding, checkpointing, and matmul — they don't need to be re-derived
at every framework boundary.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple


def _is_torch(x: Any) -> bool:
    return type(x).__module__.startswith("torch")


def _is_mlx(x: Any) -> bool:
    return type(x).__module__.startswith("mlx")


def _is_jax(x: Any) -> bool:
    mod = type(x).__module__
    return mod.startswith("jax") or mod.startswith("jaxlib")


@dataclass(frozen=True)
class ScaledTensor:
    """A low-precision tensor packed with its scale, recipe, and layout.

    Construct via ``breccia.cast(x, recipe)`` for the common path (quantize
    a high-precision tensor) or via ``breccia.from_buffer(...)`` when the
    pieces already exist (e.g., loaded from a checkpoint).

    The four fields are immutable. Cross-field validation (e.g., scale shape
    matching the layout) is delegated to ``layout.validate(data, scale)`` if
    that method exists on the layout. This keeps ``_core`` independent of
    the concrete layout types defined in :mod:`breccia.layouts`.

    Attributes
    ----------
    data : array-like
        The low-precision values. May be a native FP8 dtype on PyTorch
        (``torch.float8_e4m3fn`` / ``torch.float8_e5m2``) or ``uint8``-packed
        bytes for FP4 / INT4 / MX formats and for backends without native
        FP8 (NumPy, MLX).
    scale : array-like
        The scale tensor. Shape depends on the layout (scalar for
        ``PerTensor``, per-block for ``PerBlockK`` / ``PerBlockMN``, etc.).
    recipe : ScalingRecipe
        The quantization recipe used. Carries the format identifier and
        any recipe-specific parameters (e.g., ``amax_history`` for delayed
        scaling).
    layout : Layout
        How the scale maps to data blocks.

    Examples
    --------
    >>> import numpy as np
    >>> from breccia import from_buffer
    >>> from breccia.recipes import Float8CurrentScaling
    >>> from breccia.layouts import PerTensor
    >>> data = np.zeros((4, 8), dtype=np.uint8)
    >>> scale = np.float32(1.0)
    >>> st = from_buffer(data, scale, Float8CurrentScaling(), PerTensor())
    >>> st.shape
    (4, 8)
    """

    data: Any
    scale: Any
    recipe: Any
    layout: Any

    def __post_init__(self) -> None:
        if not hasattr(self.data, "shape") or not hasattr(self.data, "dtype"):
            raise TypeError(
                f"data must be array-like with .shape and .dtype, "
                f"got {type(self.data).__name__}"
            )
        if not hasattr(self.scale, "shape") or not hasattr(self.scale, "dtype"):
            raise TypeError(
                f"scale must be array-like with .shape and .dtype, "
                f"got {type(self.scale).__name__}"
            )
        if self.data.ndim < 1:
            raise ValueError(f"data must be at least 1-D, got {self.data.ndim}-D")
        if self.recipe is None:
            raise ValueError("recipe is required (see breccia.recipes)")
        if self.layout is None:
            raise ValueError("layout is required (see breccia.layouts)")

        validate = getattr(self.layout, "validate", None)
        if callable(validate):
            validate(self.data, self.scale)

    @property
    def shape(self) -> Tuple[int, ...]:
        """The shape of the logical (high-precision) tensor.

        Note: for FP4 / INT4 formats packed into uint8 bytes, ``data.shape``
        may differ from this — two 4-bit values share each byte. Recipes
        and bridges that produce packed data populate ``shape`` to mean the
        logical shape, not the byte-buffer shape.
        """
        return tuple(self.data.shape)

    @property
    def data_dtype(self) -> Any:
        return self.data.dtype

    @property
    def scale_dtype(self) -> Any:
        return self.scale.dtype

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def __repr__(self) -> str:
        scale_shape = tuple(self.scale.shape)
        return (
            f"breccia.ScaledTensor(shape={self.shape}, "
            f"data_dtype={self.data_dtype}, "
            f"scale_shape={scale_shape}, "
            f"recipe={type(self.recipe).__name__}, "
            f"layout={type(self.layout).__name__})"
        )


def from_buffer(data: Any, scale: Any, recipe: Any, layout: Any) -> ScaledTensor:
    """Construct a ScaledTensor from existing buffers without copying.

    Use this when you already have a quantized data buffer and a scale
    tensor (e.g., loaded from a safetensors checkpoint or handed back from
    a vendor library) and want to wrap them in the breccia primitive
    without re-quantizing.

    Parameters
    ----------
    data : array-like
        The low-precision data buffer.
    scale : array-like
        The scale tensor. Shape must satisfy the ``layout``'s validation.
    recipe : ScalingRecipe
        The recipe that produced this data.
    layout : Layout
        How the scale maps to the data.

    Returns
    -------
    ScaledTensor
    """
    return ScaledTensor(data=data, scale=scale, recipe=recipe, layout=layout)
