"""breccia — cross-framework block-scaled tensor primitive."""

from ._core import (
    ScaledTensor,
    from_buffer,
)
from .recipes import (
    ScalingRecipe,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)

__version__ = "0.0.1"

__all__ = [
    "ScaledTensor",
    "from_buffer",
    "ScalingRecipe",
    "DelayedScaling",
    "Float8CurrentScaling",
    "Float8BlockScaling",
    "MXFP8BlockScaling",
    "NVFP4BlockScaling",
    "INT4Scaling",
    "__version__",
]
