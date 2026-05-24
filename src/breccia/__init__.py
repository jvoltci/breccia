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
from .layouts import (
    Layout,
    PerTensor,
    PerBlockK,
    PerChannel,
    PerBlockMN,
)
from .kernels.reference import cast, dequantize, requantize, matmul
from .autograd import cast_ste, cast_ste_clipped

__version__ = "0.1.1"

__all__ = [
    "ScaledTensor",
    "from_buffer",
    "cast",
    "dequantize",
    "requantize",
    "matmul",
    "cast_ste",
    "cast_ste_clipped",
    "ScalingRecipe",
    "DelayedScaling",
    "Float8CurrentScaling",
    "Float8BlockScaling",
    "MXFP8BlockScaling",
    "NVFP4BlockScaling",
    "INT4Scaling",
    "Layout",
    "PerTensor",
    "PerBlockK",
    "PerChannel",
    "PerBlockMN",
    "__version__",
]
