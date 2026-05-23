"""Zero-copy data/scale exchange via the DLPack protocol.

DLPack (https://github.com/dmlc/dlpack) is the standard cross-framework
zero-copy tensor protocol. NumPy, PyTorch, MLX, and JAX all implement
the ``__dlpack__`` / ``__dlpack_device__`` interface.

These helpers move a ScaledTensor's data and scale between frameworks
without copying bytes when devices are compatible.
"""

from __future__ import annotations

from typing import Any

from breccia._core import ScaledTensor, _is_torch, _is_mlx, _is_jax


def to_dlpack(scaled: ScaledTensor) -> tuple:
    """Return ``(data_capsule, scale_capsule)`` DLPack capsules for the buffers.

    The returned capsules can be consumed by any DLPack-aware framework's
    ``from_dlpack`` function.
    """
    return scaled.data.__dlpack__(), scaled.scale.__dlpack__()


def _dlpack_to(framework: str, source: Any) -> Any:
    """Consume a DLPack-compatible source into the target framework.

    Most ``from_dlpack`` implementations want the source object (which
    has ``__dlpack__``) rather than a raw capsule. Pass the tensor
    itself; the receiving framework calls ``__dlpack__`` internally.
    """
    if framework == "torch":
        import torch

        return torch.from_dlpack(source)
    if framework == "numpy":
        import numpy as np

        return np.from_dlpack(source)
    if framework == "mlx":
        import mlx.core as mx

        return mx.from_dlpack(source)
    if framework == "jax":
        from jax.dlpack import from_dlpack as jax_from_dlpack

        return jax_from_dlpack(source)
    raise ValueError(f"unsupported framework: {framework!r}")


def from_dlpack(
    scaled: ScaledTensor,
    framework: str,
) -> ScaledTensor:
    """Move a ScaledTensor's buffers to a new framework via DLPack.

    ``framework`` is one of: ``"numpy"``, ``"torch"``, ``"mlx"``, ``"jax"``.
    Recipe and layout are unchanged.
    """
    return ScaledTensor(
        data=_dlpack_to(framework, scaled.data),
        scale=_dlpack_to(framework, scaled.scale),
        recipe=scaled.recipe,
        layout=scaled.layout,
    )
