"""Layout: how a scale tensor maps to a data tensor's blocks.

The Layout is the single source of truth for the shape relationship
between ``data`` and ``scale``. Each Layout subclass implements
``.validate(data, scale)`` and raises ``ValueError`` if the shapes don't
match. The ScaledTensor calls this validator at construction time.

Four layouts ship in v0.0.1:

========================== ========================================== ===============
Layout                     Scale shape for data shape ``(M, K)``      Used by
========================== ========================================== ===============
PerTensor                  ``()`` — a single scalar                   DelayedScaling, Float8CurrentScaling
PerBlockK(B)               ``(M, K // B)``                            Float8BlockScaling
PerChannel                 ``(M,)`` or ``(M, 1)``                     INT4Scaling
PerBlockMN(Bm, Bn)         ``(M // Bm, K // Bn)``                     MXFP8BlockScaling, NVFP4BlockScaling
========================== ========================================== ===============

For higher-rank data tensors the M / K positions are interpreted as the
last two axes; leading axes are passed through. So a 3-D activation of
shape ``(batch, M, K)`` with ``PerBlockK(B)`` has scale shape
``(batch, M, K // B)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar


class Layout:
    """Base class for layouts. Subclasses are frozen dataclasses.

    Subclasses MUST implement ``validate(data, scale) -> None``, raising
    ``ValueError`` if the scale's shape does not match what this layout
    expects given the data's shape.
    """

    name: ClassVar[str] = ""

    def validate(self, data: Any, scale: Any) -> None:  # pragma: no cover
        raise NotImplementedError("Layout subclasses must implement validate()")


@dataclass(frozen=True)
class PerTensor(Layout):
    """A single scalar scale for the entire tensor.

    Used by per-tensor recipes (``DelayedScaling``, ``Float8CurrentScaling``).
    """

    name: ClassVar[str] = "per_tensor"

    def validate(self, data: Any, scale: Any) -> None:
        if scale.ndim != 0:
            raise ValueError(
                f"PerTensor requires a scalar scale (ndim=0), "
                f"got ndim={scale.ndim} with shape={tuple(scale.shape)}"
            )


@dataclass(frozen=True)
class PerBlockK(Layout):
    """One scale per K-block along the last (contraction) dim.

    For data shape ``(..., M, K)`` and block size ``B``, scale shape is
    ``(..., M, K // B)``. Used by ``Float8BlockScaling``.
    """

    name: ClassVar[str] = "per_block_k"
    block_size: int = 128

    def __post_init__(self) -> None:
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {self.block_size}")

    def validate(self, data: Any, scale: Any) -> None:
        if data.ndim < 2:
            raise ValueError(
                f"PerBlockK requires data.ndim >= 2 (matmul shape), "
                f"got ndim={data.ndim}"
            )
        K = data.shape[-1]
        if K % self.block_size != 0:
            raise ValueError(
                f"PerBlockK: data.shape[-1]={K} must be divisible by "
                f"block_size={self.block_size}"
            )
        expected = tuple(data.shape[:-1]) + (K // self.block_size,)
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"PerBlockK: scale.shape must be {expected}, "
                f"got {tuple(scale.shape)}"
            )


@dataclass(frozen=True)
class PerChannel(Layout):
    """One scale per output channel (row).

    For data shape ``(..., M, K)``, scale shape is ``(M,)`` or
    ``(..., M, 1)``. Used by ``INT4Scaling`` weight-only quantization
    where each output row of a weight matrix gets its own scale.
    """

    name: ClassVar[str] = "per_channel"

    def validate(self, data: Any, scale: Any) -> None:
        if data.ndim < 2:
            raise ValueError(
                f"PerChannel requires data.ndim >= 2, got ndim={data.ndim}"
            )
        M = data.shape[-2]
        valid_shapes = {(M,), tuple(data.shape[:-1]) + (1,)}
        if tuple(scale.shape) not in valid_shapes:
            raise ValueError(
                f"PerChannel: scale.shape must be one of {sorted(valid_shapes)}, "
                f"got {tuple(scale.shape)}"
            )


@dataclass(frozen=True)
class PerBlockMN(Layout):
    """A 2-D grid of scales, one per ``(block_m, block_n)`` tile.

    For data shape ``(..., M, N)``, scale shape is
    ``(..., M // block_m, N // block_n)``. Used by ``MXFP8BlockScaling``
    (``block_m=1, block_n=32``) and ``NVFP4BlockScaling``
    (``block_m=1, block_n=16``).
    """

    name: ClassVar[str] = "per_block_mn"
    block_m: int = 1
    block_n: int = 32

    def __post_init__(self) -> None:
        if self.block_m <= 0 or self.block_n <= 0:
            raise ValueError(
                f"block_m and block_n must be > 0, got "
                f"({self.block_m}, {self.block_n})"
            )

    def validate(self, data: Any, scale: Any) -> None:
        if data.ndim < 2:
            raise ValueError(
                f"PerBlockMN requires data.ndim >= 2, got ndim={data.ndim}"
            )
        M = data.shape[-2]
        N = data.shape[-1]
        if M % self.block_m != 0:
            raise ValueError(
                f"PerBlockMN: data.shape[-2]={M} not divisible by "
                f"block_m={self.block_m}"
            )
        if N % self.block_n != 0:
            raise ValueError(
                f"PerBlockMN: data.shape[-1]={N} not divisible by "
                f"block_n={self.block_n}"
            )
        expected = tuple(data.shape[:-2]) + (M // self.block_m, N // self.block_n)
        if tuple(scale.shape) != expected:
            raise ValueError(
                f"PerBlockMN: scale.shape must be {expected}, "
                f"got {tuple(scale.shape)}"
            )
