"""ScalingRecipe: declarative configurations for low-precision quantization.

A recipe is **pure metadata** — it describes how a tensor was quantized
(format, block size, etc.) but contains no quantization behavior itself.
The behavior lives in :mod:`breccia.kernels.reference.cast` and dispatches
on recipe type.

Six recipes ship in v0.0.1:

================ ========================================== ===============================
Recipe           When you'd use it                          Origin
================ ========================================== ===============================
DelayedScaling   FP8 training with TE-style amax history    NVIDIA TransformerEngine
Float8Current... FP8 training with synchronous amax         TE / torchao
Float8BlockSc... FP8 weights with per-K-block scales        DeepSeek-v3 (block_k=128)
MXFP8BlockSc...  Hardware microscaling MXFP8                OCP MX standard (block_size=32)
NVFP4BlockSc...  Blackwell-class FP4 inference / training   NVIDIA Blackwell (block_size=16)
INT4Scaling      INT4 weight-only inference (GPTQ / AWQ)    Group-quant family (group=128)
================ ========================================== ===============================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar


class ScalingRecipe:
    """Base class for all recipes. Subclasses are frozen dataclasses.

    Every recipe carries a ``name`` class attribute used as a stable
    identifier for serialization, dispatch, and logging.
    """

    name: ClassVar[str] = ""


def _check_fp8_format(fmt: str) -> None:
    if fmt not in {"E4M3", "E5M2"}:
        raise ValueError(f"fp8_format must be 'E4M3' or 'E5M2', got {fmt!r}")


@dataclass(frozen=True)
class DelayedScaling(ScalingRecipe):
    """TransformerEngine-style delayed scaling.

    Uses a rolling history of recent ``amax`` values to compute the next
    scale, avoiding a synchronous reduction over the current tensor every
    step. The history is owned by the training loop, not the recipe itself
    (which is metadata).

    Parameters
    ----------
    fp8_format : {"E4M3", "E5M2"}
        FP8 representation. E4M3 has more precision, narrower range;
        E5M2 has wider range, less precision. E4M3 is the typical choice
        for forward / weights; E5M2 for backward gradients.
    amax_history_len : int
        How many previous amax values the training loop is expected to
        retain. The recipe records this number for portability; the
        actual buffer lives outside the recipe.
    margin : int
        Power-of-two exponent margin applied to the computed scale to
        reduce overflow risk. ``margin=0`` is the TE default.
    """

    name: ClassVar[str] = "delayed"
    fp8_format: str = "E4M3"
    amax_history_len: int = 16
    margin: int = 0

    def __post_init__(self) -> None:
        _check_fp8_format(self.fp8_format)
        if self.amax_history_len < 1:
            raise ValueError(
                f"amax_history_len must be >= 1, got {self.amax_history_len}"
            )


@dataclass(frozen=True)
class Float8CurrentScaling(ScalingRecipe):
    """Per-tensor FP8 scaling computed from the current tensor's amax.

    Cheapest recipe to reason about: ``scale = fp8_max / amax(x)``.
    Synchronous (every cast is a reduction), so slightly slower than
    delayed scaling in training; favored for inference / weights.
    """

    name: ClassVar[str] = "current"
    fp8_format: str = "E4M3"

    def __post_init__(self) -> None:
        _check_fp8_format(self.fp8_format)


@dataclass(frozen=True)
class Float8BlockScaling(ScalingRecipe):
    """FP8 with one scale per block along the K (contraction) dim.

    Used by DeepSeek-v3 (``block_k=128``) and TE's
    ``Float8BlockScaling``. Better dynamic range than per-tensor when
    distribution varies across rows of the weight matrix.
    """

    name: ClassVar[str] = "block"
    fp8_format: str = "E4M3"
    block_k: int = 128

    def __post_init__(self) -> None:
        _check_fp8_format(self.fp8_format)
        if self.block_k <= 0:
            raise ValueError(f"block_k must be > 0, got {self.block_k}")


@dataclass(frozen=True)
class MXFP8BlockScaling(ScalingRecipe):
    """OCP MX microscaling FP8: 32-element blocks with E8M0 (uint8) scale.

    The block size and scale encoding are fixed by the OCP MX spec; we
    raise on any other value to keep stored tensors interoperable with
    hardware MX implementations.
    """

    name: ClassVar[str] = "mxfp8"
    fp8_format: str = "E4M3"
    block_size: int = 32

    def __post_init__(self) -> None:
        _check_fp8_format(self.fp8_format)
        if self.block_size != 32:
            raise ValueError(
                "MXFP8 block_size is fixed at 32 by the OCP MX spec, "
                f"got {self.block_size}"
            )


@dataclass(frozen=True)
class NVFP4BlockScaling(ScalingRecipe):
    """NVIDIA Blackwell NVFP4: FP4 E2M1 in 16-element blocks with FP8 E4M3 scale.

    The 16-element block size and the E4M3 scale encoding are fixed by
    the Blackwell hardware spec.
    """

    name: ClassVar[str] = "nvfp4"
    fp4_format: str = "E2M1"
    block_size: int = 16
    scale_format: str = "E4M3"

    def __post_init__(self) -> None:
        if self.fp4_format != "E2M1":
            raise ValueError(
                f"NVFP4 fp4_format is fixed at 'E2M1', got {self.fp4_format!r}"
            )
        if self.block_size != 16:
            raise ValueError(f"NVFP4 block_size is fixed at 16, got {self.block_size}")
        if self.scale_format != "E4M3":
            raise ValueError(
                f"NVFP4 scale_format is fixed at 'E4M3', got {self.scale_format!r}"
            )


@dataclass(frozen=True)
class INT4Scaling(ScalingRecipe):
    """INT4 weight-only quantization (GPTQ / AWQ family).

    Groups of ``group_size`` weights along the K dim share one scale (and
    optionally a zero-point — not modeled in v0.0.1, defer to v0.1).
    """

    name: ClassVar[str] = "int4"
    group_size: int = 128
    signed: bool = True
    scale_dtype: str = "fp16"

    def __post_init__(self) -> None:
        if self.group_size <= 0:
            raise ValueError(f"group_size must be > 0, got {self.group_size}")
        if self.scale_dtype not in {"fp16", "bf16", "fp32"}:
            raise ValueError(
                "scale_dtype must be one of 'fp16', 'bf16', 'fp32', "
                f"got {self.scale_dtype!r}"
            )
