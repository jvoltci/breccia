"""Round-trip helpers between breccia's ScaledTensor and external formats.

Each module handles one external convention:

- ``_transformer_engine`` — NVIDIA TransformerEngine ``Float8Tensor``
- ``_torchao``            — PyTorch ``torchao.dtypes.AffineQuantizedTensor``
- ``_huggingface``        — HuggingFace ``safetensors`` with scale metadata
- ``_dlpack``             — zero-copy cross-framework tensor exchange
- ``_deepseek``           — DeepSeek-v3 FP8 block-scaled weight format

Each external dep is imported lazily so a base ``import breccia.bridges``
does not require any optional dep. Bridge functions raise a clear
``ImportError`` with install instructions if the dep is missing when
called.
"""

from ._transformer_engine import (
    from_transformer_engine,
    to_transformer_engine,
)
from ._torchao import from_torchao, to_torchao
from ._huggingface import save_safetensors, load_safetensors
from ._dlpack import to_dlpack, from_dlpack
from ._deepseek import from_deepseek_v3, to_deepseek_v3

__all__ = [
    "from_transformer_engine",
    "to_transformer_engine",
    "from_torchao",
    "to_torchao",
    "save_safetensors",
    "load_safetensors",
    "to_dlpack",
    "from_dlpack",
    "from_deepseek_v3",
    "to_deepseek_v3",
]
