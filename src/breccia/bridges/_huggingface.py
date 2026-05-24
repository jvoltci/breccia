"""Save/load a ``ScaledTensor`` as a safetensors file with scale metadata.

The convention:

- The data buffer is stored as a tensor named ``f"{name}.data"``.
- The scale buffer is stored as a tensor named ``f"{name}.scale"``.
- Recipe + layout configuration is serialized into the safetensors
  ``metadata`` dict under the key ``f"{name}.config"``, JSON-encoded.

This lets HuggingFace ``transformers`` / ``diffusers`` / ``accelerate``
users load and resave quantized weights without losing scale semantics.

Multiple ScaledTensors can be stored in one file by varying ``name``.

The file format is plain safetensors — readable by any safetensors
reader; the breccia-specific information is in ``metadata`` only,
which other readers ignore.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Optional

import numpy as np

from breccia._core import ScaledTensor
from breccia.layouts import (
    Layout,
    PerTensor,
    PerBlockK,
    PerChannel,
    PerBlockMN,
)
from breccia.recipes import (
    ScalingRecipe,
    DelayedScaling,
    Float8CurrentScaling,
    Float8BlockScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)


_RECIPE_REGISTRY: Dict[str, type] = {
    "delayed": DelayedScaling,
    "current": Float8CurrentScaling,
    "block": Float8BlockScaling,
    "mxfp8": MXFP8BlockScaling,
    "nvfp4": NVFP4BlockScaling,
    "int4": INT4Scaling,
}

_LAYOUT_REGISTRY: Dict[str, type] = {
    "per_tensor": PerTensor,
    "per_block_k": PerBlockK,
    "per_channel": PerChannel,
    "per_block_mn": PerBlockMN,
}


def _require_safetensors():
    try:
        import safetensors  # noqa: F401

        return safetensors
    except ImportError as e:
        raise ImportError(
            "safetensors is required for the HF bridge. "
            "Install with: pip install safetensors"
        ) from e


def _config_to_json(recipe: ScalingRecipe, layout: Layout) -> str:
    return json.dumps(
        {
            "recipe": {
                "name": recipe.name,
                "fields": {f.name: getattr(recipe, f.name) for f in _dc_fields(recipe)},
            },
            "layout": {
                "name": layout.name,
                "fields": {f.name: getattr(layout, f.name) for f in _dc_fields(layout)},
            },
        }
    )


def _json_to_config(s: str) -> tuple:
    obj = json.loads(s)
    recipe_cls = _RECIPE_REGISTRY[obj["recipe"]["name"]]
    recipe = recipe_cls(**obj["recipe"]["fields"])
    layout_cls = _LAYOUT_REGISTRY[obj["layout"]["name"]]
    layout = layout_cls(**obj["layout"]["fields"])
    return recipe, layout


def _dc_fields(obj: Any):
    import dataclasses

    if dataclasses.is_dataclass(obj):
        return dataclasses.fields(obj)
    return []


def _to_torch(arr: Any):
    """Coerce a NumPy or framework array into a torch tensor for safetensors."""
    import torch

    if isinstance(arr, torch.Tensor):
        return arr.contiguous()
    if hasattr(arr, "__array__"):
        return torch.as_tensor(np.array(arr)).contiguous()
    return torch.as_tensor(np.asarray(arr)).contiguous()


def save_safetensors(
    scaled_tensors: Mapping[str, ScaledTensor],
    path: str,
    extra_metadata: Optional[Dict[str, str]] = None,
) -> None:
    """Save a dict of ``ScaledTensor`` objects to a safetensors file.

    Parameters
    ----------
    scaled_tensors
        Mapping from ``name`` to ``ScaledTensor``. The data and scale buffers
        are saved as ``f"{name}.data"`` and ``f"{name}.scale"``.
    path
        Output filesystem path. Existing files are overwritten.
    extra_metadata
        Additional string metadata entries to merge into the safetensors
        header (e.g., user comments, model version).
    """
    _require_safetensors()
    from safetensors.torch import save_file

    tensors: Dict[str, Any] = {}
    metadata: Dict[str, str] = dict(extra_metadata or {})

    for name, st in scaled_tensors.items():
        tensors[f"{name}.data"] = _to_torch(st.data)
        tensors[f"{name}.scale"] = _to_torch(st.scale)
        if st.zero_point is not None:
            tensors[f"{name}.zero_point"] = _to_torch(st.zero_point)
        metadata[f"{name}.config"] = _config_to_json(st.recipe, st.layout)

    save_file(tensors, path, metadata=metadata)


def load_safetensors(path: str) -> Dict[str, ScaledTensor]:
    """Load ``ScaledTensor`` objects from a safetensors file.

    Returns
    -------
    dict
        Mapping from ``name`` to ``ScaledTensor`` for every name that has
        both a ``.data`` tensor and a ``.config`` metadata entry. Tensors
        without breccia config metadata are silently skipped.
    """
    _require_safetensors()
    from safetensors.torch import load_file
    from safetensors import safe_open

    tensors = load_file(path)
    with safe_open(path, framework="pt") as f:
        metadata = f.metadata() or {}

    out: Dict[str, ScaledTensor] = {}
    config_keys = [k for k in metadata if k.endswith(".config")]
    for cfg_key in config_keys:
        name = cfg_key[: -len(".config")]
        data_key = f"{name}.data"
        scale_key = f"{name}.scale"
        if data_key not in tensors or scale_key not in tensors:
            continue
        recipe, layout = _json_to_config(metadata[cfg_key])
        zp_key = f"{name}.zero_point"
        zero_point = tensors.get(zp_key, None)
        out[name] = ScaledTensor(
            data=tensors[data_key],
            scale=tensors[scale_key],
            recipe=recipe,
            layout=layout,
            zero_point=zero_point,
        )
    return out
