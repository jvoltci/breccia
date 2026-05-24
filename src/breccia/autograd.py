"""Autograd-aware quantization via the straight-through estimator (STE).

The bare :func:`breccia.cast` is non-differentiable through the
round-to-nearest step — the gradient w.r.t. ``x`` is zero. For training
loops that need gradient flow back to the high-precision parameters,
this module provides an STE wrapper.

The STE trick:

    forward:  y = dequantize(cast(x, recipe))
    backward: dx = dy  (pretend the cast was the identity function)

Implementations per backend:

- **PyTorch**: ``x + (y - x).detach()`` — gradients flow through ``x``,
  the quantization noise ``(y - x)`` is detached.
- **JAX**: ``x + jax.lax.stop_gradient(y - x)`` — equivalent.
- **NumPy** / **MLX**: no autograd → identity (same as the round-trip).

For training in FP8, the standard pattern is to wrap the cast in
``cast_ste``:

.. code-block:: python

    import breccia
    from breccia.autograd import cast_ste

    # Forward: quantize-aware activations and weights.
    A_q = cast_ste(A, breccia.Float8CurrentScaling())
    W_q = cast_ste(W, breccia.Float8CurrentScaling())
    y = breccia.matmul(A_q, W_q)
    loss = (y - target).pow(2).sum()
    loss.backward()  # gradients flow back to A and W through the cast

A note on clipping: a strict STE lets unbounded gradient flow through
values that saturated to the format's max. For most workloads this is
fine (saturation is rare with proper scaling). If your loss explodes,
use :func:`cast_ste_clipped` which zeros the gradient for values
outside ``[-fp_max, fp_max] * scale``.
"""

from __future__ import annotations

from typing import Any

from breccia._core import _is_torch, _is_jax, ScaledTensor
from breccia.kernels.reference.cast import cast, dequantize


def cast_ste(x: Any, recipe: Any) -> Any:
    """Quantization-aware cast with straight-through-estimator gradients.

    Returns a high-precision tensor with the quantization noise baked in.
    On a backward pass, the gradient w.r.t. ``x`` is passed through as
    if the cast were the identity function.

    Unlike :func:`breccia.cast`, this does NOT return a ``ScaledTensor``
    — the goal is to integrate into an autograd graph that operates on
    high-precision tensors. Internally we still build a ScaledTensor
    and dequantize it, but the wrapper hides that.

    Parameters
    ----------
    x : array (NumPy / PyTorch / MLX / JAX)
        High-precision input.
    recipe : ScalingRecipe
        Same as :func:`breccia.cast`.

    Returns
    -------
    array
        Same backend, same shape as ``x``. The values match
        ``dequantize(cast(x, recipe))``; the gradient w.r.t. ``x`` is
        the identity.
    """
    if _is_torch(x):
        return _cast_ste_torch(x, recipe)
    if _is_jax(x):
        return _cast_ste_jax(x, recipe)
    # NumPy / MLX have no autograd; round-trip is the answer.
    return dequantize(cast(x, recipe))


def _cast_ste_torch(x: Any, recipe: Any) -> Any:
    """STE via the standard `x + (y - x).detach()` trick."""
    y = dequantize(cast(x, recipe))
    return x + (y - x).detach()


def _cast_ste_jax(x: Any, recipe: Any) -> Any:
    """STE via ``jax.custom_vjp``.

    Forward: NumPy reference round-trip invoked through
    :func:`jax.pure_callback`. Backward: identity (the STE rule).

    Composes with ``jax.grad`` and ``jax.jit``.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    from breccia.kernels.reference.cast import _cast_numpy, _dequantize_numpy

    def _numpy_round_trip(x_arr):
        x_np = np.asarray(x_arr, dtype=np.float32)
        st = _cast_numpy(x_np, recipe)
        return np.asarray(_dequantize_numpy(st), dtype=np.float32)

    @jax.custom_vjp
    def _ste(x_inner):
        result_shape = jax.ShapeDtypeStruct(x_inner.shape, jnp.float32)
        return jax.pure_callback(_numpy_round_trip, result_shape, x_inner)

    def _ste_fwd(x_inner):
        return _ste(x_inner), None

    def _ste_bwd(_, g):
        return (g,)

    _ste.defvjp(_ste_fwd, _ste_bwd)
    return _ste(x)


def cast_ste_clipped(
    x: Any,
    recipe: Any,
    clip_min: float | None = None,
    clip_max: float | None = None,
) -> Any:
    """Clipped variant of :func:`cast_ste`.

    The gradient w.r.t. ``x`` is zeroed for ``x`` outside
    ``[clip_min, clip_max]`` — useful when saturation is causing gradient
    explosions. Defaults derived from the recipe's format range.

    Parameters
    ----------
    x : array
    recipe : ScalingRecipe
    clip_min, clip_max : float, optional
        Override the recipe's natural saturation bounds. Defaults to the
        format's signed range (``-fp_max .. fp_max``) before scaling.

    Returns
    -------
    array
        Same as :func:`cast_ste` but with the gradient clipped.
    """
    from breccia._formats import E4M3_MAX, E5M2_MAX, E2M1_MAX
    from breccia.recipes import (
        DelayedScaling,
        Float8CurrentScaling,
        Float8BlockScaling,
        MXFP8BlockScaling,
        NVFP4BlockScaling,
        INT4Scaling,
    )

    # Derive default clip bounds from the recipe's natural range.
    # (After the scale is applied, the format covers [-fp_max, fp_max]
    # scaled by amax/fp_max, so the natural bound IS amax. We use the
    # tensor's amax as a proxy and let the format saturation kick in
    # gracefully for anything beyond.)
    if clip_max is None:
        if isinstance(recipe, (DelayedScaling, Float8CurrentScaling, Float8BlockScaling, MXFP8BlockScaling)):
            fp_max = E4M3_MAX if recipe.fp8_format == "E4M3" else E5M2_MAX
        elif isinstance(recipe, NVFP4BlockScaling):
            fp_max = E2M1_MAX
        elif isinstance(recipe, INT4Scaling):
            fp_max = 7.0 if recipe.signed else 15.0
        else:
            fp_max = float("inf")
        # Use a generous default — clip happens AT the format saturation,
        # not at the per-element representable scale.
        clip_max = float(fp_max)
    if clip_min is None:
        clip_min = -clip_max

    if _is_torch(x):
        import torch

        y = dequantize(cast(x, recipe))
        ste = x + (y - x).detach()
        # Zero the gradient outside the clip range while preserving the value.
        clipped = torch.where(
            (x >= clip_min) & (x <= clip_max),
            ste,
            ste.detach(),
        )
        return clipped
    if _is_jax(x):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from breccia.kernels.reference.cast import _cast_numpy, _dequantize_numpy

        def _numpy_round_trip(x_arr):
            x_np = np.asarray(x_arr, dtype=np.float32)
            st = _cast_numpy(x_np, recipe)
            return np.asarray(_dequantize_numpy(st), dtype=np.float32)

        @jax.custom_vjp
        def _ste(x_inner):
            result_shape = jax.ShapeDtypeStruct(x_inner.shape, jnp.float32)
            return jax.pure_callback(_numpy_round_trip, result_shape, x_inner)

        def _ste_fwd(x_inner):
            return _ste(x_inner), (x_inner,)

        def _ste_bwd(residual, g):
            (x_inner,) = residual
            in_range = (x_inner >= clip_min) & (x_inner <= clip_max)
            return (jnp.where(in_range, g, jnp.zeros_like(g)),)

        _ste.defvjp(_ste_fwd, _ste_bwd)
        return _ste(x)
    # NumPy / MLX have no autograd
    return dequantize(cast(x, recipe))
