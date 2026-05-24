"""Tests for the straight-through estimator (STE) wrappers."""

import numpy as np
import pytest

from breccia import (
    cast_ste,
    cast_ste_clipped,
    Float8CurrentScaling,
    Float8BlockScaling,
    NVFP4BlockScaling,
    INT4Scaling,
)


# ---------- NumPy: STE degrades to round-trip (no autograd) ----------


def test_ste_numpy_returns_round_trip():
    np.random.seed(0)
    x = np.random.standard_normal((4, 64)).astype(np.float32)
    out = cast_ste(x, Float8CurrentScaling())
    assert out.shape == x.shape
    assert out.dtype == np.float32
    # Should match a plain dequantize(cast(...)).
    import breccia

    out_ref = breccia.dequantize(breccia.cast(x, Float8CurrentScaling()))
    np.testing.assert_allclose(out, out_ref, atol=1e-5)


# ---------- PyTorch: STE gives identity gradient ----------


def test_ste_torch_forward_matches_round_trip():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)
    y = cast_ste(x, Float8CurrentScaling())

    # Forward value matches the round-trip.
    import breccia

    expected = breccia.dequantize(breccia.cast(x.detach(), Float8CurrentScaling()))
    torch.testing.assert_close(y.detach(), expected, atol=1e-5, rtol=1e-5)


def test_ste_torch_gradient_is_identity():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(4, 64, dtype=torch.float32, requires_grad=True)
    y = cast_ste(x, Float8CurrentScaling())

    # Sum the output and check the gradient is all ones (identity).
    y.sum().backward()
    expected_grad = torch.ones_like(x)
    torch.testing.assert_close(x.grad, expected_grad)


def test_ste_torch_gradient_flows_through_recipe():
    """Same identity-gradient invariant for block-scaled recipes."""
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(4, 128, dtype=torch.float32, requires_grad=True)
    y = cast_ste(x, Float8BlockScaling(block_k=128))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ste_torch_with_matmul():
    """STE-aware cast composes with matmul for training-style use."""
    torch = pytest.importorskip("torch")

    import breccia

    torch.manual_seed(0)
    A = torch.randn(8, 64, dtype=torch.float32, requires_grad=True)
    W = torch.randn(64, 16, dtype=torch.float32, requires_grad=True)

    A_q = cast_ste(A, Float8CurrentScaling())
    W_q = cast_ste(W, Float8CurrentScaling())
    y = A_q @ W_q  # plain matmul, both sides are float tensors with STE noise
    loss = (y ** 2).sum()
    loss.backward()

    assert A.grad is not None
    assert W.grad is not None
    assert A.grad.shape == A.shape
    assert W.grad.shape == W.shape


# ---------- Clipped STE ----------


def test_ste_clipped_zeros_gradient_outside_range():
    torch = pytest.importorskip("torch")

    # Mix in-range and out-of-range values.
    x = torch.tensor(
        [[100.0, 0.5, -200.0, 1.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    y = cast_ste_clipped(x, Float8CurrentScaling(), clip_min=-10, clip_max=10)
    y.sum().backward()

    # In-range (0.5, 1.0) get gradient 1; out-of-range (100, -200) get 0.
    expected = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    torch.testing.assert_close(x.grad, expected)


def test_ste_clipped_defaults_to_format_range():
    """Default clip range comes from the recipe's format."""
    torch = pytest.importorskip("torch")

    x = torch.tensor([[0.5, 1.0]], dtype=torch.float32, requires_grad=True)
    y = cast_ste_clipped(x, Float8CurrentScaling())  # default clip ≈ ±448
    y.sum().backward()
    # Both 0.5 and 1.0 are well within ±448, so both get gradient 1.
    torch.testing.assert_close(x.grad, torch.ones_like(x))


# ---------- JAX: STE gives identity gradient ----------


def test_ste_jax_gradient_is_identity():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    rng = np.random.default_rng(0)
    x_np = rng.standard_normal((4, 64)).astype(np.float32)

    def f(x):
        return cast_ste(x, Float8CurrentScaling()).sum()

    g = jax.grad(f)(jnp.asarray(x_np))
    np.testing.assert_allclose(np.asarray(g), np.ones_like(x_np), atol=1e-5)


# ---------- Cross-recipe ----------


def test_ste_torch_nvfp4():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(4, 32, dtype=torch.float32, requires_grad=True)
    y = cast_ste(x, NVFP4BlockScaling())
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))


def test_ste_torch_int4():
    torch = pytest.importorskip("torch")

    torch.manual_seed(0)
    x = torch.randn(4, 128, dtype=torch.float32, requires_grad=True)
    y = cast_ste(x, INT4Scaling(group_size=128))
    y.sum().backward()
    torch.testing.assert_close(x.grad, torch.ones_like(x))
