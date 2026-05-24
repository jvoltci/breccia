"""Probe the installed TE version to discover the Float8Tensor constructor path.

One-time diagnostic script. Prints TE version, Float8Tensor signature,
and factory methods so we can pin the reverse bridge to a specific
constructor.
"""

import os

import modal


app = modal.App("breccia-te-probe")

image = (
    modal.Image.from_registry("nvcr.io/nvidia/pytorch:24.10-py3")
    .pip_install("numpy>=1.24", "safetensors>=0.4")
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/breccia",
        copy=True,
        ignore=[".git/**", ".venv/**", ".pytest_cache/**", ".hypothesis/**",
                "_site_test/**", "dist/**", "build/**", "*.egg-info/**"],
    )
    .run_commands("cd /breccia && pip install -e . --no-deps")
)


@app.function(image=image, gpu="L4", timeout=300)
def probe():
    import inspect
    import torch

    print("=" * 70)
    print("Probing TransformerEngine 2.x Float8Tensor API")
    print("=" * 70)

    import transformer_engine
    print(f"transformer_engine.__version__: {getattr(transformer_engine, '__version__', 'unknown')}")

    # Try to import Float8Tensor from the most common modern paths
    Float8Tensor = None
    found_path = None
    for module_path in [
        "transformer_engine.pytorch.tensor.float8_tensor",
        "transformer_engine.pytorch.float8_tensor",
        "transformer_engine.pytorch.tensor",
        "transformer_engine.pytorch",
    ]:
        try:
            mod = __import__(module_path, fromlist=["Float8Tensor"])
            if hasattr(mod, "Float8Tensor"):
                Float8Tensor = mod.Float8Tensor
                found_path = module_path
                break
        except ImportError:
            continue

    print(f"\nFloat8Tensor found at: {found_path}")
    if Float8Tensor is None:
        print("FATAL: Float8Tensor not found")
        return

    print(f"  class: {Float8Tensor}")
    print(f"  module: {Float8Tensor.__module__}")
    print(f"  mro: {[c.__name__ for c in Float8Tensor.__mro__]}")

    # Constructor signature
    try:
        sig = inspect.signature(Float8Tensor.__init__)
        print(f"\nFloat8Tensor.__init__ signature:")
        for pname, p in sig.parameters.items():
            print(f"    {pname}: {p.kind.name}  default={p.default}")
    except Exception as e:
        print(f"\n__init__ signature inspection failed: {e}")

    # Factory methods on the class
    print("\nFloat8Tensor classmethods / staticmethods (from_* / make_* / create_*):")
    for attr in dir(Float8Tensor):
        if attr.startswith("_"):
            continue
        if attr.startswith(("from_", "make_", "create_", "build_")):
            try:
                method = getattr(Float8Tensor, attr)
                sig = inspect.signature(method)
                print(f"    {attr}{sig}")
            except Exception:
                print(f"    {attr} (signature unavailable)")

    # Float8Quantizer
    print("\nLooking for Float8Quantizer...")
    Float8Quantizer = None
    for module_path in [
        "transformer_engine.pytorch.tensor.float8_tensor",
        "transformer_engine.pytorch.tensor",
        "transformer_engine.pytorch",
    ]:
        try:
            mod = __import__(module_path, fromlist=["Float8Quantizer"])
            if hasattr(mod, "Float8Quantizer"):
                Float8Quantizer = mod.Float8Quantizer
                print(f"  Found at: {module_path}.Float8Quantizer")
                try:
                    sig = inspect.signature(Float8Quantizer.__init__)
                    print(f"  __init__ signature:")
                    for pname, p in sig.parameters.items():
                        print(f"    {pname}: {p.kind.name}  default={p.default}")
                except Exception:
                    pass
                break
        except ImportError:
            continue
    if Float8Quantizer is None:
        print("  Float8Quantizer not found in standard paths")

    # tex.DType / Float8 dtype enum location
    print("\nLooking for FP8 dtype enum...")
    for module_path in [
        "transformer_engine_torch",
        "transformer_engine.pytorch.cpp_extensions",
        "transformer_engine.pytorch.tensor.float8_tensor",
        "transformer_engine.pytorch",
        "transformer_engine.common.recipe",
    ]:
        try:
            mod = __import__(module_path, fromlist=["DType"])
            if hasattr(mod, "DType"):
                DT = mod.DType
                print(f"  Found DType at {module_path}.DType")
                members = [a for a in dir(DT) if not a.startswith("_")]
                print(f"    members: {members}")
                break
        except ImportError:
            continue

    # Attempt actual construction with the most likely TE 2.x API
    print("\n=" * 35)
    print("Attempting Float8Tensor construction (TE 2.x patterns)")
    print("=" * 35)

    torch.manual_seed(0)
    x = torch.randn(64, 128, device="cuda", dtype=torch.float32) * 5.0
    amax = x.abs().max()
    fp8_max = 448.0
    scale = (fp8_max / amax).to(torch.float32)
    scale_inv = 1.0 / scale
    x_fp8_native = (x * scale).to(torch.float8_e4m3fn)
    data_uint8 = x_fp8_native.view(torch.uint8).contiguous()

    # Print Float8Tensor.__new__ signature
    print("\nFloat8Tensor.__new__ signature:")
    try:
        sig = inspect.signature(Float8Tensor.__new__)
        for pname, p in sig.parameters.items():
            print(f"    {pname}: {p.kind.name}  default={p.default}")
    except Exception as e:
        print(f"    failed: {e}")

    # Inspect the source if accessible
    print("\nFloat8Tensor source location:", inspect.getsourcefile(Float8Tensor))

    # Look at the constructor source
    try:
        src = inspect.getsource(Float8Tensor.__new__)
        print("\n__new__ source (first 40 lines):")
        for line in src.splitlines()[:40]:
            print(f"  {line}")
    except Exception as e:
        print(f"\n__new__ source unavailable: {e}")

    # Try construction with the discovered DType
    import transformer_engine_torch as tex

    print("\n=== Direct __new__ construction with TE 1.11 kwargs ===")
    attempts = [
        (
            "data + fp8_scale_inv + fp8_dtype kwargs",
            lambda: Float8Tensor(
                data=data_uint8,
                fp8_scale_inv=scale_inv.reshape(1),
                fp8_dtype=tex.DType.kFloat8E4M3,
            ),
        ),
        (
            "with fp8_attrs dict",
            lambda: Float8Tensor(
                data=data_uint8,
                fp8_attrs={"scale_inv": scale_inv.reshape(1), "dtype": tex.DType.kFloat8E4M3},
            ),
        ),
        (
            "with fp8_meta kwargs (legacy)",
            lambda: Float8Tensor(
                data=data_uint8,
                fp8_meta={"scaling_fwd": None},
                fp8_meta_index=0,
                fp8_dtype=tex.DType.kFloat8E4M3,
            ),
        ),
    ]
    successful_ctor = None
    for label, ctor in attempts:
        try:
            t = ctor()
            print(f"  [OK]   {label}: type={type(t).__name__}, shape={tuple(t.shape)}, dtype={t.dtype}")
            successful_ctor = label
            # Try dequantize
            try:
                deq = t.from_float8(torch.float32) if hasattr(t, "from_float8") else None
                if deq is not None:
                    print(f"    dequantize OK, shape={tuple(deq.shape)}, sample value={deq[0,0].item():.4f}")
            except Exception as e:
                print(f"    dequantize failed: {e}")
            break
        except Exception as e:
            print(f"  [FAIL] {label}: {type(e).__name__}: {str(e)[:120]}")


@app.local_entrypoint()
def main():
    probe.remote()
