"""H100 benchmark of breccia's Triton scaled_matmul vs cuBLAS FP8 GEMM.

Runs on Modal (https://modal.com). Costs ~$0.30 per full run.

v0.0.1 ships the Triton kernel but does not run this benchmark in CI
(no GPU). Run it manually when you have GPU access:

    modal run benchmarks/modal_bench.py

The script:
1. Boots a CUDA + Triton container on Modal's H100s.
2. Generates synthetic (M, K) @ (K, N) input.
3. Casts both with Float8CurrentScaling.
4. Times breccia.matmul (which calls the Triton kernel on CUDA).
5. Compares timing + correctness to cuBLAS FP8 GEMM via torch._scaled_mm.
6. Asserts max abs diff < 5e-3.
"""

from __future__ import annotations

import os

try:
    import modal
except ImportError:
    print("ERROR: modal is required to run this benchmark.")
    print("       Install with: pip install modal")
    print("       Then sign up at https://modal.com and run:")
    print("           modal run benchmarks/modal_bench.py")
    raise SystemExit(1)


app = modal.App("breccia-bench")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "torch>=2.3",
        "triton>=2.3",
        "numpy>=1.24",
        "safetensors>=0.4",
    )
    .add_local_dir(
        os.path.join(os.path.dirname(__file__), ".."),
        remote_path="/breccia",
    )
    .run_commands("cd /breccia && pip install -e .")
)


@app.function(image=image, gpu="H100", timeout=300)
def run_benchmark():
    import time

    import numpy as np
    import torch

    import breccia

    print("=" * 60)
    print("breccia Triton scaled_matmul vs cuBLAS FP8 GEMM (H100)")
    print("=" * 60)

    M, K, N = 8192, 8192, 8192
    print(f"\nShape: ({M}, {K}) @ ({K}, {N})")
    print(f"Recipe: Float8CurrentScaling (E4M3)")

    rng = torch.Generator(device="cuda").manual_seed(0)
    A = torch.randn(M, K, generator=rng, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, generator=rng, device="cuda", dtype=torch.float32)

    sa = breccia.cast(A, breccia.Float8CurrentScaling())
    sb = breccia.cast(B, breccia.Float8CurrentScaling())

    # Warmup
    for _ in range(5):
        _ = breccia.matmul(sa, sb)
    torch.cuda.synchronize()

    # Time
    n_iters = 20
    start = time.perf_counter()
    for _ in range(n_iters):
        Y_breccia = breccia.matmul(sa, sb)
    torch.cuda.synchronize()
    breccia_ms = (time.perf_counter() - start) * 1000 / n_iters
    print(f"\nbreccia.matmul: {breccia_ms:.3f} ms")

    # cuBLAS reference via torch._scaled_mm
    A_e4m3 = A.to(torch.float8_e4m3fn)
    B_e4m3 = B.to(torch.float8_e4m3fn).t()  # _scaled_mm expects column-major B
    a_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)
    b_scale = torch.tensor(1.0, device="cuda", dtype=torch.float32)

    for _ in range(5):
        _ = torch._scaled_mm(A_e4m3, B_e4m3, a_scale, b_scale, out_dtype=torch.float32)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(n_iters):
        Y_cublas = torch._scaled_mm(A_e4m3, B_e4m3, a_scale, b_scale, out_dtype=torch.float32)
    torch.cuda.synchronize()
    cublas_ms = (time.perf_counter() - start) * 1000 / n_iters
    print(f"torch._scaled_mm:  {cublas_ms:.3f} ms")
    print(f"\nRatio (breccia / cuBLAS): {breccia_ms / cublas_ms:.2f}x")

    # Correctness check
    Y_fp32 = A @ B
    max_abs_breccia = float((Y_breccia - Y_fp32).abs().max())
    max_abs_cublas = float((Y_cublas - Y_fp32).abs().max())
    print(f"\nMax abs vs FP32:")
    print(f"  breccia:        {max_abs_breccia:.4f}")
    print(f"  torch._scaled_mm: {max_abs_cublas:.4f}")

    # Cross-check breccia vs cublas
    max_abs_cross = float((Y_breccia - Y_cublas).abs().max())
    print(f"\nMax abs (breccia vs cuBLAS): {max_abs_cross:.4f}")
    assert max_abs_cross < 5e-3, f"correctness check failed: max_abs_cross={max_abs_cross}"
    print("\nCorrectness: PASS")


@app.local_entrypoint()
def main():
    run_benchmark.remote()
