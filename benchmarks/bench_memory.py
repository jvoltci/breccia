"""Memory footprint per recipe, vs FP32 / FP16 baselines.

Numbers are exact (no measurement variance) — they are a function of
the tensor shape and the recipe's scale-tensor overhead.

Run with: python benchmarks/bench_memory.py
"""

import numpy as np

import breccia

RECIPES = [
    ("Float8CurrentScaling", breccia.Float8CurrentScaling()),
    ("Float8BlockScaling(128)", breccia.Float8BlockScaling(block_k=128)),
    ("MXFP8BlockScaling(32)", breccia.MXFP8BlockScaling()),
    ("NVFP4BlockScaling(16)", breccia.NVFP4BlockScaling()),
    ("INT4Scaling(g=128)", breccia.INT4Scaling(group_size=128)),
]

SHAPES = [
    (256, 256),     # small for fast iteration
    (1024, 1024),   # mid
    (256, 4096),    # narrow projection
    (1, 4096),      # 1-row stress (lots of scale overhead per row)
]

print(f"{'shape':>16}  {'recipe':>26}  {'bytes':>12}  {'vs FP32':>9}  {'vs FP16':>9}")
print("-" * 86)

for shape in SHAPES:
    rng = np.random.default_rng(0)
    x = rng.standard_normal(shape).astype(np.float32)
    fp32_bytes = x.nbytes
    fp16_bytes = x.nbytes // 2

    # Baselines
    print(
        f"{str(shape):>16}  {'FP32':>26}  "
        f"{fp32_bytes:>12,}  {1.0:>9.3f}  {2.0:>9.3f}"
    )
    print(
        f"{'':>16}  {'FP16':>26}  "
        f"{fp16_bytes:>12,}  {0.5:>9.3f}  {1.0:>9.3f}"
    )

    for name, recipe in RECIPES:
        try:
            st = breccia.cast(x, recipe)
        except ValueError:
            # Recipe constraint not satisfied (e.g., block doesn't fit shape)
            continue
        data_bytes = st.data.nbytes
        scale_bytes = st.scale.nbytes if hasattr(st.scale, "nbytes") else np.asarray(st.scale).nbytes
        total = data_bytes + scale_bytes
        print(
            f"{'':>16}  {name:>26}  "
            f"{total:>12,}  {total / fp32_bytes:>9.3f}  {total / fp16_bytes:>9.3f}"
        )
    print()
