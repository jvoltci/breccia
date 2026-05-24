"""Accuracy degradation per recipe on a sweep of input distributions.

Reports cosine similarity and max abs error of dequantize(cast(x))
relative to x, averaged over multiple seeds.

Run with: python benchmarks/bench_accuracy.py
"""

import numpy as np

import breccia


def cos_sim(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def max_abs(a, b):
    return float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))


RECIPES = [
    ("Float8CurrentScaling (E4M3)", breccia.Float8CurrentScaling()),
    ("Float8CurrentScaling (E5M2)", breccia.Float8CurrentScaling(fp8_format="E5M2")),
    ("Float8BlockScaling(block_k=128)", breccia.Float8BlockScaling(block_k=128)),
    ("MXFP8BlockScaling", breccia.MXFP8BlockScaling()),
    ("NVFP4BlockScaling", breccia.NVFP4BlockScaling()),
    ("INT4Scaling(group_size=128)", breccia.INT4Scaling(group_size=128)),
]


def run_distribution(name, sampler):
    print(f"\n--- {name} ---")
    print(f"{'recipe':>34}  {'cos sim':>10}  {'max abs':>10}")
    for recipe_name, recipe in RECIPES:
        cs_acc = 0.0
        ma_acc = 0.0
        n_seeds = 8
        for seed in range(n_seeds):
            x = sampler(np.random.default_rng(seed))
            try:
                st = breccia.cast(x, recipe)
            except ValueError:
                cs_acc = ma_acc = float("nan")
                break
            recovered = np.asarray(breccia.dequantize(st))
            cs_acc += cos_sim(x, recovered)
            ma_acc += max_abs(x, recovered)
        print(
            f"{recipe_name:>34}  "
            f"{cs_acc / n_seeds:>10.4f}  "
            f"{ma_acc / n_seeds:>10.4f}"
        )


# Gaussian-distributed input
run_distribution(
    "Gaussian, shape (256, 256)",
    lambda rng: rng.standard_normal((256, 256)).astype(np.float32),
)

# Heavy-tail Student-t (df=2 has very fat tails)
run_distribution(
    "Heavy-tail Student-t(df=2), shape (256, 256)",
    lambda rng: rng.standard_t(df=2, size=(256, 256)).astype(np.float32),
)

# Uniform in [-1, 1]
run_distribution(
    "Uniform [-1, 1], shape (256, 256)",
    lambda rng: rng.uniform(-1, 1, (256, 256)).astype(np.float32),
)

# Gaussian with one outlier per row
run_distribution(
    "Gaussian + 1 outlier per row, shape (16, 128)",
    lambda rng: (
        rng.standard_normal((16, 128)).astype(np.float32)
        * (1 + 100 * (rng.uniform(0, 1, (16, 128)) > 0.99))
    ).astype(np.float32),
)
