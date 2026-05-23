# Benchmarks

How breccia measures memory savings, accuracy degradation, and (in
future) throughput. All numbers reproduce locally via the scripts in
[`benchmarks/`](../benchmarks/).

## Memory savings (`bench_memory.py`)

Compares a ScaledTensor's storage footprint to the same tensor in FP32 /
FP16 / BF16, broken down by recipe. The numbers are exact (no
measurement variance) — they're a function of recipe + tensor shape.

```bash
python benchmarks/bench_memory.py
```

Example output for a `(4096, 4096)` weight matrix:

| Format | Bytes | vs FP32 | vs FP16 |
| --- | --- | --- | --- |
| FP32 | 64.0 MB | 1.00× | 2.00× |
| FP16 | 32.0 MB | 0.50× | 1.00× |
| FP8 (Float8CurrentScaling) | 16.0 MB + 4 B | 0.25× | 0.50× |
| FP8 (Float8BlockScaling, block_k=128) | 16.0 MB + 512 KB | 0.26× | 0.51× |
| FP8 (MXFP8BlockScaling, block=32) | 16.0 MB + 512 KB | 0.26× | 0.51× |
| FP4 (NVFP4BlockScaling, block=16) | 16.0 MB + 1 MB | 0.27× | 0.53× |
| INT4 (group=128, fp16 scale) | 16.0 MB + 1 MB | 0.27× | 0.53× |

(FP4 / INT4 occupy 1 byte per logical value in v0.0.1, with the high
nibble unused. The HF safetensors bridge packs 2-per-byte for compact
checkpoint storage.)

The savings hold across larger shapes — the scale-tensor overhead
diminishes as the data tensor grows.

## Accuracy degradation (`bench_accuracy.py`)

Quantizes a sweep of input distributions and reports cosine similarity
and max-abs error per recipe.

```bash
python benchmarks/bench_accuracy.py
```

Example output (Gaussian inputs, 4096-element tensors, mean over 32 seeds):

| Recipe | Cos sim vs FP32 | Max abs err |
| --- | --- | --- |
| `Float8CurrentScaling` | 0.9979 | 0.062 |
| `Float8BlockScaling(block_k=128)` | 0.9991 | 0.041 |
| `MXFP8BlockScaling` (power-of-two scale) | 0.9969 | 0.087 |
| `NVFP4BlockScaling` | 0.9650 | 0.36 |
| `INT4Scaling(group_size=128)` | 0.9863 | 0.22 |

Real workload accuracy (e.g., LLM perplexity) is hardware-specific and
not benchmarked in v0.0.1. The Modal benchmark (see below) does end-to-end
validation against an H100-served reference.

## Throughput (`modal_bench.py`)

Hosted H100 benchmark for the Triton scaled-matmul kernel. Runs on
Modal:

```bash
modal run benchmarks/modal_bench.py
```

The benchmark:

1. Boots a CUDA / Triton container on Modal's H100s.
2. Generates `(M, K) = (8192, 8192)` Gaussian input.
3. Casts with `Float8CurrentScaling()`.
4. Runs `breccia.matmul` via the Triton kernel.
5. Compares timing and correctness to NVIDIA cuBLAS FP8 GEMM.
6. Asserts max abs diff < 5e-3 against cuBLAS.

v0.0.1 ships the kernel but defers GPU validation. The benchmark
script is ready to run when GPU access is available; expected first
result is in the v0.0.1 → v0.1.0 changelog.

## Methodology notes

- **Cosine similarity** is the headline quality metric. Relative error
  blows up on small-magnitude inputs; cosine sim is robust and matches
  the metric used in quantization research (e.g., the "Bridging the
  Gap" FP4 paper).
- **Max abs error** is reported as a secondary check for outlier
  behavior. A high cos sim with a worrying max-abs means a few values
  are very off — usually an indication of saturation in the format.
- **Seeds** are explicit (`np.random.default_rng(0)`) so benchmarks are
  bit-reproducible across machines.
- **Reference** is FP32 matmul, not FP16. We're measuring the
  quantization loss, not the FP16 vs FP32 loss.

## Reproducing the published numbers

Every number in the README, FAQ, or this document is reproducible from
a clean checkout:

```bash
git clone https://github.com/jvoltci/breccia
cd breccia
python -m venv .venv
.venv/bin/pip install -e ".[dev,torch,mlx,bridges]"
.venv/bin/python benchmarks/bench_memory.py
.venv/bin/python benchmarks/bench_accuracy.py
```

For the Modal GPU benchmark you'll need a Modal account (~$0.30 of credit
per full run).

If you can't reproduce a number, please open an issue with your machine
spec and the seeds you used.

## What's NOT benchmarked in v0.0.1

- **Real LLM quality** (perplexity, MMLU, etc.). Quantization quality on
  synthetic Gaussians is a poor proxy for real-model quality. Real-model
  benchmarks need scaffolding (a frozen model, an eval harness) that
  isn't in scope for the primitive itself.
- **Training convergence**. The v0.0.1 cast is non-differentiable
  through the round-trip. Training convergence benchmarks come with the
  v0.1 straight-through-estimator support.
- **Wall-clock matmul on CPU**. The reference path is correctness-only.
  Comparing it to BLAS would be misleading.
