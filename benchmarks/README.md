# Benchmarks

Three scripts cover the three things people want to know about breccia:

| Script | What | Reproduce |
| --- | --- | --- |
| `bench_memory.py` | Memory footprint per recipe vs FP32 / FP16 | `python benchmarks/bench_memory.py` |
| `bench_accuracy.py` | Cosine similarity + max abs error per recipe | `python benchmarks/bench_accuracy.py` |
| `modal_bench.py` | H100 throughput vs cuBLAS FP8 GEMM | `modal run benchmarks/modal_bench.py` |

All scripts are deterministic (explicit seeds).

## `bench_memory.py`

Reports exact byte counts — no measurement variance. The result is a
function of the tensor shape and the recipe's scale-tensor overhead.

## `bench_accuracy.py`

Sweeps four input distributions per recipe:
1. Gaussian
2. Heavy-tail Student-t (df=2)
3. Uniform [-1, 1]
4. Gaussian with sparse outliers

Reports cosine similarity (robust to small-magnitude inputs) and max
abs error (catches outlier saturation).

## `modal_bench.py`

The H100 throughput benchmark. Costs ~$0.30 of Modal credit per run.

v0.0.1 ships the Triton kernel but does not run this benchmark in CI
(no GPU). Run it manually when you have GPU access; results land in
the v0.0.1 → v0.1.0 changelog.

## Methodology notes

See [docs/benchmarks.md](../docs/benchmarks.md) for the rationale behind
the metric choices and what's deliberately *not* benchmarked in v0.0.1.
