# Changelog

All notable changes to breccia are recorded here. Format follows [Keep a
Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

Nothing yet. v0.0.2 will continue with stabilization.

## [0.0.1] — 2026-05-24

First commit. Pre-alpha. The API may change between any two commits at this stage.

### Added

- **`breccia.ScaledTensor`** dataclass — packed `(data, scale, recipe, layout)`
  representation with invariants enforced at construction
- **`breccia.from_buffer`** — zero-copy ScaledTensor constructor for
  pre-quantized data
- **Six ScalingRecipes**:
  - `DelayedScaling(fp8_format, amax_history_len, margin)` — TE main recipe
  - `Float8CurrentScaling(fp8_format)` — per-tensor synchronous amax
  - `Float8BlockScaling(fp8_format, block_k)` — DeepSeek-v3 style per-K-block
  - `MXFP8BlockScaling(fp8_format, block_size)` — OCP MX microscaling (block_size=32)
  - `NVFP4BlockScaling(fp4_format, block_size, scale_format)` — NVIDIA Blackwell NVFP4
    (block_size=16, FP4 E2M1 data, FP8 E4M3 scale)
  - `INT4Scaling(group_size, signed, scale_dtype)` — GPTQ / AWQ family INT4
- **Four Layouts**: `PerTensor`, `PerBlockK`, `PerChannel`, `PerBlockMN`
- **Core operations**: `cast`, `dequantize`, `matmul`, `requantize`
- **Reference (NumPy) kernels** in `breccia.kernels.reference`:
  - `cast.py` — quantize + dequantize per recipe (Hopper/Ada-compatible
    E4M3 / E5M2 / E2M1 LUTs + INT4 nibble encode)
  - `matmul.py` — scaled matmul, dispatches to torch/mlx/numpy backends
- **Triton kernel** (`scaled_matmul_triton`) for per-tensor FP8 scaled
  matmul on Hopper / Ada / Blackwell. GPU validation deferred to v0.1.
- **Backend dispatch** for NumPy, PyTorch, and MLX (Apple Silicon, via
  Metal). JAX skeleton in place; full wiring in v0.1.
- **Bridges** in `breccia.bridges`:
  - `_transformer_engine` — round-trip with NVIDIA TE Float8Tensor
  - `_torchao` — round-trip with `AffineQuantizedTensor` (symmetric only)
  - `_huggingface` — save / load safetensors with breccia's scale-metadata
    convention
  - `_dlpack` — zero-copy data + scale across NumPy / PyTorch / MLX / JAX
  - `_deepseek` — convenience wrapper for DeepSeek-v3 FP8 block format
- **Bit-format helpers** in `breccia._formats` — FP8 E4M3 / E5M2, FP4 E2M1,
  INT4 encode + decode, nibble pack / unpack for compact 4-bit storage
- **Benchmarks**:
  - `bench_memory.py` — exact byte counts per recipe vs FP32 / FP16
  - `bench_accuracy.py` — cosine similarity + max abs error per recipe on
    Gaussian, heavy-tail Student-t, uniform, and outlier distributions
  - `modal_bench.py` — Modal H100 benchmark of Triton scaled_matmul vs
    `torch._scaled_mm` (cuBLAS FP8 GEMM)
- **Examples**:
  - `01_quickstart.py` — minimal cast + matmul
  - `02_recipe_portable_train.py` — train MXFP8 → ship NVFP4
  - `03_checkpoint_with_scale.py` — save/load safetensors with recipes preserved
  - `04_te_migration.py` — bridge call patterns vs mock TE Float8Tensor
- **Documentation** under `docs/` (12 documents):
  index, getting-started, concepts, recipes, formats, api, bridges,
  kernels, architecture, benchmarks, numerics, faq
- **GitHub Actions CI** on Python 3.10 / 3.11 / 3.12 (Ubuntu) and 3.11 (macOS);
  runs the full test suite, all 4 examples, and the memory + accuracy
  benchmarks on every push

### Test coverage

- 192 tests passing across NumPy + PyTorch + MLX backends
- 17 hypothesis-based property tests (shape preservation, round-trip
  quality, layout shape rules, matmul shape rule, NaN-free dequantize)
- TransformerEngine bridge runtime tests are skipped without CUDA (skip
  message documents the install path); error-path test runs in CI

### Validation gaps (carrying into v0.1)

- The Triton FP8 scaled_matmul kernel ships untested on GPU (no CI GPU).
  Validation script is `benchmarks/modal_bench.py`; expected first run
  result lands in the v0.0.1 → v0.1.0 changelog entry.
- TransformerEngine round-trip is not exercised on real TE tensors in
  v0.0.1 (TE installs only on Linux + CUDA). Tested against a mock; needs
  GPU validation.
- Asymmetric quantization (non-zero zero-point) for INT4 / INT8 is not
  supported; torchao bridge accepts symmetric only.
