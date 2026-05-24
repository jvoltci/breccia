# Changelog

All notable changes to breccia are recorded here. Format follows [Keep a
Changelog](https://keepachangelog.com/); versions follow [SemVer](https://semver.org/).

## [Unreleased]

Nothing yet. v0.2 may pick up: AMD ROCm Triton backend, vLLM / SGLang
loader PRs, native MLX FP8 on Metal.

## [0.1.3] — 2026-05-24

Closes the final v0.1 🟡: the reverse-direction TE bridge
(`to_transformer_engine`) now round-trips **bit-exact** on H100.

### Fixed

- **`to_transformer_engine` pinned to TE 1.11+ constructor** — TE's
  `Float8Tensor` uses a keyword-only `__new__(*, data, fp8_scale_inv,
  fp8_dtype, ...)` signature. The bridge now discovers `Float8Tensor`
  and the `DType` enum at runtime across the few module paths TE
  exposes them at, coerces `data` to `uint8` (TE requires 1-byte
  element size and interprets bytes via `fp8_dtype`), reshapes
  `fp8_scale_inv` to the `(1,)` shape TE stores, and constructs
  via keyword args.

  Validated on Modal H100 with TE 1.11.0 from NGC PyTorch 24.10:

  - `from_transformer_engine` round-trip: max abs diff = 0.000000
  - `to_transformer_engine` round-trip: max abs diff = 0.000000
  - Full TE → breccia → TE → dequantize: max abs diff = 0.000000

  Bit-exact in both directions. All three are reproduced by
  `benchmarks/modal_te_validate.py`.

### Added

- `benchmarks/modal_te_probe.py` — one-shot diagnostic that prints the
  installed TE's `Float8Tensor.__new__` signature, factory methods,
  and `DType` enum location. Run it on a new TE version to verify the
  bridge's discovery code still finds the right symbols.

## [0.1.2] — 2026-05-24

Closes the two items that v0.1.1 left as "v0.2 TBD": **Triton AOT
compilation** and **live TransformerEngine bridge validation**. Also
ships a major documentation upgrade.

### Added

- **Triton AOT path** — `scaled_matmul_triton(a, b, autotune=False)`
  (the new default) uses a single hardcoded config
  `(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=8, num_stages=3)`.
  First call compiles only that config (~1.5s instead of ~3s for the
  5-config sweep); **warm calls run at 0.8 ms on H100 — 6× faster than
  `torch._scaled_mm`** at 2048×2048. `autotune=True` still available
  for users with hot shapes worth the warmup cost.
- **Live TransformerEngine bridge validation** — `modal_te_validate.py`
  runs on a Modal H100 with NVIDIA's NGC PyTorch image
  (`nvcr.io/nvidia/pytorch:24.10-py3`) which ships TE pre-built. The
  forward bridge (TE `Float8Tensor` → `breccia.ScaledTensor`)
  round-trips **bit-exact** (max abs diff = 0).
- **Documentation: scree-tier upgrade**:
  - Custom logo (`docs/assets/logo.svg`) + favicon
    (`docs/assets/favicon.svg`)
  - Custom brand CSS (`docs/assets/extra.css`) — purple matrix /
    orange fragment palette, gradient hero banner, 6-card metric grid,
    table styling, mkdocstrings styling, recipe/format chips
  - Hero landing page (`docs/index.md`) — tagline + badges +
    inline code preview + 3 call-to-action buttons + "At a glance"
    metrics + "Why breccia" comparison + "What you can do today"
    workflow table + examples list
  - `mkdocs.yml` upgraded: 18 navigation features, `mkdocstrings[python]`
    plugin for auto API docs from docstrings, full `pymdownx`
    extension set (mermaid, emoji, keys, mark, smartsymbols,
    tasklist), reorganized nav into Home / Get started / How-to /
    Reference / Benchmarks / FAQ

### Fixed

- **TE bridge scale shape** — TE stores `_scale_inv` as a shape-`(1,)`
  tensor; `from_transformer_engine` now squeezes to 0-D for
  `PerTensor` compatibility. Found by the live Modal validation run.
- **GH Pages docs deploy** — workflow installs `mkdocstrings[python]`
  in addition to `mkdocs-material`. Dropped the empty `custom_dir`
  reference that was crashing the build.

### Validation gaps carrying forward

- **TE bridge reverse direction** (`to_transformer_engine`) — TE 2.x
  has churned the `Float8Tensor` constructor signature; the bridge
  tries multiple paths but every TE version may need a version-specific
  shim. Forward direction is bit-exact. v0.2 should pick a specific
  TE version (e.g., 2.15) and pin the reverse path against it.
- **AOT vs autotune trade-off** — single-config AOT is fast on
  H100 for typical 2048-aligned shapes; small or odd shapes may want
  `autotune=True`. Per-silicon retuning is v0.2.

## [0.1.1] — 2026-05-24

## [0.1.1] — 2026-05-24

Re-release with the **README updated to reflect what already shipped in
0.1.0**. No code changes; the prior 0.1.0 wheel bundled an earlier
draft of the README that still showed two features as 🟡 v0.2 when
they were already implemented and validated on H100. The 0.1.1 wheel
contains the corrected README.

What 0.1.0 shipped (correctly described in 0.1.1's README):

- **Native PyTorch FP8 acceleration** — `cast` on a torch tensor now
  stays on the input device end-to-end and produces a `ScaledTensor`
  whose `data` is the native `torch.float8_e4m3fn` (or `e5m2`) dtype.
  No CPU round-trip for `DelayedScaling` / `Float8CurrentScaling` /
  `Float8BlockScaling`.
- **Block-scaled Triton kernel** —
  `breccia.kernels.triton.block_scaled_matmul_triton` implements the
  DeepSeek-v3 FP8 GEMM pattern: per-K-block scale folded into the FP32
  accumulator on every K-loop iteration. H100 validated, cos sim
  0.9813 vs FP32.

Both have been live in the master branch since the 0.1.0 commit; this
patch release just corrects the documentation.

## [0.1.0] — 2026-05-24

First beta release. The public API in `breccia.*` is now stable;
backwards-incompatible changes from here require a minor-version bump.

### Added

- **JAX backend** — `_cast_jax` / `_dequantize_jax` dispatch wired
  alongside the existing torch / MLX paths in
  `kernels/reference/cast.py`. `matmul.py` gains a JAX branch that
  returns `jnp` arrays end-to-end. 15 tests in `tests/test_jax.py`
  cover all 6 recipes.
- **Straight-through estimator (`breccia.autograd`)** — autograd-aware
  cast for quantization-aware training:
  - `cast_ste(x, recipe)` — gradient-identity through the quantization
    step on PyTorch (`x + (y - x).detach()`) and JAX (`jax.custom_vjp`
    + `jax.pure_callback`, composes with `jax.grad` / `jax.jit`).
  - `cast_ste_clipped(x, recipe, clip_min, clip_max)` — zeros the
    gradient outside the format's representable range.
- **Asymmetric INT4 quantization** — `INT4Scaling(symmetric=False)`
  with a `zero_point` field on `ScaledTensor`. Round-trip through
  the HF safetensors bridge preserves the zero-point. Better fidelity
  on skewed distributions (gamma-shaped inputs in `bench_accuracy.py`).
- **`ScaledTensor.zero_point`** — optional 5th field, default `None`
  for symmetric recipes; populated for `INT4Scaling(symmetric=False)`.
  Validated to match `scale.shape`. Carries through torch / MLX / JAX
  dispatches.
- **H100 GPU validation** of the Triton `scaled_matmul_triton` kernel
  via `benchmarks/modal_bench.py` — cos sim 0.9993 vs FP32 reference,
  Triton compilation succeeds, first-call autotune ~3s.
- **Launch artifacts** under `docs/assets/` (hero SVG),
  `.github/ISSUE_TEMPLATE/` (bug.yml), `RELEASE.md` (v0.1.0
  checklist), and `docs/vllm-integration-sketch.md`. Internal
  launch-day documents (blog draft, runbook, endorser DMs) live in
  `/v5/` private until launch.

### Changed

- `ScaledTensor` dataclass: added optional `zero_point` field
  (default `None`). Backwards-compatible — symmetric recipes ignore it.
- `from_buffer()` signature gains `zero_point=None` parameter.
- Triton kernel (`scaled_matmul_triton`): removed mask-based loads
  (Triton 3.7+ stricter type cast); requires shapes aligned to
  M%64=N%64=K%32=0.
- Test count: 192 → 252+ (added test_jax.py, test_autograd.py,
  test_asymmetric_int4.py).

### Validation gaps (carrying into v0.2)

- **Native FP8 PyTorch path** — v0.1 routes torch tensors through
  NumPy CPU for correctness. Result: ~20× slower than
  `torch._scaled_mm` on H100 even at correct cos sim. Reinterpreting
  uint8 as `torch.float8_e4m3fn` end-to-end is v0.2.
- **Triton kernel autotune cold-path** — first call is ~3s due to
  config-grid compilation. Subsequent calls are fast. AOT
  compilation or persistent kernel cache is v0.2.
- **Block-scaled Triton kernel** — v0.1 ships per-tensor only
  (`DelayedScaling` / `Float8CurrentScaling`). Per-block-K and
  MXFP8/NVFP4 variants are v0.2.
- **TransformerEngine bridge live validation** — TE requires CUDA
  toolkit in the build env; v0.1 ships bridge code + mock-tested
  unit tests, with `benchmarks/modal_te_validate.py` as the harness
  for the first real validation run.

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
