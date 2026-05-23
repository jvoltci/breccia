# FAQ

Questions that come up after a few minutes with breccia.

## What is breccia, in one sentence?

A typed primitive for block-scaled low-precision tensors — like
safetensors gave us a neutral format for `.bin` weights, breccia gives us
a neutral type for FP8 / FP4 / MXFP8 / NVFP4 / INT4 quantized tensors
across PyTorch, NumPy, MLX, and (next) JAX.

## Why is it called breccia?

Breccia is a sedimentary rock made of broken angular fragments held
together by a cementing matrix. Low-precision data fragments + the scale
matrix that gives them meaning — same structure.

It's also the natural successor to [`scree`](https://github.com/jvoltci/scree)
in the geological library family: scree handles loose variable-length
fragments; breccia handles fragments cemented by scale.

## How does breccia compare to TransformerEngine?

| | NVIDIA TransformerEngine | breccia |
| --- | --- | --- |
| Frameworks | PyTorch only | NumPy + PyTorch + MLX (JAX next) |
| Hardware | NVIDIA only | Cross-vendor (the design goal) |
| Recipes | 4 non-composable classes | 6 composable recipes |
| Autograd | Native | Via `data` field (no subclass) |
| Bridges | None | TE, torchao, HF safetensors, DLPack, DeepSeek-v3 |

`breccia.bridges.from_transformer_engine` and `to_transformer_engine`
round-trip between the two. Use whichever fits your stack today; the
bridge lets you experiment with the other.

## How does breccia compare to torchao?

torchao is PyTorch's official quantization library — autograd-native,
PyTorch-only. breccia is the cross-framework substrate. You can convert
a torchao `AffineQuantizedTensor` to a breccia `ScaledTensor` and back via
the `_torchao` bridge.

Recommendation: if you're PyTorch-only and need autograd-aware quantized
training, use torchao. If you need to ship a quantized model across
framework boundaries (e.g., trained in PyTorch, served by an MLX
inference engine), breccia is the type that survives the boundary.

## Do I have to use the Triton kernel?

No. `ScaledTensor` is a plain dataclass with NumPy / PyTorch / MLX
backends. The reference kernels (`breccia.kernels.reference.*`) work
everywhere. The Triton kernel (`breccia.kernels.triton.*`) is an
optional fast path for CUDA users.

The reference impls are the **contract**; the Triton impls are the
**speed**. Tests verify the Triton output matches the reference within
the recipe's declared tolerance.

## Can I use it for training?

Today: yes for forward passes; with caveats for backward.

- **Forward** — the cast / matmul / dequantize round-trip works
  end-to-end on torch tensors. Gradients flow through `data` (because
  the round-trip preserves the underlying tensor as a torch tensor).
- **Backward through the cast step** — v0.0.1 does NOT provide
  straight-through-estimator (STE) gradients for the quantization
  rounding. The cast operation is non-differentiable at the round-to-
  nearest step. For STE-aware training in v0.0.1, use TE or torchao;
  the breccia STE wrapper lands in v0.1.

## Can I use it for inference?

Yes. The forward path is fully supported and accuracy-validated. For
production GPU inference, the v0.0.1 caveats:

- The Triton kernel ships untested-on-GPU. Validation via Modal H100
  benchmark is the first item on the v0.1 list.
- No CUDA acceleration on torch paths in v0.0.1 — torch tensors are
  CPU-round-tripped through NumPy for correctness only.

For CPU inference or experimental work, v0.0.1 is ready today.

## What about JAX?

JAX backend is on the v0.1 roadmap. The skeleton is in place
(`_is_jax(x)` predicate, JAX-shaped paths in `cast.py`) but not wired up
to a full dispatch yet. See [architecture.md](architecture.md) →
"Adding a new backend."

## What about MLX (Apple Silicon)?

Supported in v0.0.1 — all 6 recipes work end-to-end on MLX arrays. MLX
has no native FP8 / FP4 dtype as of v0.31, so MLX paths use the same
uint8-packed representation as NumPy. The MLX path is correctness-only;
hardware-accelerated quantized matmul on Metal (via `mlx.fast.*`) is a
v0.2 item.

## How big is the v0.0.1 footprint?

```
src/breccia/_core.py        ~150 lines   ScaledTensor + invariants + predicates
src/breccia/recipes.py      ~180 lines   6 recipe dataclasses
src/breccia/layouts.py      ~150 lines   4 layout dataclasses
src/breccia/_formats.py     ~250 lines   FP8/FP4/INT4 LUTs + encode/decode + packing
src/breccia/kernels/        ~450 lines   reference cast + matmul; Triton stub
src/breccia/bridges/        ~400 lines   5 bridges
tests/                     ~1900 lines   192 tests
docs/                       12 documents
```

Smaller than `safetensors` (~3000 LOC) and an order of magnitude smaller
than `torchao` (~30k LOC).

## Does it support [my framework / hardware]?

| Framework | Status |
| --- | --- |
| NumPy | ✅ v0.0.1 |
| PyTorch (CPU) | ✅ v0.0.1 |
| PyTorch (CUDA) | ✅ via round-trip in v0.0.1; native FP8 in v0.1 |
| MLX (Apple Silicon) | ✅ v0.0.1 (correctness only) |
| JAX | 🟡 v0.1 |
| TensorFlow | not planned |

| Hardware | Triton kernel? |
| --- | --- |
| NVIDIA Hopper (H100, H200) | ✅ in v0.1 (kernel ships in v0.0.1, validated v0.1) |
| NVIDIA Ada (L4, L40) | ✅ same as Hopper |
| NVIDIA Blackwell (B100/B200) | ✅ planned with NVFP4 hardware path |
| AMD MI300 | possible (Triton has experimental ROCm) |
| Trainium2 | requires neuron-aware kernel; bridge only in v0.0.1 |
| TPU v6 | via JAX backend (v0.1) |

## Will breccia publish on conda-forge?

Not for v0.0.1. Once v0.1 ships and the API is stable, yes.

## Is breccia fast on CPU?

The reference kernels are Python-loop-based. They're correctness-only.
For CPU-bound workloads at scale, the right approach is to dequantize
once at load time and run torch's CPU kernels on the dense form, then
re-quantize for storage.

A CPU-optimized scaled-matmul (e.g., LLVM-vectorized or via oneDNN) is
not planned for v0.0.1 / v0.1.

## Why no PyPI release yet?

breccia is at v0.0.1 — API may break between any two commits. v0.1 will
be the first PyPI-published release. Install from source for now:

```bash
git clone https://github.com/jvoltci/breccia
cd breccia
pip install -e ".[torch,mlx,bridges]"
```

## How do I file a bug?

GitHub Issues: <https://github.com/jvoltci/breccia/issues>

Please include:

- Your Python version, OS, breccia version
- Backend (NumPy / PyTorch / MLX) and version of that backend
- Recipe + layout you were using
- For Triton bugs: GPU model, CUDA driver, Triton version
- A minimal reproduction (~10 lines)

## How do I propose a new feature?

Open a GitHub Discussion first (1-2 paragraphs is enough). The
maintainers will respond with whether it fits breccia's scope and what
the API should look like. Then PRs follow from the discussion.

See [../CONTRIBUTING.md](../CONTRIBUTING.md).

## What's the long-term plan?

See [architecture.md → Versioning](architecture.md#versioning).

- **v0.0.1** (now) — primitive, 6 recipes, 4 layouts, 5 bridges,
  reference kernels, Triton kernel (untested on GPU)
- **v0.1.0** — Triton kernel validated at ≤ 1.2× cuBLAS FP8 GEMM,
  JAX backend, STE wrapper for training, PyPI release
- **v0.2.0** — Native PyTorch FP8 acceleration, asymmetric INT4 (with
  zero-point), vLLM/SGLang integration sketch, MLX `mlx.fast.*` Metal
  kernel
- **v1.0.0** — API stability commitment, conda-forge, broad ecosystem
  integration
