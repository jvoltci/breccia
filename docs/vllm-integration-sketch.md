# vLLM / SGLang integration sketch

A draft architectural proposal for letting vLLM and SGLang accept
`breccia.ScaledTensor` as a canonical low-precision weight + activation
type. This is an external-facing design doc — not yet an upstream PR.

## What changes for vLLM

vLLM today handles FP8 inference via a mix of NVIDIA TransformerEngine
hooks and its own scale-tracking logic. Each quantization format
(MXFP8, NVFP4, GPTQ INT4) ships as a distinct code path:

```
HF model weights      --> safetensors with format-specific metadata
        |
        v
vLLM internal loader  --> custom dequant per format (one path per recipe)
        |
        v
GEMM kernel           --> cuBLAS FP8 GEMM, GPTQ kernel, AWQ kernel...
```

The result: every new quantization format is a separate vLLM PR adding
a new code path. breccia consolidates this at the type layer.

### Proposed change

Add a single weight-loading converter to vLLM (Python-only, no kernel
changes):

```python
# vllm/model_loader/breccia_bridge.py
from breccia.bridges import load_safetensors as breccia_load
from breccia.bridges import from_torchao, from_transformer_engine

def load_breccia_weights(path: str) -> dict[str, ScaledTensor]:
    """Load a breccia safetensors checkpoint. Recipe + layout preserved."""
    return breccia_load(path)

def from_torchao_loaded(aqt) -> ScaledTensor:
    """Wrap an existing torchao AffineQuantizedTensor as a ScaledTensor."""
    return from_torchao(aqt)
```

And the kernel-side dispatch becomes:

```python
def fp8_gemm(activation: ScaledTensor, weight: ScaledTensor) -> Tensor:
    # vLLM's existing FP8 GEMM kernel; just consumes (data, scale, recipe).
    return breccia.matmul(activation, weight)
```

That's the integration: one weight loader, one type that all FP8 / FP4
/ INT4 weights flow through. ~50-100 LOC total.

### What this unlocks for vLLM

1. **One quantization code path.** Adding a new format means adding a
   new `ScalingRecipe`, not a new vLLM module. NVFP4, MXFP8, INT4 —
   all share the same loader and the same GEMM signature.

2. **Cross-engine portable checkpoints.** A model quantized by vLLM with
   the breccia format can be loaded by SGLang, served by an
   ExecuTorch deploy, or trained against by a JAX prototype — without
   re-quantization.

3. **Vendor neutrality.** vLLM today is NVIDIA-first because TE is
   NVIDIA-only. breccia's recipes are vendor-agnostic at the type
   layer; AMD MI355 FP8 or Trainium2 FP8 backends become drop-in
   alternatives at the kernel layer.

## What changes for SGLang

SGLang's RadixAttention and prefix caching are structured around
KV-cache layouts. The integration is at the same weight-load layer:
SGLang gains the ability to accept breccia-formatted weights, recipes
flowing through to its FP8 GEMM kernels unchanged.

The deeper v0.2 conversation: could SGLang's radix-tree prefix cache
hold breccia ScaledTensor handles instead of raw tensor slices? That's
out of scope for v0.1.

## What breccia does NOT change in either engine

- Kernel selection (vLLM keeps choosing FlashAttention / cuBLAS / its
  own; breccia is just the *type* the args travel in)
- Memory management (paged KV cache, block tables — all unchanged)
- Sampling, scheduling, sequence management — all unchanged
- API surface to the application — unchanged

**The proposal is purely type-level.** Both engines already use packed
low-precision data + scale tensors; we're proposing they reuse a typed
name for it across engines.

## What we want from the vLLM / SGLang teams

1. **Validation of the type shape.** Does
   `breccia.ScaledTensor(data, scale, recipe, layout, zero_point)`
   capture every variant of low-precision tensor you ship? Speak now if
   not — we want the primitive to fit your real workloads (vLLM has
   FP8 KV cache, MXFP4 weights, AWQ INT4; SGLang has FP8 GEMM and
   torchao integration).

2. **Permission to draft a PR.** A small (~100 LOC) PR adding the
   loader functions per engine, gated behind an optional `--breccia-format`
   flag so we can test without disturbing existing users.

3. **A test workload.** Ideally a CI-runnable workload (small Llama
   model, one short prompt) that both your engine and breccia-via-your-engine
   can run, so we can lock in numerical equivalence in your test suite.

## Timeline (proposal)

If both teams are open to this:

- **Week 0 (now)**: this document, soliciting feedback
- **Week 2**: draft PR for vLLM, gated behind a flag
- **Week 4**: same for SGLang
- **Week 8**: numerical equivalence tests landed in both repos
- **Week 12**: optional `--breccia-format` flag exposed to users; gather
  feedback
- **Week 24** (post-v0.1 of breccia): consider making it the default if
  there are no perf regressions

## Open questions

- How does vLLM's FP8 KV cache interact with breccia's `ScaledTensor`
  view? Today breccia treats `data` as a contiguous buffer; vLLM's KV
  cache is paged in HBM blocks. The bridge needs to handle non-contiguous
  data either by:
  1. forcing a contiguous copy at the boundary (cheap for weights,
     potentially expensive for KV), or
  2. teaching `ScaledTensor` about non-contiguous storage (more general,
     more complex).

  Open for discussion.

- Should breccia gain a "view of view" type for slicing into a paged
  KV without copying? Useful for inference engines but adds complexity
  to the primitive.

- For asymmetric INT4 (zero-point) — vLLM AWQ uses asymmetric. breccia
  v0.1 supports asymmetric via the `zero_point` field. Is the round-trip
  through `from_torchao` clean enough for production loading of AWQ
  checkpoints?

## Contact

This document is `docs/vllm-integration-sketch.md` in the breccia repo.
Issues, comments, and counter-proposals welcome at
<https://github.com/jvoltci/breccia/discussions>.

— breccia maintainer
