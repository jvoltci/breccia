---
hide:
  - navigation
  - toc
---

<div class="breccia-hero" markdown>

# breccia

<p class="tagline" markdown>
Block-scaled tensors as a first-class type. **Triton FP8 scaled-matmul validated on H100** (cos sim 0.9993 vs FP32). Works on NumPy, PyTorch, MLX, JAX.
</p>

<div class="badges" markdown>
[![PyPI](https://img.shields.io/pypi/v/breccia.svg?style=flat-square&color=5b21b6)](https://pypi.org/project/breccia/)
[![Python](https://img.shields.io/pypi/pyversions/breccia.svg?style=flat-square&color=5b21b6)](https://pypi.org/project/breccia/)
[![License](https://img.shields.io/pypi/l/breccia.svg?style=flat-square&color=5b21b6)](https://github.com/jvoltci/breccia/blob/master/LICENSE)
[![CI](https://github.com/jvoltci/breccia/actions/workflows/ci.yml/badge.svg)](https://github.com/jvoltci/breccia/actions/workflows/ci.yml)
[![Stars](https://img.shields.io/github/stars/jvoltci/breccia.svg?style=social)](https://github.com/jvoltci/breccia)
</div>

</div>

```python
import breccia, numpy as np

# Quantize to FP8 with per-block-K scaling (DeepSeek-v3 style)
x = np.random.randn(8, 256).astype(np.float32)
st = breccia.cast(x, breccia.Float8BlockScaling(block_k=128))

# Scaled matmul: data stays in FP8, scales fold into the FP32 accumulator
A = breccia.cast(np.random.randn(16, 256).astype(np.float32), breccia.Float8CurrentScaling())
W = breccia.cast(np.random.randn(256, 128).astype(np.float32), breccia.Float8BlockScaling(block_k=128))
y = breccia.matmul(A, W)
```

[Get started →](getting-started.md){ .md-button .md-button--primary }
[GitHub →](https://github.com/jvoltci/breccia){ .md-button }
[PyPI →](https://pypi.org/project/breccia/){ .md-button }

## At a glance

<div class="breccia-metrics" markdown>

<div class="breccia-metric" markdown>
<div class="value">0.9993</div>
<div class="label">Cos sim vs FP32 (Triton on H100)</div>
</div>

<div class="breccia-metric" markdown>
<div class="value">4×</div>
<div class="label">Memory savings vs FP32 (FP8 / FP4 / INT4)</div>
</div>

<div class="breccia-metric" markdown>
<div class="value">6</div>
<div class="label">Recipes covering today's fragmentation</div>
</div>

<div class="breccia-metric" markdown>
<div class="value">4</div>
<div class="label">Backends: NumPy, PyTorch, MLX, JAX</div>
</div>

<div class="breccia-metric" markdown>
<div class="value">5</div>
<div class="label">Bridges: TE, torchao, HF, DLPack, DeepSeek</div>
</div>

<div class="breccia-metric" markdown>
<div class="value">250+</div>
<div class="label">Tests across all backends</div>
</div>

</div>

## Why breccia

Block-scaled low-precision is everywhere in modern ML — but every framework
carries its own incompatible representation. breccia is the typed primitive
that bridges them:

- **NVIDIA TransformerEngine** — 4 non-composable recipe classes, NVIDIA-only → `breccia.bridges.from_transformer_engine`
- **PyTorch torchao** — `AffineQuantizedTensor`, PyTorch-only → `breccia.bridges.from_torchao`
- **DeepSeek-v3 FP8 weights** — private block-scaled format → `breccia.bridges.from_deepseek_v3`
- **HuggingFace safetensors** — no native scale metadata → `breccia.bridges.save_safetensors` with recipe + layout preserved
- **AMD MI355 / Trainium2 / TPU v6** — incompatible scale semantics → one type, four backends today

The cross-vendor gap is *widening* through 2026–2027 with FP4. No vendor can
be the neutral substrate. breccia is the "safetensors of low-precision."

## What you can do today

| Workflow | Use case | Status |
| --- | --- | --- |
| FP8 inference | Quantize + scaled matmul end-to-end | <span class="breccia-chip accent">native torch.float8_e4m3fn</span> |
| FP8 training | Forward + STE for gradient flow on PyTorch / JAX | <span class="breccia-chip accent">`cast_ste` shipped</span> |
| DeepSeek-v3 weight loading | Bit-exact `from_deepseek_v3` round-trip | <span class="breccia-chip accent">v0.1</span> |
| Asymmetric INT4 (GPTQ / AWQ) | `INT4Scaling(symmetric=False)` + `zero_point` | <span class="breccia-chip accent">v0.1</span> |
| NVFP4 / MXFP8 quantize | Hardware-spec-locked block sizes (16 / 32) | <span class="breccia-chip accent">v0.1</span> |
| Triton FP8 scaled matmul on Hopper / Ada / Blackwell | DeepSeek-pattern block-scaled GEMM | <span class="breccia-chip accent">H100 validated</span> |
| Cross-framework prototyping | Same `ScaledTensor` on NumPy / PyTorch / MLX / JAX | <span class="breccia-chip accent">250+ tests verify</span> |

## Examples

- [01 — Quickstart](https://github.com/jvoltci/breccia/blob/master/examples/01_quickstart.py): cast + scaled matmul in 15 lines
- [02 — Recipe-portable training](https://github.com/jvoltci/breccia/blob/master/examples/02_recipe_portable_train.py): train MXFP8, ship NVFP4 (same model code)
- [03 — Checkpoint with scale](https://github.com/jvoltci/breccia/blob/master/examples/03_checkpoint_with_scale.py): safetensors round-trip preserving recipe + layout
- [04 — TE migration](https://github.com/jvoltci/breccia/blob/master/examples/04_te_migration.py): bridge TransformerEngine `Float8Tensor` → `ScaledTensor`

## The name

A *breccia* is a sedimentary rock made of broken angular fragments held
together by a cementing matrix. Low-precision data fragments + the scale
tensor that gives them meaning — same structure.

It's the natural geological successor to
[`scree`](https://github.com/jvoltci/scree): loose fragments (scree) become
breccia when cemented together.

---

<p style="text-align: center; opacity: 0.7; font-size: 0.9em;">
v0.1.1 on PyPI. Apache-2.0. <a href="https://github.com/jvoltci/breccia">Source on GitHub</a> · <a href="faq/">FAQ</a> · <a href="https://github.com/jvoltci/breccia/discussions">Discussions</a> · <a href="https://github.com/jvoltci/breccia/issues">Issues</a>
</p>
