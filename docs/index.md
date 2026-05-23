# breccia documentation

A cross-framework block-scaled tensor primitive for low-precision compute
(FP8 / FP4 / MXFP8 / NVFP4 / INT4).

Sister library to [`scree`](https://github.com/jvoltci/scree). Together:
- **scree** handles variable-length data (loose fragments).
- **breccia** handles low-precision data bound by its scale (fragments + cement).

## For users

- [**Getting started**](getting-started.md) — install, first program, common patterns
- [**Concepts**](concepts.md) — mental model: data + scale + recipe + layout
- [**Recipes**](recipes.md) — the 6 scaling recipes and when to use each
- [**API reference**](api.md) — every public function and class
- [**Bridges & migration**](bridges.md) — moving from TransformerEngine, torchao, HF
- [**FAQ**](faq.md)

## For deep dives

- [**Formats**](formats.md) — bit-level FP8 / FP4 / MXFP8 / NVFP4 / INT4 layouts
- [**Numerics**](numerics.md) — accuracy, range, and trade-offs across recipes

## For contributors

- [**Architecture**](architecture.md) — internal layout, dispatch model, design decisions
- [**Kernels**](kernels.md) — reference and Triton scaled-matmul design
- [**Benchmarks**](benchmarks.md) — methodology + reproduction
- [**Contributing**](../CONTRIBUTING.md) — how to propose changes

## Quick links

- Source on GitHub: <https://github.com/jvoltci/breccia>
- PyPI: <https://pypi.org/project/breccia/>
- Issues: <https://github.com/jvoltci/breccia/issues>
- Discussions: <https://github.com/jvoltci/breccia/discussions>
