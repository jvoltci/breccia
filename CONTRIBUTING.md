# Contributing to breccia

Thanks for considering a contribution. breccia is small enough that any
change matters; here's how to land one.

## Before you write code

For anything beyond a typo fix, **open a GitHub Discussion first**:
<https://github.com/jvoltci/breccia/discussions>. One or two paragraphs
is enough. Tell us:

- What you want to change
- Why (the problem you're hitting)
- Roughly how (the API you have in mind)

The maintainers will respond with whether it fits breccia's scope and
what the design should look like. This is the cheapest way to avoid
sunk-cost rewrites.

If your change is a clear bug fix with an obvious solution, skip the
discussion and open a PR — the bar there is "is this actually a bug" and
"is the fix surgical."

## Setting up

```bash
git clone https://github.com/jvoltci/breccia
cd breccia
python -m venv .venv
.venv/bin/pip install -e ".[dev,torch,mlx,bridges]"
.venv/bin/pytest tests/ -v
```

You should see 192+ tests pass (as of v0.0.1). On macOS without MLX or
CUDA without torch, some tests skip gracefully.

## The bar for code

breccia is intentionally small. The package philosophy lives in
[CLAUDE.md](CLAUDE.md):

- **Think before coding.** Surface tradeoffs. If unsure, ask.
- **Simplicity first.** Minimum code that solves the problem.
- **Surgical changes.** Touch only what you must.
- **Goal-driven.** Tests pass before and after.

Concretely:

- New code lives in `src/breccia/`. Tests live in `tests/`. Docs live in `docs/`.
- Every public function has a docstring with parameters and return type.
- Every behavior change has a test. Bug fixes start with a test that
  fails before the fix.
- Lint with `ruff`, format with `black` (loose: 100 cols). If your editor
  argues with the existing style, match the existing style.

## What to work on

The most impactful contributions in v0.0:

1. **JAX backend** — the skeleton is in place; wire up `_cast_jax` and
   `_dequantize_jax` in `kernels/reference/cast.py`. See
   [docs/architecture.md](docs/architecture.md) → "Adding a new backend"
   for the template (the torch path is the reference).
2. **GPU validation of the Triton kernel** — run `benchmarks/modal_bench.py`
   on H100 and post the numbers. The kernel ships untested on GPU in
   v0.0.1; the validation harness is ready.
3. **Triton kernel for block-scaled FP8** — extend `scaled_matmul_triton`
   to support `Float8BlockScaling` (per-block-K scales fold inside the
   K loop).
4. **STE wrapper for autograd-aware quantized training** — currently the
   cast operation is non-differentiable. The v0.1 design lives in a
   GitHub Discussion.
5. **Asymmetric INT4 in the torchao bridge** — v0.0.1 supports symmetric
   only. Adding zero-point support touches the `_torchao.py` bridge
   and the `INT4Scaling` recipe.

If you want something smaller to start: fix any of the open issues
marked `good first issue` on GitHub.

## The PR flow

1. Fork and branch from `main`.
2. Make the change. Keep it focused; one PR = one logical change.
3. Add or update tests. They must pass.
4. Add or update docs. If you changed the public API or behavior, this
   is mandatory.
5. Run `pytest tests/ -v` locally and confirm 192+ passing (or whatever
   the current count is).
6. Push and open a PR. Include in the description:
   - What changes and why (link to the Discussion if you opened one)
   - How you tested it
   - Any non-obvious tradeoffs
7. CI runs on Python 3.10 / 3.11 / 3.12 (Ubuntu) + 3.11 (macOS). It must
   be green before merge.

## Commit message style

Follow the convention used in the repo:

```
<area>: <one-line summary, imperative mood>

<empty line>

<body explaining what changed and why, wrapped at ~72 cols>
```

`<area>` is one of: `core`, `kernels`, `bridges`, `tests`, `docs`,
`benchmarks`, `triton`, `mlx`, `ci`. For multi-area changes, use the
most prominent one. Examples from the history:

```
core: ScaledTensor primitive + invariants
core: 6 ScalingRecipe variants
core: Layout system (PerTensor / PerBlockK / PerChannel / PerBlockMN)
kernels: reference cast (quantize + dequantize) for all 6 recipes
bridges: TransformerEngine + torchao + HF safetensors + DLPack + DeepSeek-v3
triton: scaled_matmul kernel (per-tensor FP8, validation deferred)
```

Never use `git commit --amend` on commits that have been pushed. If you
need to fix a pushed commit, push a follow-up commit.

## Don't

- Don't add backwards-compat shims for in-flux internal APIs. Anything
  in v0.0 is allowed to break.
- Don't add a new optional dependency without strong justification. Each
  one adds install friction.
- Don't refactor unrelated code. Surgical changes.
- Don't add comments that re-state what the code does. Comment the *why*
  when it's non-obvious; leave the *what* to the code.
- Don't add features beyond what was asked. If your PR description says
  "fix X," the diff should be about X.

## Reviewing other people's PRs

If you have time and the maintainers haven't reviewed a PR yet, dive in.
We don't have an "approved reviewer" gate — useful reviews are useful.
Be specific. Reference line numbers. Suggest alternative code when you
think something's wrong.

## Governance

For v0.0 through v0.5, breccia is a BDFL project (the author of the
original commits has merge authority). Starting v0.5 or v1.0 (whichever
comes first), the project moves to a formal RFC process via GitHub
Discussions for any behavior-changing PR.

The maintainers are listed in [`pyproject.toml`](pyproject.toml).
Contact: open an issue or a discussion.

## License

breccia is Apache-2.0. By contributing, you agree that your contribution
will be licensed under Apache-2.0. No CLA is required.

## Thank you

Genuinely — every PR, issue, benchmark, doc fix is what makes this go
from "one person's weekend code" to a primitive the field actually uses.
