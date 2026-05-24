# Release process

How to cut a release of `breccia`. Owner: maintainer with merge access.

## Versioning

breccia follows [SemVer](https://semver.org/):

- **v0.0.x** — pre-alpha; the API may break between any two commits.
- **v0.1.0** — first beta release; the public API in `breccia.*` becomes
  stable. Backwards-incompatible changes after this require a minor
  version bump.
- **v1.0.0** — first stable release. Backwards-incompatible changes
  require a major bump.

## v0.1.0 release checklist

This is the gate from "pre-alpha source repo" to "library people can
`pip install` and depend on."

### 1. Code readiness

- [ ] All 250+ tests passing on CI (Python 3.10/3.11/3.12 × Ubuntu + 3.11 × macOS)
- [ ] No `# TODO` or `# FIXME` comments in `src/breccia/`
- [ ] Triton scaled_matmul kernel validated on H100 with PASS correctness
      (cos sim > 0.99 vs FP32 reference). v0.0.1 H100 run logged in
      [`benchmarks/modal_bench.py`](benchmarks/modal_bench.py) output.
- [ ] Triton autotune unblocked across Hopper / Ada / Blackwell
      (kernel currently autotunes over 5 configs; on first call this
      takes ~3s of compile time per config — acceptable for v0.1).
- [ ] TransformerEngine bridge validated on a real Linux+CUDA box with
      TE installed (v0.0.1 ships with bridge code + mock-tested unit
      tests; live TE validation is the open v0.1 item).

### 2. API surface frozen

- [ ] Final review of [`src/breccia/__init__.py`](src/breccia/__init__.py)
      — anything not exported here is private and may break in v0.2
- [ ] Final review of [`src/breccia/bridges/__init__.py`](src/breccia/bridges/__init__.py)
- [ ] Public API documented in [`docs/api.md`](docs/api.md), every
      function with signature + return + invariants
- [ ] Type hints present on every public function

### 3. Documentation

- [ ] [`README.md`](README.md) reflects v0.1.0 status (status table
      updated, "pre-alpha" removed)
- [ ] [`docs/`](docs/) tree complete: index, getting-started, concepts,
      recipes, formats, api, bridges, kernels, architecture, benchmarks,
      numerics, faq
- [ ] [`CHANGELOG.md`](CHANGELOG.md) has a real v0.1.0 entry with all
      changes since v0.0.1
- [ ] All examples in [`examples/`](examples/) run cleanly on a fresh
      `pip install`
- [ ] [`CONTRIBUTING.md`](CONTRIBUTING.md) reflects the current
      contribution flow

### 4. Benchmarks

- [ ] [`benchmarks/bench_memory.py`](benchmarks/bench_memory.py)
      reproduces the README's memory-savings table
- [ ] [`benchmarks/bench_accuracy.py`](benchmarks/bench_accuracy.py)
      reproduces the README's accuracy table
- [ ] [`benchmarks/modal_bench.py`](benchmarks/modal_bench.py) runs
      end-to-end on H100 with all 4 tests passing
- [ ] [`benchmarks/modal_te_validate.py`](benchmarks/modal_te_validate.py)
      runs on a CUDA Linux machine with TE installed (currently
      requires the user to set up an image with the CUDA toolkit
      pre-installed; documented limitation)
- [ ] Benchmark numbers in README and `docs/benchmarks.md` are within
      fp tolerance of the latest run

### 5. Packaging

- [ ] [`pyproject.toml`](pyproject.toml) `version = "0.1.0"`
- [ ] Package builds cleanly: `python -m build` produces a wheel + sdist
- [ ] Install from the wheel into a fresh venv and run all examples
- [ ] Install from the sdist into a fresh venv and run the test suite
- [ ] Long-description in `pyproject.toml` matches `README.md`
- [ ] `LICENSE` is included in the sdist (verified via `tar tf`)

### 6. Pre-launch credibility

- [ ] DM list of 8–10 named individuals contacted with API critique +
      benchmark sanity check. Targeted at people who own the existing
      fragmented solutions breccia replaces:
  - Tri Dao (FlashAttention / FP8 GEMM kernels)
  - Mark Saroufim (torchao, PyTorch quantization)
  - Jerry Zhang (torchao maintainer)
  - Paulius Micikevicius (NVIDIA TransformerEngine)
  - Driss Guessous (PyTorch native FP8)
  - Tim Dettmers (bitsandbytes / quantization research)
  - Phil Wang / lucidrains (small AI demos)
  - Woosuk Kwon / Simon Mo (vLLM, FP8 inference)
- [ ] 2–3 public endorsements lined up (quote tweet or comment on launch day)
- [ ] Launch blog post drafted (lives at
      `jvoltci.github.io/breccia/launch-blog/`), reviewed by 2 outside
      readers. Draft: see `/v5/launch-blog-draft.md` (private until
      launch).

### 7. Repo hygiene

- [ ] `main` branch protected; no direct pushes
- [ ] Issue templates for bug reports and feature requests (see
      `.github/ISSUE_TEMPLATE/`)
- [ ] Pull-request template references CONTRIBUTING.md
- [ ] CODEOWNERS or maintainers file
- [ ] Discord / Discussions enabled, link in README

### 8. Release artifacts

- [ ] Git tag `v0.1.0` on the release commit
- [ ] GitHub release with auto-generated changelog
- [ ] PyPI upload via `twine upload dist/*` (after `python -m build`)
- [ ] Verify `pip install breccia==0.1.0` works on a fresh machine

## Launch day sequence (after release artifacts land)

See [/v5/launch-day-runbook.md](../launch-day-runbook.md) (private until
launch) for the minute-by-minute sequence.

Headline target slots (Pacific time):

- **05:00** — GitHub repo flips public (if not already), v0.1.0 tag pushed
- **06:00** — Blog post live at jvoltci.github.io/breccia/launch-blog/
- **06:15** — X thread with the benchmark plot
- **06:30** — Show HN submission with technical depth
- **07:00** — Email + DM the pre-launch endorser list
- **08:00** — Reddit r/MachineLearning thread
- **Throughout** — Respond to every HN comment / X reply / GitHub issue
  within 1 hour

## Hotfix releases (v0.1.x)

Any patch that fixes a bug without changing the public API:

1. Branch from the v0.1.x tag (or main if main is still on v0.1.x)
2. Fix the bug with a test that fails before the fix and passes after
3. Bump the patch version in `pyproject.toml`
4. Add a CHANGELOG entry under the new version
5. Tag and release via the same PyPI flow above

## Pre-release checks (a script)

A `scripts/check_release.py` could automate steps 1, 2, 4 mechanically
— listed in PLAN.md as a non-GPU follow-up. Until that exists, the
checklist is manual.
