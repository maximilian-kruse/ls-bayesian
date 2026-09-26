# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# ls-bayesian
Modular toolbox for large-scale Bayesian inverse problems (Python ≥3.14, NumPy/SciPy,
dolfinx FEM priors, zarr MCMC storage).

**Procedure (non-trivial tasks):** inspect code/tests and conventions → short plan →
smallest appropriate change → tests → run checks → review diff.

## Environment (pixi)
- NEVER bare `python`/`pip`/`conda`/`uv`; always `pixi run ...`. Ask before adding
  deps (`pixi add`, `--pypi` only if not on conda-forge). Never hand-edit `pixi.lock`;
  commit it with `pyproject.toml` (holds all pixi config; no `pixi.toml`).
- Envs: `default` (numpy/scipy/beartype), `prior` (+dolfinx), `mcmc` (+zarr), `dev`
  (+ruff, jupyter, plotting), `test` (all + pytest). Tools need `-e`:
  - `pixi run -e dev ruff check src tests` / `ruff format src tests`
  - `pixi run -e test pytest [path::test] [-m "unit and not slow"] [-n auto]`
  - MPI: `pixi run -e test mpirun -n 2 python -m pytest tests/spde_prior/parallel -m parallel -p no:cacheprovider`

## Architecture
Subpackages in `src/ls_bayesian/` never import each other (only `common/`). Each
defines the interfaces it consumes as ABCs; application-layer adapters (tutorials,
test helpers) wire concrete pieces together.
- `spde_prior/`: SPDE Gaussian priors on dolfinx; `SPDEPriorBuilder` + component strategies.
- `posterior/`: `LogPosterior` = `Likelihood` + `ParameterToSolutionMap` + `GaussianPrior`
  (mirrors `SPDEPrior` API); caches forward/adjoint quantities.
- `optimization/`: `BaseOptimizer` (template over `_run_impl`) on `OptimizationModel`;
  L-BFGS variants built from strategy components.
- `mcmc/`: `Sampler` drives `MCMCAlgorithm`s (pCN, MALA, pMALA); proposal reference
  measure (`measures.py`) is deliberately separate from `GaussianPrior`.
- `lowrank/`: placeholder.

Tutorial notebooks (`tutorials/<sub>/`) are executed as integration tests via
`tests/notebook_helpers.py` — keep them runnable. Tests per subpackage: `unit/`,
`integration/` (anything running a full loop), `parallel/`; helpers in `helpers.py`,
fixtures only in `conftest.py`.

## Design
- Priorities: numerical correctness > reproducibility > clear APIs > performance >
  convenience. Never trade higher for lower.
- Explicit over clever; composition over inheritance; small functions; classes only
  for real state. Builder/strategy/template patterns, one component per degree of freedom.
- Pure computation (no hidden globals, no mutating arguments); I/O separate.
- Validate once at the public API (`ValueError` with offending value), preferring
  `beartype.vale.Is` type hints over manual checks.
- Safe, explicit defaults; no magic numbers — name constants and cite source.
- Breaking public API is fine if useful.

## Code style
- Google docstrings, full type hints, `@override`. Ruff; no `# noqa` without documented reason.
- Vectorise (no loops over elements); accept array-likes, `np.asarray` once at the
  boundary, return NumPy arrays. Never silently change conventions, units, shapes, dtypes.
- Descriptive names. Import unambiguous names directly, else module-level imports.

## Docs
- Docstring everything, explaining concepts: purpose, mathematical meaning of key
  parameters, shapes/dtypes, assumptions, numerics. LaTeX (`$...$`, raw `r"""`).
- mkdocstrings links with full path, e.g.
  `` [`SPDEPrior`][ls_bayesian.spde_prior.spde_prior.SPDEPrior] ``.

## Testing
- Don't write tests until asked; every behavioral change needs tests; run relevant ones.
- Prefer mathematical invariants/analytical cases over implementation details.
  `assert_allclose` for floats; seed via `np.random.default_rng(seed)`.
- NEVER loosen tolerances, skip tests, or change reference values to pass — if a
  numerical result changes, stop and explain why.

## Numerical code
- Cite papers in docstrings and note deviations.
- Before changing an algorithm: understand the math, preserve semantics, check
  shapes/broadcasting/dtype/boundaries/indexing and conditioning, update tests.
- Don't "fix" unusual math without knowing why it's there, or swap a justified method
  for a heuristic without discussing the trade-off.
- No premature or unproven optimization; correctness first.
