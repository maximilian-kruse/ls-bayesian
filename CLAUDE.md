# Project: ls-bayesian

A modular toolbox for large-scale bayesian inverse problems.

## Working procedure
For non-trivial tasks: inspect the relevant code/tests, identify existing design
and conventions, form a short plan, implement the smallest appropriate change,
add/update tests, run relevant checks, review the final diff.

## Environment (pixi)
- NEVER use bare `python`, `pip`, `conda` or `uv` — always `pixi run <task>` /
  `pixi run python ...`.
- Add dependencies with `pixi add <pkg>` (conda-forge) or `pixi add --pypi <pkg>`
  only if not on conda-forge. Ask before adding any new dependency.
- Never hand-edit `pixi.lock`; commit it together with `pixi.toml`/`pyproject.toml`.

## Design principles
- Priorities, in order: numerical correctness > reproducibility >
  maintainability/clear APIs > performance > developer convenience. Never trade
  a higher priority for a lower one.
- Prefer explicit, readable code over clever abstractions; avoid unnecessary
  classes. Prefer composition over inheritance and small composable functions
  over large classes — use a class only when there is real state to manage.
- Facilitate composition via builder/strategy/template patterns; give each
  logical degree of freedom its own component (separation of concerns).
- Keep computation pure (inputs -> outputs, no hidden global state, no
  in-place mutation of arguments); separate I/O from computation.
- Validate inputs once, at the public API, raising `ValueError` with the
  offending value; don't re-validate internally.
- Defaults must be safe and explicit; no magic numbers — name constants and
  cite their source.
- Public API can be broken if useful.

## Coding conventions and style
- Python target per pyproject.toml; google-style docstrings; type hints on all
  methods/functions. Use `@override` where appropriate.
- Lint/format with ruff; no `# noqa` or other suppressions without a concrete,
  documented reason.
- Vectorise with NumPy/SciPy — no Python loops over array elements. Accept
  array-likes, return NumPy arrays, converting once at the boundary via
  `np.asarray`.
- Never silently change numerical conventions, units, array shapes, or dtypes.
- Compare floats with `np.testing.assert_allclose`, never `==`. Seed all
  randomness with `np.random.default_rng(seed)` in tests and examples.
- Use descriptive names (verbosity is fine). Use `beartype`
  (`beartype.vale.Is`) for runtime-validated constraints on public API type
  hints instead of manual `if`/`raise` checks where a `Vale` predicate suffices.
- Import unambiguous names directly; otherwise use module-level imports.

## Documentation
- Docstrings on every function/method/class/module, explaining concepts, not
  just restating the implementation. Cover: what it does, the mathematical
  meaning of important parameters, expected shapes/dtypes, assumptions, and
  numerical considerations.
- Cross-reference with mkdocstrings-style links using the full path under
  `ls_bayesian`, e.g. `` [`SPDEPrior`][ls_bayesian.spde_prior.spde_prior.SPDEPrior] ``
  (`docs/` is a placeholder for a future mkdocstrings/mkdocs site).
- Use LaTeX math (`$...$`, raw `r"""` docstrings) for notation.

## Testing
- Don't write tests until explicitly asked; test with pytest. Every
  behavioral change needs appropriate tests; run relevant tests after changes.
- Prefer tests of mathematical properties/invariants and known analytical
  cases over pure implementation-detail tests.
- NEVER loosen tolerances, skip tests, or change reference values to make
  tests pass — if a numerical result changes, stop and explain why.

## Scientific / numerical code
- When implementing a method from a paper, cite it in the docstring and note
  any deviations from the published algorithm.
- Before touching an algorithm: understand its mathematical formulation,
  preserve existing mathematical semantics, check shapes/broadcasting/dtype/
  boundary conditions/indexing, consider numerical stability/conditioning,
  and update tests for the behavior being changed.
- Don't "fix" unusual-looking mathematical code without first determining why
  it's written that way, and don't replace a mathematically justified
  implementation with a simpler heuristic without explicitly discussing the
  trade-off.

## Performance
- Don't optimize prematurely — establish correctness with tests first, and
  don't introduce complicated optimizations without evidence they matter.
