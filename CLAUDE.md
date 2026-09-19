# Project: ls-bayesian

A modular toolbox for large-scale bayesian inverse problems.

## Working procedure
For non-trivial tasks:
1. Inspect the relevant code and tests.
2. Identify the existing design and conventions.
3. Form a short implementation plan.
4. Implement the smallest appropriate change.
5. Add/update tests.
6. Run relevant checks.
7. Review the final diff.

## Environment (pixi)
- This project uses pixi. NEVER use bare `python`, `pip`, `conda` or `uv`.
  Always run things through pixi: `pixi run <task>` or `pixi run python ...`.
- Add dependencies with `pixi add <pkg>` (conda-forge) or `pixi add --pypi <pkg>`
  only if the package isn't on conda-forge. Ask before adding any new dependency.
- Never hand-edit `pixi.lock`. Commit it together with `pixi.toml`/`pyproject.toml` changes.

## Design Principles
- Priorities, in order:
    1. Numerical correctness
    2. Reproducibility
    3. Maintainability and clear APIs
    4. Performance
    5. Developer convenience
- Do not sacrifice numerical correctness or reproducibility for small performance 
  or implementation conveniences.
- Prefer explicit, readable code over clever abstractions.
- Keep functions reasonably small and focused.
- Avoid unnecessary classes and abstractions.
- Prefer composition over inheritance
- Prefer small composable functions over large classes. Use a class only
  when there is real state to manage 
- Implement clean, low-level libraries, facilitate composition through builder, strategy, template pattern
- Make separate component for each logic degree of freedom, adhere to separation of concerns
- Public API can be broken if useful
- Keep computation pure: functions take inputs and return outputs, with no
  hidden global state, no in-place mutation of arguments
- Separate I/O from computation
- Validate inputs at the public API and raise clear errors
  (`ValueError` with the offending value); don't validate again internally.
- Make defaults safe and explicit. No magic numbers: name constants and
  cite their source.

## Coding conventions and style
- Python target from pyproject.toml, google-style docstrings, type hints on all methods/functions
- Use ruff for linting and formatting
- Do not add `# noqa` or other lint suppressions unless there is a concrete,
  documented reason.
- Vectorise with NumPy, Scipy; avoid Python loops over array elements.
- Accept array-likes, return NumPy arrays. Convert inputs once at the
  boundary with `np.asarray`
- Compare floats with `np.testing.assert_allclose`, never `==`.
- Seed all randomness (`np.random.default_rng(seed)`) in tests and examples.
- Do not silently change numerical conventions, units, array shapes, or dtypes.
- Use descriptive names for variables, functions, methods, and classes. Verbosity is fine
- Use `beartype` (`beartype.vale.Is`) for runtime-validated constraints on public API
  type hints instead of manual `if`/`raise` checks where a `Vale` predicate suffices.
  - Use @override decorates when appropriate
  - Import necessary components directly when names are unambiguous. Otherwise use module-level imports

## Documentation
- Explain content/concepts in docstrings, not just implementation
- Add docstrings to functions, methods, classes, and modules
- Content:
    - what the function does,
    - the mathematical meaning of important parameters,
    - expected shapes/dtypes where relevant,
    - assumptions,
    - important numerical considerations.
- Cross-reference other classes/functions with mkdocstrings-style links,
  `[`Name`][full.module.path.Name]`, using the full path under `ls_bayesian`
  (e.g. `[`SPDEPrior`][ls_bayesian.spde_prior.spde_prior.SPDEPrior]`). The `docs/` directory is a
  placeholder for a future mkdocstrings/mkdocs site that will render these links.
- Use LaTeX math (`$...$`, raw docstrings via `r"""`) for mathematical notation,
  consistent with existing docstrings.

## Testing
- Do not write tests until explicitly asked
- Test with pytest
- Every behavioral change should have appropriate tests.
- Prefer tests that verify mathematical properties and invariants rather than only
  testing implementation details.
- Test known analytical cases where possible.
- Run the relevant tests after making changes.
- NEVER loosen test tolerances, skip tests or change reference values
  to make tests pass. If a numerical result changes, stop and explain why.
- Compare floats with `np.testing.assert_allclose`, never `==`.

## Scientific / numerical code
- When implementing a method from a paper, cite it in the docstring
  and note any deviations from the published algorithm.
- Guidelines for algorithms:
    1. Understand the mathematical formulation first.
    2. Preserve the mathematical semantics of the existing implementation.
    3. Check array shapes, broadcasting, dtype, boundary conditions, and indexing.
    4. Consider numerical stability and conditioning.
    5. Add or update tests for the mathematical behavior being changed.
- Do not "fix" apparently unusual mathematical code without first determining why
  it is written that way.
- Do not replace a mathematically justified implementation with a simpler heuristic
  without explicitly discussing the trade-off.

## Performance
- Do not optimize prematurely.
- First establish correctness with tests.
- Do not introduce complicated optimizations without evidence that they matter.