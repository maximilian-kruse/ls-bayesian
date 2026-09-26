r"""Bundles the pieces of a target measure that function-space MCMC algorithms need.

Classes:
    MCMCModel: Target potential, its mandatory reference measure, and an optional alternative
        Gaussian to propose from or precondition with.
"""

from dataclasses import dataclass

from ls_bayesian.mcmc.measures import GaussianMeasure, TargetMeasure


# ==================================================================================================
@dataclass(frozen=True)
class MCMCModel:
    r"""Bundles a target's potential $\Phi$ with its reference measure and, optionally, an
    alternative Gaussian -- built once, up front, and then reused across whichever
    [`MCMCAlgorithm`][ls_bayesian.mcmc.algorithm.MCMCAlgorithm] samples from it.

    `reference` ($\mu_0$, the measure relative to which $\Phi$ is stated) is mandatory: every
    algorithm in this subpackage needs it, since $\Phi$'s potential is only ever meaningful
    relative to it. `approximation` (an alternative Gaussian $\nu$) is optional; what it is used
    for, and whether it is required, differs per algorithm:

    - [`PCNAlgorithm`][ls_bayesian.mcmc.algorithms.pcn.PCNAlgorithm] proposes from `approximation`
      if given (logging an info message to that effect), else from `reference` directly. It
      corrects for a proposal drawn from `approximation` via the closed-form Gaussian
      Radon-Nikodym correction between `reference` and `approximation` (Pinski, Simpson, Stuart,
      Weber, 2015), identically $0$ when the two are equal or `approximation` is not given,
      recovering the classical pCN sampler (Cotter, Roberts, Stuart, White, 2013) exactly.
    - [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm] uses `reference` alone; if
      `approximation` is also given, it is ignored (logging an info message if a logger is
      passed), since this class has no way to correct for a mismatch.
    - [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm] additionally requires
      `approximation`, as the fixed preconditioner $\overline K$ (Beskos, Girolami, Lan, Farrell,
      Stuart, 2017). To run with a single Gaussian (i.e. $\overline K = C$, plain MALA), pass
      `reference` as `approximation` too, or prefer `MALAAlgorithm` directly.

    Attributes:
        target (TargetMeasure): Potential $\Phi(u)$ (and, for `MALAAlgorithm`/`PMALAAlgorithm`, its
            gradient) of the actual target $\mu$, relative to `reference`.
        reference (GaussianMeasure): The reference measure $\mu_0 = \mathcal N(\bar u, C)$
            relative to which $\Phi$ is stated.
        approximation (GaussianMeasure | None): Alternative Gaussian measure $\nu$ or $\overline K$,
            or `None` if not needed (see the per-algorithm behavior above). Defaults to `None`.
    """

    target: TargetMeasure
    reference: GaussianMeasure
    approximation: GaussianMeasure | None = None
