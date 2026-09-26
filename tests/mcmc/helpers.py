"""Constants, setup objects and pure helper functions for the `mcmc` tests.

This is a regular module, imported by test modules and `conftest.py` files alike. Fixtures live in
the `conftest.py` files, everything that is imported by name lives here.
"""

from dataclasses import dataclass
from typing import override

import numpy as np

from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.measures import DifferentiableTargetMeasure, GaussianMeasure, TargetMeasure
from ls_bayesian.mcmc.model import MCMCModel
from ls_bayesian.mcmc.storage import MCMCStorage
from tests import notebook_helpers

STATE_DIM = 5


# ==================================================================================================
def random_spd_matrix(rng: np.random.Generator, dim: int) -> np.ndarray:
    factor = rng.random((dim, dim))
    return factor @ factor.T + dim * np.eye(dim)


# --------------------------------------------------------------------------------------------------
def gaussian_posterior_moments(
    hessian: np.ndarray, minimizer: np.ndarray, prior_covariance: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    r"""Closed-form mean/covariance of the Gaussian posterior obtained by combining a quadratic
    potential $\Phi(u) = \frac{1}{2}(u-a)^T H (u-a)$ (see `QuadraticTargetMeasure`) with a
    centered Gaussian reference $\mathcal N(0, C)$: the product of the two densities is Gaussian
    with precision $H + C^{-1}$ and mean $(H+C^{-1})^{-1} H a$.
    """
    reference_precision = np.linalg.inv(prior_covariance)
    posterior_precision = hessian + reference_precision
    posterior_covariance = np.linalg.inv(posterior_precision)
    posterior_mean = posterior_covariance @ (hessian @ minimizer)
    return posterior_mean, posterior_covariance


# ==================================================================================================
class DenseGaussianMeasure(GaussianMeasure):
    r"""Dense Gaussian measure $\mathcal N(\bar u, C)$, so the same object can play the
    classical-prior, generalized-pCN-proposal, or PMALA-preconditioner/reference role in tests."""

    def __init__(
        self,
        mean_vector: np.ndarray,
        covariance_matrix: np.ndarray,
        seed: int,
    ) -> None:
        self.mean_vector = mean_vector
        self.covariance_matrix = covariance_matrix
        self.precision_matrix = np.linalg.inv(covariance_matrix)
        self.covariance_factor = np.linalg.cholesky(covariance_matrix)
        self._rng = np.random.default_rng(seed)

    @property
    @override
    def mean(self) -> np.ndarray:
        return self.mean_vector

    @property
    @override
    def random_vector_size(self) -> int:
        return self.covariance_factor.shape[1]

    @override
    def apply_covariance_factorization(self, random_vector: np.ndarray) -> np.ndarray:
        return self.covariance_factor @ random_vector

    @override
    def apply_covariance_operator(self, vector: np.ndarray) -> np.ndarray:
        return self.covariance_matrix @ vector

    @override
    def apply_precision_operator(self, vector: np.ndarray) -> np.ndarray:
        return self.precision_matrix @ vector

    def generate_random_increment(self) -> np.ndarray:
        """Draw one i.i.d. standard normal vector of `random_vector_size`, from this instance's
        own seeded RNG."""
        return self._rng.standard_normal(self.random_vector_size)


# ==================================================================================================
class QuadraticTargetMeasure(DifferentiableTargetMeasure):
    r"""Quadratic potential $\Phi(u) = \frac{1}{2}(u-a)^T H (u-a)$, exact gradient $H(u-a)$.

    A quadratic $\Phi$ combined with a Gaussian reference measure makes the resulting target $\mu$
    itself Gaussian, with moments given in closed form by `gaussian_posterior_moments` -- used as
    the independent oracle for detailed-balance and stationarity checks.
    """

    def __init__(self, matrix: np.ndarray, minimizer: np.ndarray) -> None:
        self.matrix = matrix
        self.minimizer = minimizer

    @override
    def evaluate_potential(self, state: np.ndarray) -> float:
        difference = state - self.minimizer
        return float(0.5 * difference @ self.matrix @ difference)

    @override
    def evaluate_gradient(self, state: np.ndarray) -> np.ndarray:
        return self.matrix @ (state - self.minimizer)


# ==================================================================================================
@dataclass
class QuadraticGaussianSetup:
    """A quadratic target combined with a centered dense Gaussian reference, shared across
    multiple test files. `reference` is a fully-capable `GaussianMeasure`, so `to_model()` (with
    `approximation` left `None`) suffices for the classical/plain fallback of every algorithm."""

    target: QuadraticTargetMeasure
    reference: DenseGaussianMeasure
    hessian: np.ndarray
    minimizer: np.ndarray
    covariance: np.ndarray

    def to_model(self, approximation: GaussianMeasure | None = None) -> MCMCModel:
        """Build an `MCMCModel` from this setup's `target`/`reference`, optionally overriding
        `approximation`."""
        return MCMCModel(target=self.target, reference=self.reference, approximation=approximation)


# --------------------------------------------------------------------------------------------------
def create_quadratic_gaussian_setup(seed: int) -> QuadraticGaussianSetup:
    """Build a `QuadraticGaussianSetup` with random SPD target Hessian and reference covariance,
    and a random target minimizer, all derived from `seed`."""
    rng = np.random.default_rng(seed)
    hessian = random_spd_matrix(rng, STATE_DIM)
    covariance = random_spd_matrix(rng, STATE_DIM)
    minimizer = rng.standard_normal(STATE_DIM)
    reference = DenseGaussianMeasure(np.zeros(STATE_DIM), covariance, seed=seed + 1000)
    target = QuadraticTargetMeasure(hessian, minimizer)
    return QuadraticGaussianSetup(
        target=target,
        reference=reference,
        hessian=hessian,
        minimizer=minimizer,
        covariance=covariance,
    )


# ==================================================================================================
class PotentialOnlyTargetMeasure(TargetMeasure):
    """A `TargetMeasure` that is not a `DifferentiableTargetMeasure` -- for asserting that
    algorithms requiring a gradient reject it at construction."""

    @override
    def evaluate_potential(self, state: np.ndarray) -> float:
        return 0.0


# ==================================================================================================
class ZeroTargetMeasure(DifferentiableTargetMeasure):
    """Potential and gradient identically zero -- isolates the pure prior-reverting proposal from
    any drift contribution."""

    @override
    def evaluate_potential(self, state: np.ndarray) -> float:
        return 0.0

    @override
    def evaluate_gradient(self, state: np.ndarray) -> np.ndarray:
        return np.zeros_like(state)


# ==================================================================================================
class CallCountingTargetMeasure(DifferentiableTargetMeasure):
    """Wraps another `DifferentiableTargetMeasure`, counting calls to
    `evaluate_potential`/`evaluate_gradient` -- for asserting that MALA-family algorithms do not
    re-evaluate an already-cached state."""

    def __init__(self, wrapped: DifferentiableTargetMeasure) -> None:
        self._wrapped = wrapped
        self.num_evaluate_potential_calls = 0
        self.num_evaluate_gradient_calls = 0

    @override
    def evaluate_potential(self, state: np.ndarray) -> float:
        self.num_evaluate_potential_calls += 1
        return self._wrapped.evaluate_potential(state)

    @override
    def evaluate_gradient(self, state: np.ndarray) -> np.ndarray:
        self.num_evaluate_gradient_calls += 1
        return self._wrapped.evaluate_gradient(state)


# ==================================================================================================
class FakeMCMCAlgorithm(MCMCAlgorithm):
    """Deterministic double: constant acceptance probability, proposal obtained by adding
    `proposal_increment` to the current state. Records every `_update_cache` call for orchestration
    tests. `step_width` is fixed at `1.0`, unused beyond satisfying the ABC constructor."""

    def __init__(self, acceptance_probability: float, proposal_increment: float = 1.0) -> None:
        super().__init__(step_width=1.0)
        self.acceptance_probability = acceptance_probability
        self.proposal_increment = proposal_increment
        self.update_cache_calls: list[bool] = []

    @override
    def _create_proposal(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return state + self.proposal_increment

    @override
    def _evaluate_acceptance_probability(
        self, current_state: np.ndarray, proposal: np.ndarray
    ) -> float:
        return self.acceptance_probability

    @override
    def _update_cache(self, accepted: bool) -> None:
        self.update_cache_calls.append(accepted)


# ==================================================================================================
class FailingAfterNStepsAlgorithm(MCMCAlgorithm):
    """Always accepts and advances by `proposal_increment`, but raises `RuntimeError` on the
    `fail_at_call`-th call to `_create_proposal` -- exercises `Sampler.run`'s flush-on-exception
    guarantee."""

    def __init__(self, fail_at_call: int, proposal_increment: float = 1.0) -> None:
        super().__init__(step_width=1.0)
        self.fail_at_call = fail_at_call
        self.proposal_increment = proposal_increment
        self.num_create_proposal_calls = 0

    @override
    def _create_proposal(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        self.num_create_proposal_calls += 1
        if self.num_create_proposal_calls == self.fail_at_call:
            raise RuntimeError("synthetic failure")
        return state + self.proposal_increment

    @override
    def _evaluate_acceptance_probability(
        self, current_state: np.ndarray, proposal: np.ndarray
    ) -> float:
        return 1.0

    @override
    def _update_cache(self, accepted: bool) -> None:
        pass


# ==================================================================================================
class RecordingStorage(MCMCStorage):
    """Wraps another `MCMCStorage`, recording how often `flush` was called -- for testing
    `Sampler.run`'s flush-on-exception guarantee."""

    def __init__(self, wrapped: MCMCStorage) -> None:
        self._wrapped = wrapped
        self.flush_call_count = 0

    @override
    def store(self, sample: np.ndarray) -> None:
        self._wrapped.store(sample)

    @property
    @override
    def values(self) -> object:
        return self._wrapped.values

    @override
    def flush(self) -> None:
        self.flush_call_count += 1
        self._wrapped.flush()


# ==================================================================================================
MCMC_TUTORIALS_DIR = notebook_helpers.REPO_ROOT / "tutorials" / "mcmc"
PCN_NOTEBOOK = MCMC_TUTORIALS_DIR / "pcn.ipynb"
MALA_NOTEBOOK = MCMC_TUTORIALS_DIR / "mala.ipynb"
PMALA_NOTEBOOK = MCMC_TUTORIALS_DIR / "pmala.ipynb"
NOTEBOOK_EXECUTION_TIMEOUT_SECONDS = 120
