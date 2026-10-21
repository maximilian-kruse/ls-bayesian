import numpy as np
import pytest

from ls_bayesian.lowrank.eigensolvers import (
    GeneralizedEigensolver,
    LanczosSolver,
    RandomizedDoublePassSolver,
    compute_generalized_eigenpairs,
)
from ls_bayesian.lowrank.gaussian import LowRankGaussian
from tests.lowrank import helpers

pytestmark = pytest.mark.integration

SOLVERS = {
    "lanczos": LanczosSolver(),
    "double_pass": RandomizedDoublePassSolver(oversampling=4),
    # Sketch larger than the Hessian rank: rank-deficient sketch must be handled.
    "double_pass_rank_deficient_sketch": RandomizedDoublePassSolver(oversampling=10),
}


# ==================================================================================================
def _compute(
    setup: helpers.DenseLowRankSetup, solver: GeneralizedEigensolver, num_eigenvalues: int
):
    return compute_generalized_eigenpairs(
        setup.apply_hessian,
        setup.reference.apply_precision_operator,
        setup.reference.apply_covariance_operator,
        helpers.PARAMETER_SIZE,
        num_eigenvalues,
        solver,
        minimum_eigenvalue=1e-8,
    )


# ==================================================================================================
@pytest.mark.parametrize("solver", SOLVERS.values(), ids=SOLVERS.keys())
@pytest.mark.parametrize("num_eigenvalues", [3, helpers.HESSIAN_RANK])
def test_eigenpairs_solve_generalized_problem(
    dense_setup: helpers.DenseLowRankSetup, solver: GeneralizedEigensolver, num_eigenvalues: int
) -> None:
    result = _compute(dense_setup, solver, num_eigenvalues)
    precision = dense_setup.reference.precision_matrix

    assert result.eigenvalues.shape == (num_eigenvalues,)
    assert np.all(np.diff(result.eigenvalues) < 0)
    np.testing.assert_allclose(
        result.eigenvalues, dense_setup.eigenvalues[:num_eigenvalues], rtol=1e-6
    )
    np.testing.assert_allclose(
        dense_setup.hessian @ result.eigenvectors,
        precision @ result.eigenvectors * result.eigenvalues,
        atol=1e-6 * dense_setup.eigenvalues[0],
    )
    np.testing.assert_allclose(
        result.eigenvectors.T @ precision @ result.eigenvectors,
        np.eye(num_eigenvalues),
        atol=1e-8,
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("solver", SOLVERS.values(), ids=SOLVERS.keys())
def test_eigenpairs_yield_dense_posterior_covariance(
    dense_setup: helpers.DenseLowRankSetup, solver: GeneralizedEigensolver
) -> None:
    result = _compute(dense_setup, solver, helpers.HESSIAN_RANK)
    gaussian = LowRankGaussian(dense_setup.reference, result)
    identity = np.eye(helpers.PARAMETER_SIZE)

    covariance = np.column_stack([gaussian.apply_covariance_operator(c) for c in identity])

    np.testing.assert_allclose(covariance, dense_setup.posterior_covariance, atol=1e-8)


# --------------------------------------------------------------------------------------------------
def test_rank_deficient_hessian_is_truncated_to_its_rank(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    result = _compute(dense_setup, RandomizedDoublePassSolver(oversampling=10), 10)

    assert result.eigenvalues.size == helpers.HESSIAN_RANK


# --------------------------------------------------------------------------------------------------
def test_sample_statistics_match_posterior(dense_setup: helpers.DenseLowRankSetup) -> None:
    """Sample mean/covariance deviate from the exact ones by at most five standard errors."""
    num_samples = 20_000
    mean = np.random.default_rng(10).standard_normal(helpers.PARAMETER_SIZE)
    gaussian = LowRankGaussian(
        dense_setup.reference,
        _compute(dense_setup, LanczosSolver(), helpers.HESSIAN_RANK),
        mean,
        seed=11,
    )
    covariance = dense_setup.posterior_covariance
    variance = np.diag(covariance)

    samples = np.array([gaussian.generate_sample() for _ in range(num_samples)])

    mean_error = np.abs(samples.mean(axis=0) - mean)
    assert np.all(mean_error < 5 * np.sqrt(variance / num_samples))
    covariance_error = np.abs(np.cov(samples, rowvar=False) - covariance)
    standard_error = np.sqrt((np.outer(variance, variance) + covariance**2) / num_samples)
    assert np.all(covariance_error < 5 * standard_error)


# --------------------------------------------------------------------------------------------------
def test_estimated_pointwise_variance_is_within_statistical_error(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    """Standard error of the Rademacher diagonal estimate is $\\sqrt{\\sum_{j\\ne i}C_{ij}^2/N}$."""
    num_probes = 2000
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)
    reference_covariance = dense_setup.reference.covariance_matrix
    off_diagonal = reference_covariance - np.diag(np.diag(reference_covariance))

    variance = gaussian.compute_pointwise_variance(num_probes=num_probes)

    standard_error = np.sqrt(np.sum(off_diagonal**2, axis=1) / num_probes)
    assert np.all(np.abs(variance - np.diag(dense_setup.posterior_covariance)) < 5 * standard_error)
