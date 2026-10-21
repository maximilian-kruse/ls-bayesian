import numpy as np
import pytest

from ls_bayesian.lowrank.eigensolvers import GeneralizedEigenpairs
from ls_bayesian.lowrank.gaussian import LowRankGaussian
from tests.lowrank import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def _truncated_formulas(
    setup: helpers.DenseLowRankSetup, rank: int
) -> tuple[GeneralizedEigenpairs, np.ndarray, np.ndarray]:
    """Dense covariance/precision of the rank-`rank` truncation, from the textbook formulas."""
    eigenvalues = setup.eigenvalues[:rank]
    eigenvectors = setup.eigenvectors[:, :rank]
    precision = setup.reference.precision_matrix
    covariance = (
        setup.reference.covariance_matrix
        - eigenvectors @ np.diag(eigenvalues / (1 + eigenvalues)) @ eigenvectors.T
    )
    posterior_precision = (
        precision + precision @ eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T @ precision
    )
    return GeneralizedEigenpairs(eigenvalues, eigenvectors), covariance, posterior_precision


# ==================================================================================================
def test_covariance_operator_matches_dense_posterior(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)
    identity = np.eye(helpers.PARAMETER_SIZE)

    covariance = np.column_stack([gaussian.apply_covariance_operator(c) for c in identity])

    np.testing.assert_allclose(covariance, dense_setup.posterior_covariance, atol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_precision_operator_matches_dense_posterior(dense_setup: helpers.DenseLowRankSetup) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)
    identity = np.eye(helpers.PARAMETER_SIZE)

    precision = np.column_stack([gaussian.apply_precision_operator(c) for c in identity])

    np.testing.assert_allclose(precision, dense_setup.posterior_precision, atol=1e-11)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rank", [1, 3, helpers.HESSIAN_RANK])
def test_covariance_and_precision_are_inverse_of_each_other(
    dense_setup: helpers.DenseLowRankSetup, rank: int
) -> None:
    eigenpairs, _, _ = _truncated_formulas(dense_setup, rank)
    gaussian = LowRankGaussian(dense_setup.reference, eigenpairs)
    vector = np.random.default_rng(1).standard_normal(helpers.PARAMETER_SIZE)

    recovered = gaussian.apply_covariance_operator(gaussian.apply_precision_operator(vector))

    np.testing.assert_allclose(recovered, vector, atol=1e-10)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rank", [1, 3, helpers.HESSIAN_RANK])
def test_truncated_operators_match_textbook_formulas(
    dense_setup: helpers.DenseLowRankSetup, rank: int
) -> None:
    eigenpairs, expected_covariance, expected_precision = _truncated_formulas(dense_setup, rank)
    gaussian = LowRankGaussian(dense_setup.reference, eigenpairs)
    vector = np.random.default_rng(2).standard_normal(helpers.PARAMETER_SIZE)

    np.testing.assert_allclose(
        gaussian.apply_covariance_operator(vector), expected_covariance @ vector, atol=1e-12
    )
    np.testing.assert_allclose(
        gaussian.apply_precision_operator(vector), expected_precision @ vector, atol=1e-11
    )


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("rank", [0, 2, helpers.HESSIAN_RANK])
def test_factorization_reproduces_covariance(
    dense_setup: helpers.DenseLowRankSetup, rank: int
) -> None:
    """$\\widehat\\Gamma\\widehat\\Gamma^T = \\Gamma$, also for a non-symmetric reference factor."""
    eigenpairs, expected_covariance, _ = _truncated_formulas(dense_setup, rank)
    gaussian = LowRankGaussian(dense_setup.reference, eigenpairs)
    identity = np.eye(gaussian.random_vector_size)

    factor = np.column_stack([gaussian.apply_covariance_factorization(c) for c in identity])

    np.testing.assert_allclose(factor @ factor.T, expected_covariance, atol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_evaluate_cost_matches_quadratic_form(dense_setup: helpers.DenseLowRankSetup) -> None:
    rng = np.random.default_rng(3)
    mean = rng.standard_normal(helpers.PARAMETER_SIZE)
    state = rng.standard_normal(helpers.PARAMETER_SIZE)
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs, mean=mean)

    expected_cost = 0.5 * (state - mean) @ dense_setup.posterior_precision @ (state - mean)

    cost = gaussian.evaluate_cost(state)

    assert isinstance(cost, float)
    np.testing.assert_allclose(cost, expected_cost, rtol=1e-11)


# --------------------------------------------------------------------------------------------------
def test_evaluate_cost_is_zero_at_the_mean(dense_setup: helpers.DenseLowRankSetup) -> None:
    mean = np.random.default_rng(4).standard_normal(helpers.PARAMETER_SIZE)
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs, mean=mean)

    assert gaussian.evaluate_cost(mean) == 0.0


# --------------------------------------------------------------------------------------------------
def test_mean_defaults_to_zero_and_is_protected_from_mutation(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)

    np.testing.assert_array_equal(gaussian.mean, np.zeros(helpers.PARAMETER_SIZE))
    gaussian.mean[:] = 1.0
    np.testing.assert_array_equal(gaussian.mean, np.zeros(helpers.PARAMETER_SIZE))


# --------------------------------------------------------------------------------------------------
def test_pointwise_variance_with_given_prior_variance(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)

    variance = gaussian.compute_pointwise_variance(
        prior_variance=np.diag(dense_setup.reference.covariance_matrix)
    )

    np.testing.assert_allclose(variance, np.diag(dense_setup.posterior_covariance), atol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_estimated_pointwise_variance_is_exact_for_diagonal_reference() -> None:
    """Rademacher probing is exact for a diagonal matrix, since $z \\circ Dz = D$."""
    rng = np.random.default_rng(5)
    prior_variance = rng.uniform(1.0, 2.0, helpers.PARAMETER_SIZE)
    reference = helpers.DenseReferenceCovariance(np.diag(prior_variance))
    eigenvectors = rng.standard_normal((helpers.PARAMETER_SIZE, 2))
    eigenpairs = GeneralizedEigenpairs(np.array([2.0, 0.5]), eigenvectors)
    gaussian = LowRankGaussian(reference, eigenpairs)

    estimated = gaussian.compute_pointwise_variance(num_probes=3)
    given = gaussian.compute_pointwise_variance(prior_variance=prior_variance)

    np.testing.assert_allclose(estimated, given, rtol=1e-12)


# --------------------------------------------------------------------------------------------------
def test_log_determinant_ratio_matches_dense_determinants(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)

    _, logdet_posterior = np.linalg.slogdet(dense_setup.posterior_covariance)
    _, logdet_reference = np.linalg.slogdet(dense_setup.reference.covariance_matrix)

    np.testing.assert_allclose(
        gaussian.compute_log_determinant_ratio(), logdet_posterior - logdet_reference, rtol=1e-11
    )


# --------------------------------------------------------------------------------------------------
def test_generate_sample_is_reproducible_for_equal_seed_and_varies_between_calls(
    dense_setup: helpers.DenseLowRankSetup,
) -> None:
    first = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs, seed=7)
    second = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs, seed=7)

    first_sample, first_sample_again = first.generate_sample(), first.generate_sample()

    np.testing.assert_array_equal(first_sample, second.generate_sample())
    assert not np.allclose(first_sample, first_sample_again)


# --------------------------------------------------------------------------------------------------
def test_negative_eigenvalue_raises(dense_setup: helpers.DenseLowRankSetup) -> None:
    eigenpairs = GeneralizedEigenpairs(np.array([1.0, -1e-3]), dense_setup.eigenvectors[:, :2])

    with pytest.raises(ValueError, match="non-negative"):
        LowRankGaussian(dense_setup.reference, eigenpairs)


# --------------------------------------------------------------------------------------------------
def test_mean_of_wrong_shape_raises(dense_setup: helpers.DenseLowRankSetup) -> None:
    with pytest.raises(ValueError, match="Mean has shape"):
        LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs, mean=np.zeros(3))


# --------------------------------------------------------------------------------------------------
def test_prior_variance_of_wrong_shape_raises(dense_setup: helpers.DenseLowRankSetup) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)

    with pytest.raises(ValueError, match="Prior variance has shape"):
        gaussian.compute_pointwise_variance(prior_variance=np.ones(3))


# --------------------------------------------------------------------------------------------------
def test_non_positive_number_of_probes_raises(dense_setup: helpers.DenseLowRankSetup) -> None:
    gaussian = LowRankGaussian(dense_setup.reference, dense_setup.exact_eigenpairs)

    with pytest.raises(ValueError, match="probes"):
        gaussian.compute_pointwise_variance(num_probes=0)
