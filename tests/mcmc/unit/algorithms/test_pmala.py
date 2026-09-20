import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from ls_bayesian.mcmc.algorithms.pmala import PMALAAlgorithm
from tests.mcmc import helpers

pytestmark = pytest.mark.unit

# Double-precision round-off accumulated over dense O(STATE_DIM) linear algebra (matrix-vector
# products, matrix inverses) on both sides of the comparison.
_DENSE_LINALG_RTOL = 1e-9


# ==================================================================================================
def test_acceptance_probability_matches_exact_density_ratio_with_mismatched_preconditioner() -> None:
    """Detailed-balance check with a preconditioner genuinely different from the prior covariance
    (not merely a rescaling): this is the case `MALAAlgorithm` cannot handle and that this class
    exists to cover."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(20)
    covariance = helpers.random_spd_matrix(rng, dim)
    precision = np.linalg.inv(covariance)
    preconditioner_covariance = helpers.random_spd_matrix(rng, dim)
    preconditioner = helpers.DenseGaussianMeasure(
        np.zeros(dim), preconditioner_covariance, seed=21
    )
    reference = helpers.DenseGaussianMeasure(np.zeros(dim), covariance, seed=22)
    hessian = helpers.random_spd_matrix(rng, dim)
    minimizer = rng.standard_normal(dim)
    target = helpers.QuadraticTargetMeasure(hessian, minimizer)
    step_width = 0.29
    algorithm_under_test = PMALAAlgorithm(target, reference, preconditioner, step_width)

    u = rng.standard_normal(dim)
    v = rng.standard_normal(dim)

    def log_pi(state: np.ndarray) -> float:
        target_difference = state - minimizer
        return -(0.5 * target_difference @ hessian @ target_difference) - (
            0.5 * state @ precision @ state
        )

    rho = (2 - step_width) / (2 + step_width)
    sqrt_one_minus_rho_squared = np.sqrt(8 * step_width) / (2 + step_width)
    transition_covariance = sqrt_one_minus_rho_squared**2 * preconditioner_covariance
    transition_precision = np.linalg.inv(transition_covariance)
    drift_coefficient = 2 * step_width / (2 + step_width)

    def transition_mean(state: np.ndarray) -> np.ndarray:
        score = hessian @ (state - minimizer) + precision @ state
        return state - drift_coefficient * (preconditioner_covariance @ score)

    def log_q(to_state: np.ndarray, from_state: np.ndarray) -> float:
        difference = to_state - transition_mean(from_state)
        return -0.5 * difference @ transition_precision @ difference

    expected_exponent = (log_pi(v) + log_q(u, v)) - (log_pi(u) + log_q(v, u))

    acceptance_u_to_v = algorithm_under_test._evaluate_acceptance_probability(u, v)
    algorithm_under_test_reverse = PMALAAlgorithm(target, reference, preconditioner, step_width)
    acceptance_v_to_u = algorithm_under_test_reverse._evaluate_acceptance_probability(v, u)

    np.testing.assert_allclose(
        np.log(acceptance_u_to_v) - np.log(acceptance_v_to_u),
        expected_exponent,
        rtol=_DENSE_LINALG_RTOL,
    )


# ==================================================================================================
def test_collapses_to_mala_algorithm_when_preconditioner_and_reference_equal_prior() -> None:
    """With `preconditioner` and `reference_measure` both wrapping the target's actual prior --
    the same measure `MALAAlgorithm` would take as `proposal_measure` -- PMALA must reduce to
    plain MALA exactly (Beskos et al.'s Remark 3.8, `K(u) = C`)."""
    setup = helpers.create_quadratic_gaussian_setup(seed=30)
    step_width = 0.33

    mala_algorithm = MALAAlgorithm(setup.target, setup.reference, step_width)
    pmala_algorithm = PMALAAlgorithm(setup.target, setup.reference, setup.reference, step_width)

    state = np.random.default_rng(31).standard_normal(helpers.STATE_DIM)
    mala_proposal = mala_algorithm._create_proposal(state, np.random.default_rng(32))
    pmala_proposal = pmala_algorithm._create_proposal(state, np.random.default_rng(32))

    np.testing.assert_allclose(pmala_proposal, mala_proposal, rtol=1e-12)

    other_state = np.random.default_rng(33).standard_normal(helpers.STATE_DIM)
    mala_acceptance = mala_algorithm._evaluate_acceptance_probability(state, other_state)
    pmala_acceptance = pmala_algorithm._evaluate_acceptance_probability(state, other_state)

    np.testing.assert_allclose(pmala_acceptance, mala_acceptance, rtol=1e-10)


# ==================================================================================================
def test_preconditioner_mean_and_evaluate_cost_are_ignored() -> None:
    """Two preconditioners that differ only in `mean`/`evaluate_cost` (never in covariance or
    precision) must give identical proposals and acceptance probabilities -- both are documented
    as unread in the preconditioner role."""
    setup = helpers.create_quadratic_gaussian_setup(seed=40)
    covariance = helpers.random_spd_matrix(np.random.default_rng(41), helpers.STATE_DIM)
    preconditioner_a = helpers.DenseGaussianMeasure(
        np.zeros(helpers.STATE_DIM), covariance, seed=42
    )
    preconditioner_b = helpers.DenseGaussianMeasure(
        mean_vector=np.random.default_rng(43).standard_normal(helpers.STATE_DIM) * 1000.0,
        covariance_matrix=covariance,
        seed=42,
        correction_matrix=helpers.random_spd_matrix(np.random.default_rng(44), helpers.STATE_DIM),
    )
    step_width = 0.25
    algorithm_a = PMALAAlgorithm(setup.target, setup.reference, preconditioner_a, step_width)
    algorithm_b = PMALAAlgorithm(setup.target, setup.reference, preconditioner_b, step_width)

    state = np.random.default_rng(45).standard_normal(helpers.STATE_DIM)
    proposal_a = algorithm_a._create_proposal(state, np.random.default_rng(46))
    proposal_b = algorithm_b._create_proposal(state, np.random.default_rng(46))

    np.testing.assert_allclose(proposal_a, proposal_b, rtol=1e-12)

    other_state = np.random.default_rng(47).standard_normal(helpers.STATE_DIM)
    acceptance_a = algorithm_a._evaluate_acceptance_probability(state, other_state)
    acceptance_b = algorithm_b._evaluate_acceptance_probability(state, other_state)

    np.testing.assert_allclose(acceptance_a, acceptance_b, rtol=1e-12)


# ==================================================================================================
def test_gradient_evaluated_once_per_step_regardless_of_accept_reject_history() -> None:
    """Same caching contract as `MALAAlgorithm`: N manually-driven steps must evaluate the target's
    gradient/cost exactly N+1 times, never re-evaluating a state already in cache."""
    setup = helpers.create_quadratic_gaussian_setup(seed=50)
    counting_target = helpers.CallCountingTargetMeasure(setup.target)
    algorithm_under_test = PMALAAlgorithm(
        counting_target, setup.reference, setup.reference, step_width=0.2
    )
    rng = np.random.default_rng(51)
    state = rng.standard_normal(helpers.STATE_DIM)

    num_steps = 4
    for step in range(num_steps):
        proposal = algorithm_under_test._create_proposal(state, rng)
        algorithm_under_test._evaluate_acceptance_probability(state, proposal)
        accepted = step % 2 == 0
        algorithm_under_test._update_cache(accepted)
        state = proposal if accepted else state

    assert counting_target.num_evaluate_gradient_calls == num_steps + 1
    assert counting_target.num_evaluate_cost_calls == num_steps + 1


# ==================================================================================================
@pytest.mark.parametrize("step_width", [0.0, -1.0], ids=["zero", "negative"])
def test_step_width_must_be_positive(step_width: float) -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises(BeartypeCallHintViolation):
        PMALAAlgorithm(setup.target, setup.reference, setup.reference, step_width=step_width)
