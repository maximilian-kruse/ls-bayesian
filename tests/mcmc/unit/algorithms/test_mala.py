import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.mcmc.algorithms.mala import MALAAlgorithm
from tests.mcmc import helpers

pytestmark = pytest.mark.unit

# Double-precision round-off accumulated over dense O(STATE_DIM) linear algebra (matrix-vector
# products, one matrix inverse) on both sides of the comparison.
_DENSE_LINALG_RTOL = 1e-9


# ==================================================================================================
def test_acceptance_probability_matches_exact_gaussian_transition_density_ratio() -> None:
    """Detailed-balance check: the MH exponent `log a(u,v) - log a(v,u)` implemented by
    `_evaluate_acceptance_probability` must equal `[log pi(v) + log q(u|v)] - [log pi(u) +
    log q(v|u)]` for the *exact* (unnormalized) target density pi and transition kernel q,
    computed independently here with dense NumPy rather than via the code under test."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(11)
    covariance = helpers.random_spd_matrix(rng, dim)
    precision = np.linalg.inv(covariance)
    mean_vector = rng.standard_normal(dim)
    proposal_measure = helpers.DenseGaussianMeasure(mean_vector, covariance, seed=12)
    hessian = helpers.random_spd_matrix(rng, dim)
    minimizer = rng.standard_normal(dim)
    target = helpers.QuadraticTargetMeasure(hessian, minimizer)
    step_width = 0.37
    algorithm_under_test = MALAAlgorithm(target, proposal_measure, step_width)

    u = rng.standard_normal(dim)
    v = rng.standard_normal(dim)

    def log_pi(state: np.ndarray) -> float:
        target_difference = state - minimizer
        reference_difference = state - mean_vector
        return -(0.5 * target_difference @ hessian @ target_difference) - (
            0.5 * reference_difference @ precision @ reference_difference
        )

    sqrt_one_minus_rho_squared = np.sqrt(8 * step_width) / (2 + step_width)
    transition_covariance = sqrt_one_minus_rho_squared**2 * covariance
    transition_precision = np.linalg.inv(transition_covariance)

    def transition_mean(state: np.ndarray) -> np.ndarray:
        gradient = hessian @ (state - minimizer)
        return mean_vector + (
            (2 - step_width) * (state - mean_vector) - 2 * step_width * (covariance @ gradient)
        ) / (2 + step_width)

    def log_q(to_state: np.ndarray, from_state: np.ndarray) -> float:
        difference = to_state - transition_mean(from_state)
        return -0.5 * difference @ transition_precision @ difference

    expected_exponent = (log_pi(v) + log_q(u, v)) - (log_pi(u) + log_q(v, u))

    # `_evaluate_acceptance_probability` caches quantities for its `current_state` argument across
    # calls, assuming that argument never changes; a fresh instance per direction keeps the two
    # evaluations independent, exactly as they would be on two separate chains.
    acceptance_u_to_v = algorithm_under_test._evaluate_acceptance_probability(u, v)
    algorithm_under_test_reverse = MALAAlgorithm(target, proposal_measure, step_width)
    acceptance_v_to_u = algorithm_under_test_reverse._evaluate_acceptance_probability(v, u)

    np.testing.assert_allclose(
        np.log(acceptance_u_to_v) - np.log(acceptance_v_to_u),
        expected_exponent,
        rtol=_DENSE_LINALG_RTOL,
    )


# --------------------------------------------------------------------------------------------------
def test_acceptance_is_one_when_proposal_equals_current_state() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)
    algorithm_under_test = MALAAlgorithm(setup.target, setup.reference, step_width=0.2)
    state = np.random.default_rng(1).standard_normal(helpers.STATE_DIM)

    acceptance_probability = algorithm_under_test._evaluate_acceptance_probability(state, state)

    assert acceptance_probability == 1.0


# ==================================================================================================
def test_proposal_with_zero_gradient_is_pure_prior_reverting_step() -> None:
    """With `Phi` identically zero, the drift term vanishes and the proposal reduces to the
    classical pCN-type AR(1) step around `proposal_measure.mean`."""
    dim = helpers.STATE_DIM
    rng_setup = np.random.default_rng(5)
    covariance = helpers.random_spd_matrix(rng_setup, dim)
    mean_vector = rng_setup.standard_normal(dim)
    proposal_measure = helpers.DenseGaussianMeasure(mean_vector, covariance, seed=7)
    target = helpers.ZeroTargetMeasure()
    step_width = 0.4
    algorithm_under_test = MALAAlgorithm(target, proposal_measure, step_width)
    state = rng_setup.standard_normal(dim)

    proposal = algorithm_under_test._create_proposal(state, np.random.default_rng(9))

    random_increment = proposal_measure.apply_covariance_factorization(
        np.random.default_rng(9).normal(size=proposal_measure.random_vector_size)
    )
    rho = (2 - step_width) / (2 + step_width)
    sqrt_one_minus_rho_squared = np.sqrt(8 * step_width) / (2 + step_width)
    expected_proposal = (
        mean_vector + rho * (state - mean_vector) + sqrt_one_minus_rho_squared * random_increment
    )

    np.testing.assert_allclose(proposal, expected_proposal, rtol=1e-12)


# ==================================================================================================
def test_gradient_evaluated_once_per_step_regardless_of_accept_reject_history() -> None:
    """After N manually-driven steps (accept/reject forced deterministically, not via `rng`), the
    target's gradient/cost must be evaluated exactly N+1 times -- once for the initial current
    state, once per step for its proposal -- never re-evaluating a state already in cache. Twice
    that count (2N) would mean the cache introduced in `algorithm.py` is not actually being hit."""
    setup = helpers.create_quadratic_gaussian_setup(seed=4)
    counting_target = helpers.CallCountingTargetMeasure(setup.target)
    algorithm_under_test = MALAAlgorithm(counting_target, setup.reference, step_width=0.2)
    rng = np.random.default_rng(3)
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
        MALAAlgorithm(setup.target, setup.reference, step_width=step_width)


# --------------------------------------------------------------------------------------------------
def test_proposal_shape_matches_state() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)
    algorithm_under_test = MALAAlgorithm(setup.target, setup.reference, step_width=0.2)
    state = np.random.default_rng(2).standard_normal(helpers.STATE_DIM)

    proposal = algorithm_under_test._create_proposal(state, np.random.default_rng(3))

    assert proposal.shape == state.shape
