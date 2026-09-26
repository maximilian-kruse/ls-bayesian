from pathlib import Path

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.common.logging import BaseLogger, LoggerSettings
from ls_bayesian.mcmc.algorithms.pcn import PCNAlgorithm
from ls_bayesian.mcmc.model import MCMCModel
from tests.mcmc import helpers

pytestmark = pytest.mark.unit

# Double-precision round-off accumulated over dense O(STATE_DIM) linear algebra (matrix-vector
# products, one matrix inverse) on both sides of the comparison.
_DENSE_LINALG_RTOL = 1e-9


# ==================================================================================================
def test_acceptance_probability_matches_exact_density_ratio_with_generalized_proposal() -> None:
    """Detailed-balance check with a genuinely different `approximation` ($\\nu \\neq \\mu_0$): the
    MH exponent implemented by `_evaluate_acceptance_probability` must equal `[log pi(v) +
    log q(u|v)] - [log pi(u) + log q(v|u)]` for the *actual* target density `pi` (relative to
    `mu_0`, computed independently here with dense NumPy) and the transition kernel `q`
    drawn from `approximation` -- i.e. generalized pCN must actually sample `mu`, not merely be
    self-consistent with whatever correction it computes."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(60)
    reference_covariance = helpers.random_spd_matrix(rng, dim)
    reference_precision = np.linalg.inv(reference_covariance)
    reference = helpers.DenseGaussianMeasure(np.zeros(dim), reference_covariance, seed=61)
    approximation_covariance = helpers.random_spd_matrix(rng, dim)
    mean_vector = rng.standard_normal(dim)
    approximation = helpers.DenseGaussianMeasure(mean_vector, approximation_covariance, seed=62)
    hessian = helpers.random_spd_matrix(rng, dim)
    minimizer = rng.standard_normal(dim)
    target = helpers.QuadraticTargetMeasure(hessian, minimizer)
    step_width = 0.4
    model = MCMCModel(target=target, reference=reference, approximation=approximation)
    algorithm_under_test = PCNAlgorithm(model, step_width)

    u = rng.standard_normal(dim)
    v = rng.standard_normal(dim)

    def log_pi(state: np.ndarray) -> float:
        target_difference = state - minimizer
        return -(0.5 * target_difference @ hessian @ target_difference) - (
            0.5 * state @ reference_precision @ state
        )

    transition_covariance = step_width**2 * approximation_covariance
    transition_precision = np.linalg.inv(transition_covariance)

    def transition_mean(state: np.ndarray) -> np.ndarray:
        return mean_vector + np.sqrt(1 - step_width**2) * (state - mean_vector)

    def log_q(to_state: np.ndarray, from_state: np.ndarray) -> float:
        difference = to_state - transition_mean(from_state)
        return -0.5 * difference @ transition_precision @ difference

    expected_exponent = (log_pi(v) + log_q(u, v)) - (log_pi(u) + log_q(v, u))

    acceptance_u_to_v = algorithm_under_test._evaluate_acceptance_probability(u, v)
    algorithm_under_test_reverse = PCNAlgorithm(model, step_width)
    acceptance_v_to_u = algorithm_under_test_reverse._evaluate_acceptance_probability(v, u)

    np.testing.assert_allclose(
        np.log(acceptance_u_to_v) - np.log(acceptance_v_to_u),
        expected_exponent,
        rtol=_DENSE_LINALG_RTOL,
    )


# ==================================================================================================
def test_acceptance_probability_matches_exact_density_ratio_with_noncentered_reference() -> None:
    """Same detailed-balance check as
    `test_acceptance_probability_matches_exact_density_ratio_with_generalized_proposal`, but with a
    `reference` measure that has a non-zero mean -- catches the correction formula silently
    assuming a centered reference, which the all-zero-mean reference in that test cannot."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(63)
    reference_covariance = helpers.random_spd_matrix(rng, dim)
    reference_precision = np.linalg.inv(reference_covariance)
    reference_mean = rng.standard_normal(dim)
    reference = helpers.DenseGaussianMeasure(reference_mean, reference_covariance, seed=64)
    approximation_covariance = helpers.random_spd_matrix(rng, dim)
    approximation_mean = rng.standard_normal(dim)
    approximation = helpers.DenseGaussianMeasure(
        approximation_mean, approximation_covariance, seed=65
    )
    hessian = helpers.random_spd_matrix(rng, dim)
    minimizer = rng.standard_normal(dim)
    target = helpers.QuadraticTargetMeasure(hessian, minimizer)
    step_width = 0.4
    model = MCMCModel(target=target, reference=reference, approximation=approximation)
    algorithm_under_test = PCNAlgorithm(model, step_width)

    u = rng.standard_normal(dim)
    v = rng.standard_normal(dim)

    def log_pi(state: np.ndarray) -> float:
        target_difference = state - minimizer
        reference_difference = state - reference_mean
        return -(0.5 * target_difference @ hessian @ target_difference) - (
            0.5 * reference_difference @ reference_precision @ reference_difference
        )

    transition_covariance = step_width**2 * approximation_covariance
    transition_precision = np.linalg.inv(transition_covariance)

    def transition_mean(state: np.ndarray) -> np.ndarray:
        return approximation_mean + np.sqrt(1 - step_width**2) * (state - approximation_mean)

    def log_q(to_state: np.ndarray, from_state: np.ndarray) -> float:
        difference = to_state - transition_mean(from_state)
        return -0.5 * difference @ transition_precision @ difference

    expected_exponent = (log_pi(v) + log_q(u, v)) - (log_pi(u) + log_q(v, u))

    acceptance_u_to_v = algorithm_under_test._evaluate_acceptance_probability(u, v)
    algorithm_under_test_reverse = PCNAlgorithm(model, step_width)
    acceptance_v_to_u = algorithm_under_test_reverse._evaluate_acceptance_probability(v, u)

    np.testing.assert_allclose(
        np.log(acceptance_u_to_v) - np.log(acceptance_v_to_u),
        expected_exponent,
        rtol=_DENSE_LINALG_RTOL,
    )


# ==================================================================================================
def test_generalized_potential_reduces_to_target_cost_when_correction_is_zero() -> None:
    """With `approximation=None` (the default), `Phi_nu` must equal `Phi` exactly -- the
    classical-pCN fallback."""
    setup = helpers.create_quadratic_gaussian_setup(seed=70)
    algorithm_under_test = PCNAlgorithm(setup.to_model(), step_width=0.5)
    state = np.random.default_rng(71).standard_normal(helpers.STATE_DIM)

    generalized_potential = algorithm_under_test._evaluate_generalized_potential(state)

    assert generalized_potential == setup.target.evaluate_potential(state)


# ==================================================================================================
def test_logs_info_when_both_reference_and_approximation_given(tmp_path: Path) -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=70)
    approximation = helpers.DenseGaussianMeasure(
        np.random.default_rng(72).standard_normal(helpers.STATE_DIM),
        helpers.random_spd_matrix(np.random.default_rng(73), helpers.STATE_DIM),
        seed=74,
    )
    logfile_path = tmp_path / "pcn.log"
    logger_settings = LoggerSettings(print_to_console=False, logfile_path=logfile_path)
    model = MCMCModel(target=setup.target, reference=setup.reference, approximation=approximation)

    with BaseLogger(logger_settings, prefix="pcn") as logger:
        PCNAlgorithm(model, step_width=0.5, logger=logger)

    assert "approximation" in logfile_path.read_text()


# ==================================================================================================
def test_target_cost_evaluated_once_per_step_regardless_of_accept_reject_history() -> None:
    """Same caching contract as `MALAAlgorithm`/`PMALAAlgorithm`: N manually-driven steps must
    evaluate the target's cost exactly N+1 times, never re-evaluating a state already in cache --
    including across a rejected step, where the (unchanged) current state's potential must be
    reused rather than recomputed."""
    setup = helpers.create_quadratic_gaussian_setup(seed=80)
    counting_target = helpers.CallCountingTargetMeasure(setup.target)
    model = MCMCModel(target=counting_target, reference=setup.reference)
    algorithm_under_test = PCNAlgorithm(model, step_width=0.5)
    rng = np.random.default_rng(81)
    state = rng.standard_normal(helpers.STATE_DIM)

    num_steps = 4
    for step in range(num_steps):
        proposal = algorithm_under_test._create_proposal(state, rng)
        algorithm_under_test._evaluate_acceptance_probability(state, proposal)
        accepted = step % 2 == 0
        algorithm_under_test._update_cache(accepted)
        state = proposal if accepted else state

    assert counting_target.num_evaluate_potential_calls == num_steps + 1


# ==================================================================================================
@pytest.mark.parametrize("step_width", [0.0, 1.0, -0.1, 1.1], ids=["zero", "one", "below", "above"])
def test_step_width_must_be_in_open_unit_interval(step_width: float) -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises(BeartypeCallHintViolation):
        PCNAlgorithm(setup.to_model(), step_width=step_width)
