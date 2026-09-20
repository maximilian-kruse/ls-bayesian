import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.mcmc.algorithms.pcn import PCNAlgorithm
from tests.mcmc import helpers

pytestmark = pytest.mark.unit

# Double-precision round-off accumulated over dense O(STATE_DIM) linear algebra (matrix-vector
# products, one matrix inverse) on both sides of the comparison.
_DENSE_LINALG_RTOL = 1e-9


# ==================================================================================================
def test_acceptance_probability_matches_exact_gaussian_transition_density_ratio_with_nontrivial_correction() -> (
    None
):
    """Detailed-balance check treating `proposal_measure` (with a genuine nonzero `evaluate_cost`
    correction) as the base comparison measure directly: the classical pCN kernel drawn from
    `proposal_measure` is exactly reversible with respect to it, so `Phi_nu(u) - Phi_nu(v)` must be
    the exact MH exponent for the density `exp(-Phi_nu) d(proposal_measure)`, independently of the
    generalized-pCN theory the code implements."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(60)
    covariance = helpers.random_spd_matrix(rng, dim)
    precision = np.linalg.inv(covariance)
    mean_vector = rng.standard_normal(dim)
    correction_matrix = helpers.random_spd_matrix(rng, dim)
    proposal_measure = helpers.DenseGaussianMeasure(
        mean_vector, covariance, seed=61, correction_matrix=correction_matrix
    )
    hessian = helpers.random_spd_matrix(rng, dim)
    minimizer = rng.standard_normal(dim)
    target = helpers.QuadraticTargetMeasure(hessian, minimizer)
    step_width = 0.4
    algorithm_under_test = PCNAlgorithm(target, proposal_measure, step_width)

    u = rng.standard_normal(dim)
    v = rng.standard_normal(dim)

    def phi_nu(state: np.ndarray) -> float:
        target_difference = state - minimizer
        correction_difference = state - mean_vector
        return (
            0.5 * target_difference @ hessian @ target_difference
            - 0.5 * correction_difference @ correction_matrix @ correction_difference
        )

    def log_pi(state: np.ndarray) -> float:
        reference_difference = state - mean_vector
        return -phi_nu(state) - 0.5 * reference_difference @ precision @ reference_difference

    transition_covariance = step_width**2 * covariance
    transition_precision = np.linalg.inv(transition_covariance)

    def transition_mean(state: np.ndarray) -> np.ndarray:
        return mean_vector + np.sqrt(1 - step_width**2) * (state - mean_vector)

    def log_q(to_state: np.ndarray, from_state: np.ndarray) -> float:
        difference = to_state - transition_mean(from_state)
        return -0.5 * difference @ transition_precision @ difference

    expected_exponent = (log_pi(v) + log_q(u, v)) - (log_pi(u) + log_q(v, u))

    acceptance_u_to_v = algorithm_under_test._evaluate_acceptance_probability(u, v)
    algorithm_under_test_reverse = PCNAlgorithm(target, proposal_measure, step_width)
    acceptance_v_to_u = algorithm_under_test_reverse._evaluate_acceptance_probability(v, u)

    np.testing.assert_allclose(
        np.log(acceptance_u_to_v) - np.log(acceptance_v_to_u),
        expected_exponent,
        rtol=_DENSE_LINALG_RTOL,
    )


# ==================================================================================================
def test_generalized_potential_reduces_to_target_cost_when_correction_is_zero() -> None:
    """With `evaluate_cost` identically zero (`QuadraticGaussianSetup.reference`), `Phi_nu` must
    equal `Phi` exactly -- the classical-pCN fallback."""
    setup = helpers.create_quadratic_gaussian_setup(seed=70)
    algorithm_under_test = PCNAlgorithm(setup.target, setup.reference, step_width=0.5)
    state = np.random.default_rng(71).standard_normal(helpers.STATE_DIM)

    generalized_potential = algorithm_under_test._evaluate_generalized_potential(state)

    assert generalized_potential == setup.target.evaluate_cost(state)


# ==================================================================================================
def test_current_potential_not_recomputed_after_an_accepted_step() -> None:
    """`_PCNStateCache` is only ever populated by `_update_cache` on acceptance (unlike MALA/PMALA,
    which also cache a rejected-and-persisted current state) -- exactly the reuse
    `algorithm.py`'s `_update_cache` docstring documents, no more. This locks in that precise,
    weaker contract so a future change is a deliberate decision, not an accidental regression."""
    setup = helpers.create_quadratic_gaussian_setup(seed=80)
    counting_target = helpers.CallCountingTargetMeasure(setup.target)
    algorithm_under_test = PCNAlgorithm(counting_target, setup.reference, step_width=0.5)
    rng = np.random.default_rng(81)
    state = rng.standard_normal(helpers.STATE_DIM)

    proposal = algorithm_under_test._create_proposal(state, rng)
    algorithm_under_test._evaluate_acceptance_probability(state, proposal)
    algorithm_under_test._update_cache(accepted=True)
    evaluations_after_first_step = counting_target.num_evaluate_cost_calls

    next_proposal = algorithm_under_test._create_proposal(proposal, rng)
    algorithm_under_test._evaluate_acceptance_probability(proposal, next_proposal)

    assert counting_target.num_evaluate_cost_calls == evaluations_after_first_step + 1


# ==================================================================================================
@pytest.mark.parametrize("step_width", [0.0, 1.0, -0.1, 1.1], ids=["zero", "one", "below", "above"])
def test_step_width_must_be_in_open_unit_interval(step_width: float) -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises(BeartypeCallHintViolation):
        PCNAlgorithm(setup.target, setup.reference, step_width=step_width)
