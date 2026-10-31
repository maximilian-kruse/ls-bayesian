from typing import override

import numpy as np
import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.mcmc.algorithms.phmc import PHMCAlgorithm
from ls_bayesian.mcmc.measures import DifferentiableTargetMeasure
from ls_bayesian.mcmc.model import MCMCModel
from tests.mcmc import helpers

pytestmark = pytest.mark.unit

# Double-precision round-off accumulated over dense O(STATE_DIM) linear algebra on both sides.
_DENSE_LINALG_RTOL = 1e-9


# ==================================================================================================
class _LogCoshTarget(DifferentiableTargetMeasure):
    r"""Non-quadratic potential $\Phi(u) = \sum_i \log\cosh(w_i^T u - b_i)$."""

    def __init__(self, weights: np.ndarray, offsets: np.ndarray) -> None:
        self._weights = weights
        self._offsets = offsets

    @override
    def evaluate_potential(self, state: np.ndarray) -> float:
        return float(np.sum(np.log(np.cosh(self._weights @ state - self._offsets))))

    @override
    def evaluate_gradient(self, state: np.ndarray) -> np.ndarray:
        return self._weights.T @ np.tanh(self._weights @ state - self._offsets)


# --------------------------------------------------------------------------------------------------
def _nonlinear_model(seed: int) -> tuple[MCMCModel, helpers.DenseGaussianMeasure, np.ndarray]:
    """A log-cosh target, a dense reference and a dense preconditioner that differs from it."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(seed)
    reference = helpers.DenseGaussianMeasure(np.zeros(dim), helpers.random_spd_matrix(rng, dim), 1)
    preconditioner = helpers.DenseGaussianMeasure(
        np.zeros(dim), 0.3 * helpers.random_spd_matrix(rng, dim), 2
    )
    target = _LogCoshTarget(rng.standard_normal((3, dim)), rng.standard_normal(3))
    model = MCMCModel(target=target, reference=reference, approximation=preconditioner)
    return model, preconditioner, rng.standard_normal(dim)


# --------------------------------------------------------------------------------------------------
def _plain_energy(model: MCMCModel, position: np.ndarray, velocity: np.ndarray) -> float:
    r"""$H(u, v) = \Phi(u) + \frac12\langle u, C^{-1}u\rangle + \frac12\langle v, \overline
    K^{-1}v\rangle$, with the dense measures of the model."""
    assert model.approximation is not None
    return float(
        model.target.evaluate_potential(position)
        + 0.5 * position @ model.reference.apply_precision_operator(position)
        + 0.5 * velocity @ model.approximation.apply_precision_operator(velocity)
    )


# ==================================================================================================
def test_energy_error_matches_exact_density_ratio_for_one_step() -> None:
    """For one leapfrog step the proposal is Gaussian, $u' \\sim \\mathcal N(\\cos\\varepsilon\\,u +
    \\sin\\varepsilon\\,\\frac{\\varepsilon}{2}g(u), \\sin^2\\varepsilon\\,\\overline K)$, so
    $\\exp(-\\Delta H)$ must equal the Metropolis-Hastings ratio of this proposal. Preconditioner
    and reference differ genuinely."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(60)
    covariance = helpers.random_spd_matrix(rng, dim)
    precision = np.linalg.inv(covariance)
    preconditioner_covariance = helpers.random_spd_matrix(rng, dim)
    preconditioner = helpers.DenseGaussianMeasure(np.zeros(dim), preconditioner_covariance, 61)
    reference = helpers.DenseGaussianMeasure(np.zeros(dim), covariance, 62)
    hessian = helpers.random_spd_matrix(rng, dim)
    minimizer = rng.standard_normal(dim)
    target = helpers.QuadraticTargetMeasure(hessian, minimizer)
    model = MCMCModel(target=target, reference=reference, approximation=preconditioner)
    angle = 0.37
    algorithm = PHMCAlgorithm(model, angle, num_steps=1)

    position = rng.standard_normal(dim)
    velocity = np.linalg.cholesky(preconditioner_covariance) @ rng.standard_normal(dim)
    trajectory = algorithm.integrate(position, velocity, angle, 1)

    def log_pi(state: np.ndarray) -> float:
        difference = state - minimizer
        return -0.5 * difference @ hessian @ difference - 0.5 * state @ precision @ state

    transition_precision = np.linalg.inv(np.sin(angle) ** 2 * preconditioner_covariance)

    def log_q(to_state: np.ndarray, from_state: np.ndarray) -> float:
        score = hessian @ (from_state - minimizer) + precision @ from_state
        drift = from_state - preconditioner_covariance @ score
        mean = np.cos(angle) * from_state + np.sin(angle) * angle / 2 * drift
        difference = to_state - mean
        return -0.5 * difference @ transition_precision @ difference

    new_position = trajectory.position
    expected = (log_pi(new_position) + log_q(position, new_position)) - (
        log_pi(position) + log_q(new_position, position)
    )

    np.testing.assert_allclose(-trajectory.energy_change, expected, rtol=_DENSE_LINALG_RTOL)


# ==================================================================================================
@pytest.mark.parametrize("record_path", [False, True])
def test_telescoped_energy_error_equals_plain_energy_difference(record_path: bool) -> None:
    """The dimension-robust form of Beskos et al.'s Theorem 3.10 (ii) equals $H(u_I, v_I) -
    H(u_0, v_0)$ for a non-quadratic potential and several steps, after every step."""
    model, preconditioner, position = _nonlinear_model(seed=70)
    angle, num_steps = 0.21, 7
    algorithm = PHMCAlgorithm(model, angle, num_steps)
    velocity = preconditioner.apply_covariance_factorization(
        np.random.default_rng(71).standard_normal(helpers.STATE_DIM)
    )

    trajectory = algorithm.integrate(position, velocity, angle, num_steps, record_path)

    expected = _plain_energy(model, trajectory.position, trajectory.velocity) - _plain_energy(
        model, position, velocity
    )
    np.testing.assert_allclose(trajectory.energy_change, expected, rtol=1e-8, atol=1e-10)
    if record_path:
        assert trajectory.energy_changes is not None
        assert trajectory.energy_changes.shape == (num_steps,)
        np.testing.assert_allclose(trajectory.energy_changes[-1], trajectory.energy_change)
        partial = algorithm.integrate(position, velocity, angle, 3)
        np.testing.assert_allclose(trajectory.energy_changes[2], partial.energy_change, rtol=1e-8)


# ==================================================================================================
def test_energy_is_conserved_when_preconditioner_is_the_target_covariance() -> None:
    """For a centered Gaussian target and $\\overline K$ equal to its covariance, $g \\equiv 0$:
    the leapfrog is the exact rotation and every trajectory is accepted."""
    dim = helpers.STATE_DIM
    rng = np.random.default_rng(80)
    covariance = helpers.random_spd_matrix(rng, dim)
    hessian = helpers.random_spd_matrix(rng, dim)
    posterior_covariance = np.linalg.inv(hessian + np.linalg.inv(covariance))
    reference = helpers.DenseGaussianMeasure(np.zeros(dim), covariance, 81)
    preconditioner = helpers.DenseGaussianMeasure(np.zeros(dim), posterior_covariance, 82)
    target = helpers.QuadraticTargetMeasure(hessian, np.zeros(dim))
    model = MCMCModel(target=target, reference=reference, approximation=preconditioner)
    algorithm = PHMCAlgorithm(model, 0.9, num_steps=5)
    state = rng.standard_normal(dim)

    proposal = algorithm._create_proposal(state, np.random.default_rng(83))

    assert algorithm._evaluate_acceptance_probability(state, proposal) == pytest.approx(1.0)


# ==================================================================================================
def test_energy_error_is_second_order_in_the_step_angle() -> None:
    """At a fixed total angle, halving the step angle reduces the root mean square energy error by
    about a factor of four."""
    model, preconditioner, position = _nonlinear_model(seed=90)
    total_angle = 0.8
    rng = np.random.default_rng(91)
    velocities = [
        preconditioner.apply_covariance_factorization(rng.standard_normal(helpers.STATE_DIM))
        for _ in range(30)
    ]

    def rms_error(num_steps: int) -> float:
        algorithm = PHMCAlgorithm(model, total_angle / num_steps, num_steps)
        errors = [
            algorithm.integrate(position, v, total_angle / num_steps, num_steps).energy_change
            for v in velocities
        ]
        return float(np.sqrt(np.mean(np.square(errors))))

    ratio = rms_error(8) / rms_error(16)

    assert 3.0 < ratio < 5.0


# ==================================================================================================
def test_trajectory_is_reversible_under_velocity_flip() -> None:
    model, preconditioner, position = _nonlinear_model(seed=100)
    angle, num_steps = 0.3, 6
    algorithm = PHMCAlgorithm(model, angle, num_steps)
    velocity = preconditioner.apply_covariance_factorization(
        np.random.default_rng(101).standard_normal(helpers.STATE_DIM)
    )

    forward = algorithm.integrate(position, velocity, angle, num_steps)
    backward = algorithm.integrate(forward.position, -forward.velocity, angle, num_steps)

    np.testing.assert_allclose(backward.position, position, atol=1e-10)
    np.testing.assert_allclose(backward.velocity, -velocity, atol=1e-10)
    np.testing.assert_allclose(backward.energy_change, -forward.energy_change, atol=1e-10)


# ==================================================================================================
def test_gradient_evaluated_once_per_leapfrog_step_and_cached_between_steps() -> None:
    """N manually driven steps of L leapfrog steps evaluate the target N * L + 1 times, however the
    steps are accepted or rejected: the current state is cached."""
    setup = helpers.create_quadratic_gaussian_setup(seed=110)
    counting_target = helpers.CallCountingTargetMeasure(setup.target)
    model = MCMCModel(
        target=counting_target, reference=setup.reference, approximation=setup.reference
    )
    num_leapfrog = 3
    algorithm = PHMCAlgorithm(model, 0.2, num_leapfrog, jitter=0.3)
    rng = np.random.default_rng(111)
    state = rng.standard_normal(helpers.STATE_DIM)

    num_steps = 4
    for step in range(num_steps):
        proposal = algorithm._create_proposal(state, rng)
        algorithm._evaluate_acceptance_probability(state, proposal)
        accepted = step % 2 == 0
        algorithm._update_cache(accepted)
        state = proposal if accepted else state

    expected = num_steps * num_leapfrog + 1
    assert counting_target.num_evaluate_gradient_calls == expected
    assert counting_target.num_evaluate_potential_calls == expected
    assert algorithm.num_gradient_evaluations == expected


# ==================================================================================================
def test_jitter_changes_the_step_angle_but_not_its_range() -> None:
    """With jitter the proposals of one state differ from the ones without, but are produced from
    the same velocity stream: the jitter consumes exactly one uniform draw."""
    setup = helpers.create_quadratic_gaussian_setup(seed=120)
    model = setup.to_model(setup.reference)
    state = np.random.default_rng(121).standard_normal(helpers.STATE_DIM)

    plain = PHMCAlgorithm(model, 0.3, 2)._create_proposal(state, np.random.default_rng(122))
    jittered = PHMCAlgorithm(model, 0.3, 2, jitter=0.5)._create_proposal(
        state, np.random.default_rng(122)
    )

    assert not np.allclose(plain, jittered)


# ==================================================================================================
@pytest.mark.parametrize("step_width", [0.0, -1.0], ids=["zero", "negative"])
def test_step_width_must_be_positive(step_width: float) -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises(BeartypeCallHintViolation):
        PHMCAlgorithm(setup.to_model(setup.reference), step_width, 2)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("num_steps", "jitter"), [(0, 0.0), (2, -0.1), (2, 1.0)], ids=["no-steps", "neg", "full"]
)
def test_steps_and_jitter_are_validated(num_steps: int, jitter: float) -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises(BeartypeCallHintViolation):
        PHMCAlgorithm(setup.to_model(setup.reference), 0.2, num_steps, jitter)


# --------------------------------------------------------------------------------------------------
def test_step_angle_must_stay_below_pi() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)

    with pytest.raises(ValueError, match="below pi"):
        PHMCAlgorithm(setup.to_model(setup.reference), 3.0, 2, jitter=0.1)


# --------------------------------------------------------------------------------------------------
def test_rejects_non_differentiable_target() -> None:
    reference = helpers.DenseGaussianMeasure(
        np.zeros(helpers.STATE_DIM), np.eye(helpers.STATE_DIM), seed=0
    )
    model = MCMCModel(
        target=helpers.PotentialOnlyTargetMeasure(), reference=reference, approximation=reference
    )

    with pytest.raises(ValueError, match="DifferentiableTargetMeasure"):
        PHMCAlgorithm(model, 0.2, 2)


# --------------------------------------------------------------------------------------------------
def test_rejects_model_without_approximation() -> None:
    setup = helpers.create_quadratic_gaussian_setup(seed=0)
    model = MCMCModel(target=setup.target, reference=setup.reference)

    with pytest.raises(ValueError, match="requires model.approximation"):
        PHMCAlgorithm(model, 0.2, 2)
