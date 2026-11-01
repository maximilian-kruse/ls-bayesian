import numpy as np
import pytest

from ls_bayesian.optimization.algorithms.newton_cg import (
    NewtonCGOptimizer,
    NewtonCGSettings,
    TrustRegionSettings,
)
from ls_bayesian.optimization.components.line_search import (
    ArmijoBacktrackingLineSearch,
    ArmijoBacktrackingLineSearchSettings,
)
from tests.optimization import helpers

pytestmark = pytest.mark.unit

DIMENSION = 8
CONVERGENCE_ABSOLUTE_TOLERANCE = 1e-6


def _optimizer(settings: NewtonCGSettings | None = None) -> NewtonCGOptimizer:
    return NewtonCGOptimizer(
        settings or NewtonCGSettings(gradient_norm_tolerance=1e-8),
        ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings()),
    )


# ==================================================================================================
def test_requires_hessian() -> None:
    assert NewtonCGOptimizer.requires_hessian is True


# --------------------------------------------------------------------------------------------------
def test_settings_reject_inverted_forcing_bounds() -> None:
    with pytest.raises(ValueError, match="minimum_forcing_term"):
        NewtonCGSettings(minimum_forcing_term=0.6, maximum_forcing_term=0.5)


# --------------------------------------------------------------------------------------------------
def test_forcing_term_is_relative_to_the_initial_gradient_and_clipped() -> None:
    optimizer = _optimizer(NewtonCGSettings(maximum_forcing_term=0.5, minimum_forcing_term=1e-3))
    # same ratio of gradient norms, very different scales: the forcing term must not change
    assert optimizer._forcing_term(1e6, 1e6) == 0.5
    assert optimizer._forcing_term(1e-6, 1e-6) == 0.5
    assert optimizer._forcing_term(4.0, 100.0) == pytest.approx(0.2)
    assert optimizer._forcing_term(4e-6, 1e-4) == pytest.approx(0.2)
    assert optimizer._forcing_term(1e-12, 1.0) == 1e-3


# --------------------------------------------------------------------------------------------------
def test_quadratic_converges_in_one_iteration_with_exact_cg() -> None:
    rng = np.random.default_rng(0)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    minimizer = rng.random(DIMENSION)
    model = helpers.QuadraticModel(matrix, minimizer)
    settings = NewtonCGSettings(
        gradient_norm_tolerance=1e-6, maximum_forcing_term=1e-8, minimum_forcing_term=1e-10
    )

    result = _optimizer(settings).run(np.zeros(DIMENSION), model)

    assert result.success
    np.testing.assert_allclose(result.result, minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE)
    assert result.num_iterations == 1


# --------------------------------------------------------------------------------------------------
def test_quadratic_in_cameron_martin_like_geometry_needs_few_cg_iterations() -> None:
    rng = np.random.default_rng(1)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    minimizer = rng.random(DIMENSION)
    # inner product = Hessian: the gradient representer is m - m*, the Hessian action the identity
    model = helpers.QuadraticModel(matrix, minimizer, inner_product_matrix=matrix)

    result = _optimizer().run(np.zeros(DIMENSION), model)

    assert result.success
    assert result.num_iterations == 1
    np.testing.assert_allclose(result.result, minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_line_search_failure_returns_last_iterate() -> None:
    class AscentModel(helpers.QuadraticModel):
        def evaluate_cost(self, parameter_vector):  # noqa: ANN001, ANN201
            # the quadratic model's descent direction increases this loss
            return float(-np.sum(parameter_vector**2))

    rng = np.random.default_rng(2)
    matrix = helpers.random_spd_matrix(rng, 3)
    model = AscentModel(matrix, rng.random(3))

    result = _optimizer().run(np.ones(3), model)

    assert not result.success
    assert "Line search failed" in result.status_message


# ==================================================================================================
def _trust_region_optimizer(
    trust_region: TrustRegionSettings | None = None, tolerance: float = 1e-8
) -> NewtonCGOptimizer:
    return NewtonCGOptimizer(
        NewtonCGSettings(
            gradient_norm_tolerance=tolerance, trust_region=trust_region or TrustRegionSettings()
        )
    )


# --------------------------------------------------------------------------------------------------
def test_exactly_one_globalization_is_required() -> None:
    line_search = ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings())
    with pytest.raises(ValueError, match="Exactly one"):
        NewtonCGOptimizer(NewtonCGSettings())
    with pytest.raises(ValueError, match="Exactly one"):
        NewtonCGOptimizer(NewtonCGSettings(trust_region=TrustRegionSettings()), line_search)


# --------------------------------------------------------------------------------------------------
def test_trust_region_settings_reject_inconsistent_thresholds() -> None:
    with pytest.raises(ValueError, match="expand_threshold"):
        TrustRegionSettings(shrink_threshold=0.8, expand_threshold=0.75)
    with pytest.raises(ValueError, match="shrink_threshold"):
        TrustRegionSettings(acceptance_threshold=0.2, shrink_threshold=0.1)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("ratio", "reached_boundary", "expected"),
    [
        (0.1, True, 0.5),  # poor agreement: shrink
        (0.1, False, 0.5),
        (0.5, True, 1.0),  # acceptable: keep
        (0.9, False, 1.0),  # good, but the step was interior: keep
        (0.9, True, 2.0),  # good and at the boundary: expand
        (-1.0, True, 0.5),  # the loss increased
    ],
)
def test_radius_update_rules(ratio: float, reached_boundary: bool, expected: float) -> None:
    optimizer = _trust_region_optimizer()
    assert optimizer._updated_radius(1.0, ratio, reached_boundary) == expected


# --------------------------------------------------------------------------------------------------
def test_trust_region_quadratic_converges_and_takes_the_full_first_step() -> None:
    rng = np.random.default_rng(0)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    minimizer = rng.random(DIMENSION)
    model = helpers.QuadraticModel(matrix, minimizer)
    optimizer = NewtonCGOptimizer(
        NewtonCGSettings(
            gradient_norm_tolerance=1e-6,
            maximum_forcing_term=1e-8,
            minimum_forcing_term=1e-10,
            trust_region=TrustRegionSettings(),
        )
    )

    result = optimizer.run(np.zeros(DIMENSION), model)

    assert result.success
    assert result.num_iterations == 1  # exact model, first step unconstrained, ratio 1
    np.testing.assert_allclose(result.result, minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE)


# --------------------------------------------------------------------------------------------------
def test_trust_region_rejection_cap_returns_last_iterate() -> None:
    class DeceptiveModel(helpers.QuadraticModel):
        def evaluate_cost(self, parameter_vector):  # noqa: ANN001, ANN201
            # the quadratic model promises a decrease that the loss never delivers
            return float(-np.sum(parameter_vector**2))

    rng = np.random.default_rng(2)
    model = DeceptiveModel(helpers.random_spd_matrix(rng, 3), rng.random(3))
    optimizer = _trust_region_optimizer(TrustRegionSettings(maximum_num_consecutive_rejections=3))

    result = optimizer.run(np.ones(3), model)

    assert not result.success
    assert "Trust region failed" in result.status_message
    assert result.num_iterations == 0
    np.testing.assert_allclose(result.result, np.ones(3))


# ==================================================================================================
@pytest.mark.parametrize("use_trust_region", [False, True])
def test_early_stopping_on_small_predicted_reduction_counts_as_converged(
    use_trust_region: bool,
) -> None:
    rng = np.random.default_rng(0)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    model = helpers.QuadraticModel(matrix, rng.random(DIMENSION))
    settings = NewtonCGSettings(
        gradient_norm_tolerance=1e-12,
        minimum_predicted_reduction=1e6,  # any Newton step predicts less than this
        trust_region=TrustRegionSettings() if use_trust_region else None,
    )
    line_search = (
        None
        if use_trust_region
        else ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings())
    )

    result = NewtonCGOptimizer(settings, line_search).run(np.zeros(DIMENSION), model)

    assert result.success
    assert result.num_iterations == 0
    assert "Stopped early" in result.status_message
    np.testing.assert_allclose(result.result, np.zeros(DIMENSION))


# --------------------------------------------------------------------------------------------------
def test_early_stopping_leaves_runs_with_large_predicted_reduction_alone() -> None:
    rng = np.random.default_rng(1)
    matrix = helpers.random_spd_matrix(rng, DIMENSION)
    minimizer = rng.random(DIMENSION)
    model = helpers.QuadraticModel(matrix, minimizer)
    settings = NewtonCGSettings(
        gradient_norm_tolerance=1e-6,
        maximum_forcing_term=1e-8,
        minimum_forcing_term=1e-10,
        minimum_predicted_reduction=1e-12,
    )
    line_search = ArmijoBacktrackingLineSearch(ArmijoBacktrackingLineSearchSettings())

    result = NewtonCGOptimizer(settings, line_search).run(np.zeros(DIMENSION), model)

    assert result.success
    assert "Converged" in result.status_message
    np.testing.assert_allclose(result.result, minimizer, atol=CONVERGENCE_ABSOLUTE_TOLERANCE)
