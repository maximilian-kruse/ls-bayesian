import numpy as np
import pytest

from ls_bayesian.optimization.components.seed_scaling import (
    BarzilaiBorweinSeedScaling,
    BarzilaiBorweinSeedScalingSettings,
    NoSeedScaling,
)
from tests.optimization import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_no_seed_scaling_returns_one_without_a_correction_pair() -> None:
    strategy = NoSeedScaling()
    model = helpers.ZeroModel()

    assert strategy.compute_scale_factor(None, None, model) == 1.0


# --------------------------------------------------------------------------------------------------
def test_no_seed_scaling_returns_one_with_a_correction_pair() -> None:
    strategy = NoSeedScaling()
    model = helpers.ZeroModel()

    factor = strategy.compute_scale_factor(np.array([1.0, 0.5]), np.array([0.3, 0.8]), model)

    assert factor == 1.0


# ==================================================================================================
def test_barzilai_borwein_settings_rejects_gamma_min_not_less_than_gamma_max() -> None:
    with pytest.raises(ValueError, match="gamma_min"):
        BarzilaiBorweinSeedScalingSettings(gamma_min=1.0, gamma_max=1.0)


# --------------------------------------------------------------------------------------------------
def test_barzilai_borwein_settings_rejects_fallback_value_outside_the_bounds() -> None:
    with pytest.raises(ValueError, match="fallback_value"):
        BarzilaiBorweinSeedScalingSettings(gamma_min=1e-2, gamma_max=1e2, fallback_value=1e-6)
    with pytest.raises(ValueError, match="fallback_value"):
        BarzilaiBorweinSeedScalingSettings(gamma_min=1e-2, gamma_max=1e2, fallback_value=1e3)


# --------------------------------------------------------------------------------------------------
def test_barzilai_borwein_seed_scaling_returns_one_without_a_correction_pair() -> None:
    strategy = BarzilaiBorweinSeedScaling(BarzilaiBorweinSeedScalingSettings())
    model = helpers.ZeroModel()

    assert strategy.compute_scale_factor(None, None, model) == 1.0


# --------------------------------------------------------------------------------------------------
def test_barzilai_borwein_seed_scaling_returns_fallback_value_without_a_correction_pair() -> None:
    settings = BarzilaiBorweinSeedScalingSettings(gamma_min=1e-10, fallback_value=1e-6)
    strategy = BarzilaiBorweinSeedScaling(settings)
    model = helpers.ZeroModel()

    assert strategy.compute_scale_factor(None, None, model) == 1e-6


# --------------------------------------------------------------------------------------------------
def test_barzilai_borwein_seed_scaling_ignores_fallback_value_with_a_correction_pair() -> None:
    settings = BarzilaiBorweinSeedScalingSettings(gamma_min=1e-10, fallback_value=1e-6)
    strategy = BarzilaiBorweinSeedScaling(settings)
    model = helpers.ZeroModel()
    state_difference = np.array([1.0, 0.5])
    gradient_difference = np.array([0.3, 0.8])
    expected_gamma = np.dot(state_difference, gradient_difference) / np.dot(
        gradient_difference, gradient_difference
    )

    factor = strategy.compute_scale_factor(state_difference, gradient_difference, model)

    np.testing.assert_allclose(factor, expected_gamma)


# --------------------------------------------------------------------------------------------------
def test_barzilai_borwein_seed_scaling_matches_the_closed_form_ratio_within_bounds() -> None:
    strategy = BarzilaiBorweinSeedScaling(BarzilaiBorweinSeedScalingSettings())
    model = helpers.ZeroModel()
    state_difference = np.array([1.0, 0.5])
    gradient_difference = np.array([0.3, 0.8])
    expected_gamma = np.dot(state_difference, gradient_difference) / np.dot(
        gradient_difference, gradient_difference
    )

    factor = strategy.compute_scale_factor(state_difference, gradient_difference, model)

    np.testing.assert_allclose(factor, expected_gamma)


# --------------------------------------------------------------------------------------------------
def test_barzilai_borwein_seed_scaling_clamps_out_of_range_gamma() -> None:
    """A near-degenerate `gradient_difference` (small `‖y‖²` relative to `(s, y)`) drives the raw
    ratio far outside `[gamma_min, gamma_max]`; the returned factor must be the clamped value."""
    settings = BarzilaiBorweinSeedScalingSettings()
    strategy = BarzilaiBorweinSeedScaling(settings)
    model = helpers.ZeroModel()
    state_difference = np.array([1.0, 0.5])
    gradient_difference = np.array([1e-3, 5e-4])
    raw_gamma = np.dot(state_difference, gradient_difference) / np.dot(
        gradient_difference, gradient_difference
    )
    assert raw_gamma > settings.gamma_max, "test setup must actually trigger clamping"

    factor = strategy.compute_scale_factor(state_difference, gradient_difference, model)

    assert factor == settings.gamma_max
