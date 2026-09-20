import numpy as np
import pytest

from tests.optimization import helpers

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_evaluate_norm_matches_sqrt_of_inner_product_for_weighted_metric() -> None:
    weight_matrix = np.diag([1.0, 4.0, 9.0])
    model = helpers.WeightedInnerProductModel(weight_matrix)
    vector = np.array([1.0, 1.0, 1.0])
    expected_norm = np.sqrt(14.0)  # 1*1 + 4*1 + 9*1, computed independently of evaluate_norm

    norm = model.evaluate_norm(vector)

    # Pure float64 arithmetic over a handful of flops: bound by a few multiples of machine epsilon.
    np.testing.assert_allclose(norm, expected_norm, rtol=1e-14)


# --------------------------------------------------------------------------------------------------
def test_evaluate_norm_of_zero_vector_is_zero() -> None:
    weight_matrix = np.diag([1.0, 4.0, 9.0])
    model = helpers.WeightedInnerProductModel(weight_matrix)
    vector = np.zeros(3)

    norm = model.evaluate_norm(vector)

    np.testing.assert_allclose(norm, 0.0, atol=1e-14)


# --------------------------------------------------------------------------------------------------
def test_evaluate_norm_returns_python_float() -> None:
    model = helpers.WeightedInnerProductModel(np.eye(2))
    vector = np.array([1.0, 0.0])

    norm = model.evaluate_norm(vector)

    assert isinstance(norm, float)


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("scale", [2.0, -3.0, 0.5], ids=["c=2.0", "c=-3.0", "c=0.5"])
def test_evaluate_norm_scales_linearly_with_vector_magnitude(scale: float) -> None:
    weight_matrix = np.diag([1.0, 4.0, 9.0])
    model = helpers.WeightedInnerProductModel(weight_matrix)
    vector = np.array([1.0, 2.0, -1.0])

    norm_of_vector = model.evaluate_norm(vector)
    norm_of_scaled_vector = model.evaluate_norm(scale * vector)

    np.testing.assert_allclose(norm_of_scaled_vector, abs(scale) * norm_of_vector, rtol=1e-13)


# ==================================================================================================
def test_evaluate_norm_clamps_roundoff_negative_squared_norm_to_zero() -> None:
    # Ten times smaller in magnitude than the guard's round-off tolerance (1e2 * eps ~= 2.22e-14):
    # must be treated as round-off noise around zero, not a genuine sign error.
    model = helpers.ConstantInnerProductModel(inner_product_value=-1e-15)
    vector = np.array([1.0, 0.0])

    norm = model.evaluate_norm(vector)

    np.testing.assert_allclose(norm, 0.0, atol=0.0)


# --------------------------------------------------------------------------------------------------
def test_evaluate_norm_raises_for_negative_inner_product() -> None:
    # Many orders of magnitude larger than the round-off tolerance: not explainable by float64
    # round-off, so this must be treated as `evaluate_inner_product` violating positive
    # semi-definiteness rather than clamped away.
    model = helpers.ConstantInnerProductModel(inner_product_value=-1.0)
    vector = np.array([1.0, 0.0])

    with pytest.raises(ValueError, match="non-negative"):
        model.evaluate_norm(vector)
