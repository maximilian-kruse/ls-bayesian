import numpy as np
import pytest

from ls_bayesian.posterior import cache

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_retrieve_quantity_returns_none_for_empty_cache() -> None:
    evaluation_cache = cache.EvaluationCache()

    assert evaluation_cache.retrieve_quantity("solution", np.array([1.0, 2.0])) is None


# --------------------------------------------------------------------------------------------------
def test_retrieve_quantity_returns_stored_value() -> None:
    evaluation_cache = cache.EvaluationCache()
    parameter_vector = np.array([1.0, 2.0])
    quantity_value = np.array([3.0, 4.0, 5.0])

    evaluation_cache.store_quantity("solution", parameter_vector, quantity_value)

    np.testing.assert_allclose(
        evaluation_cache.retrieve_quantity("solution", parameter_vector.copy()), quantity_value
    )


# --------------------------------------------------------------------------------------------------
def test_retrieve_quantity_returns_none_for_unknown_name() -> None:
    evaluation_cache = cache.EvaluationCache()
    parameter_vector = np.array([1.0])
    evaluation_cache.store_quantity("solution", parameter_vector, np.array([1.0]))

    assert evaluation_cache.retrieve_quantity("gradient", parameter_vector) is None


# --------------------------------------------------------------------------------------------------
def test_quantities_are_stored_independently() -> None:
    evaluation_cache = cache.EvaluationCache()
    parameter_vector = np.array([1.0])

    evaluation_cache.store_quantity("solution", parameter_vector, np.array([1.0]))
    evaluation_cache.store_quantity("gradient", parameter_vector, np.array([2.0]))

    np.testing.assert_allclose(
        evaluation_cache.retrieve_quantity("solution", parameter_vector), 1.0
    )
    np.testing.assert_allclose(
        evaluation_cache.retrieve_quantity("gradient", parameter_vector), 2.0
    )


# --------------------------------------------------------------------------------------------------
def test_retrieve_quantity_for_other_parameter_keeps_cache() -> None:
    evaluation_cache = cache.EvaluationCache()
    parameter_vector = np.array([1.0])
    evaluation_cache.store_quantity("solution", parameter_vector, np.array([3.0]))

    assert evaluation_cache.retrieve_quantity("solution", np.array([2.0])) is None
    np.testing.assert_allclose(
        evaluation_cache.retrieve_quantity("solution", parameter_vector), 3.0
    )


# --------------------------------------------------------------------------------------------------
def test_store_quantity_for_new_parameter_discards_previous_quantities() -> None:
    evaluation_cache = cache.EvaluationCache()
    first_parameter_vector = np.array([1.0])
    second_parameter_vector = np.array([2.0])
    evaluation_cache.store_quantity("solution", first_parameter_vector, np.array([1.0]))
    evaluation_cache.store_quantity("gradient", first_parameter_vector, np.array([1.0]))

    evaluation_cache.store_quantity("solution", second_parameter_vector, np.array([2.0]))

    assert evaluation_cache.retrieve_quantity("solution", first_parameter_vector) is None
    assert evaluation_cache.retrieve_quantity("gradient", first_parameter_vector) is None
    assert evaluation_cache.retrieve_quantity("gradient", second_parameter_vector) is None


# --------------------------------------------------------------------------------------------------
def test_parameter_vectors_are_compared_exactly() -> None:
    """A tolerance would alias finite-difference perturbations with the unperturbed parameter."""
    evaluation_cache = cache.EvaluationCache()
    parameter_vector = np.array([1.0, 2.0])
    evaluation_cache.store_quantity("solution", parameter_vector, np.array([1.0]))

    perturbed_parameter_vector = parameter_vector + np.array([1e-12, 0.0])

    assert evaluation_cache.retrieve_quantity("solution", perturbed_parameter_vector) is None


# --------------------------------------------------------------------------------------------------
def test_parameter_vectors_of_different_shape_do_not_match() -> None:
    evaluation_cache = cache.EvaluationCache()
    evaluation_cache.store_quantity("solution", np.array([1.0, 1.0]), np.array([1.0]))

    assert evaluation_cache.retrieve_quantity("solution", np.array([1.0, 1.0, 1.0])) is None


# --------------------------------------------------------------------------------------------------
def test_quantities_are_stored_without_copying() -> None:
    evaluation_cache = cache.EvaluationCache()
    parameter_vector = np.array([1.0])
    quantity_value = np.array([3.0, 4.0])

    evaluation_cache.store_quantity("solution", parameter_vector, quantity_value)

    assert evaluation_cache.retrieve_quantity("solution", parameter_vector) is quantity_value
