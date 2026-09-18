"""Caching of quantities that derive from a single parameter vector.

Classes:
    EvaluationCache: Stores named quantities for the most recently evaluated parameter vector.
"""

from collections.abc import Hashable

import numpy as np


# ==================================================================================================
class EvaluationCache:
    """Stores named quantities for the most recently evaluated parameter vector.

    We typically evaluate cost, gradient and Hessian-vector products at the same parameter
    vector, and all of them need the same expensive intermediate results, most prominently the
    solution of the forward problem. The cache stores such results under a name, for exactly one
    parameter vector. Storing a quantity for a different parameter vector discards all quantities
    of the previous one. The cache does not compute anything, callers decide what to store.

    Methods:
        store_quantity: Store a quantity for a parameter vector.
        retrieve_quantity: Retrieve a quantity stored for a parameter vector.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize an empty cache."""
        self._cached_parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None
        self._cached_quantities: dict[Hashable, np.ndarray[tuple[int], np.dtype[np.float64]]] = {}

    # ----------------------------------------------------------------------------------------------
    def store_quantity(
        self,
        quantity_name: Hashable,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        quantity_value: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> None:
        """Store a quantity for a parameter vector.

        If the parameter vector differs from the cached one, all previously stored quantities are
        discarded first. The arrays are stored as given, without copying. Callers must not modify
        them in-place afterwards.

        Args:
            quantity_name (Hashable): Name under which the quantity is stored.
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter vector the
                quantity was computed for.
            quantity_value (np.ndarray[tuple[int], np.dtype[np.float64]]): Value of the quantity.
        """
        if not self._is_cached_parameter_vector(parameter_vector):
            self._cached_parameter_vector = parameter_vector
            self._cached_quantities.clear()
        self._cached_quantities[quantity_name] = quantity_value

    # ----------------------------------------------------------------------------------------------
    def retrieve_quantity(
        self,
        quantity_name: Hashable,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]] | None:
        """Retrieve a quantity stored for a parameter vector.

        Parameter vectors are compared for exact equality. Retrieval does not modify the cache.

        Args:
            quantity_name (Hashable): Name under which the quantity was stored.
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter vector the
                quantity is requested for.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]] | None: The stored array, or `None` if
                the quantity has not been stored for this parameter vector.
        """
        if not self._is_cached_parameter_vector(parameter_vector):
            return None
        return self._cached_quantities.get(quantity_name)

    # ----------------------------------------------------------------------------------------------
    def _is_cached_parameter_vector(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> bool:
        """Check whether the given parameter vector equals the cached one."""
        return self._cached_parameter_vector is not None and np.array_equal(
            parameter_vector, self._cached_parameter_vector
        )
