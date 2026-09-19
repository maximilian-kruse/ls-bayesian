r"""The objective an optimizer minimizes, bundled with the geometry it is expressed in.

Classes:
    OptimizationModel: ABC interface for an objective, together with the inner-product space its
        gradient and Hessian-vector product are expressed in.
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class OptimizationModel(ABC):
    r"""ABC interface for an objective an optimizer minimizes, together with the inner-product
    space its gradient and Hessian-vector product are expressed in.

    A gradient (or Hessian-vector product) is only meaningful together with the inner product it
    is the Riesz representer under: an optimizer that evaluates bilinear forms of these
    representers (a two-loop recursion, a line search's directional derivative, cautious updating,
    ...) silently produces wrong results if `evaluate_gradient`/`evaluate_hessian_vector_product`
    and `evaluate_inner_product` disagree. Bundling all four into one interface removes the
    possibility of pairing them up inconsistently. Implementations from other subpackages are not
    coupled to this module directly; they are connected through thin adapters, mirroring
    [`ls_bayesian.posterior.interfaces`][ls_bayesian.posterior.interfaces].

    Methods:
        evaluate_cost: Evaluate the objective at a point.
        evaluate_gradient: Evaluate the gradient at a point, as the Riesz representer under
            `evaluate_inner_product`.
        evaluate_hessian_vector_product: Evaluate a Hessian-vector product at a point and
            direction, as the Riesz representer under `evaluate_inner_product`.
        evaluate_inner_product: Evaluate the inner product between two vectors.
        evaluate_norm: Evaluate the norm of a vector, induced by `evaluate_inner_product`.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        """Evaluate the objective at a point.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Point to evaluate at.

        Returns:
            float: Objective value.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Evaluate the gradient at a point, as the Riesz representer under
        `evaluate_inner_product`.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Point to evaluate at.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Gradient, same shape as the parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_hessian_vector_product(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Evaluate a Hessian-vector product at a point and direction, as the Riesz representer
        under `evaluate_inner_product`.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Point the Hessian is
                evaluated at.
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Direction.

        Raises:
            NotImplementedError: If this model does not provide Hessian-vector products.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Hessian-vector product, same shape as
                the parameter.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate_inner_product(
        self,
        first_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        second_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> float:
        """Evaluate the inner product between two vectors.

        Args:
            first_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): First vector.
            second_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Second vector.

        Returns:
            float: Inner product value.
        """

    # ----------------------------------------------------------------------------------------------
    def evaluate_norm(self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate the norm $\|v\| = \sqrt{(v, v)}$ induced by `evaluate_inner_product`.

        Concrete by default; a subclass may override this if a numerically stabler or cheaper norm
        exists for its inner product than forming the bilinear form and taking a square root.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to evaluate the norm of.

        Returns:
            float: Norm of the vector.
        """
        return float(np.sqrt(self.evaluate_inner_product(vector, vector)))
