r"""ABC interface of the reference covariance that low-rank Gaussian approximations build upon.

Classes:
    ReferenceCovariance: ABC interface for the covariance $\Gamma_0$ of a reference Gaussian, e.g.
        a prior, which is corrected by a low-rank update.
"""

from abc import ABC, abstractmethod

import numpy as np


# ==================================================================================================
class ReferenceCovariance(ABC):
    r"""ABC interface for the covariance $\Gamma_0$ of the Gaussian that a
    [`LowRankGaussian`][ls_bayesian.lowrank.gaussian.LowRankGaussian] corrects.

    The low-rank formulas (Villa, Petra, Ghattas, 2021; Saibaba, Lee, Kitanidis, 2016) need
    $\Gamma_0$ and $\Gamma_0^{-1}$ to be exact inverses of each other and symmetric positive
    definite, as matrices acting on the same (coefficient) vector space the Hessian is expressed in.
    Iterative approximations of the actions are fine as long as their error is small compared to
    the tolerance of the eigensolver. The interface carries no mean: the mean of a low-rank
    Gaussian is passed separately, as it is generally not the mean of the reference (e.g. the MAP
    point of a Laplace approximation).

    Methods:
        apply_covariance_factorization: Apply the covariance factorization $\widehat{\Gamma}_0$.
        apply_covariance_operator: Apply the covariance operator $\Gamma_0$.
        apply_precision_operator: Apply the precision operator $\Gamma_0^{-1}$.

    Attributes:
        random_vector_size (int): Size of the i.i.d. standard normal vector required by
            `apply_covariance_factorization`.
    """

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def random_vector_size(self) -> int:
        r"""Return the size of the i.i.d. standard normal vector required by
        `apply_covariance_factorization`."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance factorization $\widehat{\Gamma}_0$,
        $\widehat{\Gamma}_0\widehat{\Gamma}_0^T = \Gamma_0$. It does not have to be symmetric or
        square.

        Args:
            random_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): I.i.d. standard normal
                vector, shape `(random_vector_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, shape `(parameter_size,)`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_covariance_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance operator $\Gamma_0$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the covariance
                operator to, shape `(parameter_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, shape `(parameter_size,)`.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the precision operator $\Gamma_0^{-1}$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector to apply the precision
                operator to, shape `(parameter_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, shape `(parameter_size,)`.
        """
