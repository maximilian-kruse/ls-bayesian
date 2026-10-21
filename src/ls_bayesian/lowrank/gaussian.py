r"""Gaussian measure defined as a low-rank correction of a reference covariance.

Classes:
    LowRankGaussian: Gaussian with covariance $\Gamma_0 - V D V^T$, the low-rank Laplace
        approximation of a posterior.
"""

import numpy as np

from ls_bayesian.lowrank.eigensolvers import GeneralizedEigenpairs
from ls_bayesian.lowrank.interfaces import ReferenceCovariance


# ==================================================================================================
class LowRankGaussian:
    r"""Gaussian $\mathcal{N}(\bar m, \Gamma)$ whose precision is the sum of a reference precision
    and a low-rank positive semi-definite Hessian, $\Gamma^{-1} = \Gamma_0^{-1} + H_r$.

    This is the Laplace approximation of a posterior with prior covariance $\Gamma_0$ and
    (Gauss-Newton) Hessian $H$ of the data misfit at the MAP point $\bar m$ (Bui-Thanh, Ghattas,
    Martin, Stadler, 2013; Villa, Petra, Ghattas, 2021). With the dominant eigenpairs
    $(\lambda_i, v_i)$ of $Hv = \lambda \Gamma_0^{-1}v$, normalized as $V^T\Gamma_0^{-1}V = I$,
    the Hessian is approximated by $H_r = \Gamma_0^{-1} V \Lambda V^T \Gamma_0^{-1}$, and the
    Sherman-Morrison-Woodbury identity gives

    $$
    \Gamma = \Gamma_0 - V D V^T, \qquad D = \mathrm{diag}\left(\frac{\lambda_i}{1+\lambda_i}\right).
    $$

    Samples are $\bar m + \widehat{\Gamma}\xi$ with the factorization
    $\widehat{\Gamma}\widehat{\Gamma}^T = \Gamma$,

    $$
    \widehat{\Gamma} = \left(I - V\,\mathrm{diag}\left(1 - \frac{1}{\sqrt{1+\lambda_i}}\right)
    V^T\Gamma_0^{-1}\right)\widehat{\Gamma}_0.
    $$

    Vectors are in the same (coefficient) space the Hessian and the reference covariance are
    expressed in; all vector arguments have shape `(parameter_size,)`, where `parameter_size` is
    the number of rows of the eigenvectors. The class is independent of the origin of the
    reference and the eigenpairs, see
    [`compute_generalized_eigenpairs`]
    [ls_bayesian.lowrank.eigensolvers.compute_generalized_eigenpairs]. The accuracy of the
    covariance is limited by the number of eigenpairs, and by how well the reference's covariance
    and precision actions invert each other.

    Methods:
        apply_covariance_factorization: Apply the covariance factorization $\widehat{\Gamma}$.
        apply_covariance_operator: Apply the covariance operator $\Gamma$.
        apply_precision_operator: Apply the precision operator $\Gamma^{-1}$.
        evaluate_cost: Evaluate $\frac{1}{2}(m-\bar m)^T\Gamma^{-1}(m-\bar m)$.
        generate_sample: Draw a sample from the Gaussian.
        compute_pointwise_variance: Compute the diagonal of $\Gamma$.
        compute_log_determinant_ratio: Compute $\log\det\Gamma - \log\det\Gamma_0$.

    Attributes:
        mean (np.ndarray[tuple[int], np.dtype[np.float64]]): Mean $\bar m$.
        random_vector_size (int): Size of the i.i.d. standard normal vector required by
            `apply_covariance_factorization`.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        reference: ReferenceCovariance,
        eigenpairs: GeneralizedEigenpairs,
        mean: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        seed: int = 0,
    ) -> None:
        r"""Initialize the low-rank Gaussian.

        Args:
            reference (ReferenceCovariance): Reference covariance $\Gamma_0$.
            eigenpairs (GeneralizedEigenpairs): Dominant eigenpairs of $Hv = \lambda\Gamma_0^{-1}v$,
                with non-negative eigenvalues.
            mean (np.ndarray[tuple[int], np.dtype[np.float64]] | None): Mean $\bar m$, shape
                `(parameter_size,)`. Defaults to zero.
            seed (int): Seed of the generator of `generate_sample`. Defaults to `0`.

        Raises:
            ValueError: If an eigenvalue is negative or not finite, or the shape of the mean does
                not match the eigenvectors.
        """
        eigenvalues = np.asarray(eigenpairs.eigenvalues, dtype=np.float64)
        eigenvectors = np.asarray(eigenpairs.eigenvectors, dtype=np.float64)
        parameter_size = eigenvectors.shape[0]
        if not np.all(np.isfinite(eigenvalues) & (eigenvalues >= 0.0)):
            raise ValueError(
                f"Eigenvalues must be finite and non-negative, but are {eigenvalues}. "
                "Discard non-positive eigenvalues, e.g. with `compute_generalized_eigenpairs`."
            )
        mean = np.zeros(parameter_size) if mean is None else np.asarray(mean, dtype=np.float64)
        if mean.shape != (parameter_size,):
            raise ValueError(
                f"Mean has shape {mean.shape}, but the eigenvectors have {parameter_size} rows."
            )
        self._reference = reference
        self._eigenvalues = eigenvalues
        self._eigenvectors = eigenvectors
        self._mean = mean
        self._rng = np.random.default_rng(seed)
        self._covariance_weights = eigenvalues / (1.0 + eigenvalues)
        self._factorization_weights = 1.0 - 1.0 / np.sqrt(1.0 + eigenvalues)

    # ----------------------------------------------------------------------------------------------
    @property
    def mean(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Return a copy of the mean $\bar m$."""
        return self._mean.copy()

    # ----------------------------------------------------------------------------------------------
    @property
    def random_vector_size(self) -> int:
        r"""Return the size of the i.i.d. standard normal vector required by
        `apply_covariance_factorization`."""
        return self._reference.random_vector_size

    # ----------------------------------------------------------------------------------------------
    def apply_covariance_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance $\Gamma = \Gamma_0 - VDV^T$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector, shape
                `(parameter_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, shape `(parameter_size,)`.
        """
        vector = np.asarray(vector, dtype=np.float64)
        return self._reference.apply_covariance_operator(vector) - self._eigenvectors @ (
            self._covariance_weights * (self._eigenvectors.T @ vector)
        )

    # ----------------------------------------------------------------------------------------------
    def apply_precision_operator(
        self, vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the precision $\Gamma^{-1} = \Gamma_0^{-1} + \Gamma_0^{-1}V\Lambda V^T
        \Gamma_0^{-1}$.

        Args:
            vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Vector, shape
                `(parameter_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, shape `(parameter_size,)`.
        """
        precision_applied = self._reference.apply_precision_operator(
            np.asarray(vector, dtype=np.float64)
        )
        correction = self._eigenvectors @ (
            self._eigenvalues * (self._eigenvectors.T @ precision_applied)
        )
        return precision_applied + self._reference.apply_precision_operator(correction)

    # ----------------------------------------------------------------------------------------------
    def apply_covariance_factorization(
        self, random_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance factorization $\widehat{\Gamma}$, $\widehat{\Gamma}
        \widehat{\Gamma}^T = \Gamma$.

        Args:
            random_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): I.i.d. standard normal
                vector, shape `(random_vector_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result, shape `(parameter_size,)`.
        """
        reference_applied = self._reference.apply_covariance_factorization(
            np.asarray(random_vector, dtype=np.float64)
        )
        projection = self._eigenvectors.T @ self._reference.apply_precision_operator(
            reference_applied
        )
        return reference_applied - self._eigenvectors @ (self._factorization_weights * projection)

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(self, state: np.ndarray[tuple[int], np.dtype[np.float64]]) -> float:
        r"""Evaluate the potential $\frac{1}{2}(m-\bar m)^T\Gamma^{-1}(m-\bar m)$, i.e. the negative
        log-density up to an additive constant.

        With $d = m-\bar m$ the quadratic form is evaluated as $\frac{1}{2}(d^T\Gamma_0^{-1}d +
        \sum_i\lambda_i (v_i^T\Gamma_0^{-1}d)^2)$, which needs a single application of
        $\Gamma_0^{-1}$.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): State $m$, shape
                `(parameter_size,)`.

        Returns:
            float: Potential value.
        """
        difference = np.asarray(state, dtype=np.float64) - self._mean
        precision_applied = self._reference.apply_precision_operator(difference)
        projection = self._eigenvectors.T @ precision_applied
        return float(
            0.5 * (difference @ precision_applied + np.sum(self._eigenvalues * projection**2))
        )

    # ----------------------------------------------------------------------------------------------
    def generate_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Draw a sample $\bar m + \widehat{\Gamma}\xi$. Successive calls draw different samples,
        reproducible for a given seed.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Sample, shape `(parameter_size,)`.
        """
        random_vector = self._rng.standard_normal(self.random_vector_size)
        return self._mean + self.apply_covariance_factorization(random_vector)

    # ----------------------------------------------------------------------------------------------
    def compute_pointwise_variance(
        self,
        prior_variance: np.ndarray[tuple[int], np.dtype[np.float64]] | None = None,
        num_probes: int = 128,
        seed: int = 0,
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Compute the diagonal of $\Gamma$, $\mathrm{diag}(\Gamma_0) - \mathrm{diag}(VDV^T)$.

        The diagonal of the low-rank correction is exact. If the reference variance is not given,
        $\mathrm{diag}(\Gamma_0)$ is estimated with Rademacher probing vectors $z_j$ (Bekas,
        Kokiopoulou, Saad, 2007), $\mathrm{diag}(\Gamma_0) \approx \frac{1}{N}\sum_j z_j \circ
        \Gamma_0 z_j$. The estimate is unbiased with a standard error that decays as $N^{-1/2}$,
        and is not guaranteed to be positive, so prefer passing the exact diagonal where it is
        available. A truncated spectral representation of $\Gamma_0$ is not a substitute, since it
        underestimates the variance by the discarded tail.

        Args:
            prior_variance (np.ndarray[tuple[int], np.dtype[np.float64]] | None): Exact diagonal
                of the reference covariance $\Gamma_0$, shape `(parameter_size,)`. Estimated if
                `None`. Defaults to `None`.
            num_probes (int): Number $N$ of probing vectors for the estimate. Defaults to `128`.
            seed (int): Seed of the probing vectors. Defaults to `0`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Pointwise variance, shape
                `(parameter_size,)`.

        Raises:
            ValueError: If `prior_variance` has the wrong shape or `num_probes` is not positive.
        """
        parameter_size = self._eigenvectors.shape[0]
        if prior_variance is None:
            if num_probes <= 0:
                raise ValueError(f"Number of probes must be positive, but is {num_probes}.")
            rng = np.random.default_rng(seed)
            probes = rng.choice([-1.0, 1.0], size=(num_probes, parameter_size))
            prior_variance = np.mean(
                [probe * self._reference.apply_covariance_operator(probe) for probe in probes],
                axis=0,
            )
        else:
            prior_variance = np.asarray(prior_variance, dtype=np.float64)
            if prior_variance.shape != (parameter_size,):
                raise ValueError(
                    f"Prior variance has shape {prior_variance.shape}, "
                    f"but the parameter size is {parameter_size}."
                )
        return prior_variance - self._eigenvectors**2 @ self._covariance_weights

    # ----------------------------------------------------------------------------------------------
    def compute_log_determinant_ratio(self) -> float:
        r"""Compute $\log\det\Gamma - \log\det\Gamma_0 = -\sum_i\log(1+\lambda_i)$.

        The determinant of the reference itself is not accessible through
        [`ReferenceCovariance`][ls_bayesian.lowrank.interfaces.ReferenceCovariance], hence only
        the ratio is provided. It is the quantity needed for e.g. the Kullback-Leibler divergence
        to the reference, or the Occam factor of the Laplace evidence.

        Returns:
            float: Log-determinant ratio, non-positive.
        """
        return float(-np.sum(np.log1p(self._eigenvalues)))
