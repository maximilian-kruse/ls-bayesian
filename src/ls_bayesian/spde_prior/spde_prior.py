"""Gaussian prior measure based on an SPDE representation.

Classes:
    SPDEPrior: SPDE-based prior distribution class.
"""

import numpy as np

from ls_bayesian.spde_prior import components, fem


# ==================================================================================================
class SPDEPrior:
    r"""SPDE-based prior distribution class.

    This class implements functionality of a Gaussian prior measure that is typically required in
    the context of Bayesian inference. Its covariance structure arises as the solution operator of
    an SPDE. In the spirit of separation of concerns, the prior component is rather stupid. It
    simply utilizes objects for the (representation of) a covariance operator $\mathcal{C}$, its
    factorization $\widehat{\mathcal{C}}$, and a precision operator $\mathcal{C}^{-1}$. These
    components have to adhere to the interface prescribed by the
    [`InterfaceComponent`][ls_bayesian.spde_prior.components.InterfaceComponent] class. The
    necessary objects can be manually assembled and be handed to the `SPDEPrior` class for maximum
    flexibility. On the other hand, the [`builder`][ls_bayesian.spde_prior.builder] module provides
    a convenient alternative for the setup of a preconfigured prior object.

    All public vectors are given on the mesh vertices. Internally, they are converted to the DoFs
    of the underlying function space via a
    [`FEMConverter`][ls_bayesian.spde_prior.fem.FEMConverter].

    Methods:
        evaluate_cost: Evaluate the cost/negative log-probability for a given parameter vector.
        evaluate_gradient: Evaluate the gradient of the cost functional with respect to
            a given parameter vector.
        evaluate_hessian_vector_product: Evaluate the Hessian-vector product of the cost functional
            in the given direction.
        generate_sample: Generate a sample from the prior distribution.
        apply_covariance_operator: Apply the covariance operator $\mathcal{C}$.
        apply_covariance_factorization: Apply the covariance factorization $\widehat{\mathcal{C}}$.
        apply_precision_operator: Apply the precision operator $\mathcal{C}^{-1}$.

    Attributes:
        random_vector_size: Size of the i.i.d. normal vector required for sampling.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        mean_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
        precision_operator: components.InterfaceComponent,
        covariance_operator: components.InterfaceComponent,
        covariance_factorization: components.InterfaceComponent,
        fem_converter: fem.FEMConverter,
        seed: int,
    ) -> None:
        r"""Initialize SPDEPrior distribution object.

        Args:
            mean_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Mean vector $\overline{m}$
                of the prior measure, given on mesh vertices.
            precision_operator (components.InterfaceComponent): Representation of the precision
                operator $\mathcal{C}^{-1}$.
            covariance_operator (components.InterfaceComponent): Representation of the covariance
                operator $\mathcal{C}$.
            covariance_factorization (components.InterfaceComponent): Representation of a
                factorization $\widehat{\mathcal{C}}$ of the covariance operator for sampling.
            fem_converter (fem.FEMConverter): Converter object to switch between vertex-
                and DoF-based representation of vectors.
            seed (int): Random seed for the internal random number generator.

        Raises:
            ValueError: Checks that the mean vector is given on the mesh vertices.
            ValueError: Checks that precision operator has the correct shape.
            ValueError: Checks that covariance operator has the correct shape.
            ValueError: Checks that covariance factor has the correct shape.
        """
        self._fem_converter = fem_converter
        self._mean_vector = self._fem_converter.convert_vertex_values_to_dofs(mean_vector)
        mean_vector_dim = self._fem_converter.global_dof_space_dim
        if not precision_operator.shape == (mean_vector_dim, mean_vector_dim):
            raise ValueError(
                f"Precision operator shape {precision_operator.shape} does not match "
                f"the mean vector dimension {mean_vector_dim}."
            )
        if not covariance_operator.shape == (mean_vector_dim, mean_vector_dim):
            raise ValueError(
                f"Covariance operator shape {covariance_operator.shape} does not match "
                f"the mean vector dimension {mean_vector_dim}."
            )
        if not covariance_factorization.shape[0] == mean_vector_dim:
            raise ValueError(
                f"Covariance factorization output dimension {covariance_factorization.shape[0]} "
                f"does not match the mean vector dimension {mean_vector_dim}."
            )
        self._precision_operator = precision_operator
        self._covariance_operator = covariance_operator
        self._covariance_factorization = covariance_factorization
        self._prng = np.random.default_rng(seed)

    # ----------------------------------------------------------------------------------------------
    @property
    def random_vector_size(self) -> int:
        """Return the required size of the random vector for sampling."""
        return self._covariance_factorization.shape[1]

    # ----------------------------------------------------------------------------------------------
    def evaluate_cost(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> float:
        r"""Evaluate the cost functional, i.e. the negative log probability for given input.

        Computes $\frac{1}{2} (m-\overline{m})^T \mathcal{C}^{-1} (m-\overline{m})$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter candidate for
                which to evaluate the cost/negative log probability, given on mesh vertices.

        Returns:
            float: Cost/negative log probability, up to an additive constant.
        """
        parameter_vector_dof = self._fem_converter.convert_vertex_values_to_dofs(parameter_vector)
        difference_vector = parameter_vector_dof - self._mean_vector
        local_cost = np.inner(difference_vector, self._precision_operator.apply(difference_vector))
        cost = 0.5 * self._fem_converter.comm.allreduce(local_cost)
        assert cost >= 0, f"Cost needs to be non-negative, but is {cost}."
        return float(cost)

    # ----------------------------------------------------------------------------------------------
    def evaluate_gradient(
        self, parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]]
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the gradient of the cost/negative log-probability w.r.t. the parameter vector.

        Computes $\mathcal{C}^{-1} (m - \overline{m})$.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter candidate for
                which to evaluate the gradient, given on mesh vertices.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Gradient of the cost/negative
                log-probability, given on mesh vertices.
        """
        parameter_vector_dof = self._fem_converter.convert_vertex_values_to_dofs(parameter_vector)
        difference_vector = parameter_vector_dof - self._mean_vector
        gradient_dof = self._precision_operator.apply(difference_vector)
        gradient = self._fem_converter.convert_dofs_to_vertex_values(gradient_dof)
        return gradient

    # ----------------------------------------------------------------------------------------------
    def evaluate_hessian_vector_product(
        self,
        direction_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Evaluate the application of the Hessian of the cost functional on a given direction.

        Computes $\mathcal{C}^{-1} m_{\text{dir}}$.

        Args:
            direction_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Direction for which
                to evaluate the Hessian-vector product, given on mesh vertices.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Hessian-vector product,
                given on mesh vertices.
        """
        hessian_vector_product = self.apply_precision_operator(direction_vector)
        return hessian_vector_product

    # ----------------------------------------------------------------------------------------------
    def generate_sample(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Generate a sample from the prior measure.

        Computes sample in two-step procedure:

        1. Generate i.i.d. normal vector $\xi$ matching input dimension of $\widehat{\mathcal{C}}$.
        2. Multiply with covariance factorization $\widehat{\mathcal{C}}$ and add mean.

        Samples are reproducible for a given seed, but successive calls draw different samples.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Sample vector, given on mesh vertices.
        """
        random_vector = self._prng.normal(loc=0.0, scale=1.0, size=self.random_vector_size)
        sample_vector_dof = self._covariance_factorization.apply(random_vector)
        sample_vector_dof += self._mean_vector
        sample_vector = self._fem_converter.convert_dofs_to_vertex_values(sample_vector_dof)
        return sample_vector

    # ----------------------------------------------------------------------------------------------
    def apply_covariance_operator(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Apply the covariance operator to a given parameter vector.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter candidate for
                which to apply the covariance operator, given on mesh vertices.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result of applying the covariance
                operator, given on mesh vertices.
        """
        parameter_vector_dof = self._fem_converter.convert_vertex_values_to_dofs(parameter_vector)
        covariance_applied_dof = self._covariance_operator.apply(parameter_vector_dof)
        covariance_applied = self._fem_converter.convert_dofs_to_vertex_values(
            covariance_applied_dof
        )
        return covariance_applied

    # ----------------------------------------------------------------------------------------------
    def apply_covariance_factorization(
        self,
        random_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        r"""Apply the covariance factorization $\widehat{\mathcal{C}}$ to a given random vector.

        Args:
            random_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Random vector for
                which to apply the covariance factorization, shape `(random_vector_size,)`.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result of applying the covariance
                factorization, given on mesh vertices.

        Raises:
            ValueError: Checks that the random vector has the correct shape.
        """
        if random_vector.shape != (self.random_vector_size,):
            raise ValueError(
                f"Random vector shape {random_vector.shape} does not match "
                "the covariance factorization input dimension "
                f"{self.random_vector_size}."
            )
        covariance_factorization_applied_dof = self._covariance_factorization.apply(random_vector)
        covariance_factorization_applied = self._fem_converter.convert_dofs_to_vertex_values(
            covariance_factorization_applied_dof
        )
        return covariance_factorization_applied

    # ----------------------------------------------------------------------------------------------
    def apply_precision_operator(
        self,
        parameter_vector: np.ndarray[tuple[int], np.dtype[np.float64]],
    ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Apply the precision operator to a given parameter vector.

        Args:
            parameter_vector (np.ndarray[tuple[int], np.dtype[np.float64]]): Parameter candidate for
                which to apply the precision operator, given on mesh vertices.

        Returns:
            np.ndarray[tuple[int], np.dtype[np.float64]]: Result of applying the precision
                operator, given on mesh vertices.
        """
        parameter_vector_dof = self._fem_converter.convert_vertex_values_to_dofs(parameter_vector)
        precision_applied_dof = self._precision_operator.apply(parameter_vector_dof)
        precision_applied = self._fem_converter.convert_dofs_to_vertex_values(precision_applied_dof)
        return precision_applied
