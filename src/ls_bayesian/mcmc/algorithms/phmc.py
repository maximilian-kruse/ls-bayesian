r"""Preconditioned Hamiltonian Monte Carlo on function space with a fixed Gaussian preconditioner.

Classes:
    PHMCTrajectory: End point and energy error of one leapfrog trajectory.
    PHMCAlgorithm: Function-space HMC with a fixed Gaussian $\overline K \neq C$.
"""

from dataclasses import dataclass
from numbers import Real
from typing import Annotated, override

import numpy as np
from beartype.vale import Is

from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.measures import DifferentiableTargetMeasure
from ls_bayesian.mcmc.model import MCMCModel

_VectorType = np.ndarray[tuple[int], np.dtype[np.float64]]


# ==================================================================================================
@dataclass(frozen=True)
class _PHMCStateCache:
    r"""Quantities at a position, so a trajectory evaluates each state once.

    Attributes:
        potential (float): $\Phi(u)$.
        drift (np.ndarray): $g(u) = u - \overline K s(u)$ with $s(u) = \nabla\Phi(u) + C^{-1}u$.
        reference_precision_drift (np.ndarray): $C^{-1}g(u)$.
        drift_norm_squared (float): $\langle g(u), C^{-1}g(u)\rangle$.
    """

    potential: float
    drift: _VectorType
    reference_precision_drift: _VectorType
    drift_norm_squared: float


# ==================================================================================================
@dataclass(frozen=True)
class PHMCTrajectory:
    r"""A leapfrog trajectory.

    Attributes:
        position (np.ndarray): The position $u_I$ at the end.
        velocity (np.ndarray): The velocity $v_I$ at the end.
        energy_change (float): $\Delta H(u_0, v_0)$ of the whole trajectory, in the telescoped form
            of Beskos et al. (2017), Theorem 3.10 (ii).
        energy_changes (np.ndarray | None): $\Delta H$ of the trajectory cut after each step, shape
            `(num_steps,)`, if the path was recorded, else `None`.
        positions (np.ndarray | None): The positions after each step, shape `(num_steps, size)`,
            if the path was recorded, else `None`.
    """

    position: _VectorType
    velocity: _VectorType
    energy_change: float
    energy_changes: np.ndarray | None
    positions: np.ndarray | None = None


# ==================================================================================================
class PHMCAlgorithm(MCMCAlgorithm):
    r"""Hamiltonian Monte Carlo on function space, preconditioned by a fixed Gaussian $\overline K$.

    Implements the state-independent case of the $\infty$-mHMC of Beskos, Girolami, Lan, Farrell,
    Stuart (2017), Sec. 3.3: $K(u) \equiv \overline K$, as
    [`PMALAAlgorithm`][ls_bayesian.mcmc.algorithms.pmala.PMALAAlgorithm] is their $\infty$-mMALA
    with a fixed $\overline K$. For $\overline K = C$ it is the $\infty$-HMC of Beskos, Pinski,
    Sanz-Serna, Stuart (2011). The potential $\Phi$ stays relative to the centered reference
    $\mu_0 = \mathcal N(0, C)$; the fixed $\overline K$ makes the determinant terms of the energy
    state-independent, so they cancel.

    The dynamics are $u'' + \overline K\{C^{-1}u + \nabla\Phi(u)\} = 0$ with the velocity drawn from
    $\mathcal N(0, \overline K)$. With $g(u) = -\overline K\{(C^{-1} - \overline K^{-1})u +
    \nabla\Phi(u)\} = u - \overline K s(u)$ one step of angle $\varepsilon$ (eq. (30) there) is

    $$
    v^- = v + \tfrac{\varepsilon}{2}g(u), \quad
    \begin{pmatrix}u_\varepsilon \\ v^+\end{pmatrix} =
    \begin{pmatrix}\cos\varepsilon & \sin\varepsilon \\ -\sin\varepsilon & \cos\varepsilon
    \end{pmatrix}\begin{pmatrix}u \\ v^-\end{pmatrix}, \quad
    v_\varepsilon = v^+ + \tfrac{\varepsilon}{2}g(u_\varepsilon),
    $$

    and the proposal is the position after `num_steps` such steps, accepted with probability $1
    \wedge \exp(-\Delta H)$, where $\Delta H$ is the energy error in the form of their Theorem 3.10
    (ii), which only involves finite quantities in infinite dimensions:

    $$
    \Delta H = \Phi(u_I) - \Phi(u_0) + \tfrac12\langle v_I, (\overline K^{-1} - C^{-1})v_I\rangle
        - \tfrac12\langle v_0, (\overline K^{-1} - C^{-1})v_0\rangle
        - \tfrac{\varepsilon^2}{8}\big(|g(u_I)|^2_{C^{-1}} - |g(u_0)|^2_{C^{-1}}\big)
        + \tfrac{\varepsilon}{2}\sum_{i=0}^{I-1}\big(\langle g(u_i), C^{-1}v_i\rangle
        + \langle g(u_{i+1}), C^{-1}v_{i+1}\rangle\big).
    $$

    The step angle can be jittered: for each proposal $\varepsilon$ is drawn uniformly from
    $[(1 - j)\varepsilon_0, (1 + j)\varepsilon_0]$, independently of the state, which avoids
    resonances of the trajectory length with the period $2\pi$ of the rotation. The forward and
    the reverse trajectory use the same angle, so the acceptance probability is unchanged.

    The prior mean is not read: as for PMALA, the target is stated relative to the centered
    reference. A trajectory takes `num_steps` evaluations of the gradient (the current state is
    cached from the previous step). Inner products are the ordinary Euclidean dot product of the
    coefficient vectors, as in
    [`MALAAlgorithm`][ls_bayesian.mcmc.algorithms.mala.MALAAlgorithm].

    Methods:
        compute_step: Compute one step of MCMC.
        integrate: The leapfrog trajectory from a position and a velocity.

    Attributes:
        num_gradient_evaluations (int): Number of evaluations of the potential and its gradient.
        num_divergences (int): Number of proposals with a non-finite energy error, which are
            rejected.

    References:
        Beskos, Pinski, Sanz-Serna, Stuart (2011). *Hybrid Monte Carlo on Hilbert spaces.*
        Stochastic Processes and their Applications 121(10).
        Beskos, Girolami, Lan, Farrell, Stuart (2017). *Geometric MCMC for Infinite-Dimensional
        Inverse Problems.* Journal of Computational Physics 335, 327-351.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        model: MCMCModel,
        step_width: Annotated[Real, Is[lambda x: x > 0]],
        num_steps: Annotated[int, Is[lambda x: x > 0]],
        jitter: Annotated[Real, Is[lambda x: 0 <= x < 1]] = 0.0,
    ) -> None:
        r"""Initialize the preconditioned HMC algorithm.

        Args:
            model (MCMCModel): Target (must be a `DifferentiableTargetMeasure`), the reference
                measure $\mu_0 = \mathcal N(0, C)$ and the fixed Gaussian $\overline K$ in
                `approximation`, all on the same coefficient space; only $\overline K$'s
                covariance, precision and factorization are used, `mean` is ignored.
            step_width (Real): Step angle $\varepsilon > 0$ of one leapfrog step; the rotation of
                the Gaussian part is exact, so for a Gaussian target the error comes from
                $\overline K \neq$ the target covariance only.
            num_steps (int): Number of leapfrog steps $I \geq 1$ per proposal.
            jitter (Real, optional): Relative half width $j \in [0, 1)$ of the uniform distribution
                of the step angle per proposal. Defaults to `0.0`, no jitter.

        Raises:
            ValueError: If `model.approximation` is `None`, `model.target` is not a
                `DifferentiableTargetMeasure`, or the largest step angle is not below $\pi$.
        """
        if model.approximation is None:
            raise ValueError("PHMCAlgorithm requires model.approximation to be given.")
        if not isinstance(model.target, DifferentiableTargetMeasure):
            raise ValueError(
                "PHMCAlgorithm requires model.target to be a DifferentiableTargetMeasure."
            )
        if (1 + jitter) * step_width >= np.pi:
            raise ValueError(
                f"The largest step angle (1 + jitter) * step_width = {(1 + jitter) * step_width} "
                "must be below pi."
            )
        super().__init__(step_width)
        self._target_model = model.target
        self._reference_measure = model.reference
        self._preconditioner = model.approximation
        self._num_steps = num_steps
        self._jitter = float(jitter)
        self._current_cache: _PHMCStateCache | None = None
        self._pending: tuple[float, _PHMCStateCache] | None = None
        self.num_gradient_evaluations = 0
        self.num_divergences = 0

    # ----------------------------------------------------------------------------------------------
    def integrate(
        self,
        position: _VectorType,
        velocity: _VectorType,
        step_angle: float,
        num_steps: int,
        record_path: bool = False,
    ) -> PHMCTrajectory:
        r"""Integrate the leapfrog trajectory of `num_steps` steps of angle `step_angle`.

        Args:
            position (np.ndarray): Start position $u_0$.
            velocity (np.ndarray): Start velocity $v_0$.
            step_angle (float): Step angle $\varepsilon$.
            num_steps (int): Number of steps $I$.
            record_path (bool, optional): Whether to evaluate $\Delta H$ after every step too, at
                the cost of two precision applications per step, and to keep the positions.
                Defaults to `False`.

        Returns:
            PHMCTrajectory: The end point and the energy error.
        """
        cache = self._evaluate_state(position)
        return self._integrate_trajectory(
            position, cache, velocity, step_angle, num_steps, record_path
        )[0]

    # ----------------------------------------------------------------------------------------------
    @override
    def _create_proposal(
        self,
        state: _VectorType,
        rng: np.random.Generator,
    ) -> _VectorType:
        if self._current_cache is None:
            self._current_cache = self._evaluate_state(state)
        step_angle = float(self._step_width) * (1 + self._jitter * (2 * rng.uniform() - 1))
        velocity = self._preconditioner.apply_covariance_factorization(
            rng.normal(size=self._preconditioner.random_vector_size)
        )
        trajectory, end_cache = self._integrate_trajectory(
            state, self._current_cache, velocity, step_angle, self._num_steps, False
        )
        self._pending = (trajectory.energy_change, end_cache)
        return trajectory.position

    # ----------------------------------------------------------------------------------------------
    @override
    def _evaluate_acceptance_probability(
        self,
        current_state: _VectorType,
        proposal: _VectorType,
    ) -> float:
        if self._pending is None:
            raise RuntimeError("The acceptance probability needs the proposal of the same step.")
        energy_change = self._pending[0]
        if not np.isfinite(energy_change):
            self.num_divergences += 1
            return 0.0
        return float(min(1.0, np.exp(-energy_change)))

    # ----------------------------------------------------------------------------------------------
    @override
    def _update_cache(self, accepted: bool) -> None:
        if accepted and self._pending is not None:
            self._current_cache = self._pending[1]
        self._pending = None

    # ----------------------------------------------------------------------------------------------
    def _evaluate_state(self, state: _VectorType) -> _PHMCStateCache:
        r"""Evaluate $\Phi(u)$, $g(u) = u - \overline K s(u)$ and $C^{-1}g(u)$ at `state`."""
        self.num_gradient_evaluations += 1
        gradient = self._target_model.evaluate_gradient(state)
        score = gradient + self._reference_measure.apply_precision_operator(state)
        drift = state - self._preconditioner.apply_covariance_operator(score)
        reference_precision_drift = self._reference_measure.apply_precision_operator(drift)
        return _PHMCStateCache(
            potential=self._target_model.evaluate_potential(state),
            drift=drift,
            reference_precision_drift=reference_precision_drift,
            drift_norm_squared=float(np.dot(drift, reference_precision_drift)),
        )

    # ----------------------------------------------------------------------------------------------
    def _velocity_form(self, velocity: _VectorType) -> float:
        r"""$\langle v, (\overline K^{-1} - C^{-1})v\rangle$."""
        return float(
            np.dot(velocity, self._preconditioner.apply_precision_operator(velocity))
            - np.dot(velocity, self._reference_measure.apply_precision_operator(velocity))
        )

    # ----------------------------------------------------------------------------------------------
    def _integrate_trajectory(
        self,
        position: _VectorType,
        cache: _PHMCStateCache,
        velocity: _VectorType,
        step_angle: float,
        num_steps: int,
        record_path: bool,
    ) -> tuple[PHMCTrajectory, _PHMCStateCache]:
        """The trajectory and the cache of its end point."""
        cosine, sine = np.cos(step_angle), np.sin(step_angle)
        form_start = self._velocity_form(velocity)
        coupling = float(np.dot(cache.reference_precision_drift, velocity))
        accumulated = 0.0  # eps/2 sum (<g_i, C^-1 v_i> + <g_{i+1}, C^-1 v_{i+1}>)
        start_potential = cache.potential
        start_norm = cache.drift_norm_squared
        energy_changes: list[float] = []
        positions: list[_VectorType] = []
        current_position, current_velocity, current_cache = position, velocity, cache
        for _ in range(num_steps):
            kicked = current_velocity + 0.5 * step_angle * current_cache.drift
            next_position = cosine * current_position + sine * kicked
            rotated = cosine * kicked - sine * current_position
            next_cache = self._evaluate_state(next_position)
            next_velocity = rotated + 0.5 * step_angle * next_cache.drift
            next_coupling = float(np.dot(next_cache.reference_precision_drift, next_velocity))
            accumulated += 0.5 * step_angle * (coupling + next_coupling)
            coupling = next_coupling
            current_position, current_velocity, current_cache = (
                next_position,
                next_velocity,
                next_cache,
            )
            if record_path:
                positions.append(current_position)
                energy_changes.append(
                    self._energy_change(
                        start_potential,
                        start_norm,
                        form_start,
                        current_cache,
                        current_velocity,
                        step_angle,
                        accumulated,
                    )
                )
        energy_change = (
            energy_changes[-1]
            if record_path
            else self._energy_change(
                start_potential,
                start_norm,
                form_start,
                current_cache,
                current_velocity,
                step_angle,
                accumulated,
            )
        )
        trajectory = PHMCTrajectory(
            position=current_position,
            velocity=current_velocity,
            energy_change=energy_change,
            energy_changes=np.array(energy_changes) if record_path else None,
            positions=np.array(positions) if record_path else None,
        )
        return trajectory, current_cache

    # ----------------------------------------------------------------------------------------------
    def _energy_change(
        self,
        start_potential: float,
        start_norm: float,
        form_start: float,
        end_cache: _PHMCStateCache,
        end_velocity: _VectorType,
        step_angle: float,
        accumulated: float,
    ) -> float:
        r"""$\Delta H$ of the trajectory that ends at `end_cache`, see the class docstring."""
        return float(
            end_cache.potential
            - start_potential
            + 0.5 * (self._velocity_form(end_velocity) - form_start)
            - step_angle**2 / 8 * (end_cache.drift_norm_squared - start_norm)
            + accumulated
        )
