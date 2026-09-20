"""MCMC output tracking: quantities of interest and statistics computed from them.

An output generally consists of two components: a Quantity of Interest (QoI), computed solely
from the chain state, and a statistic, derived from a sequence of QoI values. For example, to
track the acceptance rate, the QoI is "was the sample accepted?" and the statistic is the running
mean over all samples so far.

Classes:
    MCMCQoI: ABC interface for a Quantity of Interest.
    MCMCStatistic: ABC interface for a statistic.
    MCMCOutput: Output that combines a QoI and a statistic.
    ComponentQoI: A specific component of the state as a QoI.
    MeanQoI: The mean of the state as a QoI.
    AcceptanceQoI: Whether the proposal was accepted, as a QoI.
    IdentityStatistic: The QoI value itself, unchanged.
    RunningMeanStatistic: A running mean of QoI values.
    BatchMeanStatistic: A batch mean of QoI values.

Functions:
    build: Build a logged `MCMCOutput` from a QoI and a statistic, deriving its column label and
        format automatically.
"""

from abc import ABC, abstractmethod
from numbers import Number

import numpy as np


# ==================================================================================================
class MCMCQoI(ABC):
    """ABC interface for a Quantity of Interest (QoI) computed from an MCMC chain state.

    Methods:
        evaluate: Evaluate the QoI from a state.
        name: Name of the QoI, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]], accepted: bool
    ) -> Number:
        """Evaluate the QoI from a chain state.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): Current chain state.
            accepted (bool): Whether the current state was just accepted as a proposal.

        Returns:
            Number: QoI value.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def name(self) -> str:
        """Return the name of the QoI, used for logging."""

    # ----------------------------------------------------------------------------------------------
    def __str__(self) -> str:
        """Return the QoI's name."""
        return self.name()


# --------------------------------------------------------------------------------------------------
class ComponentQoI(MCMCQoI):
    """QoI extracting a single component from the state vector.

    Methods:
        evaluate: Evaluate the QoI from a state.
        name: Name of the QoI, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, component: int) -> None:
        """Initialize the QoI.

        Args:
            component (int): Index of the component to extract from the state vector.
        """
        self._component = component

    # ----------------------------------------------------------------------------------------------
    def evaluate(
        self, state: np.ndarray[tuple[int], np.dtype[np.float64]], accepted: bool
    ) -> float:
        """Return `state[component]`."""
        return float(state[self._component])

    # ----------------------------------------------------------------------------------------------
    def name(self) -> str:
        """Return `"component <index>"`."""
        return f"component {self._component}"


# --------------------------------------------------------------------------------------------------
class MeanQoI(MCMCQoI):
    """QoI computing the mean of all state components.

    Methods:
        evaluate: Evaluate the QoI from a state.
        name: Name of the QoI, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def evaluate(state: np.ndarray[tuple[int], np.dtype[np.float64]], accepted: bool) -> float:
        """Return `np.mean(state)`."""
        return float(np.mean(state))

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def name() -> str:
        """Return `"mean"`."""
        return "mean"


# --------------------------------------------------------------------------------------------------
class AcceptanceQoI(MCMCQoI):
    """QoI reporting whether the proposal was accepted.

    Methods:
        evaluate: Evaluate the QoI from a state.
        name: Name of the QoI, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def evaluate(state: np.ndarray[tuple[int], np.dtype[np.float64]], accepted: bool) -> float:
        """Return `1.0` if `accepted`, `0.0` otherwise."""
        return float(accepted)

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def name() -> str:
        """Return `"acceptance"`."""
        return "acceptance"


# ==================================================================================================
class MCMCStatistic(ABC):
    """ABC interface for a statistic computed from a sequence of QoI values.

    Methods:
        evaluate: Update the statistic with a new QoI value.
        name: Name of the statistic, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def evaluate(self, qoi_value: Number) -> Number:
        """Update the statistic with a new QoI value and return the updated value.

        Args:
            qoi_value (Number): Newly evaluated QoI value.

        Returns:
            Number: Updated statistic value.
        """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def name(self) -> str:
        """Return the name of the statistic, used for logging."""

    # ----------------------------------------------------------------------------------------------
    def __str__(self) -> str:
        """Return the statistic's name."""
        return self.name()


# --------------------------------------------------------------------------------------------------
class IdentityStatistic(MCMCStatistic):
    """Statistic returning the QoI value unchanged.

    Methods:
        evaluate: Update the statistic with a new QoI value.
        name: Name of the statistic, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def evaluate(qoi_value: Number) -> Number:
        """Return `qoi_value` unchanged."""
        return qoi_value

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def name() -> str:
        """Return `"identity"`."""
        return "identity"


# --------------------------------------------------------------------------------------------------
class RunningMeanStatistic(MCMCStatistic):
    """Statistic computing the running mean of all QoI values seen so far.

    Methods:
        evaluate: Update the statistic with a new QoI value.
        name: Name of the statistic, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize the running mean at zero, with zero samples seen."""
        self._running_value = 0.0
        self._num_samples = 0

    # ----------------------------------------------------------------------------------------------
    def evaluate(self, qoi_value: Number) -> float:
        """Update and return the running mean, given one new QoI value."""
        self._running_value = self._num_samples / (
            self._num_samples + 1
        ) * self._running_value + qoi_value / (self._num_samples + 1)
        self._num_samples += 1
        return self._running_value

    # ----------------------------------------------------------------------------------------------
    def name(self) -> str:
        """Return `"mean"`."""
        return "mean"


# --------------------------------------------------------------------------------------------------
class BatchMeanStatistic(MCMCStatistic):
    """Statistic computing the mean of QoI values within fixed-size, non-overlapping batches.

    Methods:
        evaluate: Update the statistic with a new QoI value.
        name: Name of the statistic, used for logging.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self, batch_size: int) -> None:
        """Initialize the statistic.

        Args:
            batch_size (int): Number of QoI values per batch.

        Raises:
            ValueError: If `batch_size` is not greater than zero.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be greater than zero, got {batch_size}.")
        self._batch_size = batch_size
        self._values: list[Number] = []
        self._batch_mean = 0.0

    # ----------------------------------------------------------------------------------------------
    def evaluate(self, qoi_value: Number) -> float:
        """Update the current batch and return the most recently completed batch's mean."""
        self._values.append(qoi_value)
        if len(self._values) >= self._batch_size:
            self._batch_mean = float(np.mean(self._values))
            self._values.clear()
        return self._batch_mean

    # ----------------------------------------------------------------------------------------------
    def name(self) -> str:
        """Return `"BM<batch size>"`."""
        return f"BM{self._batch_size}"


# ==================================================================================================
class MCMCOutput:
    """Combines a QoI and a statistic into one tracked, optionally logged, output.

    Methods:
        update: Update the output with a new chain state.

    Attributes:
        value (Number): Most recently computed output value.
        all_values (np.ndarray[tuple[int], np.dtype[np.float64]]): All computed output values, in
            call order.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        qoi: MCMCQoI,
        statistic: MCMCStatistic,
        str_id: str | None = None,
        str_format: str | None = None,
        log: bool = False,
    ) -> None:
        """Initialize the output.

        Args:
            qoi (MCMCQoI): Quantity of interest to evaluate from each chain state.
            statistic (MCMCStatistic): Statistic to compute from the sequence of QoI values.
            str_id (str | None, optional): Column label, used when `log` is `True`. Defaults to
                `None`.
            str_format (str | None, optional): Format spec for the value column, used when `log`
                is `True`. Defaults to `None`.
            log (bool, optional): Whether a logger should include this output. Defaults to
                `False`.

        Raises:
            ValueError: If `log` is `True` and `str_id` or `str_format` is `None`.
        """
        if log and str_id is None:
            raise ValueError("str_id must be given if log=True.")
        if log and str_format is None:
            raise ValueError("str_format must be given if log=True.")
        self.str_id = str_id
        self.str_format = str_format
        self.log = log
        self._qoi = qoi
        self._statistic = statistic
        self._values: list[Number] = []

    # ----------------------------------------------------------------------------------------------
    def update(self, state: np.ndarray[tuple[int], np.dtype[np.float64]], accepted: bool) -> None:
        """Evaluate the QoI at `state`, update the statistic, and record the resulting value.

        Args:
            state (np.ndarray[tuple[int], np.dtype[np.float64]]): Current chain state.
            accepted (bool): Whether `state` was just accepted as a proposal.
        """
        qoi_value = self._qoi.evaluate(state, accepted)
        self._values.append(self._statistic.evaluate(qoi_value))

    # ----------------------------------------------------------------------------------------------
    @property
    def value(self) -> Number:
        """Return the most recently computed output value."""
        return self._values[-1]

    # ----------------------------------------------------------------------------------------------
    @property
    def all_values(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        """Return all computed output values, in call order, as a new array."""
        return np.array(self._values, dtype=np.float64)


# ==================================================================================================
def build(qoi: MCMCQoI, statistic: MCMCStatistic) -> MCMCOutput:
    """Build a logged `MCMCOutput` from a QoI and a statistic, deriving its column label and
    format automatically (e.g. `build(AcceptanceQoI(), RunningMeanStatistic())` for the running
    mean acceptance rate, `build(ComponentQoI(0), IdentityStatistic())` for the raw value of state
    component 0 at every step).

    Args:
        qoi (MCMCQoI): Quantity of interest to evaluate from each chain state.
        statistic (MCMCStatistic): Statistic to compute from the sequence of QoI values.

    Returns:
        MCMCOutput: Output with a column label/format derived from `qoi`/`statistic`.
    """
    str_id = f"{qoi}" if isinstance(statistic, IdentityStatistic) else f"{statistic} of {qoi}"
    str_id = f"{str_id:<12}"
    return MCMCOutput(
        qoi=qoi,
        statistic=statistic,
        str_id=str_id,
        str_format=f"<+{max(12, len(str_id))}.3e",
        log=True,
    )
