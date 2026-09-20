"""Driver that runs an MCMC algorithm for a fixed number of samples.

Classes:
    SamplerSettings: Settings for a sampling run.
    Sampler: Runs an `MCMCAlgorithm`, dispatching each new state to storage, outputs, and a
        logger.
"""

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Annotated

import numpy as np
from beartype.vale import Is

from ls_bayesian.common.logging import BaseLogger
from ls_bayesian.mcmc.algorithm import MCMCAlgorithm
from ls_bayesian.mcmc.output import MCMCOutput
from ls_bayesian.mcmc.storage import MCMCStorage

# Column widths for the progress table logged by `_log_header`/`_log_iteration`. Wide enough to
# fit the header labels and typical numeric values in scientific notation (e.g. "1.234568e+00")
# with one space of padding.
_ITERATION_COLUMN_WIDTH = 10
_TIME_COLUMN_WIDTH = 12


# ==================================================================================================
@dataclass
class SamplerSettings:
    """Settings for an MCMC sampling run.

    The field constraints are validated on initialization.

    Attributes:
        num_samples (int): Number of states to generate, including the initial state.
        store_interval (int): Store every `store_interval`-th state (in addition to the initial
            state). Defaults to `1`, storing every state.
        log_interval (int): Log every `log_interval`-th state (in addition to the initial state).
            Defaults to `1`, logging every state.
    """

    num_samples: Annotated[int, Is[lambda x: x > 0]]
    store_interval: Annotated[int, Is[lambda x: x > 0]] = 1
    log_interval: Annotated[int, Is[lambda x: x > 0]] = 1


# ==================================================================================================
class Sampler:
    """Runs an `MCMCAlgorithm` for a fixed number of samples, dispatching each new state to
    storage, outputs, and a logger.

    Methods:
        run: Run the chain from an initial state for `settings.num_samples` states.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        algorithm: MCMCAlgorithm,
        storage: MCMCStorage | None = None,
        outputs: Iterable[MCMCOutput] = (),
        logger: BaseLogger | None = None,
    ) -> None:
        """Initialize the sampler.

        Args:
            algorithm (MCMCAlgorithm): Algorithm advancing the chain by one step at a time.
            storage (MCMCStorage | None, optional): Storage for samples. Nothing is stored if
                `None`. The caller owns the storage's lifetime; the sampler never constructs its
                own. Defaults to `None`.
            outputs (Iterable[MCMCOutput], optional): Outputs updated with every generated state.
                Defaults to `()`.
            logger (BaseLogger | None, optional): Logger for iteration-by-iteration progress
                reports. Nothing is logged if `None`. The caller owns the logger's lifetime.
                Defaults to `None`.
        """
        self._algorithm = algorithm
        self._storage = storage
        self._outputs = tuple(outputs)
        self._logger = logger

    # ----------------------------------------------------------------------------------------------
    def run(
        self,
        initial_state: np.ndarray[tuple[int], np.dtype[np.float64]],
        settings: SamplerSettings,
        seed: int = 0,
    ) -> None:
        """Run the chain from `initial_state` for `settings.num_samples` states.

        The initial state is always stored and logged; the accept/reject step only applies to
        subsequently generated states, so it is treated as unconditionally accepted. `storage` (if
        given) is flushed once the run finishes, including if it is interrupted by an exception,
        so no sample buffered but not yet written to a backing store is lost.

        Args:
            initial_state (np.ndarray[tuple[int], np.dtype[np.float64]]): State the chain starts
                from.
            settings (SamplerSettings): Settings for the run.
            seed (int, optional): Seed for the chain's random number generator. Defaults to `0`.

        Raises:
            ValueError: If `settings.store_interval` or `settings.log_interval` exceeds
                `settings.num_samples`.
        """
        self._validate_settings(settings)
        rng = np.random.default_rng(seed=seed)
        current_state = np.asarray(initial_state, dtype=np.float64)
        start_time = time.monotonic()

        try:
            self._dispatch(
                0, current_state, accepted=True, settings=settings, start_time=start_time
            )
            for iteration in range(1, settings.num_samples):
                current_state, accepted = self._algorithm.compute_step(current_state, rng)
                self._dispatch(iteration, current_state, accepted, settings, start_time)
        finally:
            if self._storage is not None:
                self._storage.flush()

    # ----------------------------------------------------------------------------------------------
    @staticmethod
    def _validate_settings(settings: SamplerSettings) -> None:
        """Raise if a logging/storage interval exceeds the number of samples to generate."""
        if settings.store_interval > settings.num_samples:
            raise ValueError(
                f"store_interval ({settings.store_interval}) must not exceed num_samples "
                f"({settings.num_samples})."
            )
        if settings.log_interval > settings.num_samples:
            raise ValueError(
                f"log_interval ({settings.log_interval}) must not exceed num_samples "
                f"({settings.num_samples})."
            )

    # ----------------------------------------------------------------------------------------------
    def _dispatch(
        self,
        iteration: int,
        state: np.ndarray[tuple[int], np.dtype[np.float64]],
        accepted: bool,
        settings: SamplerSettings,
        start_time: float,
    ) -> None:
        """Update outputs, store, and log the current state, if the respective interval elapsed."""
        for output in self._outputs:
            output.update(state, accepted)
        if self._storage is not None and iteration % settings.store_interval == 0:
            self._storage.store(state)
        if iteration % settings.log_interval == 0:
            if iteration == 0:
                self._log_header()
            self._log_iteration(iteration, time.monotonic() - start_time)

    # ----------------------------------------------------------------------------------------------
    def _log_header(self) -> None:
        """Log the column header for iteration-by-iteration progress reports. No-op if
        `self._logger` is `None`."""
        if self._logger is None:
            return
        header = f"{'Iteration':>{_ITERATION_COLUMN_WIDTH}} {'Time [s]':>{_TIME_COLUMN_WIDTH}} "
        header += " ".join(output.str_id for output in self._outputs if output.log)
        self._logger.info(header)
        self._logger.info("-" * len(header))

    # ----------------------------------------------------------------------------------------------
    def _log_iteration(self, iteration: int, elapsed_time_seconds: float) -> None:
        """Log one row of iteration-by-iteration progress. No-op if `self._logger` is `None`."""
        if self._logger is None:
            return
        row = (
            f"{iteration:>{_ITERATION_COLUMN_WIDTH}d} "
            f"{elapsed_time_seconds:>{_TIME_COLUMN_WIDTH}.3f} "
        )
        row += " ".join(
            f"{output.value:{output.str_format}}" for output in self._outputs if output.log
        )
        self._logger.info(row)
