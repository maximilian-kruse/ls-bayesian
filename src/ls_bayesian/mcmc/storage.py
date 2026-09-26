"""Storage backends for MCMC samples.

Classes:
    MCMCStorage: ABC interface for sample storage.
    NumpyStorage: In-memory storage.
    ZarrStorage: Chunked, disk-backed storage via Zarr.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, override

import numpy as np
import zarr


# ==================================================================================================
class MCMCStorage(ABC):
    """ABC interface for MCMC sample storage.

    `values` is intentionally typed `Any`: different backends hand back fundamentally different
    array types (an in-memory NumPy array vs. a lazy, on-disk Zarr array), and forcing one common
    type would either misrepresent a backend's actual return value or require materializing a
    disk-backed dataset in memory just to satisfy the type hint. Implementations must ensure
    `values` reflects every sample stored so far, including any not yet flushed to a backing
    store, and must return `None` if no sample has been stored yet.

    A backend that buffers samples before writing them to a backing store (e.g. `ZarrStorage`)
    only guarantees durability up to the last `flush()`; callers that need every stored sample to
    survive a crash must call `flush()` themselves once done storing.

    Methods:
        store: Store one sample.
        flush: Flush any buffered samples to their backing store.

    Attributes:
        values: All stored samples, in a backend-specific array type, or `None` if empty.
    """

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        """Store one sample.

        Args:
            sample (np.ndarray[tuple[int], np.dtype[np.float64]]): Sample to store.
        """

    # ----------------------------------------------------------------------------------------------
    @property
    @abstractmethod
    def values(self) -> Any:
        """Return all stored samples, in a backend-specific array type, or `None` if empty."""

    # ----------------------------------------------------------------------------------------------
    @abstractmethod
    def flush(self) -> None:
        """Flush any buffered samples to their backing store."""


# ==================================================================================================
class NumpyStorage(MCMCStorage):
    """In-memory storage backed by a plain Python list, stacked into a NumPy array on demand.

    Methods:
        store: Store one sample.
        flush: No-op; `NumpyStorage` has no backing store to flush to.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(self) -> None:
        """Initialize empty storage."""
        self._samples: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []

    # ----------------------------------------------------------------------------------------------
    @override
    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        """Append `sample` to the in-memory buffer."""
        self._samples.append(sample)

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def values(self) -> np.ndarray[tuple[int, int], np.dtype[np.float64]] | None:
        """Return all stored samples, stacked along a new leading axis, as a NumPy array.

        Returns:
            np.ndarray[tuple[int, int], np.dtype[np.float64]] | None: Stored samples of shape
                `(num_samples, sample_dim)`, or `None` if no sample has been stored yet.
        """
        if not self._samples:
            return None
        return np.stack(self._samples, axis=0)

    # ----------------------------------------------------------------------------------------------
    @override
    def flush(self) -> None:
        """No-op; `NumpyStorage` has no backing store to flush to. Use `ZarrStorage` for
        disk-backed storage."""


# ==================================================================================================
class ZarrStorage(MCMCStorage):
    """Disk-backed storage via Zarr.

    `chunk_size` and `buffer_size` are independent: `chunk_size` is a Zarr storage-layout
    parameter (how many samples form one on-disk chunk, trading write/read granularity against
    compression and indexing overhead), while `buffer_size` controls only how many samples
    accumulate in memory before a write to disk is even attempted. A single `store()` call can
    therefore write less than one full chunk (Zarr fills it incrementally across calls) -- keeping
    `buffer_size` small does not require small on-disk chunks, and a large `chunk_size` does not
    force buffering that many samples in memory (and losing them if the process is interrupted
    before the next flush).

    Methods:
        store: Buffer one sample, writing to disk once the buffer is full.
        flush: Write any buffered samples to disk.
    """

    # ----------------------------------------------------------------------------------------------
    def __init__(
        self,
        save_directory: Path,
        chunk_size: int,
        buffer_size: int = 1,
        overwrite: bool = False,
    ) -> None:
        """Initialize the storage, creating (or opening) the Zarr store at `save_directory`.

        Args:
            save_directory (Path): Directory the Zarr store is created in. Missing directories,
                including `save_directory` itself, are created.
            chunk_size (int): Length of the on-disk Zarr chunk along the sample axis.
            buffer_size (int, optional): Number of samples accumulated in memory before writing to
                disk. Defaults to `1`, writing every sample immediately: the safe default, since a
                larger value trades durability (samples not yet written are lost if the process is
                interrupted) for fewer, larger writes.
            overwrite (bool, optional): Whether to overwrite an existing store at
                `save_directory`. Defaults to `False`, which appends to an existing store; its
                samples are then immediately available via `values`.

        Raises:
            ValueError: If `chunk_size` or `buffer_size` is not greater than zero, or if an
                existing dataset at `save_directory` is chunked differently from `chunk_size`.
        """
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be greater than zero, got {chunk_size}.")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be greater than zero, got {buffer_size}.")
        self._chunk_size = chunk_size
        self._buffer_size = buffer_size
        self._buffer: list[np.ndarray[tuple[int], np.dtype[np.float64]]] = []
        store = zarr.storage.LocalStore(save_directory)
        self._root = zarr.open_group(store, mode="w" if overwrite else "a")
        self._dataset: zarr.Array | None = self._open_existing_dataset()

    # ----------------------------------------------------------------------------------------------
    @override
    def store(self, sample: np.ndarray[tuple[int], np.dtype[np.float64]]) -> None:
        """Buffer `sample`, writing to disk once `buffer_size` samples are buffered."""
        self._buffer.append(sample)
        if len(self._buffer) >= self._buffer_size:
            self.flush()

    # ----------------------------------------------------------------------------------------------
    @property
    @override
    def values(self) -> zarr.Array | None:
        """Flush any buffered samples, then return the full on-disk dataset.

        Returns:
            zarr.Array | None: The on-disk dataset, or `None` if no sample has been stored yet.
        """
        self.flush()
        return self._dataset

    # ----------------------------------------------------------------------------------------------
    @override
    def flush(self) -> None:
        """Write any buffered samples to disk, appending to the dataset if it already exists."""
        if not self._buffer:
            return
        chunk = np.stack(self._buffer, axis=0)
        self._buffer.clear()
        if self._dataset is None:
            self._dataset = self._create_dataset(chunk)
        else:
            self._dataset.append(chunk)

    # ----------------------------------------------------------------------------------------------
    def _open_existing_dataset(self) -> zarr.Array | None:
        """Open the on-disk dataset if the store already holds one, else return `None`.

        Unlike creating a new dataset, opening an existing one needs no sample: its shape, chunking
        and dtype are stored on disk.

        Raises:
            ValueError: If the existing dataset is chunked differently from `chunk_size`.
        """
        if "data" not in self._root:
            return None
        dataset = self._root["data"]
        existing_chunk_size = dataset.chunks[0]
        if existing_chunk_size != self._chunk_size:
            raise ValueError(
                f"chunk_size {self._chunk_size} does not match the existing dataset's chunk size "
                f"{existing_chunk_size}."
            )
        return dataset

    # ----------------------------------------------------------------------------------------------
    def _create_dataset(
        self, first_chunk: np.ndarray[tuple[int, int], np.dtype[np.float64]]
    ) -> zarr.Array:
        """Create the on-disk dataset and write `first_chunk` to it.

        Creation is deferred to the first flush because the sample shape and dtype are only known
        once the first sample has been stored.
        """
        sample_shape = first_chunk.shape[1:]
        dataset = self._root.create_array(
            "data",
            shape=(0, *sample_shape),
            chunks=(self._chunk_size, *sample_shape),
            dtype=first_chunk.dtype,
        )
        dataset.append(first_chunk)
        return dataset
