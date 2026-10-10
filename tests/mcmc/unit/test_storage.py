from pathlib import Path

import numpy as np
import pytest
import zarr

from ls_bayesian.mcmc.storage import NumpyStorage, ZarrStorage, open_zarr_samples

pytestmark = pytest.mark.unit


# ==================================================================================================
def test_numpy_storage_values_is_none_before_any_store() -> None:
    storage_under_test = NumpyStorage()

    assert storage_under_test.values is None


# --------------------------------------------------------------------------------------------------
def test_numpy_storage_stacks_samples_in_call_order() -> None:
    storage_under_test = NumpyStorage()

    storage_under_test.store(np.array([1.0, 2.0]))
    storage_under_test.store(np.array([3.0, 4.0]))

    np.testing.assert_allclose(storage_under_test.values, [[1.0, 2.0], [3.0, 4.0]])


# --------------------------------------------------------------------------------------------------
def test_numpy_storage_flush_is_noop() -> None:
    storage_under_test = NumpyStorage()
    storage_under_test.store(np.array([1.0]))

    storage_under_test.flush()

    np.testing.assert_allclose(storage_under_test.values, [[1.0]])


# ==================================================================================================
def test_zarr_storage_values_is_none_before_any_store(tmp_path: Path) -> None:
    storage_under_test = ZarrStorage(tmp_path, chunk_size=4)

    assert storage_under_test.values is None


# --------------------------------------------------------------------------------------------------
def _read_raw_on_disk_row_count(save_directory: Path) -> int:
    """Read the row count of the on-disk Zarr dataset directly, bypassing `ZarrStorage.values`
    (which would itself trigger a flush) -- exposes whether a buffered-but-unflushed sample has
    actually reached disk."""
    root = zarr.open_group(zarr.storage.LocalStore(save_directory), mode="r")
    if "data" not in root:
        return 0
    return root["data"].shape[0]


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_buffers_until_buffer_size_then_writes(tmp_path: Path) -> None:
    storage_under_test = ZarrStorage(tmp_path, chunk_size=4, buffer_size=3)

    storage_under_test.store(np.array([1.0]))
    storage_under_test.store(np.array([2.0]))
    assert _read_raw_on_disk_row_count(tmp_path) == 0

    storage_under_test.store(np.array([3.0]))
    assert _read_raw_on_disk_row_count(tmp_path) == 3


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_values_flushes_residual_buffer(tmp_path: Path) -> None:
    storage_under_test = ZarrStorage(tmp_path, chunk_size=4, buffer_size=5)
    storage_under_test.store(np.array([1.0]))
    storage_under_test.store(np.array([2.0]))

    values = storage_under_test.values

    assert values.shape == (2, 1)


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_chunk_size_sets_on_disk_chunking(tmp_path: Path) -> None:
    storage_under_test = ZarrStorage(tmp_path, chunk_size=7, buffer_size=1)
    storage_under_test.store(np.array([1.0]))

    assert storage_under_test.values.chunks[0] == 7


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_appends_to_existing_store(tmp_path: Path) -> None:
    first_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1)
    first_storage.store(np.array([1.0]))
    first_storage.store(np.array([2.0]))

    second_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1, overwrite=False)
    second_storage.store(np.array([3.0]))

    np.testing.assert_allclose(second_storage.values, [[1.0], [2.0], [3.0]])


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_values_returns_existing_samples_before_any_store(tmp_path: Path) -> None:
    first_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1)
    first_storage.store(np.array([1.0]))
    first_storage.store(np.array([2.0]))

    second_storage = ZarrStorage(tmp_path, chunk_size=4, overwrite=False)

    np.testing.assert_allclose(second_storage.values, [[1.0], [2.0]])


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_rejects_chunk_size_mismatching_existing_store(tmp_path: Path) -> None:
    first_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1)
    first_storage.store(np.array([1.0]))

    with pytest.raises(ValueError, match="chunk_size 5"):
        ZarrStorage(tmp_path, chunk_size=5, overwrite=False)


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_overwrite_true_ignores_existing_chunk_size(tmp_path: Path) -> None:
    first_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1)
    first_storage.store(np.array([1.0]))

    second_storage = ZarrStorage(tmp_path, chunk_size=5, buffer_size=1, overwrite=True)
    second_storage.store(np.array([2.0]))

    assert second_storage.values.chunks[0] == 5


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_overwrite_true_truncates_existing_store(tmp_path: Path) -> None:
    first_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1)
    first_storage.store(np.array([1.0]))
    first_storage.store(np.array([2.0]))

    second_storage = ZarrStorage(tmp_path, chunk_size=4, buffer_size=1, overwrite=True)
    second_storage.store(np.array([3.0]))

    np.testing.assert_allclose(second_storage.values, [[3.0]])


# --------------------------------------------------------------------------------------------------
def test_zarr_storage_creates_missing_nested_directory(tmp_path: Path) -> None:
    save_directory = tmp_path / "a" / "b"
    storage_under_test = ZarrStorage(save_directory, chunk_size=4, buffer_size=1)

    storage_under_test.store(np.array([1.0]))

    np.testing.assert_allclose(storage_under_test.values, [[1.0]])


# ==================================================================================================
@pytest.mark.parametrize(
    ("chunk_size", "buffer_size"),
    [(0, 1), (-1, 1), (1, 0), (1, -1)],
    ids=["chunk_zero", "chunk_negative", "buffer_zero", "buffer_negative"],
)
def test_zarr_storage_rejects_non_positive_chunk_or_buffer_size(
    tmp_path: Path, chunk_size: int, buffer_size: int
) -> None:
    with pytest.raises(ValueError, match="greater than zero"):
        ZarrStorage(tmp_path, chunk_size=chunk_size, buffer_size=buffer_size)


# ==================================================================================================
def test_open_zarr_samples_reads_what_zarr_storage_wrote(tmp_path: Path) -> None:
    storage_under_test = ZarrStorage(tmp_path / "samples.zarr", chunk_size=2)
    samples = np.arange(10.0).reshape(5, 2)
    for sample in samples:
        storage_under_test.store(sample)

    opened = open_zarr_samples(tmp_path / "samples.zarr")

    np.testing.assert_array_equal(opened[:], samples)


# --------------------------------------------------------------------------------------------------
def test_open_zarr_samples_rejects_missing_store_and_store_without_samples(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No Zarr store"):
        open_zarr_samples(tmp_path / "missing.zarr")
    ZarrStorage(tmp_path / "empty.zarr", chunk_size=2)
    with pytest.raises(FileNotFoundError, match="holds no samples"):
        open_zarr_samples(tmp_path / "empty.zarr")
