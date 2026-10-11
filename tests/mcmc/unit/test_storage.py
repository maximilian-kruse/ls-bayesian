from pathlib import Path

import numpy as np
import pytest
import zarr

from ls_bayesian.mcmc.storage import (
    MappedSampleStore,
    NumpyStorage,
    SampleStore,
    ZarrStorage,
    iterate_sample_blocks,
    open_zarr_samples,
)

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


# ==================================================================================================
def test_numpy_and_zarr_arrays_satisfy_sample_store_protocol(tmp_path: Path) -> None:
    zarr_array = zarr.create_array(tmp_path / "samples.zarr", shape=(3, 2), dtype=np.float64)

    assert isinstance(np.zeros((3, 2)), SampleStore)
    assert isinstance(zarr_array, SampleStore)
    assert not isinstance(object(), SampleStore)


# ==================================================================================================
@pytest.mark.parametrize("block_size", [1, 3, 7, 10, 100])
def test_iterate_sample_blocks_concatenates_to_samples_after_burn_in(block_size: int) -> None:
    samples = np.arange(20.0).reshape(10, 2)

    blocks = list(iterate_sample_blocks(samples, burn_in=2, block_size=block_size))

    assert all(0 < block.shape[0] <= block_size for block in blocks)
    np.testing.assert_array_equal(np.concatenate(blocks), samples[2:])


# --------------------------------------------------------------------------------------------------
def test_iterate_sample_blocks_reads_zarr_store_like_numpy_array(tmp_path: Path) -> None:
    samples = np.random.default_rng(0).normal(size=(25, 3))
    zarr_array = zarr.create_array(tmp_path / "samples.zarr", shape=samples.shape, dtype=np.float64)
    zarr_array[:] = samples

    blocks = list(iterate_sample_blocks(zarr_array, burn_in=4, block_size=6))

    assert all(isinstance(block, np.ndarray) for block in blocks)
    np.testing.assert_array_equal(np.concatenate(blocks), samples[4:])


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"burn_in": -1}, "burn_in"),
        ({"burn_in": 5}, "burn_in"),
        ({"burn_in": 6}, "burn_in"),
        ({"block_size": 0}, "block_size"),
        ({"block_size": -3}, "block_size"),
    ],
)
def test_iterate_sample_blocks_rejects_invalid_arguments(arguments: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        next(iterate_sample_blocks(np.zeros((5, 2)), **arguments))


# ==================================================================================================
def test_mapped_sample_store_applies_transform_to_every_slice() -> None:
    samples = np.arange(12.0).reshape(6, 2)

    mapped = MappedSampleStore(samples, lambda block: block**2)

    assert mapped.shape == samples.shape
    np.testing.assert_allclose(mapped[1:4], samples[1:4] ** 2)
    np.testing.assert_allclose(mapped[:], samples**2)


# --------------------------------------------------------------------------------------------------
def test_mapped_sample_store_streams_through_iterate_sample_blocks() -> None:
    samples = np.random.default_rng(1).normal(size=(17, 3))

    mapped = MappedSampleStore(samples, np.sin)
    blocks = list(iterate_sample_blocks(mapped, burn_in=3, block_size=5))

    np.testing.assert_allclose(np.concatenate(blocks), np.sin(samples[3:]))
    assert isinstance(mapped, SampleStore)


# --------------------------------------------------------------------------------------------------
def test_mapped_sample_store_does_not_modify_base_samples() -> None:
    samples = np.arange(6.0).reshape(3, 2)
    original = samples.copy()

    MappedSampleStore(samples, lambda block: block * 2.0)[:]

    np.testing.assert_array_equal(samples, original)
