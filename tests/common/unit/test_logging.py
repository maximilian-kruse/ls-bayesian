import gc
import os
import sys
from pathlib import Path
from typing import Literal

import pytest
from beartype.roar import BeartypeCallHintViolation

from ls_bayesian.common.logging import BaseLogger, LoggerSettings

pytestmark = pytest.mark.unit


# ==================================================================================================
def _check_if_file_handle_is_open(path: Path) -> bool:
    """Check whether the current process holds an open file descriptor for the given file."""
    file_descriptor_directory = Path("/proc/self/fd")
    for file_descriptor in file_descriptor_directory.iterdir():
        try:
            if os.readlink(file_descriptor) == str(path):
                return True
        except FileNotFoundError:
            continue
    return False


def _create_logger_with_file(
    logfile_path: Path, *, write_mode: Literal["w", "a"] = "w"
) -> BaseLogger:
    return BaseLogger(
        LoggerSettings(print_to_console=False, logfile_path=logfile_path, write_mode=write_mode),
        prefix="test",
    )


# ==================================================================================================
def test_logger_writes_prefixed_messages(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"
    logger = _create_logger_with_file(logfile_path)

    logger.info("info message")
    logger.warning("warning message")
    logger.debug("debug message")
    logger.error("error message")
    logger.close()

    assert logfile_path.read_text().splitlines() == [
        "[TEST] info message",
        "[TEST][WARNING] warning message",
        "[TEST][DEBUG] debug message",
        "[TEST][ERROR] error message",
    ]


# --------------------------------------------------------------------------------------------------
def test_close_releases_logfile(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"
    logger = _create_logger_with_file(logfile_path)
    assert _check_if_file_handle_is_open(logfile_path)

    logger.close()

    assert logger.closed
    assert not _check_if_file_handle_is_open(logfile_path)


# --------------------------------------------------------------------------------------------------
def test_context_manager_closes_logger(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"

    with _create_logger_with_file(logfile_path) as logger:
        logger.info("message")

    assert logger.closed
    assert not _check_if_file_handle_is_open(logfile_path)
    assert logfile_path.read_text() == "[TEST] message\n"


# --------------------------------------------------------------------------------------------------
def test_garbage_collection_releases_logfile(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"
    logger = _create_logger_with_file(logfile_path)
    assert _check_if_file_handle_is_open(logfile_path)

    del logger
    gc.collect()

    assert not _check_if_file_handle_is_open(logfile_path)


# --------------------------------------------------------------------------------------------------
def test_close_is_idempotent(tmp_path: Path) -> None:
    logger = _create_logger_with_file(tmp_path / "test.log")

    logger.close()
    logger.close()

    assert logger.closed


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("method_name", ["info", "debug", "warning", "error", "exception"])
def test_logging_to_closed_logger_raises(tmp_path: Path, method_name: str) -> None:
    logger = _create_logger_with_file(tmp_path / "test.log")
    logger.close()

    with pytest.raises(ValueError, match="closed logger"):
        getattr(logger, method_name)("message")


# --------------------------------------------------------------------------------------------------
def test_close_keeps_console_stream_open() -> None:
    console_stream = sys.stdout
    logger = BaseLogger(LoggerSettings(print_to_console=True), prefix="test")

    logger.close()

    assert not console_stream.closed


# --------------------------------------------------------------------------------------------------
def test_console_logger_prints_prefixed_messages(capsys: pytest.CaptureFixture[str]) -> None:
    logger = BaseLogger(LoggerSettings(print_to_console=True), prefix="test")

    logger.info("info message")
    logger.warning("warning message")
    logger.debug("debug message")
    logger.error("error message")
    logger.close()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        "[TEST] info message",
        "[TEST][WARNING] warning message",
        "[TEST][DEBUG] debug message",
        "[TEST][ERROR] error message",
    ]


# --------------------------------------------------------------------------------------------------
def test_logger_writes_to_both_console_and_file_simultaneously(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    logfile_path = tmp_path / "test.log"
    expected_lines = ["[TEST] info message", "[TEST][WARNING] warning message"]
    logger = BaseLogger(
        LoggerSettings(print_to_console=True, logfile_path=logfile_path), prefix="test"
    )

    logger.info("info message")
    logger.warning("warning message")
    logger.close()

    captured = capsys.readouterr()
    assert captured.out.splitlines() == expected_lines
    assert logfile_path.read_text().splitlines() == expected_lines


# --------------------------------------------------------------------------------------------------
def test_write_mode_w_overwrites_existing_logfile(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"
    logfile_path.write_text("stale\n")

    with _create_logger_with_file(logfile_path, write_mode="w") as logger:
        logger.info("new message")

    assert logfile_path.read_text() == "[TEST] new message\n"


# --------------------------------------------------------------------------------------------------
def test_write_mode_a_appends_to_existing_logfile(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"
    logfile_path.write_text("stale\n")

    with _create_logger_with_file(logfile_path, write_mode="a") as logger:
        logger.info("new message")

    assert logfile_path.read_text() == "stale\n[TEST] new message\n"


# --------------------------------------------------------------------------------------------------
def test_exception_includes_traceback(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"
    logger = _create_logger_with_file(logfile_path)

    try:
        raise ValueError("boom")
    except ValueError:
        logger.exception("context")
    logger.close()

    logged_text = logfile_path.read_text()
    assert "[TEST][ERROR] context" in logged_text
    assert "ValueError: boom" in logged_text


# --------------------------------------------------------------------------------------------------
def test_messages_do_not_propagate_to_root_logger(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    logger = _create_logger_with_file(tmp_path / "test.log")

    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.close()

    assert caplog.records == []


# --------------------------------------------------------------------------------------------------
def test_two_loggers_with_same_prefix_do_not_share_handlers(tmp_path: Path) -> None:
    first_logfile_path = tmp_path / "first.log"
    second_logfile_path = tmp_path / "second.log"
    first_logger = _create_logger_with_file(first_logfile_path)
    second_logger = _create_logger_with_file(second_logfile_path)

    first_logger.info("first message")
    second_logger.info("second message")
    first_logger.close()

    assert first_logfile_path.read_text() == "[TEST] first message\n"
    assert second_logfile_path.read_text() == "[TEST] second message\n"
    assert first_logger.closed
    assert not second_logger.closed

    second_logger.close()


# --------------------------------------------------------------------------------------------------
def test_missing_parent_directories_are_created(tmp_path: Path) -> None:
    logfile_path = tmp_path / "a" / "b" / "c.log"

    with _create_logger_with_file(logfile_path) as logger:
        logger.info("message")

    assert logfile_path.read_text() == "[TEST] message\n"


# --------------------------------------------------------------------------------------------------
def test_context_manager_propagates_exceptions(tmp_path: Path) -> None:
    logfile_path = tmp_path / "test.log"

    with (
        pytest.raises(RuntimeError, match="boom"),
        _create_logger_with_file(logfile_path) as logger,
    ):
        raise RuntimeError("boom")

    assert logger.closed


# --------------------------------------------------------------------------------------------------
def test_logger_settings_defaults() -> None:
    settings = LoggerSettings()

    assert settings.print_to_console is True
    assert settings.logfile_path is None
    assert settings.write_mode == "w"


# --------------------------------------------------------------------------------------------------
def test_no_logfile_handler_when_path_is_none(tmp_path: Path) -> None:
    logger = BaseLogger(LoggerSettings(print_to_console=False, logfile_path=None), prefix="test")

    logger.info("message")
    logger.close()

    assert list(tmp_path.iterdir()) == []


# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("invalid_write_mode", ["x", "r", "append"])
def test_invalid_write_mode_raises_beartype_violation(invalid_write_mode: str) -> None:
    with pytest.raises(BeartypeCallHintViolation):
        LoggerSettings(write_mode=invalid_write_mode)
