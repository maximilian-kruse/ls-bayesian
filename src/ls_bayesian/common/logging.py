"""Logging utilities shared across subpackages.

Classes:
    LoggerSettings: Settings for the output channels of a logger.
    BaseLogger: Logger writing prefixed messages to the console and/or a log file.
"""

import itertools
import logging
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Literal, Self, override

_instance_counter = itertools.count()


# ==================================================================================================
@dataclass
class LoggerSettings:
    """Settings for the output channels of a logger.

    Attributes:
        print_to_console (bool): Whether to print messages to standard output. Defaults to `True`.
        logfile_path (Path | None): File to write messages to. Missing parent directories are
            created. If `None`, no log file is written. Defaults to `None`.
        write_mode (Literal["w", "a"]): Whether to overwrite (`"w"`) or append to (`"a"`) an
            existing log file. Defaults to `"w"`.
    """

    print_to_console: bool = True
    logfile_path: Path | None = None
    write_mode: Literal["w", "a"] = "w"


# ==================================================================================================
class _LevelAwareFormatter(logging.Formatter):
    """Formatter that prefixes messages, and only shows the level for non-info messages."""

    def __init__(self, prefix: str) -> None:
        """Initialize the formatter.

        Args:
            prefix (str): Prefix prepended to every message, e.g. the name of the component.
        """
        super().__init__()
        self._info_formatter = logging.Formatter(f"[{prefix}] %(message)s")
        self._level_formatter = logging.Formatter(f"[{prefix}][%(levelname)s] %(message)s")

    @override
    def format(self, record: logging.LogRecord) -> str:
        """Format a record as `[PREFIX] message`, or `[PREFIX][LEVEL] message` for non-info levels.

        Args:
            record (logging.LogRecord): Record to format.

        Returns:
            str: Formatted message.
        """
        formatter = (
            self._info_formatter if record.levelno == logging.INFO else self._level_formatter
        )
        return formatter.format(record)


# ==================================================================================================
class BaseLogger:
    """Logger writing prefixed messages to the console and/or a log file.

    Every instance owns a separate, non-propagating Python logger, so that instances with different
    prefixes and output channels do not interfere with each other or with the root logger. All
    levels, including debug messages, are emitted.

    The logger holds an open log file until it is closed, either explicitly via
    [`close`][ls_bayesian.common.logging.BaseLogger.close], by using it as a context manager, or
    at the latest when it is garbage collected or the interpreter exits. Logging to a closed logger
    raises an error.

    Methods:
        info: Log an info message.
        debug: Log a debug message.
        warning: Log a warning.
        exception: Log an error message together with the traceback of the current exception.
        error: Log an error message.
        close: Flush and release all output channels.

    Attributes:
        closed (bool): Whether the logger has been closed.
    """

    def __init__(self, logger_settings: LoggerSettings, prefix: str) -> None:
        """Initialize the logger and its output channels.

        Args:
            logger_settings (LoggerSettings): Output channel settings.
            prefix (str): Prefix prepended to every message, converted to upper case.
        """
        pylogger_name = f"{__name__}.{next(_instance_counter)}"
        self._pylogger = logging.getLogger(pylogger_name)
        self._pylogger.setLevel(logging.DEBUG)
        self._pylogger.propagate = False
        formatter = _LevelAwareFormatter(prefix=prefix.upper())

        if logger_settings.print_to_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(formatter)
            self._pylogger.addHandler(console_handler)

        if logger_settings.logfile_path is not None:
            logger_settings.logfile_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                logger_settings.logfile_path, mode=logger_settings.write_mode
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            self._pylogger.addHandler(file_handler)

        # The finalizer must not reference self, otherwise the logger is never garbage collected.
        self._finalizer = weakref.finalize(self, _remove_handlers, self._pylogger)

    @property
    def closed(self) -> bool:
        """Whether the logger has been closed."""
        return not self._finalizer.alive

    def close(self) -> None:
        """Flush and release all output channels.

        The log file is closed, the console stream is left open. Closing an already closed logger
        has no effect.
        """
        self._finalizer()

    def __enter__(self) -> Self:
        """Return the logger for use in a `with` statement, which closes it on exit."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the logger when leaving the `with` statement."""
        self.close()

    def info(self, message: str) -> None:
        """Log an info message.

        Args:
            message (str): Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.info(message)

    def debug(self, message: str) -> None:
        """Log a debug message.

        Args:
            message (str): Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.debug(message)

    def warning(self, message: str) -> None:
        """Log a warning.

        Args:
            message (str): Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.warning(message)

    def exception(self, message: str) -> None:
        """Log an error message together with the traceback of the current exception.

        Only call this from within an exception handler.

        Args:
            message (str): Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.exception(message)

    def error(self, message: str) -> None:
        """Log an error message.

        Args:
            message (str): Message to log.

        Raises:
            ValueError: If the logger has been closed.
        """
        self._check_open()
        self._pylogger.error(message)

    def _check_open(self) -> None:
        """Raise if the logger has been closed.

        Without handlers, Python would silently route warnings and errors to its last-resort
        handler on stderr instead.
        """
        if self.closed:
            raise ValueError("Cannot log to a closed logger.")


# ==================================================================================================
def _remove_handlers(pylogger: logging.Logger) -> None:
    """Detach, flush and close all handlers of a Python logger."""
    for handler in list(pylogger.handlers):
        pylogger.removeHandler(handler)
        handler.close()
