"""Comprehensive file-based logging for NavIRL.

Provides multiple file logger implementations:

- :class:`RotatingFileLogger` -- Plain-text log files with size-based rotation
  and optional gzip compression.
- :class:`CSVLogger` -- Structured CSV logging with automatic header management.
- :class:`JSONLLogger` -- JSON-Lines (newline-delimited JSON) logging with
  buffered writes and atomic file operations.
- :class:`HDF5Logger` -- Stub for HDF5-based logging (requires ``h5py``).

All loggers support log levels, structured context, buffered writes, and
safe file operations via temporary files and atomic renames.
"""

from __future__ import annotations

import csv
import gzip
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, TextIO

import numpy as np

# ---------------------------------------------------------------------------
# Log levels
# ---------------------------------------------------------------------------


class LogLevel(IntEnum):
    """Numeric log levels mirroring the standard library."""

    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50

    @classmethod
    def from_string(cls, name: str) -> LogLevel:
        """Parse a level from its name string (case-insensitive).

        Args:
            name: Level name such as ``"INFO"`` or ``"debug"``.

        Returns:
            Corresponding ``LogLevel`` member.

        Raises:
            ValueError: If *name* is not a recognised level.
        """
        upper = name.upper()
        try:
            return cls[upper]
        except KeyError:
            raise ValueError(
                f"Unknown log level {name!r}. " f"Valid levels: {', '.join(cls.__members__)}"
            ) from None


# ---------------------------------------------------------------------------
# Structured log record
# ---------------------------------------------------------------------------


@dataclass
class LogRecord:
    """Structured log record used across all file loggers.

    Attributes:
        timestamp: Unix timestamp.
        level: Numeric log level.
        message: Human-readable message.
        context: Additional structured data.
        logger_name: Name of the logger that produced this record.
    """

    timestamp: float
    level: int
    message: str
    context: dict[str, Any] = field(default_factory=dict)
    logger_name: str = ""

    @property
    def level_name(self) -> str:
        """Human-readable level name.

        Returns:
            Level name string (e.g. ``"INFO"``).
        """
        try:
            return LogLevel(self.level).name
        except ValueError:
            return f"LEVEL_{self.level}"

    @property
    def iso_timestamp(self) -> str:
        """ISO-8601 formatted timestamp string.

        Returns:
            Timestamp string.
        """
        return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(self.timestamp))

    def to_dict(self) -> dict[str, Any]:
        """Serialize the record to a dictionary.

        Returns:
            Flat dictionary representation.
        """
        d: dict[str, Any] = {
            "timestamp": self.timestamp,
            "iso_time": self.iso_timestamp,
            "level": self.level,
            "level_name": self.level_name,
            "message": self.message,
        }
        if self.logger_name:
            d["logger"] = self.logger_name
        if self.context:
            d["context"] = self.context
        return d

    def format_text(self) -> str:
        """Format as a human-readable text line.

        Returns:
            Formatted string.
        """
        parts = [self.iso_timestamp, self.level_name.ljust(8), self.message]
        if self.logger_name:
            parts.insert(2, f"[{self.logger_name}]")
        if self.context:
            parts.append(json.dumps(self.context, default=str))
        return " ".join(parts)


# ---------------------------------------------------------------------------
# Base file logger
# ---------------------------------------------------------------------------


class BaseFileLogger:
    """Abstract base for file loggers.

    Provides level filtering, structured context attachment, and lifecycle
    management.

    Args:
        level: Minimum log level to record.
        name: Logger name included in every record.
    """

    def __init__(self, level: int | LogLevel = LogLevel.DEBUG, name: str = "") -> None:
        self._level = int(level)
        self._name = name
        self._context: dict[str, Any] = {}
        self._closed = False
        self._lock = threading.Lock()

    @property
    def level(self) -> int:
        """Current minimum log level."""
        return self._level

    @level.setter
    def level(self, value: int | LogLevel) -> None:
        self._level = int(value)

    def set_context(self, **kwargs: Any) -> None:
        """Attach persistent context fields to all future records.

        Args:
            **kwargs: Key-value pairs added to every log record's context.
        """
        self._context.update(kwargs)

    def clear_context(self) -> None:
        """Remove all persistent context fields."""
        self._context.clear()

    @contextmanager
    def context(self, **kwargs: Any) -> Generator[None, None, None]:
        """Temporarily add context fields for the duration of a block.

        Args:
            **kwargs: Context fields.

        Yields:
            Nothing. Context is removed when the block exits.
        """
        old = dict(self._context)
        self._context.update(kwargs)
        try:
            yield
        finally:
            self._context = old

    def _make_record(self, level: int, message: str, **extra: Any) -> LogRecord:
        """Create a log record, merging persistent and ad-hoc context.

        Args:
            level: Log level.
            message: Log message.
            **extra: Additional context fields for this record only.

        Returns:
            A ``LogRecord`` instance.
        """
        ctx = {**self._context, **extra}
        return LogRecord(
            timestamp=time.time(),
            level=level,
            message=message,
            context=ctx,
            logger_name=self._name,
        )

    def _should_log(self, level: int) -> bool:
        return level >= self._level and not self._closed

    # Convenience methods
    def debug(self, message: str, **extra: Any) -> None:
        """Log at DEBUG level."""
        self._log(LogLevel.DEBUG, message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        """Log at INFO level."""
        self._log(LogLevel.INFO, message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        """Log at WARNING level."""
        self._log(LogLevel.WARNING, message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        """Log at ERROR level."""
        self._log(LogLevel.ERROR, message, **extra)

    def critical(self, message: str, **extra: Any) -> None:
        """Log at CRITICAL level."""
        self._log(LogLevel.CRITICAL, message, **extra)

    def _log(self, level: int, message: str, **extra: Any) -> None:
        """Internal dispatch -- subclasses override :meth:`_write_record`."""
        if not self._should_log(level):
            return
        record = self._make_record(level, message, **extra)
        with self._lock:
            self._write_record(record)

    def _write_record(self, record: LogRecord) -> None:
        """Write a record to the backing store.  Override in subclasses."""
        raise NotImplementedError

    def close(self) -> None:
        """Close the logger and release resources.  Safe to call repeatedly."""
        self._closed = True

    @property
    def is_closed(self) -> bool:
        """Whether this logger has been closed."""
        return self._closed

    def __enter__(self) -> BaseFileLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


# ---------------------------------------------------------------------------
# Rotating file logger
# ---------------------------------------------------------------------------


class RotatingFileLogger(BaseFileLogger):
    """Plain-text file logger with size-based rotation and optional compression.

    When the active log file exceeds *max_bytes*, it is rotated.  Up to
    *backup_count* old files are kept.  If *compress* is ``True``, rotated
    files are gzip-compressed.

    Args:
        path: Path to the primary log file.
        max_bytes: Maximum file size in bytes before rotation (0 = no limit).
        backup_count: Number of rotated backups to keep.
        compress: Whether to gzip-compress rotated files.
        level: Minimum log level.
        name: Logger name.
        encoding: File encoding.

    Example::

        with RotatingFileLogger("app.log", max_bytes=5_000_000) as log:
            log.info("Application started", version="1.0")
    """

    def __init__(
        self,
        path: str | Path,
        max_bytes: int = 10_000_000,
        backup_count: int = 5,
        compress: bool = True,
        level: int | LogLevel = LogLevel.DEBUG,
        name: str = "",
        encoding: str = "utf-8",
    ) -> None:
        super().__init__(level=level, name=name)
        self._path = Path(path)
        self._max_bytes = max_bytes
        self._backup_count = backup_count
        self._compress = compress
        self._encoding = encoding

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file: TextIO = self._path.open("a", encoding=self._encoding)
        self._current_size = self._path.stat().st_size if self._path.exists() else 0

    def _write_record(self, record: LogRecord) -> None:
        """Write a formatted text line, rotating if necessary."""
        line = record.format_text() + "\n"
        encoded_len = len(line.encode(self._encoding))

        if self._max_bytes > 0 and self._current_size + encoded_len > self._max_bytes:
            self._rotate()

        self._file.write(line)
        self._file.flush()
        self._current_size += encoded_len

    def _rotate(self) -> None:
        """Rotate log files, compressing old ones if enabled."""
        self._file.close()

        # Shift existing backups
        for i in range(self._backup_count - 1, 0, -1):
            src = self._backup_path(i)
            dst = self._backup_path(i + 1)
            if src.exists():
                src.rename(dst)

        # Rotate current to .1
        first_backup = self._backup_path(1)
        if self._path.exists():
            if self._compress:
                gz_path = first_backup.with_suffix(first_backup.suffix + ".gz")
                with self._path.open("rb") as f_in, gzip.open(gz_path, "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)
                self._path.unlink()
            else:
                self._path.rename(first_backup)

        # Remove excess backups
        self._cleanup_old_backups()

        # Re-open the primary file
        self._file = self._path.open("a", encoding=self._encoding)
        self._current_size = 0

    def _backup_path(self, index: int) -> Path:
        """Return the path for a backup at a given index.

        Args:
            index: Backup index (1-based).

        Returns:
            Path to the backup file.
        """
        return self._path.with_suffix(f".{index}{self._path.suffix}")

    def _cleanup_old_backups(self) -> None:
        """Remove backup files beyond the retention limit."""
        for i in range(self._backup_count + 1, self._backup_count + 20):
            p = self._backup_path(i)
            for candidate in [p, p.with_suffix(p.suffix + ".gz")]:
                if candidate.exists():
                    candidate.unlink()

    def flush(self) -> None:
        """Flush the underlying file."""
        if not self._closed:
            self._file.flush()

    def close(self) -> None:
        """Close the log file."""
        if self._closed:
            return
        with self._lock:
            self._file.flush()
            self._file.close()
        super().close()


# ---------------------------------------------------------------------------
# CSV logger
# ---------------------------------------------------------------------------


class CSVLogger(BaseFileLogger):
    """Structured CSV logger with automatic header management.

    The first call to :meth:`log_row` establishes the column schema.
    Subsequent rows must use the same columns (extra columns are ignored,
    missing columns get empty strings).

    Args:
        path: Path to the CSV file.
        columns: Optional pre-defined column names.  If ``None``, columns
            are inferred from the first logged row.
        level: Minimum log level (applies to level-based convenience methods).
        name: Logger name.
        buffer_size: Rows to buffer before flushing to disk.
        delimiter: CSV field delimiter.

    Example::

        with CSVLogger("metrics.csv", columns=["step", "loss", "lr"]) as csv_log:
            csv_log.log_row(step=1, loss=0.5, lr=1e-3)
    """

    def __init__(
        self,
        path: str | Path,
        columns: list[str] | None = None,
        level: int | LogLevel = LogLevel.DEBUG,
        name: str = "",
        buffer_size: int = 50,
        delimiter: str = ",",
    ) -> None:
        super().__init__(level=level, name=name)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._columns: list[str] | None = columns
        self._buffer_size = buffer_size
        self._delimiter = delimiter
        self._buffer: list[dict[str, Any]] = []
        self._row_count = 0

        # If columns provided, write header immediately
        if self._columns:
            self._write_header()

    def _write_header(self) -> None:
        """Write the CSV header row."""
        with self._path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns, delimiter=self._delimiter)
            writer.writeheader()

    def log_row(self, **fields: Any) -> None:
        """Log a single row of data.

        On the first call, column names are inferred from *fields* if they
        were not provided at construction.

        Args:
            **fields: Column name-value pairs.
        """
        if self._closed:
            return

        with self._lock:
            if self._columns is None:
                self._columns = list(fields.keys())
                self._write_header()

            self._buffer.append(fields)
            self._row_count += 1

            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def log_dict(self, data: dict[str, Any]) -> None:
        """Log a row from a dictionary.

        Args:
            data: Column name-value mapping.
        """
        self.log_row(**data)

    def log_rows(self, rows: list[dict[str, Any]]) -> None:
        """Log multiple rows at once.

        Args:
            rows: List of row dictionaries.
        """
        for row in rows:
            self.log_row(**row)

    def _write_record(self, record: LogRecord) -> None:
        """Write a log record as a CSV row (level-based methods)."""
        data = {
            "timestamp": record.iso_timestamp,
            "level": record.level_name,
            "message": record.message,
        }
        data.update(record.context)
        self.log_row(**data)

    def _flush_buffer(self) -> None:
        """Flush buffered rows to disk."""
        if not self._buffer or self._columns is None:
            return
        with self._path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=self._columns,
                delimiter=self._delimiter,
                extrasaction="ignore",
            )
            for row in self._buffer:
                # Fill missing columns with empty string
                full_row = {c: row.get(c, "") for c in self._columns}
                writer.writerow(full_row)
        self._buffer.clear()

    def flush(self) -> None:
        """Flush buffered rows to disk."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Flush and close."""
        if self._closed:
            return
        with self._lock:
            self._flush_buffer()
        super().close()

    @property
    def row_count(self) -> int:
        """Total number of rows logged."""
        return self._row_count

    @property
    def columns(self) -> list[str] | None:
        """Current column schema, or ``None`` if not yet established."""
        return list(self._columns) if self._columns else None

    def read_all(self) -> list[dict[str, str]]:
        """Read all rows back from the CSV file.

        Returns:
            List of row dictionaries.
        """
        self.flush()
        if not self._path.exists():
            return []
        with self._path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter=self._delimiter)
            return list(reader)


# ---------------------------------------------------------------------------
# JSONL logger
# ---------------------------------------------------------------------------


class JSONLLogger(BaseFileLogger):
    """JSON-Lines (newline-delimited JSON) logger.

    Each call appends one JSON object per line.  Supports buffered writes and
    atomic file operations (via write-to-temp then rename).

    Args:
        path: Path to the ``.jsonl`` file.
        level: Minimum log level.
        name: Logger name.
        buffer_size: Lines to buffer before flushing.
        atomic_writes: If ``True``, flush operations write to a temporary file
            and atomically rename to the target path.  Provides crash safety
            at a slight performance cost.
        ensure_ascii: JSON ``ensure_ascii`` flag.

    Example::

        with JSONLLogger("events.jsonl") as jl:
            jl.log_entry({"event": "start", "time": 0.0})
            jl.info("Simulation started")
    """

    def __init__(
        self,
        path: str | Path,
        level: int | LogLevel = LogLevel.DEBUG,
        name: str = "",
        buffer_size: int = 100,
        atomic_writes: bool = False,
        ensure_ascii: bool = False,
    ) -> None:
        super().__init__(level=level, name=name)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._buffer_size = buffer_size
        self._atomic = atomic_writes
        self._ensure_ascii = ensure_ascii
        self._buffer: list[str] = []
        self._entry_count = 0

        # Open in append mode
        if not self._atomic:
            self._file: TextIO | None = self._path.open("a", encoding="utf-8")
        else:
            self._file = None
            # Ensure file exists
            if not self._path.exists():
                self._path.touch()

    def log_entry(self, data: dict[str, Any]) -> None:
        """Append a JSON object as a single line.

        Args:
            data: Dictionary to serialize.
        """
        if self._closed:
            return
        line = json.dumps(data, default=str, ensure_ascii=self._ensure_ascii) + "\n"
        with self._lock:
            self._buffer.append(line)
            self._entry_count += 1
            if len(self._buffer) >= self._buffer_size:
                self._flush_buffer()

    def log_entries(self, entries: list[dict[str, Any]]) -> None:
        """Append multiple JSON objects.

        Args:
            entries: List of dictionaries.
        """
        for entry in entries:
            self.log_entry(entry)

    def _write_record(self, record: LogRecord) -> None:
        """Write a ``LogRecord`` as a JSON line."""
        self.log_entry(record.to_dict())

    def _flush_buffer(self) -> None:
        """Flush the in-memory buffer to disk."""
        if not self._buffer:
            return

        if self._atomic:
            self._atomic_append()
        else:
            if self._file is not None and not self._file.closed:
                self._file.writelines(self._buffer)
                self._file.flush()
        self._buffer.clear()

    def _atomic_append(self) -> None:
        """Append buffered lines using a temporary file for crash safety.

        Reads existing content, appends new lines, writes to a temp file,
        and atomically renames.
        """
        existing = ""
        if self._path.exists():
            existing = self._path.read_text(encoding="utf-8")

        new_content = existing + "".join(self._buffer)

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._path.parent),
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as tmp_f:
                tmp_f.write(new_content)
            os.replace(tmp_path, str(self._path))
        except Exception:
            # Clean up temp file on error
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def flush(self) -> None:
        """Flush buffered entries to disk."""
        with self._lock:
            self._flush_buffer()

    def close(self) -> None:
        """Flush and close."""
        if self._closed:
            return
        with self._lock:
            self._flush_buffer()
            if self._file is not None and not self._file.closed:
                self._file.close()
        super().close()

    @property
    def entry_count(self) -> int:
        """Total number of entries logged."""
        return self._entry_count

    def read_all(self) -> list[dict[str, Any]]:
        """Read all entries from the JSONL file.

        Returns:
            List of parsed dictionaries.
        """
        self.flush()
        if not self._path.exists():
            return []
        entries = []
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
        return entries

    def iter_entries(self) -> Generator[dict[str, Any], None, None]:
        """Iterate over entries from the JSONL file.

        Yields:
            Parsed dictionaries.
        """
        self.flush()
        if not self._path.exists():
            return
        with self._path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


# ---------------------------------------------------------------------------
# HDF5 logger (stub)
# ---------------------------------------------------------------------------


class HDF5Logger(BaseFileLogger):
    """Stub for HDF5-based logging.

    Requires the ``h5py`` package.  If ``h5py`` is not installed, the logger
    degrades to a no-op with a warning.

    This class provides an interface for logging numerical arrays and metadata
    to HDF5 files.  It is intended for high-volume, high-dimensional data
    such as trajectories and sensor readings.

    Args:
        path: Path to the ``.h5`` file.
        level: Minimum log level.
        name: Logger name.
        compression: HDF5 compression filter (e.g. ``"gzip"``).
        compression_level: Compression level (0-9 for gzip).

    Example::

        with HDF5Logger("data.h5") as h5:
            h5.log_array("positions", positions_array)
            h5.log_scalar_series("rewards", reward_array)
    """

    def __init__(
        self,
        path: str | Path,
        level: int | LogLevel = LogLevel.DEBUG,
        name: str = "",
        compression: str = "gzip",
        compression_level: int = 4,
    ) -> None:
        super().__init__(level=level, name=name)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._compression = compression
        self._compression_level = compression_level
        self._h5py_available = False
        self._file_handle: Any = None

        try:
            import h5py

            self._h5py_available = True
            self._h5 = h5py
            self._file_handle = h5py.File(str(self._path), "a")
        except ImportError:
            logging.getLogger(__name__).warning(
                "h5py is not installed. HDF5Logger will operate as a no-op. "
                "Install with: pip install h5py"
            )

    def log_array(
        self,
        name: str,
        data: np.ndarray,
        group: str | None = None,
        attrs: dict[str, Any] | None = None,
    ) -> None:
        """Store a numpy array as an HDF5 dataset.

        Args:
            name: Dataset name.
            data: Array to store.
            group: Optional HDF5 group path.
            attrs: Optional attributes to attach to the dataset.
        """
        if not self._h5py_available or self._file_handle is None:
            return
        with self._lock:
            target = self._file_handle
            if group:
                target = self._file_handle.require_group(group)

            if name in target:
                del target[name]
            ds = target.create_dataset(
                name,
                data=data,
                compression=self._compression,
                compression_opts=self._compression_level,
            )
            if attrs:
                for k, v in attrs.items():
                    ds.attrs[k] = v

    def log_scalar_series(
        self,
        name: str,
        data: np.ndarray,
        group: str | None = None,
    ) -> None:
        """Store a 1-D series of scalar values.

        Args:
            name: Dataset name.
            data: 1-D array of values.
            group: Optional group path.
        """
        arr = np.asarray(data, dtype=np.float64).ravel()
        self.log_array(name, arr, group=group)

    def append_to_dataset(
        self,
        name: str,
        data: np.ndarray,
        group: str | None = None,
        axis: int = 0,
    ) -> None:
        """Append data to an existing (or new) resizable dataset.

        Creates the dataset on first call with ``maxshape=None`` on the
        append axis to allow unlimited growth.

        Args:
            name: Dataset name.
            data: Array to append.
            group: Optional group path.
            axis: Axis along which to append.
        """
        if not self._h5py_available or self._file_handle is None:
            return
        with self._lock:
            target = self._file_handle
            if group:
                target = self._file_handle.require_group(group)

            data = np.asarray(data)
            if name not in target:
                maxshape = list(data.shape)
                maxshape[axis] = None  # unlimited
                target.create_dataset(
                    name,
                    data=data,
                    maxshape=tuple(maxshape),
                    compression=self._compression,
                    compression_opts=self._compression_level,
                )
            else:
                ds = target[name]
                old_shape = list(ds.shape)
                new_shape = list(old_shape)
                new_shape[axis] += data.shape[axis]
                ds.resize(new_shape)
                slices = [slice(None)] * len(old_shape)
                slices[axis] = slice(old_shape[axis], new_shape[axis])
                ds[tuple(slices)] = data

    def log_metadata(self, group: str, attrs: dict[str, Any]) -> None:
        """Attach metadata attributes to an HDF5 group.

        Args:
            group: Group path.
            attrs: Attribute key-value pairs.
        """
        if not self._h5py_available or self._file_handle is None:
            return
        with self._lock:
            grp = self._file_handle.require_group(group)
            for k, v in attrs.items():
                if isinstance(v, (str, int, float, bool)):
                    grp.attrs[k] = v
                else:
                    grp.attrs[k] = json.dumps(v, default=str)

    def _write_record(self, record: LogRecord) -> None:
        """Write a log record.  HDF5Logger stores structured log records
        as JSON strings appended to a ``_logs`` dataset."""
        if not self._h5py_available or self._file_handle is None:
            return
        line = json.dumps(record.to_dict(), default=str)
        # Store as a simple attribute for simplicity
        log_grp = self._file_handle.require_group("_logs")
        idx = len(log_grp.attrs)
        log_grp.attrs[f"entry_{idx}"] = line

    def flush(self) -> None:
        """Flush the HDF5 file."""
        if self._file_handle is not None:
            self._file_handle.flush()

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._closed:
            return
        with self._lock:
            if self._file_handle is not None:
                self._file_handle.flush()
                self._file_handle.close()
                self._file_handle = None
        super().close()

    @property
    def available(self) -> bool:
        """Whether ``h5py`` is installed and the file is open."""
        return self._h5py_available and self._file_handle is not None


# ---------------------------------------------------------------------------
# Composite logger
# ---------------------------------------------------------------------------


class CompositeFileLogger:
    """Fan-out logger that writes to multiple file loggers simultaneously.

    Args:
        loggers: Variable number of ``BaseFileLogger`` instances.

    Example::

        composite = CompositeFileLogger(
            RotatingFileLogger("app.log"),
            JSONLLogger("structured.jsonl"),
        )
        composite.info("Application started")
    """

    def __init__(self, *loggers: BaseFileLogger) -> None:
        self._loggers = list(loggers)

    def add_logger(self, logger_inst: BaseFileLogger) -> None:
        """Add another logger to the composite.

        Args:
            logger_inst: Logger to add.
        """
        self._loggers.append(logger_inst)

    def debug(self, message: str, **extra: Any) -> None:
        """Log at DEBUG level to all loggers."""
        for lg in self._loggers:
            lg.debug(message, **extra)

    def info(self, message: str, **extra: Any) -> None:
        """Log at INFO level to all loggers."""
        for lg in self._loggers:
            lg.info(message, **extra)

    def warning(self, message: str, **extra: Any) -> None:
        """Log at WARNING level to all loggers."""
        for lg in self._loggers:
            lg.warning(message, **extra)

    def error(self, message: str, **extra: Any) -> None:
        """Log at ERROR level to all loggers."""
        for lg in self._loggers:
            lg.error(message, **extra)

    def critical(self, message: str, **extra: Any) -> None:
        """Log at CRITICAL level to all loggers."""
        for lg in self._loggers:
            lg.critical(message, **extra)

    def set_context(self, **kwargs: Any) -> None:
        """Set context on all child loggers."""
        for lg in self._loggers:
            lg.set_context(**kwargs)

    def flush(self) -> None:
        """Flush all child loggers."""
        for lg in self._loggers:
            if hasattr(lg, "flush"):
                lg.flush()

    def close(self) -> None:
        """Close all child loggers."""
        for lg in self._loggers:
            lg.close()

    def __enter__(self) -> CompositeFileLogger:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
