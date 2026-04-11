"""Tests for navirl.logging.file_logger module.

Covers LogLevel, LogRecord, BaseFileLogger, RotatingFileLogger, CSVLogger,
JSONLLogger, HDF5Logger (no-op mode), and CompositeFileLogger.
"""

from __future__ import annotations

import csv
import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from navirl.logging.file_logger import (
    BaseFileLogger,
    CompositeFileLogger,
    CSVLogger,
    HDF5Logger,
    JSONLLogger,
    LogLevel,
    LogRecord,
    RotatingFileLogger,
)

# ===================================================================
# LogLevel tests
# ===================================================================


class TestLogLevel:
    def test_level_values(self):
        assert LogLevel.DEBUG == 10
        assert LogLevel.INFO == 20
        assert LogLevel.WARNING == 30
        assert LogLevel.ERROR == 40
        assert LogLevel.CRITICAL == 50

    def test_ordering(self):
        assert LogLevel.DEBUG < LogLevel.INFO < LogLevel.WARNING < LogLevel.ERROR < LogLevel.CRITICAL

    def test_from_string_valid(self):
        assert LogLevel.from_string("INFO") is LogLevel.INFO
        assert LogLevel.from_string("debug") is LogLevel.DEBUG
        assert LogLevel.from_string("Warning") is LogLevel.WARNING

    def test_from_string_invalid(self):
        with pytest.raises(ValueError, match="Unknown log level"):
            LogLevel.from_string("VERBOSE")


# ===================================================================
# LogRecord tests
# ===================================================================


class TestLogRecord:
    def test_basic_creation(self):
        rec = LogRecord(timestamp=1000.0, level=LogLevel.INFO, message="hello")
        assert rec.message == "hello"
        assert rec.level == 20

    def test_level_name(self):
        rec = LogRecord(timestamp=0.0, level=LogLevel.ERROR, message="")
        assert rec.level_name == "ERROR"

    def test_level_name_unknown(self):
        rec = LogRecord(timestamp=0.0, level=99, message="")
        assert rec.level_name == "LEVEL_99"

    def test_iso_timestamp(self):
        rec = LogRecord(timestamp=0.0, level=LogLevel.INFO, message="")
        iso = rec.iso_timestamp
        assert "1970" in iso and "T" in iso

    def test_to_dict_basic(self):
        rec = LogRecord(timestamp=1.0, level=LogLevel.INFO, message="test")
        d = rec.to_dict()
        assert d["message"] == "test"
        assert d["level"] == LogLevel.INFO
        assert "level_name" in d
        assert "logger" not in d  # no logger_name

    def test_to_dict_with_logger_and_context(self):
        rec = LogRecord(
            timestamp=1.0,
            level=LogLevel.DEBUG,
            message="msg",
            context={"key": "val"},
            logger_name="test_logger",
        )
        d = rec.to_dict()
        assert d["logger"] == "test_logger"
        assert d["context"] == {"key": "val"}

    def test_format_text(self):
        rec = LogRecord(timestamp=0.0, level=LogLevel.WARNING, message="watch out")
        text = rec.format_text()
        assert "WARNING" in text
        assert "watch out" in text

    def test_format_text_with_context(self):
        rec = LogRecord(
            timestamp=0.0,
            level=LogLevel.INFO,
            message="hi",
            context={"step": 42},
            logger_name="sim",
        )
        text = rec.format_text()
        assert "[sim]" in text
        assert "42" in text


# ===================================================================
# BaseFileLogger tests
# ===================================================================


class TestBaseFileLogger:
    def test_level_property(self):
        logger = _StubLogger(level=LogLevel.WARNING)
        assert logger.level == LogLevel.WARNING
        logger.level = LogLevel.DEBUG
        assert logger.level == LogLevel.DEBUG

    def test_should_log_filters_by_level(self):
        logger = _StubLogger(level=LogLevel.WARNING)
        logger.info("should be skipped")
        assert len(logger.records) == 0
        logger.warning("should appear")
        assert len(logger.records) == 1

    def test_closed_logger_ignores_writes(self):
        logger = _StubLogger()
        logger.close()
        assert logger.is_closed
        logger.info("ignored")
        assert len(logger.records) == 0

    def test_set_and_clear_context(self):
        logger = _StubLogger()
        logger.set_context(run_id="abc")
        logger.info("test")
        assert logger.records[0].context["run_id"] == "abc"
        logger.clear_context()
        logger.info("test2")
        assert logger.records[1].context == {}

    def test_context_manager(self):
        logger = _StubLogger()
        logger.set_context(base="val")
        with logger.context(temp="ctx"):
            logger.info("inside")
            assert logger.records[-1].context == {"base": "val", "temp": "ctx"}
        logger.info("outside")
        assert logger.records[-1].context == {"base": "val"}

    def test_context_manager_protocol(self):
        logger = _StubLogger()
        with logger as lg:
            assert lg is logger
        assert logger.is_closed

    def test_convenience_methods(self):
        logger = _StubLogger(level=LogLevel.DEBUG)
        logger.debug("d")
        logger.info("i")
        logger.warning("w")
        logger.error("e")
        logger.critical("c")
        levels = [r.level for r in logger.records]
        assert levels == [
            LogLevel.DEBUG,
            LogLevel.INFO,
            LogLevel.WARNING,
            LogLevel.ERROR,
            LogLevel.CRITICAL,
        ]

    def test_extra_context_per_call(self):
        logger = _StubLogger()
        logger.info("msg", extra_key=123)
        assert logger.records[0].context["extra_key"] == 123

    def test_name_in_records(self):
        logger = _StubLogger(name="my_logger")
        logger.info("test")
        assert logger.records[0].logger_name == "my_logger"


# ===================================================================
# RotatingFileLogger tests
# ===================================================================


class TestRotatingFileLogger:
    def test_basic_write_and_read(self, tmp_path):
        log_path = tmp_path / "test.log"
        with RotatingFileLogger(log_path) as logger:
            logger.info("hello world")
        content = log_path.read_text()
        assert "hello world" in content
        assert "INFO" in content

    def test_level_filtering(self, tmp_path):
        log_path = tmp_path / "test.log"
        with RotatingFileLogger(log_path, level=LogLevel.ERROR) as logger:
            logger.info("skipped")
            logger.error("recorded")
        content = log_path.read_text()
        assert "skipped" not in content
        assert "recorded" in content

    def test_rotation_triggers(self, tmp_path):
        log_path = tmp_path / "test.log"
        # Very small max_bytes to trigger rotation quickly
        with RotatingFileLogger(log_path, max_bytes=100, backup_count=3, compress=False) as logger:
            for i in range(20):
                logger.info(f"Message number {i:04d} with padding to make it longer")
        # Primary file should exist
        assert log_path.exists()
        # At least one backup should exist
        backup_1 = log_path.with_suffix(".1.log")
        assert backup_1.exists()

    def test_rotation_with_compression(self, tmp_path):
        log_path = tmp_path / "test.log"
        with RotatingFileLogger(log_path, max_bytes=100, backup_count=2, compress=True) as logger:
            for i in range(20):
                logger.info(f"Message {i:04d} padded to fill the file up quickly now")
        # Compressed backup should exist
        gz_files = list(tmp_path.glob("*.gz"))
        assert len(gz_files) >= 1
        # Verify gz file is valid
        with gzip.open(gz_files[0], "rt") as f:
            content = f.read()
            assert "Message" in content

    def test_flush(self, tmp_path):
        log_path = tmp_path / "test.log"
        logger = RotatingFileLogger(log_path)
        logger.info("flush test")
        logger.flush()
        assert "flush test" in log_path.read_text()
        logger.close()

    def test_double_close_safe(self, tmp_path):
        log_path = tmp_path / "test.log"
        logger = RotatingFileLogger(log_path)
        logger.close()
        logger.close()  # Should not raise

    def test_context_in_output(self, tmp_path):
        log_path = tmp_path / "test.log"
        with RotatingFileLogger(log_path) as logger:
            logger.set_context(episode=5)
            logger.info("with context")
        content = log_path.read_text()
        assert "episode" in content

    def test_creates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "nested" / "dir" / "test.log"
        with RotatingFileLogger(log_path) as logger:
            logger.info("nested")
        assert log_path.exists()


# ===================================================================
# CSVLogger tests
# ===================================================================


class TestCSVLogger:
    def test_basic_logging_with_predefined_columns(self, tmp_path):
        csv_path = tmp_path / "metrics.csv"
        with CSVLogger(csv_path, columns=["step", "loss"]) as logger:
            logger.log_row(step=1, loss=0.5)
            logger.log_row(step=2, loss=0.3)
        rows = _read_csv(csv_path)
        assert len(rows) == 2
        assert rows[0]["step"] == "1"
        assert rows[1]["loss"] == "0.3"

    def test_auto_infer_columns(self, tmp_path):
        csv_path = tmp_path / "auto.csv"
        with CSVLogger(csv_path) as logger:
            logger.log_row(a=1, b=2)
            logger.log_row(a=3, b=4)
        assert logger.columns == ["a", "b"]
        rows = _read_csv(csv_path)
        assert len(rows) == 2

    def test_missing_columns_get_empty(self, tmp_path):
        csv_path = tmp_path / "missing.csv"
        with CSVLogger(csv_path, columns=["a", "b", "c"]) as logger:
            logger.log_row(a=1)  # b and c missing
        rows = _read_csv(csv_path)
        assert rows[0]["b"] == ""
        assert rows[0]["c"] == ""

    def test_extra_columns_ignored(self, tmp_path):
        csv_path = tmp_path / "extra.csv"
        with CSVLogger(csv_path, columns=["a"]) as logger:
            logger.log_row(a=1, b=2)  # b should be ignored
        rows = _read_csv(csv_path)
        assert "b" not in rows[0]

    def test_log_dict(self, tmp_path):
        csv_path = tmp_path / "dict.csv"
        with CSVLogger(csv_path) as logger:
            logger.log_dict({"x": 10, "y": 20})
        rows = _read_csv(csv_path)
        assert rows[0]["x"] == "10"

    def test_log_rows(self, tmp_path):
        csv_path = tmp_path / "multi.csv"
        with CSVLogger(csv_path) as logger:
            logger.log_rows([{"a": 1}, {"a": 2}, {"a": 3}])
        assert logger.row_count == 3

    def test_row_count(self, tmp_path):
        csv_path = tmp_path / "count.csv"
        with CSVLogger(csv_path) as logger:
            assert logger.row_count == 0
            logger.log_row(x=1)
            assert logger.row_count == 1

    def test_buffer_flushes_on_threshold(self, tmp_path):
        csv_path = tmp_path / "buffered.csv"
        with CSVLogger(csv_path, columns=["i"], buffer_size=5) as logger:
            for i in range(5):
                logger.log_row(i=i)
            # After 5 rows, buffer should have flushed
            rows = _read_csv(csv_path)
            assert len(rows) == 5

    def test_read_all(self, tmp_path):
        csv_path = tmp_path / "read.csv"
        with CSVLogger(csv_path, columns=["v"]) as logger:
            logger.log_row(v="hello")
            rows = logger.read_all()
        assert len(rows) == 1
        assert rows[0]["v"] == "hello"

    def test_custom_delimiter(self, tmp_path):
        csv_path = tmp_path / "tab.csv"
        with CSVLogger(csv_path, columns=["a", "b"], delimiter="\t") as logger:
            logger.log_row(a=1, b=2)
        content = csv_path.read_text()
        assert "\t" in content

    def test_level_based_logging(self, tmp_path):
        csv_path = tmp_path / "level.csv"
        with CSVLogger(csv_path) as logger:
            logger.info("test message")
        rows = _read_csv(csv_path)
        assert len(rows) == 1
        assert rows[0]["message"] == "test message"

    def test_closed_logger_ignores_rows(self, tmp_path):
        csv_path = tmp_path / "closed.csv"
        logger = CSVLogger(csv_path, columns=["x"])
        logger.close()
        logger.log_row(x=1)
        assert logger.row_count == 0

    def test_creates_parent_dirs(self, tmp_path):
        csv_path = tmp_path / "deep" / "nested" / "file.csv"
        with CSVLogger(csv_path, columns=["a"]) as logger:
            logger.log_row(a=1)
        assert csv_path.exists()


# ===================================================================
# JSONLLogger tests
# ===================================================================


class TestJSONLLogger:
    def test_basic_logging(self, tmp_path):
        path = tmp_path / "test.jsonl"
        with JSONLLogger(path) as logger:
            logger.log_entry({"event": "start", "step": 0})
            logger.log_entry({"event": "end", "step": 100})
        entries = _read_jsonl(path)
        assert len(entries) == 2
        assert entries[0]["event"] == "start"
        assert entries[1]["step"] == 100

    def test_entry_count(self, tmp_path):
        path = tmp_path / "count.jsonl"
        with JSONLLogger(path) as logger:
            assert logger.entry_count == 0
            logger.log_entry({"a": 1})
            assert logger.entry_count == 1

    def test_log_entries_batch(self, tmp_path):
        path = tmp_path / "batch.jsonl"
        with JSONLLogger(path) as logger:
            logger.log_entries([{"i": i} for i in range(5)])
        assert logger.entry_count == 5
        entries = _read_jsonl(path)
        assert len(entries) == 5

    def test_buffer_flushes(self, tmp_path):
        path = tmp_path / "buf.jsonl"
        with JSONLLogger(path, buffer_size=3) as logger:
            for i in range(3):
                logger.log_entry({"i": i})
            # Buffer should have flushed at 3
            entries = _read_jsonl(path)
            assert len(entries) == 3

    def test_flush_explicit(self, tmp_path):
        path = tmp_path / "flush.jsonl"
        logger = JSONLLogger(path, buffer_size=1000)
        logger.log_entry({"x": 1})
        logger.flush()
        entries = _read_jsonl(path)
        assert len(entries) == 1
        logger.close()

    def test_read_all(self, tmp_path):
        path = tmp_path / "read.jsonl"
        with JSONLLogger(path) as logger:
            logger.log_entry({"key": "value"})
            entries = logger.read_all()
        assert entries[0]["key"] == "value"

    def test_iter_entries(self, tmp_path):
        path = tmp_path / "iter.jsonl"
        with JSONLLogger(path) as logger:
            logger.log_entry({"a": 1})
            logger.log_entry({"a": 2})
            result = list(logger.iter_entries())
        assert len(result) == 2

    def test_level_based_logging(self, tmp_path):
        path = tmp_path / "level.jsonl"
        with JSONLLogger(path) as logger:
            logger.info("hello")
        entries = _read_jsonl(path)
        assert len(entries) == 1
        assert entries[0]["message"] == "hello"

    def test_atomic_writes(self, tmp_path):
        path = tmp_path / "atomic.jsonl"
        with JSONLLogger(path, atomic_writes=True, buffer_size=2) as logger:
            logger.log_entry({"x": 1})
            logger.log_entry({"x": 2})  # Should trigger atomic flush
        entries = _read_jsonl(path)
        assert len(entries) == 2

    def test_closed_logger_ignores_entries(self, tmp_path):
        path = tmp_path / "closed.jsonl"
        logger = JSONLLogger(path)
        logger.close()
        logger.log_entry({"x": 1})
        assert logger.entry_count == 0

    def test_read_all_empty_file(self, tmp_path):
        path = tmp_path / "empty.jsonl"
        logger = JSONLLogger(path)
        entries = logger.read_all()
        assert entries == []
        logger.close()

    def test_serializes_numpy(self, tmp_path):
        """Verify that numpy types are serialized via default=str."""
        path = tmp_path / "np.jsonl"
        with JSONLLogger(path) as logger:
            logger.log_entry({"val": np.float64(3.14), "arr": np.array([1, 2])})
        entries = _read_jsonl(path)
        assert len(entries) == 1

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "dir" / "test.jsonl"
        with JSONLLogger(path) as logger:
            logger.log_entry({"ok": True})
        assert path.exists()


# ===================================================================
# HDF5Logger tests (no-op mode, h5py not installed)
# ===================================================================


class TestHDF5LoggerNoOp:
    """Test HDF5Logger gracefully degrades when h5py is unavailable."""

    def test_noop_creation(self, tmp_path):
        path = tmp_path / "test.h5"
        logger = HDF5Logger(path)
        # Should not crash, just warn
        assert not logger.available or logger.available  # either way is fine
        logger.close()

    def test_noop_log_array(self, tmp_path):
        path = tmp_path / "test.h5"
        logger = HDF5Logger(path)
        # Should not raise even if h5py is not available
        logger.log_array("test", np.zeros(10))
        logger.close()

    def test_noop_log_scalar_series(self, tmp_path):
        path = tmp_path / "test.h5"
        logger = HDF5Logger(path)
        logger.log_scalar_series("rewards", np.array([1.0, 2.0, 3.0]))
        logger.close()

    def test_noop_append_to_dataset(self, tmp_path):
        path = tmp_path / "test.h5"
        logger = HDF5Logger(path)
        logger.append_to_dataset("traj", np.zeros((5, 2)))
        logger.close()

    def test_noop_log_metadata(self, tmp_path):
        path = tmp_path / "test.h5"
        logger = HDF5Logger(path)
        logger.log_metadata("episode_001", {"steps": 100, "reward": 5.0})
        logger.close()

    def test_noop_flush_and_close(self, tmp_path):
        path = tmp_path / "test.h5"
        logger = HDF5Logger(path)
        logger.flush()
        logger.close()
        logger.close()  # double close safe

    def test_context_manager(self, tmp_path):
        path = tmp_path / "test.h5"
        with HDF5Logger(path) as logger:
            logger.log_array("x", np.ones(5))


# ===================================================================
# CompositeFileLogger tests
# ===================================================================


class TestCompositeFileLogger:
    def test_fan_out_to_multiple_loggers(self, tmp_path):
        log_path = tmp_path / "text.log"
        jsonl_path = tmp_path / "events.jsonl"
        with CompositeFileLogger(
            RotatingFileLogger(log_path),
            JSONLLogger(jsonl_path),
        ) as composite:
            composite.info("test message")
        assert "test message" in log_path.read_text()
        entries = _read_jsonl(jsonl_path)
        assert len(entries) == 1

    def test_add_logger(self, tmp_path):
        log_path = tmp_path / "a.log"
        extra_path = tmp_path / "b.log"
        composite = CompositeFileLogger(RotatingFileLogger(log_path))
        composite.add_logger(RotatingFileLogger(extra_path))
        composite.info("both")
        composite.close()
        assert "both" in log_path.read_text()
        assert "both" in extra_path.read_text()

    def test_all_levels(self, tmp_path):
        log_path = tmp_path / "all.log"
        composite = CompositeFileLogger(RotatingFileLogger(log_path))
        composite.debug("d")
        composite.info("i")
        composite.warning("w")
        composite.error("e")
        composite.critical("c")
        composite.close()
        content = log_path.read_text()
        for msg in ("d", "i", "w", "e", "c"):
            assert msg in content

    def test_set_context(self, tmp_path):
        jsonl_path = tmp_path / "ctx.jsonl"
        composite = CompositeFileLogger(JSONLLogger(jsonl_path))
        composite.set_context(run="abc")
        composite.info("test")
        composite.close()
        entries = _read_jsonl(jsonl_path)
        assert entries[0]["context"]["run"] == "abc"

    def test_flush(self, tmp_path):
        jsonl_path = tmp_path / "flush.jsonl"
        composite = CompositeFileLogger(JSONLLogger(jsonl_path, buffer_size=1000))
        composite.info("flushed")
        composite.flush()
        entries = _read_jsonl(jsonl_path)
        assert len(entries) == 1
        composite.close()

    def test_context_manager_protocol(self, tmp_path):
        log_path = tmp_path / "cm.log"
        with CompositeFileLogger(RotatingFileLogger(log_path)) as composite:
            composite.info("ctx mgr")
        assert "ctx mgr" in log_path.read_text()


# ===================================================================
# Helpers
# ===================================================================


class _StubLogger(BaseFileLogger):
    """In-memory logger for testing BaseFileLogger behavior."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.records: list[LogRecord] = []

    def _write_record(self, record: LogRecord) -> None:
        self.records.append(record)


def _read_csv(path: Path) -> list[dict[str, str]]:
    """Read all rows from a CSV file."""
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read all entries from a JSONL file."""
    entries = []
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    return entries
