from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours


def _touch_dir(path: Path, mtime: float) -> None:
    path.mkdir(parents=True, exist_ok=True)
    os.utime(path, (mtime, mtime))


def test_prune_old_run_dirs_respects_prefix_and_ttl(tmp_path: Path):
    now = time.time()
    stale = tmp_path / "tune_old"
    fresh = tmp_path / "tune_new"
    other = tmp_path / "logs_old"

    _touch_dir(stale, now - 7200.0)
    _touch_dir(fresh, now - 300.0)
    _touch_dir(other, now - 7200.0)

    removed = prune_old_run_dirs(tmp_path, ttl_hours=1.0, prefixes=("tune_",), keep_latest=0)

    assert stale in removed
    assert not stale.exists()
    assert fresh.exists()
    assert other.exists()


def test_prune_old_run_dirs_keep_latest(tmp_path: Path):
    now = time.time()
    first = tmp_path / "tune_a"
    second = tmp_path / "tune_b"
    third = tmp_path / "tune_c"

    _touch_dir(first, now - 10800.0)
    _touch_dir(second, now - 7200.0)
    _touch_dir(third, now - 3600.0)

    removed = prune_old_run_dirs(tmp_path, ttl_hours=0.5, prefixes=("tune_",), keep_latest=1)

    assert first in removed
    assert second in removed
    assert third.exists()


def test_resolve_retention_hours_precedence(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("NAVIRL_TEST_RETENTION", "12")
    assert resolve_retention_hours(None, env_var="NAVIRL_TEST_RETENTION", default_hours=4.0) == 12.0
    assert resolve_retention_hours(2.5, env_var="NAVIRL_TEST_RETENTION", default_hours=4.0) == 2.5
