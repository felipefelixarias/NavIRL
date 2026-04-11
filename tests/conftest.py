from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")


def _make_mplconfigdir() -> Path:
    """Create a per-process matplotlib config directory.

    Each process (controller and every xdist worker) gets its own directory so
    that atexit cleanup in one process does not remove the directory while
    another process still needs it.
    """
    d = Path(tempfile.mkdtemp(prefix="navirl-mplconfig-"))
    atexit.register(shutil.rmtree, d, ignore_errors=True)
    os.environ["MPLCONFIGDIR"] = str(d)
    os.environ["NAVIRL_MPLCONFIGDIR"] = str(d)
    return d


_MPLCONFIGDIR = _make_mplconfigdir()

import matplotlib
import pytest

# Ensure rendering tests are stable in headless environments.
matplotlib.use("Agg", force=True)


def _is_xdist_worker(config) -> bool:
    """Return True if running inside a pytest-xdist worker process."""
    return hasattr(config, "workerinput")


def pytest_configure(config) -> None:
    """Give each xdist worker its own matplotlib config directory.

    Module-level ``_make_mplconfigdir()`` only runs once in the controller.
    Forked workers inherit the same path and atexit handler, so the first
    worker to exit would delete the directory for all others.  Re-creating
    the directory here ensures each worker has an independent copy.
    """
    if _is_xdist_worker(config):
        _make_mplconfigdir()


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session) -> None:
    # Only prune artifact directories from the controller process (or when
    # running without xdist). Workers inherit the cleaned state and should not
    # race on directory removal.
    if _is_xdist_worker(session.config):
        return

    from navirl.artifacts import prune_old_run_dirs, resolve_retention_hours

    ttl_hours = resolve_retention_hours(
        None,
        env_var="NAVIRL_TEST_ARTIFACT_TTL_HOURS",
        default_hours=24.0,
    )
    repo_root = Path.cwd()
    prune_old_run_dirs(repo_root / "logs", ttl_hours=ttl_hours, keep_latest=4)
    prune_old_run_dirs(
        repo_root / "out" / "tune",
        ttl_hours=ttl_hours,
        prefixes=("tune_",),
        keep_latest=2,
    )
    prune_old_run_dirs(
        repo_root / "out" / "verify",
        ttl_hours=ttl_hours,
        keep_latest=2,
    )
