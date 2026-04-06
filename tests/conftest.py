from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
_MPLCONFIGDIR = Path(tempfile.mkdtemp(prefix="navirl-mplconfig-"))
atexit.register(shutil.rmtree, _MPLCONFIGDIR, ignore_errors=True)
os.environ["MPLCONFIGDIR"] = str(_MPLCONFIGDIR)
os.environ["NAVIRL_MPLCONFIGDIR"] = str(_MPLCONFIGDIR)

import matplotlib
import pytest

# Ensure rendering tests are stable in headless environments.
matplotlib.use("Agg", force=True)


@pytest.hookimpl(tryfirst=True)
def pytest_sessionstart(session) -> None:
    _ = session
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
