from __future__ import annotations

import os
import shutil
import time
from pathlib import Path


def resolve_retention_hours(
    requested_hours: float | None,
    *,
    env_var: str,
    default_hours: float | None,
) -> float | None:
    """Resolve retention hours from CLI value, env var, or default."""
    if requested_hours is not None:
        hours = float(requested_hours)
        if hours < 0:
            raise ValueError("retention hours must be >= 0")
        return hours

    raw = os.getenv(env_var, "").strip()
    if not raw:
        return default_hours

    try:
        hours = float(raw)
    except ValueError as exc:
        raise ValueError(f"{env_var} must be a number of hours, got {raw!r}") from exc

    if hours < 0:
        raise ValueError(f"{env_var} must be >= 0")
    return hours


def prune_old_run_dirs(
    root: str | Path,
    *,
    ttl_hours: float | None,
    prefixes: tuple[str, ...] = (),
    keep_latest: int = 0,
) -> list[Path]:
    """Delete stale run directories under root, keeping newest entries if requested."""
    if ttl_hours is None or ttl_hours <= 0:
        return []

    root_path = Path(root)
    if not root_path.exists() or not root_path.is_dir():
        return []

    candidates: list[tuple[float, Path]] = []
    for child in root_path.iterdir():
        if not child.is_dir() or child.is_symlink():
            continue
        if prefixes and not child.name.startswith(prefixes):
            continue
        try:
            mtime = child.stat().st_mtime
        except OSError:
            continue
        candidates.append((mtime, child))

    if not candidates:
        return []

    candidates.sort(key=lambda item: item[0], reverse=True)
    keep_count = max(0, int(keep_latest))
    kept = {path for _, path in candidates[:keep_count]}
    cutoff = time.time() - float(ttl_hours) * 3600.0

    removed: list[Path] = []
    for mtime, path in candidates:
        if path in kept or mtime >= cutoff:
            continue
        try:
            shutil.rmtree(path)
            removed.append(path)
        except OSError:
            continue

    return removed
