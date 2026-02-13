"""NavIRL: Indoor social navigation simulation and evaluation toolkit."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ._version import __version__

os.environ.setdefault("MPLBACKEND", "Agg")


def _resolve_mplconfig_dir() -> Path:
    explicit = os.environ.get("NAVIRL_MPLCONFIGDIR", "").strip()
    if explicit:
        return Path(explicit)

    xdg_cache = os.environ.get("XDG_CACHE_HOME", "").strip()
    candidates = []
    if xdg_cache:
        candidates.append(Path(xdg_cache) / "navirl" / "mplconfig")
    else:
        candidates.append(Path.home() / ".cache" / "navirl" / "mplconfig")
    candidates.append(Path(tempfile.gettempdir()) / "navirl-mplconfig")

    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            return cand
        except OSError:
            continue

    return Path(tempfile.mkdtemp(prefix="navirl-mplconfig-"))


os.environ.setdefault("MPLCONFIGDIR", str(_resolve_mplconfig_dir()))

__all__ = ["__version__"]
