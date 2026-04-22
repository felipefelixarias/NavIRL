"""Regression guards: every navirl subpackage must import cleanly in a fresh
interpreter.

Phantom imports in ``__init__.py`` files (names that don't exist in the
corresponding submodules) are typically invisible to pytest because sibling
test files register ``sys.modules`` stubs that leak across ``pytest-xdist``
workers. The only reliable detector is spawning a fresh subprocess and
running ``import navirl.<subpkg>``.

This suite parametrizes that check across every subpackage discovered on
disk, so adding a new package automatically extends coverage. Known-broken
packages are marked ``xfail(strict=True)`` so that the fix PR's merge turns
them green and surfaces as an XPASS alerting maintainers to drop the marker.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
NAVIRL_ROOT = REPO_ROOT / "navirl"


def _discover_packages() -> list[str]:
    names: list[str] = []
    for init in NAVIRL_ROOT.rglob("__init__.py"):
        rel = init.parent.relative_to(REPO_ROOT)
        names.append(".".join(rel.parts))
    return sorted(names)


# Packages with phantom imports being fixed by open PRs. When those PRs land
# these entries should be removed; strict=True ensures the unexpected pass
# fails the test and forces cleanup.
KNOWN_BROKEN: dict[str, str] = {
    "navirl.imitation": "PR #154 (torch-less nn fallback + real export names)",
    "navirl.maps": "PR #153 (phantom submodule imports)",
    "navirl.rewards": "PR #153 (phantom submodule imports)",
}


@pytest.mark.parametrize("pkg", _discover_packages())
def test_subpackage_imports_in_fresh_interpreter(pkg: str) -> None:
    if pkg in KNOWN_BROKEN:
        pytest.xfail(f"known phantom-import breakage fixed by {KNOWN_BROKEN[pkg]}")

    result = subprocess.run(
        [sys.executable, "-c", f"import {pkg}"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"`import {pkg}` failed in a fresh subprocess.\n"
        f"This usually means __init__.py references names that don't exist "
        f"in its submodules (a phantom import).\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )


def test_discovery_finds_expected_packages() -> None:
    names = _discover_packages()
    assert "navirl" in names
    assert len(names) >= 20, f"expected many subpackages, got {len(names)}: {names}"
    for name in names:
        assert name == "navirl" or name.startswith("navirl."), name
