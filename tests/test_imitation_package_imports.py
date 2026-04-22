"""Regression tests for navirl.imitation package-level imports.

These tests guard against two classes of bug that have bitten this package
before:

1. ``__init__.py`` references names that do not exist in the submodules
   (phantom imports — masked by test bootstrap stubs that leak across
   pytest-xdist workers).
2. Submodules define ``nn.Module`` subclasses at module top level but guard
   ``import torch`` with a ``try/except ImportError`` that leaves ``nn``
   undefined, so ``import navirl.imitation`` blows up on torch-less envs.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import textwrap


def test_navirl_imitation_imports_cleanly() -> None:
    """``import navirl.imitation`` must succeed in-process."""
    mod = importlib.import_module("navirl.imitation")
    assert mod.__all__, "navirl.imitation should expose a non-empty __all__"


def test_navirl_imitation_all_names_resolve() -> None:
    """Every name listed in ``__all__`` must be attached to the module."""
    mod = importlib.import_module("navirl.imitation")
    missing = [name for name in mod.__all__ if not hasattr(mod, name)]
    assert not missing, f"navirl.imitation.__all__ references missing names: {missing}"


def test_navirl_imitation_imports_in_fresh_interpreter() -> None:
    """Spawn a fresh interpreter to avoid sys.modules workarounds masking breakage.

    Test files in this suite register stubs in ``sys.modules`` before loading
    submodules directly from disk to work around historical ``__init__.py``
    phantom imports. Those stubs leak between tests via pytest-xdist workers,
    so a regression in ``__init__.py`` can be invisible to the main suite.
    A subprocess guarantees a clean import.
    """
    script = textwrap.dedent(
        """
        import navirl.imitation
        missing = [n for n in navirl.imitation.__all__
                   if not hasattr(navirl.imitation, n)]
        assert not missing, f"missing: {missing}"
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"navirl.imitation failed to import in a fresh interpreter.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
