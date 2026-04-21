"""Reproducibility package templates.

Provides standard templates for checklist reports and package
documentation that accompany published reproducibility packages.
"""

from __future__ import annotations

from pathlib import Path

TEMPLATES_DIR = Path(__file__).parent


def get_checklist_template() -> str:
    """Return the contents of the standard checklist template."""
    return (TEMPLATES_DIR / "CHECKLIST.md").read_text(encoding="utf-8")


def get_package_readme_template() -> str:
    """Return the contents of the package README template."""
    return (TEMPLATES_DIR / "PACKAGE_README.md").read_text(encoding="utf-8")
