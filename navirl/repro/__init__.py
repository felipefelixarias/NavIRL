"""Reproducibility package tools for NavIRL.

Provides utilities to bundle validated experiment runs into publishable
reproducibility packages with environment pins, checksums, and
publication readiness checklists.
"""

from __future__ import annotations

from navirl.repro.checklist import ChecklistReport, CheckResult, run_checklist
from navirl.repro.package import (
    ArtifactEntry,
    EnvironmentPin,
    ReproPackage,
    build_repro_package,
    verify_repro_package,
)

__all__ = [
    "ArtifactEntry",
    "ChecklistReport",
    "CheckResult",
    "EnvironmentPin",
    "ReproPackage",
    "build_repro_package",
    "run_checklist",
    "verify_repro_package",
]
