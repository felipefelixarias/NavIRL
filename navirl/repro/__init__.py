"""Reproducibility package tools for NavIRL.

Provides utilities to bundle validated experiment runs into publishable
reproducibility packages with environment pins, checksums, publication
readiness checklists, replay validation, and compliance scanning.
"""

from __future__ import annotations

from navirl.repro.checklist import ChecklistReport, CheckResult, run_checklist
from navirl.repro.compliance import ComplianceFinding, ComplianceReport, scan_compliance
from navirl.repro.package import (
    ArtifactEntry,
    EnvironmentPin,
    ReproPackage,
    build_repro_package,
    verify_repro_package,
)
from navirl.repro.replay import (
    MetricComparison,
    ReplayReport,
    ReplayResult,
    replay_package,
)

__all__ = [
    "ArtifactEntry",
    "ChecklistReport",
    "CheckResult",
    "ComplianceFinding",
    "ComplianceReport",
    "EnvironmentPin",
    "MetricComparison",
    "ReplayReport",
    "ReplayResult",
    "ReproPackage",
    "build_repro_package",
    "replay_package",
    "run_checklist",
    "scan_compliance",
    "verify_repro_package",
]
