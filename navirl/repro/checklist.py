"""Publication readiness checklist for reproducibility packages.

Automates verification of whether a reproducibility package meets
the minimum requirements for publication alongside a study.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CheckResult:
    """Result of a single checklist item."""

    name: str
    passed: bool
    message: str


@dataclass
class ChecklistReport:
    """Complete checklist evaluation report."""

    package_name: str
    results: list[CheckResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_name": self.package_name,
            "passed": self.passed,
            "total_checks": self.total,
            "passed_checks": self.passed_count,
            "results": [
                {"name": r.name, "passed": r.passed, "message": r.message}
                for r in self.results
            ],
        }

    def to_markdown(self) -> str:
        lines = [
            f"# Reproducibility Checklist: {self.package_name}",
            "",
            f"**Status**: {'PASS' if self.passed else 'FAIL'} "
            f"({self.passed_count}/{self.total} checks passed)",
            "",
            "| Check | Status | Details |",
            "|-------|--------|---------|",
        ]
        for r in self.results:
            icon = "PASS" if r.passed else "FAIL"
            lines.append(f"| {r.name} | {icon} | {r.message} |")
        lines.append("")
        return "\n".join(lines)


def run_checklist(package_dir: Path) -> ChecklistReport:
    """Run the full publication readiness checklist.

    Parameters
    ----------
    package_dir:
        Root directory of the reproducibility package.

    Returns
    -------
    ChecklistReport with all check results.
    """
    manifest_path = package_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return ChecklistReport(
            package_name=package_dir.name,
            results=[CheckResult("manifest_exists", False, "MANIFEST.json not found")],
        )

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    name = data.get("name", package_dir.name)
    results: list[CheckResult] = []

    # 1. Manifest exists and is valid JSON
    results.append(CheckResult("manifest_exists", True, "MANIFEST.json present and valid"))

    # 2. Package has a name and version
    has_name = bool(data.get("name"))
    has_version = bool(data.get("version"))
    results.append(
        CheckResult(
            "identity",
            has_name and has_version,
            f"name={data.get('name', '?')}, version={data.get('version', '?')}",
        )
    )

    # 3. Environment pins captured
    env = data.get("environment", {})
    has_python = bool(env.get("python_version"))
    has_platform = bool(env.get("platform_system"))
    has_packages = bool(env.get("packages"))
    results.append(
        CheckResult(
            "environment_pins",
            has_python and has_platform,
            f"python={'yes' if has_python else 'missing'}, "
            f"platform={'yes' if has_platform else 'missing'}, "
            f"packages={len(env.get('packages', {}))} pinned",
        )
    )

    # 4. Scenarios included
    scenarios_dir = package_dir / "scenarios"
    scenario_files = list(scenarios_dir.glob("*.yaml")) if scenarios_dir.is_dir() else []
    results.append(
        CheckResult(
            "scenarios_included",
            len(scenario_files) > 0,
            f"{len(scenario_files)} scenario file(s) found",
        )
    )

    # 5. Results present
    results_dir = package_dir / "results"
    result_files = list(results_dir.glob("*.json")) if results_dir.is_dir() else []
    results.append(
        CheckResult(
            "results_present",
            len(result_files) > 0,
            f"{len(result_files)} result file(s) found",
        )
    )

    # 6. Artifact checksums recorded
    artifacts = data.get("artifacts", [])
    all_have_hash = all(bool(a.get("sha256")) for a in artifacts)
    results.append(
        CheckResult(
            "artifact_checksums",
            len(artifacts) > 0 and all_have_hash,
            f"{len(artifacts)} artifact(s) with checksums",
        )
    )

    # 7. Expected metrics documented
    expected = data.get("expected_metrics", {})
    results.append(
        CheckResult(
            "expected_metrics",
            len(expected) > 0,
            f"{len(expected)} metric(s) with expected values",
        )
    )

    # 8. Package-level description provided
    has_desc = bool(data.get("description"))
    results.append(
        CheckResult(
            "description",
            has_desc,
            "description provided" if has_desc else "no description",
        )
    )

    # 9. Package includes pinned package list
    results.append(
        CheckResult(
            "package_pins",
            has_packages,
            f"{len(env.get('packages', {}))} packages pinned"
            if has_packages
            else "no package pins",
        )
    )

    # 10. Checksum is present
    has_checksum = bool(data.get("checksum"))
    results.append(
        CheckResult(
            "package_checksum",
            has_checksum,
            f"checksum={data.get('checksum', '?')[:16]}..."
            if has_checksum
            else "no checksum",
        )
    )

    return ChecklistReport(package_name=name, results=results)
