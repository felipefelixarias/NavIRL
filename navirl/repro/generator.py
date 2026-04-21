"""Auto-generate reproducibility packages from experiment runs.

Provides a higher-level interface than :func:`build_repro_package` that
can discover scenario files and run outputs automatically, generate
documentation from templates, and produce publication-ready packages
in a single call.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from navirl.repro.package import ReproPackage, build_repro_package
from navirl.repro.templates import get_checklist_template, get_package_readme_template


@dataclass
class GeneratorConfig:
    """Configuration for reproducibility package generation.

    Parameters
    ----------
    name:
        Package name (e.g. ``"hallway-study-2024"``).
    version:
        Semantic version string.
    description:
        Human-readable description for the package.
    run_dir:
        Directory containing experiment run outputs.
    out_dir:
        Output directory for the generated package.
    scenario_paths:
        Explicit scenario YAML paths. If ``None``, auto-discovered
        from *run_dir*.
    pack_result_path:
        Optional path to ``pack_results.json`` for expected metrics.
    metadata:
        Free-form metadata dict to embed in the package.
    include_checklist:
        Whether to include the checklist template in the package.
    include_readme:
        Whether to generate a README from the template.
    """

    name: str
    version: str = "1.0"
    description: str = ""
    run_dir: Path = field(default_factory=lambda: Path("."))
    out_dir: Path = field(default_factory=lambda: Path("out/repro"))
    scenario_paths: list[Path] | None = None
    pack_result_path: Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    include_checklist: bool = True
    include_readme: bool = True


def discover_run_dirs(root: Path) -> list[Path]:
    """Find experiment run directories under *root*.

    A run directory is identified by containing a ``summary.json`` file.

    Returns
    -------
    list[Path]
        Sorted list of directories that contain run outputs.
    """
    if not root.is_dir():
        return []
    return sorted({p.parent for p in root.rglob("summary.json")})


def discover_scenarios(root: Path) -> list[Path]:
    """Find scenario YAML files under *root*.

    Looks for files named ``scenario.yaml`` or ``*.yaml`` directly
    inside run directories.

    Returns
    -------
    list[Path]
        Sorted list of discovered scenario files.
    """
    if not root.is_dir():
        return []
    # Prefer scenario.yaml inside run bundles
    scenarios = sorted(root.rglob("scenario.yaml"))
    if scenarios:
        return scenarios
    # Fall back to any yaml files
    return sorted(root.rglob("*.yaml"))


def _format_metrics_table(expected_metrics: dict[str, dict[str, float]]) -> str:
    """Format expected metrics as a Markdown table."""
    if not expected_metrics:
        return "No expected metrics recorded."
    lines = ["| Metric | Mean | Std |", "|--------|------|-----|"]
    for metric, values in sorted(expected_metrics.items()):
        mean = values.get("mean", 0.0)
        std = values.get("std", 0.0)
        lines.append(f"| {metric} | {mean:.4f} | {std:.4f} |")
    return "\n".join(lines)


def generate_repro_package(config: GeneratorConfig) -> ReproPackage:
    """Generate a complete reproducibility package from experiment outputs.

    This is the main entry point for package generation. It:

    1. Builds the core package via :func:`build_repro_package`
    2. Optionally writes a README from the package template
    3. Optionally includes the publication readiness checklist

    Parameters
    ----------
    config:
        Generation configuration.

    Returns
    -------
    ReproPackage
        The assembled package.
    """
    package = build_repro_package(
        name=config.name,
        version=config.version,
        description=config.description,
        run_dir=config.run_dir,
        scenario_paths=config.scenario_paths,
        pack_result_path=config.pack_result_path,
        out_dir=config.out_dir,
        metadata=config.metadata,
    )

    if config.include_checklist:
        _write_checklist(config.out_dir)

    if config.include_readme:
        _write_readme(package, config.out_dir)

    return package


def _write_checklist(out_dir: Path) -> Path:
    """Write the checklist template into the package directory."""
    checklist_path = out_dir / "CHECKLIST.md"
    checklist_path.write_text(get_checklist_template(), encoding="utf-8")
    return checklist_path


def _write_readme(package: ReproPackage, out_dir: Path) -> Path:
    """Generate a README from the package template and write it."""
    template = get_package_readme_template()
    metrics_table = _format_metrics_table(package.expected_metrics)

    readme = template.format(
        name=package.name,
        version=package.version,
        created_at=package.created_at or "N/A",
        description=package.description or "N/A",
        python_version=package.environment.python_version or "3.x",
        platform_system=package.environment.platform_system or "N/A",
        platform_machine=package.environment.platform_machine or "N/A",
        package_dir=out_dir.name,
        metrics_table=metrics_table,
    )

    readme_path = out_dir / "README.md"
    readme_path.write_text(readme, encoding="utf-8")
    return readme_path


def generate_canonical_package(
    scenario_path: Path,
    out_dir: Path,
    *,
    name: str | None = None,
    version: str = "1.0",
    description: str = "",
    metadata: dict[str, Any] | None = None,
) -> ReproPackage:
    """Generate a repro package from a single canonical scenario file.

    This is a convenience wrapper for the common case of producing a
    reproducibility package from one scenario YAML without first running
    the full pipeline. It bundles the scenario config and creates
    the package structure ready for results to be added after replay.

    Parameters
    ----------
    scenario_path:
        Path to a scenario YAML file.
    out_dir:
        Output directory.
    name:
        Package name. Defaults to the scenario file stem.
    version:
        Semantic version.
    description:
        Human-readable description.
    metadata:
        Optional metadata dict.

    Returns
    -------
    ReproPackage
        The assembled package.
    """
    if not scenario_path.is_file():
        raise FileNotFoundError(f"Scenario not found: {scenario_path}")

    pkg_name = name or scenario_path.stem

    # Create a minimal run directory with the scenario
    run_dir = out_dir / "_staging"
    run_dir.mkdir(parents=True, exist_ok=True)
    bundle = run_dir / pkg_name
    bundle.mkdir(exist_ok=True)
    shutil.copy2(scenario_path, bundle / "scenario.yaml")

    config = GeneratorConfig(
        name=pkg_name,
        version=version,
        description=description or f"Reproducibility package for {scenario_path.name}",
        run_dir=run_dir,
        out_dir=out_dir,
        metadata=metadata or {"source_scenario": str(scenario_path)},
    )

    package = generate_repro_package(config)

    # Clean up staging directory
    shutil.rmtree(run_dir, ignore_errors=True)

    return package
