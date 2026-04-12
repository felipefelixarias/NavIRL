"""Reproducibility package builder.

Bundles validated experiment runs into a self-contained, publishable
reproducibility package containing configs, environment pins, expected
outputs, and checksums.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class EnvironmentPin:
    """Frozen snapshot of the execution environment."""

    python_version: str = ""
    platform_system: str = ""
    platform_machine: str = ""
    platform_release: str = ""
    packages: dict[str, str] = field(default_factory=dict)

    @classmethod
    def capture(cls) -> EnvironmentPin:
        """Capture current environment information."""
        packages: dict[str, str] = {}
        try:
            from importlib.metadata import distributions

            for dist in distributions():
                packages[dist.metadata["Name"]] = dist.metadata["Version"]
        except Exception:
            pass

        return cls(
            python_version=sys.version,
            platform_system=platform.system(),
            platform_machine=platform.machine(),
            platform_release=platform.release(),
            packages=dict(sorted(packages.items())),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_version": self.python_version,
            "platform_system": self.platform_system,
            "platform_machine": self.platform_machine,
            "platform_release": self.platform_release,
            "packages": self.packages,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnvironmentPin:
        return cls(
            python_version=data.get("python_version", ""),
            platform_system=data.get("platform_system", ""),
            platform_machine=data.get("platform_machine", ""),
            platform_release=data.get("platform_release", ""),
            packages=data.get("packages", {}),
        )


@dataclass
class ArtifactEntry:
    """A tracked artifact within the reproducibility package."""

    relative_path: str
    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ArtifactEntry:
        return cls(
            relative_path=data["relative_path"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
        )


@dataclass
class ReproPackage:
    """A self-contained reproducibility package.

    Bundles scenario configs, result summaries, environment pins,
    and artifact checksums into a publishable directory.
    """

    name: str
    version: str = "1.0"
    description: str = ""
    created_at: str = ""
    environment: EnvironmentPin = field(default_factory=EnvironmentPin)
    artifacts: list[ArtifactEntry] = field(default_factory=list)
    expected_metrics: dict[str, dict[str, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def checksum(self) -> str:
        """Compute deterministic SHA-256 of the package definition."""
        obj = {
            "name": self.name,
            "version": self.version,
            "artifacts": [a.to_dict() for a in self.artifacts],
            "expected_metrics": self.expected_metrics,
        }
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "checksum": self.checksum(),
            "environment": self.environment.to_dict(),
            "artifacts": [a.to_dict() for a in self.artifacts],
            "expected_metrics": self.expected_metrics,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ReproPackage:
        return cls(
            name=data["name"],
            version=data.get("version", "1.0"),
            description=data.get("description", ""),
            created_at=data.get("created_at", ""),
            environment=EnvironmentPin.from_dict(data.get("environment", {})),
            artifacts=[ArtifactEntry.from_dict(a) for a in data.get("artifacts", [])],
            expected_metrics=data.get("expected_metrics", {}),
            metadata=data.get("metadata", {}),
        )


def _hash_file(path: Path) -> str:
    """Compute SHA-256 of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_repro_package(
    *,
    name: str,
    version: str = "1.0",
    description: str = "",
    run_dir: Path,
    scenario_paths: list[Path] | None = None,
    pack_result_path: Path | None = None,
    out_dir: Path,
    metadata: dict[str, Any] | None = None,
) -> ReproPackage:
    """Build a reproducibility package from validated experiment outputs.

    Parameters
    ----------
    name:
        Package name (e.g. ``"hallway-study-2024"``).
    version:
        Semantic version for the package.
    description:
        Human-readable description.
    run_dir:
        Directory containing experiment run outputs (state logs, summaries).
    scenario_paths:
        Optional explicit list of scenario YAML files to include.
        If not provided, discovers ``*.yaml`` files in run_dir.
    pack_result_path:
        Optional path to a pack_results.json to embed expected metrics from.
    out_dir:
        Output directory for the reproducibility package.
    metadata:
        Free-form metadata dict.

    Returns
    -------
    ReproPackage:
        The assembled package descriptor.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture environment
    env = EnvironmentPin.capture()

    # Collect scenario files
    scenarios_dir = out_dir / "scenarios"
    scenarios_dir.mkdir(exist_ok=True)

    if scenario_paths:
        for sp in scenario_paths:
            if sp.is_file():
                shutil.copy2(sp, scenarios_dir / sp.name)
    else:
        # Discover scenarios from run_dir
        for sp in sorted(run_dir.glob("**/scenario.yaml")):
            dest_name = f"{sp.parent.name}_scenario.yaml"
            shutil.copy2(sp, scenarios_dir / dest_name)

    # Copy run summaries
    results_dir = out_dir / "results"
    results_dir.mkdir(exist_ok=True)

    for summary in sorted(run_dir.glob("**/summary.json")):
        dest_name = f"{summary.parent.name}_summary.json"
        shutil.copy2(summary, results_dir / dest_name)

    # Load expected metrics from pack results if available
    expected_metrics: dict[str, dict[str, float]] = {}
    if pack_result_path and pack_result_path.is_file():
        with pack_result_path.open("r", encoding="utf-8") as f:
            pack_data = json.load(f)
        # Extract aggregated metrics if present
        if "runs" in pack_data:
            from navirl.packs.schema import PackResult, PackRunResult

            runs = [
                PackRunResult(
                    entry_id=r["entry_id"],
                    seed=r["seed"],
                    metrics=r.get("metrics", {}),
                    status=r.get("status", "completed"),
                    error=r.get("error"),
                )
                for r in pack_data["runs"]
            ]
            pr = PackResult(
                manifest_name=pack_data.get("manifest_name", name),
                manifest_version=pack_data.get("manifest_version", version),
                manifest_checksum=pack_data.get("manifest_checksum", ""),
                runs=runs,
                timestamp=pack_data.get("timestamp", ""),
            )
            metric_keys = list({k for r in runs if r.status == "completed" for k in r.metrics})
            expected_metrics = pr.aggregate(sorted(metric_keys))
        shutil.copy2(pack_result_path, results_dir / "pack_results.json")

    # Hash all files in the package
    artifacts: list[ArtifactEntry] = []
    for fpath in sorted(out_dir.rglob("*")):
        if fpath.is_file() and fpath.name != "MANIFEST.json":
            rel = str(fpath.relative_to(out_dir))
            artifacts.append(
                ArtifactEntry(
                    relative_path=rel,
                    sha256=_hash_file(fpath),
                    size_bytes=fpath.stat().st_size,
                )
            )

    now = datetime.now(UTC).isoformat()
    package = ReproPackage(
        name=name,
        version=version,
        description=description,
        created_at=now,
        environment=env,
        artifacts=artifacts,
        expected_metrics=expected_metrics,
        metadata=metadata or {},
    )

    # Write manifest
    manifest_path = out_dir / "MANIFEST.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(package.to_dict(), f, indent=2, sort_keys=True)

    return package


def verify_repro_package(package_dir: Path) -> tuple[bool, list[str]]:
    """Verify integrity of a reproducibility package.

    Checks that all artifacts exist and their checksums match.

    Parameters
    ----------
    package_dir:
        Root directory of the reproducibility package.

    Returns
    -------
    tuple[bool, list[str]]:
        (all_ok, list_of_issues). If all_ok is True, list is empty.
    """
    manifest_path = package_dir / "MANIFEST.json"
    if not manifest_path.is_file():
        return False, ["MANIFEST.json not found"]

    with manifest_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    package = ReproPackage.from_dict(data)
    issues: list[str] = []

    for artifact in package.artifacts:
        fpath = package_dir / artifact.relative_path
        if not fpath.is_file():
            issues.append(f"Missing artifact: {artifact.relative_path}")
            continue
        actual_hash = _hash_file(fpath)
        if actual_hash != artifact.sha256:
            issues.append(
                f"Checksum mismatch: {artifact.relative_path} "
                f"(expected {artifact.sha256[:16]}..., got {actual_hash[:16]}...)"
            )
        actual_size = fpath.stat().st_size
        if actual_size != artifact.size_bytes:
            issues.append(
                f"Size mismatch: {artifact.relative_path} "
                f"(expected {artifact.size_bytes}, got {actual_size})"
            )

    return len(issues) == 0, issues
