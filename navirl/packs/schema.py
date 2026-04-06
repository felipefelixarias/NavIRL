"""Experiment pack schema definitions.

An experiment pack is a standardized, versioned collection of scenarios
with fixed seeds and metric selections designed for cross-lab reproducibility.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PackScenarioEntry:
    """A single scenario entry within an experiment pack.

    Parameters
    ----------
    id:
        Short identifier for this entry (e.g. ``"hallway_pass"``).
    path:
        Path to the scenario YAML file, relative to the scenarios library
        or absolute.
    seeds:
        List of integer seeds to run for this scenario.
    """

    id: str
    path: str
    seeds: list[int] = field(default_factory=lambda: [42])


@dataclass
class PackManifest:
    """Top-level experiment pack definition.

    Parameters
    ----------
    name:
        Machine-readable pack name (e.g. ``"social-nav-baseline"``).
    version:
        Semantic version string (e.g. ``"1.0"``).
    description:
        Human-readable description of the pack's purpose.
    scenarios:
        Ordered list of scenario entries with seeds.
    metrics:
        Which metric keys to track and report.
    metadata:
        Free-form metadata (authors, purpose, notes).
    """

    name: str
    version: str = "1.0"
    description: str = ""
    scenarios: list[PackScenarioEntry] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_runs(self) -> int:
        """Total number of (scenario, seed) combinations."""
        return sum(len(e.seeds) for e in self.scenarios)

    def checksum(self) -> str:
        """Compute a deterministic SHA-256 checksum of the pack definition.

        This captures pack identity (name, version, scenarios, metrics) so
        that results can be tied back to the exact pack configuration.
        """
        obj = {
            "name": self.name,
            "version": self.version,
            "scenarios": [
                {"id": e.id, "path": e.path, "seeds": e.seeds}
                for e in self.scenarios
            ],
            "metrics": self.metrics,
        }
        blob = json.dumps(obj, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@dataclass
class PackRunResult:
    """Result of a single (scenario, seed) execution within a pack."""

    entry_id: str
    seed: int
    metrics: dict[str, Any] = field(default_factory=dict)
    status: str = "completed"
    error: str | None = None


@dataclass
class PackResult:
    """Complete results for an experiment pack execution."""

    manifest_name: str
    manifest_version: str
    manifest_checksum: str
    runs: list[PackRunResult] = field(default_factory=list)
    timestamp: str = ""

    def aggregate(self, metric_names: list[str]) -> dict[str, dict[str, float]]:
        """Compute per-metric aggregated statistics across all completed runs.

        Returns a dict mapping metric names to ``{mean, std, min, max}``
        computed over all completed runs that reported that metric.
        """
        import math

        import numpy as np

        completed = [r for r in self.runs if r.status == "completed"]
        result: dict[str, dict[str, float]] = {}
        for key in metric_names:
            vals = [
                float(r.metrics[key])
                for r in completed
                if key in r.metrics and isinstance(r.metrics[key], (int, float))
                and math.isfinite(float(r.metrics[key]))
            ]
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                result[key] = {
                    "mean": float(arr.mean()),
                    "std": float(arr.std()),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                }
            else:
                result[key] = {
                    "mean": float("nan"),
                    "std": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "manifest_name": self.manifest_name,
            "manifest_version": self.manifest_version,
            "manifest_checksum": self.manifest_checksum,
            "timestamp": self.timestamp,
            "total_runs": len(self.runs),
            "completed_runs": sum(1 for r in self.runs if r.status == "completed"),
            "failed_runs": sum(1 for r in self.runs if r.status == "failed"),
            "runs": [
                {
                    "entry_id": r.entry_id,
                    "seed": r.seed,
                    "metrics": r.metrics,
                    "status": r.status,
                    "error": r.error,
                }
                for r in self.runs
            ],
        }
