"""Experiment pack manifest schema and data classes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class PackScenarioEntry:
    """A single scenario within an experiment pack.

    Attributes:
        id: Unique identifier for this entry within the pack.
        path: Path to the scenario YAML file (relative to pack manifest).
        seeds: List of seeds to run this scenario with.
    """

    id: str
    path: str
    seeds: list[int] = field(default_factory=lambda: [7])


@dataclass
class PackManifest:
    """Top-level experiment pack manifest.

    Attributes:
        name: Human-readable pack name.
        version: Semantic version string.
        description: Explanation of what this pack evaluates.
        scenarios: List of scenario entries to execute.
        metrics: Which metrics to include in the final report.
        metadata: Authorship and provenance information.
    """

    name: str
    version: str
    description: str = ""
    scenarios: list[PackScenarioEntry] = field(default_factory=list)
    metrics: list[str] = field(
        default_factory=lambda: [
            "success_rate",
            "collision_count",
            "intrusion_rate",
            "avg_robot_human_min_dist",
        ]
    )
    metadata: dict[str, str] = field(default_factory=dict)
