"""Execute an experiment pack and collect per-run results."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from navirl.metrics import compute_metrics_from_bundle
from navirl.packs.schema import PackManifest
from navirl.pipeline import run_scenario_dict
from navirl.scenarios.load import load_scenario


@dataclass
class PackRunResult:
    """Result of executing a single scenario+seed within a pack."""

    entry_id: str
    seed: int
    bundle_dir: str
    metrics: dict[str, Any]


@dataclass
class PackResult:
    """Aggregated result of running an entire experiment pack."""

    pack_name: str
    pack_version: str
    runs: list[PackRunResult] = field(default_factory=list)
    checksum: str = ""

    def aggregate(self, metric_names: list[str] | None = None) -> dict[str, float]:
        """Compute per-metric averages across all runs."""
        if not self.runs:
            return {}
        all_keys = metric_names or sorted(
            {k for r in self.runs for k in r.metrics if isinstance(r.metrics.get(k), (int, float))}
        )
        agg: dict[str, float] = {}
        for key in all_keys:
            vals = [
                r.metrics[key]
                for r in self.runs
                if key in r.metrics and isinstance(r.metrics[key], (int, float))
            ]
            if vals:
                agg[f"avg_{key}"] = sum(vals) / len(vals)
        return agg

    def to_dict(self) -> dict[str, Any]:
        return {
            "pack_name": self.pack_name,
            "pack_version": self.pack_version,
            "checksum": self.checksum,
            "num_runs": len(self.runs),
            "runs": [
                {
                    "entry_id": r.entry_id,
                    "seed": r.seed,
                    "bundle_dir": r.bundle_dir,
                    "metrics": r.metrics,
                }
                for r in self.runs
            ],
        }


def _compute_pack_checksum(manifest: PackManifest) -> str:
    """Deterministic hash of the pack definition for reproducibility tracking."""
    payload = json.dumps(
        {
            "name": manifest.name,
            "version": manifest.version,
            "scenarios": [
                {"id": e.id, "seeds": e.seeds} for e in manifest.scenarios
            ],
            "metrics": manifest.metrics,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def run_pack(
    manifest: PackManifest,
    out_root: str | Path = "out/packs",
    render: bool = False,
    video: bool = False,
) -> PackResult:
    """Execute all scenario+seed combinations defined in a pack.

    Args:
        manifest: Loaded pack manifest.
        out_root: Base output directory for run bundles.
        render: Enable rendering during execution.
        video: Enable video recording.

    Returns:
        PackResult with per-run metrics and aggregate data.
    """
    out_root = Path(out_root)
    result = PackResult(
        pack_name=manifest.name,
        pack_version=manifest.version,
        checksum=_compute_pack_checksum(manifest),
    )

    for entry in manifest.scenarios:
        scenario = load_scenario(entry.path)
        for seed in entry.seeds:
            scenario_copy = dict(scenario)
            scenario_copy["seed"] = seed
            run_id = f"pack_{manifest.name}_{entry.id}_seed{seed}"

            log = run_scenario_dict(
                scenario=scenario_copy,
                out_root=str(out_root),
                run_id=run_id,
                render_override=render,
                video_override=video,
            )

            state_path = Path(log.bundle_dir) / "state.jsonl"
            if state_path.exists():
                metrics = compute_metrics_from_bundle(state_path)
            else:
                metrics = {}

            result.runs.append(
                PackRunResult(
                    entry_id=entry.id,
                    seed=seed,
                    bundle_dir=str(log.bundle_dir),
                    metrics=metrics,
                )
            )

    return result
