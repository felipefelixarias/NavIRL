"""Load and validate experiment pack manifests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from navirl.packs.schema import PackManifest, PackScenarioEntry

_REQUIRED_FIELDS = {"name", "version", "scenarios"}


def validate_pack(data: dict[str, Any]) -> list[str]:
    """Validate a raw pack dict and return a list of error messages (empty = valid)."""
    errors: list[str] = []

    for f in _REQUIRED_FIELDS:
        if f not in data:
            errors.append(f"Missing required field: {f}")

    if not isinstance(data.get("scenarios"), list):
        errors.append("'scenarios' must be a list")
    else:
        for i, entry in enumerate(data["scenarios"]):
            if not isinstance(entry, dict):
                errors.append(f"scenarios[{i}]: must be a mapping")
                continue
            if "id" not in entry:
                errors.append(f"scenarios[{i}]: missing 'id'")
            if "path" not in entry:
                errors.append(f"scenarios[{i}]: missing 'path'")
            if "seeds" in entry and not isinstance(entry["seeds"], list):
                errors.append(f"scenarios[{i}]: 'seeds' must be a list")

    version = data.get("version", "")
    if isinstance(version, str) and not version:
        errors.append("'version' must be a non-empty string")

    return errors


def load_pack(path: str | Path) -> PackManifest:
    """Load an experiment pack manifest from a YAML file.

    Args:
        path: Path to the pack YAML manifest.

    Returns:
        Parsed PackManifest.

    Raises:
        ValueError: If the manifest is invalid.
        FileNotFoundError: If the manifest or referenced scenarios are missing.
    """
    pack_path = Path(path).resolve()
    if not pack_path.exists():
        raise FileNotFoundError(f"Pack manifest not found: {pack_path}")

    with pack_path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Pack manifest must be a YAML mapping: {pack_path}")

    errors = validate_pack(raw)
    if errors:
        raise ValueError(
            f"Invalid pack manifest ({pack_path}):\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # Resolve scenario paths relative to the manifest directory.
    pack_dir = pack_path.parent
    entries: list[PackScenarioEntry] = []
    for entry_raw in raw["scenarios"]:
        scenario_path = Path(entry_raw["path"])
        if not scenario_path.is_absolute():
            scenario_path = (pack_dir / scenario_path).resolve()
        if not scenario_path.exists():
            raise FileNotFoundError(
                f"Scenario not found: {scenario_path} "
                f"(referenced by pack entry '{entry_raw['id']}')"
            )
        entries.append(
            PackScenarioEntry(
                id=entry_raw["id"],
                path=str(scenario_path),
                seeds=entry_raw.get("seeds", [7]),
            )
        )

    return PackManifest(
        name=raw["name"],
        version=raw["version"],
        description=raw.get("description", ""),
        scenarios=entries,
        metrics=raw.get("metrics", [
            "success_rate",
            "collision_count",
            "intrusion_rate",
            "avg_robot_human_min_dist",
        ]),
        metadata=raw.get("metadata", {}),
    )
