"""Load and validate experiment pack manifests from YAML files."""

from __future__ import annotations

from pathlib import Path

import yaml

from navirl.packs.schema import PackManifest, PackScenarioEntry

# Default metrics tracked by packs when none are specified.
DEFAULT_PACK_METRICS: list[str] = [
    "success_rate",
    "collisions_agent_agent",
    "collisions_agent_obstacle",
    "intrusion_rate",
    "min_dist_robot_human_min",
    "min_dist_robot_human_mean",
    "oscillation_score",
    "jerk_proxy",
    "path_length_robot",
    "time_to_goal_robot",
    "deadlock_count",
]

# Location of built-in scenario library.
_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "scenarios" / "library"


def load_pack(path: str | Path) -> PackManifest:
    """Load a pack manifest from a YAML file.

    Scenario paths are resolved relative to the built-in scenario library
    unless they are absolute paths or the file exists relative to the
    manifest's own directory.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    ValueError
        If required fields are missing or malformed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Pack manifest not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError("Pack manifest must be a YAML mapping")

    name = data.get("name")
    if not name:
        raise ValueError("Pack manifest must have a 'name' field")

    version = str(data.get("version", "1.0"))
    description = data.get("description", "")
    metadata = data.get("metadata", {})

    raw_scenarios = data.get("scenarios", [])
    if not raw_scenarios:
        raise ValueError("Pack manifest must define at least one scenario")

    entries: list[PackScenarioEntry] = []
    for raw in raw_scenarios:
        if not isinstance(raw, dict) or "id" not in raw:
            raise ValueError(f"Each scenario entry must be a mapping with an 'id' field: {raw}")

        entry_id = raw["id"]
        entry_path = raw.get("path", f"{entry_id}.yaml")
        seeds = raw.get("seeds", [42])

        if not isinstance(seeds, list) or not all(isinstance(s, int) for s in seeds):
            raise ValueError(f"Seeds must be a list of integers for entry '{entry_id}'")

        # Resolve the scenario path
        resolved = _resolve_scenario_path(entry_path, pack_dir=path.parent)
        entries.append(PackScenarioEntry(id=entry_id, path=str(resolved), seeds=seeds))

    metrics = data.get("metrics", DEFAULT_PACK_METRICS)

    return PackManifest(
        name=name,
        version=version,
        description=description,
        scenarios=entries,
        metrics=metrics,
        metadata=metadata,
    )


def _resolve_scenario_path(entry_path: str, pack_dir: Path) -> Path:
    """Resolve a scenario path, checking multiple locations."""
    p = Path(entry_path)

    # Absolute path — use as-is
    if p.is_absolute() and p.exists():
        return p

    # Relative to pack directory
    relative_to_pack = pack_dir / p
    if relative_to_pack.exists():
        return relative_to_pack.resolve()

    # In the built-in scenario library
    in_library = _LIBRARY_DIR / p
    if in_library.exists():
        return in_library.resolve()

    # Try appending .yaml if not already present
    if not p.suffix:
        for candidate_dir in (pack_dir, _LIBRARY_DIR):
            candidate = candidate_dir / f"{p}.yaml"
            if candidate.exists():
                return candidate.resolve()

    raise FileNotFoundError(
        f"Cannot resolve scenario path '{entry_path}': "
        f"checked {pack_dir}, {_LIBRARY_DIR}"
    )
