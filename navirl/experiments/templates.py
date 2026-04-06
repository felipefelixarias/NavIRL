"""Batch experiment template definitions.

A BatchTemplate specifies a reproducible experiment manifest: which scenarios
to run, which seeds to sweep, and optional parameter overrides to apply as
a grid.  Templates can be defined programmatically or loaded from YAML files
under ``research/templates/``.
"""

from __future__ import annotations

import copy
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BatchTemplate:
    """Reproducible experiment manifest.

    Parameters
    ----------
    name:
        Human-readable template name (e.g. ``"hallway_seed_sweep"``).
    description:
        One-line description of what this template tests.
    scenarios:
        List of scenario file paths (relative to the repo or absolute).
        May use the special value ``"library"`` to include all built-in
        scenarios.
    seeds:
        List of integer seeds.  Each scenario is run once per seed.
    param_grid:
        Optional mapping of dotted parameter paths to lists of values.
        The Cartesian product of these values is applied as overrides to
        every (scenario, seed) pair.  Example::

            {"scene.orca.neighbor_dist": [2.0, 4.0]}

    tags:
        Free-form tags for filtering / grouping.
    """

    name: str
    description: str = ""
    scenarios: list[str] = field(default_factory=list)
    seeds: list[int] = field(default_factory=lambda: [42])
    param_grid: dict[str, list[Any]] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> BatchTemplate:
        """Load a template from a YAML file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(
            name=data["name"],
            description=data.get("description", ""),
            scenarios=data.get("scenarios", []),
            seeds=data.get("seeds", [42]),
            param_grid=data.get("param_grid", {}),
            tags=data.get("tags", []),
        )

    def to_yaml(self, path: str | Path) -> None:
        """Serialize the template to a YAML file."""
        data = {
            "name": self.name,
            "description": self.description,
            "scenarios": self.scenarios,
            "seeds": self.seeds,
        }
        if self.param_grid:
            data["param_grid"] = self.param_grid
        if self.tags:
            data["tags"] = self.tags
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)

    # ------------------------------------------------------------------
    # Scenario resolution
    # ------------------------------------------------------------------

    def resolve_scenarios(self) -> list[Path]:
        """Resolve scenario paths, expanding ``"library"`` to built-ins."""
        library_dir = Path(__file__).resolve().parent.parent / "scenarios" / "library"
        paths: list[Path] = []
        for entry in self.scenarios:
            if entry == "library":
                paths.extend(sorted(library_dir.glob("*.yaml")))
            else:
                p = Path(entry)
                if not p.is_absolute():
                    p = library_dir / p
                paths.append(p)
        return paths

    # ------------------------------------------------------------------
    # Task expansion
    # ------------------------------------------------------------------

    def expand_tasks(self) -> list[dict[str, Any]]:
        """Expand the template into a flat list of run tasks.

        Each task is a dict with keys ``scenario``, ``seed``, and
        ``overrides`` (a flat dict of dotted-path parameter overrides).

        Returns
        -------
        list[dict]
            One entry per (scenario, seed, param_combo) combination.
        """
        scenarios = self.resolve_scenarios()
        if not scenarios:
            return []

        override_combos = self._expand_param_grid()

        tasks: list[dict[str, Any]] = []
        for scenario_path in scenarios:
            for seed in self.seeds:
                for overrides in override_combos:
                    tasks.append(
                        {
                            "scenario": scenario_path,
                            "seed": seed,
                            "overrides": copy.deepcopy(overrides),
                        }
                    )
        return tasks

    def _expand_param_grid(self) -> list[dict[str, Any]]:
        """Return Cartesian product of param_grid values."""
        if not self.param_grid:
            return [{}]

        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        combos: list[dict[str, Any]] = []
        for combo in itertools.product(*values):
            combos.append(dict(zip(keys, combo, strict=False)))
        return combos

    @property
    def total_runs(self) -> int:
        """Total number of individual runs this template will produce."""
        n_scenarios = len(self.resolve_scenarios())
        n_combos = max(1, len(self._expand_param_grid()))
        return n_scenarios * len(self.seeds) * n_combos
