"""Experiment management and hyperparameter search for NavIRL."""

from __future__ import annotations

import copy
import itertools
import json
import random
import sqlite3
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Exports: Experiment, ExperimentGrid, ExperimentRandom, ResultsDB

__all__ = [
    "Experiment",
    "ExperimentGrid",
    "ExperimentRandom",
    "ResultsDB",
]


class ExperimentStatus(str, Enum):
    """Possible states of an experiment."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Experiment:
    """A single experiment run.

    Parameters
    ----------
    name:
        Human-readable experiment name.
    config:
        Dictionary of hyperparameters / configuration values.
    status:
        Current status of the experiment.
    results:
        Dictionary of result metrics (populated after completion).
    timestamps:
        Dictionary of notable timestamps (``started``, ``completed``, etc.).
    error:
        Error message if the experiment failed.
    """

    name: str
    config: Dict[str, Any] = field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.PENDING
    results: Dict[str, Any] = field(default_factory=dict)
    timestamps: Dict[str, float] = field(default_factory=dict)
    error: Optional[str] = None

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Mark the experiment as running."""
        self.status = ExperimentStatus.RUNNING
        self.timestamps["started"] = time.time()

    def complete(self, results: Dict[str, Any]) -> None:
        """Mark the experiment as completed with *results*."""
        self.status = ExperimentStatus.COMPLETED
        self.results.update(results)
        self.timestamps["completed"] = time.time()

    def fail(self, error: Union[str, Exception]) -> None:
        """Mark the experiment as failed with an *error* message."""
        self.status = ExperimentStatus.FAILED
        self.error = str(error)
        self.timestamps["failed"] = time.time()

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a JSON-friendly dictionary."""
        return {
            "name": self.name,
            "config": self.config,
            "status": self.status.value,
            "results": self.results,
            "timestamps": self.timestamps,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Experiment":
        """Deserialise from a dictionary."""
        return cls(
            name=data["name"],
            config=data.get("config", {}),
            status=ExperimentStatus(data.get("status", "pending")),
            results=data.get("results", {}),
            timestamps=data.get("timestamps", {}),
            error=data.get("error"),
        )


class ExperimentGrid:
    """Grid search over hyperparameter combinations.

    Parameters
    ----------
    param_grid:
        Mapping of parameter names to lists of candidate values.
    base_config:
        Optional base configuration that each generated config is merged into.
    name_template:
        Format string for experiment names.  Receives all grid parameters as
        keyword arguments.
    """

    def __init__(
        self,
        param_grid: Dict[str, List[Any]],
        base_config: Optional[Dict[str, Any]] = None,
        name_template: str = "grid_{index}",
    ) -> None:
        self.param_grid = param_grid
        self.base_config = base_config or {}
        self.name_template = name_template

    def generate_configs(self) -> List[Dict[str, Any]]:
        """Return the full Cartesian product of parameter values.

        Returns
        -------
        list[dict]
            One configuration dictionary per combination.
        """
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        configs: List[Dict[str, Any]] = []

        for index, combo in enumerate(itertools.product(*values)):
            cfg = copy.deepcopy(self.base_config)
            params = dict(zip(keys, combo))
            cfg.update(params)
            cfg["_grid_index"] = index
            cfg["_grid_params"] = params
            configs.append(cfg)

        return configs

    def generate_experiments(self) -> List[Experiment]:
        """Generate :class:`Experiment` objects for every combination."""
        experiments: List[Experiment] = []
        for index, cfg in enumerate(self.generate_configs()):
            name = self.name_template.format(
                index=index, **cfg.get("_grid_params", {})
            )
            experiments.append(Experiment(name=name, config=cfg))
        return experiments

    @property
    def total_combinations(self) -> int:
        """Total number of parameter combinations."""
        count = 1
        for vals in self.param_grid.values():
            count *= len(vals)
        return count


class ExperimentRandom:
    """Random search over hyperparameter distributions.

    Parameters
    ----------
    param_distributions:
        Mapping of parameter names to distributions.  Each distribution can be:
        - A list of values (one is sampled uniformly).
        - A tuple ``(low, high)`` for uniform float sampling.
        - A callable ``() -> value``.
    base_config:
        Optional base configuration merged into each generated config.
    seed:
        Random seed for reproducibility.
    name_template:
        Format string for experiment names.
    """

    def __init__(
        self,
        param_distributions: Dict[str, Any],
        base_config: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        name_template: str = "random_{index}",
    ) -> None:
        self.param_distributions = param_distributions
        self.base_config = base_config or {}
        self.name_template = name_template
        self._rng = random.Random(seed)

    def _sample_param(self, dist: Any) -> Any:
        """Sample a single parameter value from *dist*."""
        if isinstance(dist, (list, tuple)) and len(dist) == 2 and all(
            isinstance(v, (int, float)) for v in dist
        ):
            # Uniform float in [low, high)
            low, high = dist
            return self._rng.uniform(float(low), float(high))
        if isinstance(dist, (list, tuple)):
            return self._rng.choice(dist)
        if callable(dist):
            return dist()
        return dist

    def generate_configs(self, n_trials: int) -> List[Dict[str, Any]]:
        """Generate *n_trials* random configurations.

        Parameters
        ----------
        n_trials:
            Number of random configurations to produce.

        Returns
        -------
        list[dict]
            Sampled configuration dictionaries.
        """
        configs: List[Dict[str, Any]] = []
        for index in range(n_trials):
            cfg = copy.deepcopy(self.base_config)
            params: Dict[str, Any] = {}
            for name, dist in self.param_distributions.items():
                params[name] = self._sample_param(dist)
            cfg.update(params)
            cfg["_random_index"] = index
            cfg["_random_params"] = params
            configs.append(cfg)
        return configs

    def generate_experiments(self, n_trials: int) -> List[Experiment]:
        """Generate :class:`Experiment` objects for *n_trials* random configs."""
        experiments: List[Experiment] = []
        for index, cfg in enumerate(self.generate_configs(n_trials)):
            name = self.name_template.format(index=index)
            experiments.append(Experiment(name=name, config=cfg))
        return experiments


class ResultsDB:
    """SQLite-backed experiment results storage.

    Parameters
    ----------
    db_path:
        Path to the SQLite database file.  Created if it does not exist.
    """

    _CREATE_TABLE = """
        CREATE TABLE IF NOT EXISTS experiments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            config      TEXT NOT NULL,
            status      TEXT NOT NULL,
            results     TEXT NOT NULL,
            timestamps  TEXT NOT NULL,
            error       TEXT,
            created_at  REAL NOT NULL
        )
    """

    def __init__(self, db_path: Union[str, Path]) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(self._CREATE_TABLE)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_experiment(self, experiment: Experiment) -> int:
        """Insert or update an experiment record.

        Returns
        -------
        int
            Row id of the inserted/updated record.
        """
        cursor = self._conn.execute(
            """
            INSERT INTO experiments (name, config, status, results, timestamps, error, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                experiment.name,
                json.dumps(experiment.config),
                experiment.status.value,
                json.dumps(experiment.results),
                json.dumps(experiment.timestamps),
                experiment.error,
                time.time(),
            ),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def update_experiment(self, row_id: int, experiment: Experiment) -> None:
        """Update an existing experiment row."""
        self._conn.execute(
            """
            UPDATE experiments
            SET name = ?, config = ?, status = ?, results = ?, timestamps = ?, error = ?
            WHERE id = ?
            """,
            (
                experiment.name,
                json.dumps(experiment.config),
                experiment.status.value,
                json.dumps(experiment.results),
                json.dumps(experiment.timestamps),
                experiment.error,
                row_id,
            ),
        )
        self._conn.commit()

    def get_best(
        self,
        metric: str,
        n: int = 1,
        mode: str = "max",
    ) -> List[Experiment]:
        """Return the top-*n* experiments ranked by *metric*.

        Parameters
        ----------
        metric:
            Key inside the experiment's ``results`` dictionary.
        n:
            Number of experiments to return.
        mode:
            ``"max"`` for highest-is-best, ``"min"`` for lowest-is-best.

        Returns
        -------
        list[Experiment]
        """
        rows = self._conn.execute(
            "SELECT * FROM experiments WHERE status = ?",
            (ExperimentStatus.COMPLETED.value,),
        ).fetchall()

        experiments: List[Tuple[float, Experiment]] = []
        for row in rows:
            exp = self._row_to_experiment(row)
            if metric in exp.results:
                experiments.append((float(exp.results[metric]), exp))

        reverse = mode == "max"
        experiments.sort(key=lambda t: t[0], reverse=reverse)
        return [exp for _, exp in experiments[:n]]

    def query(self, filters: Optional[Dict[str, Any]] = None) -> List[Experiment]:
        """Return experiments matching the given *filters*.

        Parameters
        ----------
        filters:
            Mapping of column names to required values.  Supported keys:
            ``"name"``, ``"status"``.  If ``None``, return all experiments.

        Returns
        -------
        list[Experiment]
        """
        sql = "SELECT * FROM experiments"
        params: List[Any] = []
        if filters:
            clauses: List[str] = []
            for col, val in filters.items():
                if col in ("name", "status"):
                    clauses.append(f"{col} = ?")
                    params.append(str(val))
            if clauses:
                sql += " WHERE " + " AND ".join(clauses)

        rows = self._conn.execute(sql, params).fetchall()
        return [self._row_to_experiment(r) for r in rows]

    def to_dataframe(self) -> Any:
        """Return all experiments as a ``pandas.DataFrame``.

        Requires ``pandas`` to be installed.  Returns a DataFrame with
        experiment metadata plus flattened config and result columns.
        """
        import pandas as pd  # optional dependency

        rows = self._conn.execute("SELECT * FROM experiments").fetchall()
        records: List[Dict[str, Any]] = []
        for row in rows:
            exp = self._row_to_experiment(row)
            record: Dict[str, Any] = {
                "id": row["id"],
                "name": exp.name,
                "status": exp.status.value,
                "error": exp.error,
                "created_at": row["created_at"],
            }
            for k, v in exp.config.items():
                record[f"config.{k}"] = v
            for k, v in exp.results.items():
                record[f"result.{k}"] = v
            for k, v in exp.timestamps.items():
                record[f"ts.{k}"] = v
            records.append(record)
        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_experiment(row: sqlite3.Row) -> Experiment:
        return Experiment(
            name=row["name"],
            config=json.loads(row["config"]),
            status=ExperimentStatus(row["status"]),
            results=json.loads(row["results"]),
            timestamps=json.loads(row["timestamps"]),
            error=row["error"],
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "ResultsDB":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass
