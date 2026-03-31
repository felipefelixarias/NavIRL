"""Runtime safety monitoring for navigation agents.

Tracks safety metrics, records violations, and logs safety-relevant events
during execution.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


class Severity(str, Enum):
    """Severity levels for safety alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class SafetyAlert:
    """Record of a single safety event.

    Attributes
    ----------
    timestamp : float
        Wall-clock or simulation time of the event.
    severity : Severity
        How serious the event is.
    constraint_name : str
        Which constraint triggered the alert.
    details : dict[str, Any]
        Arbitrary extra information (state, action, distances, etc.).
    """

    timestamp: float
    severity: Severity
    constraint_name: str
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# SafetyMonitor
# ---------------------------------------------------------------------------


class SafetyMonitor:
    """Tracks safety metrics during execution.

    Call :meth:`record_step` after every environment step.  At any point you
    can query accumulated violations and aggregate statistics.
    """

    def __init__(self) -> None:
        self._violations: list[SafetyAlert] = []
        self._step_count: int = 0
        self._min_obstacle_distances: list[float] = []
        self._speeds: list[float] = []
        self._shield_interventions: int = 0

    # -- recording ----------------------------------------------------------

    def record_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> None:
        """Record a single environment step.

        Parameters
        ----------
        state : np.ndarray
            Agent state after the step.
        action : np.ndarray
            Action that was executed.
        info : dict, optional
            Extra information from the environment / shield.  Recognised keys:
            ``"min_obstacle_dist"``, ``"shield_intervened"``,
            ``"violation"`` (a :class:`SafetyAlert` or dict).
        """
        info = info or {}
        self._step_count += 1

        # Speed tracking.
        speed = float(np.linalg.norm(action[:2])) if action.shape[0] >= 2 else 0.0
        self._speeds.append(speed)

        # Obstacle distance tracking.
        if "min_obstacle_dist" in info:
            self._min_obstacle_distances.append(float(info["min_obstacle_dist"]))

        # Shield intervention tracking.
        if info.get("shield_intervened", False):
            self._shield_interventions += 1

        # Explicit violation.
        violation = info.get("violation")
        if violation is not None:
            if isinstance(violation, SafetyAlert):
                self._violations.append(violation)
            elif isinstance(violation, dict):
                self._violations.append(
                    SafetyAlert(
                        timestamp=violation.get("timestamp", time.time()),
                        severity=Severity(violation.get("severity", "warning")),
                        constraint_name=violation.get("constraint_name", "unknown"),
                        details=violation.get("details", {}),
                    )
                )

    # -- queries ------------------------------------------------------------

    def get_violations(self) -> list[SafetyAlert]:
        """Return all recorded violations."""
        return list(self._violations)

    def get_statistics(self) -> dict[str, Any]:
        """Return aggregate safety statistics.

        Returns a dict with keys such as ``total_steps``,
        ``num_violations``, ``violation_rate``, ``shield_intervention_rate``,
        ``mean_speed``, ``max_speed``, ``min_obstacle_distance``, etc.
        """
        stats: dict[str, Any] = {
            "total_steps": self._step_count,
            "num_violations": len(self._violations),
            "violation_rate": (
                len(self._violations) / self._step_count if self._step_count > 0 else 0.0
            ),
            "shield_interventions": self._shield_interventions,
            "shield_intervention_rate": (
                self._shield_interventions / self._step_count if self._step_count > 0 else 0.0
            ),
        }

        if self._speeds:
            stats["mean_speed"] = float(np.mean(self._speeds))
            stats["max_speed"] = float(np.max(self._speeds))

        if self._min_obstacle_distances:
            stats["min_obstacle_distance"] = float(np.min(self._min_obstacle_distances))
            stats["mean_obstacle_distance"] = float(np.mean(self._min_obstacle_distances))

        # Break down violations by severity.
        for sev in Severity:
            count = sum(1 for v in self._violations if v.severity == sev)
            stats[f"violations_{sev.value}"] = count

        return stats

    def reset(self) -> None:
        """Clear all recorded data."""
        self._violations.clear()
        self._step_count = 0
        self._min_obstacle_distances.clear()
        self._speeds.clear()
        self._shield_interventions = 0


# ---------------------------------------------------------------------------
# SafetyLogger
# ---------------------------------------------------------------------------


class SafetyLogger:
    """Logs safety events using the standard :mod:`logging` module.

    Parameters
    ----------
    name : str
        Logger name (passed to ``logging.getLogger``).
    """

    _SEVERITY_MAP = {
        Severity.INFO: logging.INFO,
        Severity.WARNING: logging.WARNING,
        Severity.CRITICAL: logging.CRITICAL,
    }

    def __init__(self, name: str = "navirl.safety") -> None:
        self._logger = logging.getLogger(name)

    def log_alert(self, alert: SafetyAlert) -> None:
        """Log a :class:`SafetyAlert` at the appropriate severity level."""
        level = self._SEVERITY_MAP.get(alert.severity, logging.WARNING)
        self._logger.log(
            level,
            "[%s] constraint=%s | %s",
            alert.severity.value.upper(),
            alert.constraint_name,
            alert.details,
        )

    def log_statistics(self, stats: dict[str, Any]) -> None:
        """Log aggregate safety statistics at INFO level."""
        self._logger.info("Safety statistics: %s", stats)
