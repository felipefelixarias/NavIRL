"""Extended tests for navirl/safety/monitoring.py.

Covers: SafetyMonitor.get_statistics(), get_violations(), reset(),
violation dict-to-SafetyAlert conversion, SafetyLogger.log_alert()
and log_statistics(), severity breakdown, and edge cases.
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pytest

from navirl.safety.monitoring import SafetyAlert, SafetyLogger, SafetyMonitor, Severity

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def monitor() -> SafetyMonitor:
    return SafetyMonitor()


@pytest.fixture()
def logger() -> SafetyLogger:
    return SafetyLogger()


def _step(action=(1.0, 0.0), **info_kw):
    """Helper to build state/action/info for record_step."""
    return {
        "state": np.array([0.0, 0.0], dtype=np.float64),
        "action": np.array(action, dtype=np.float64),
        "info": info_kw if info_kw else None,
    }


# ---------------------------------------------------------------------------
# Severity enum
# ---------------------------------------------------------------------------


class TestSeverity:
    def test_values(self):
        assert Severity.INFO == "info"
        assert Severity.WARNING == "warning"
        assert Severity.CRITICAL == "critical"

    def test_str_enum(self):
        assert str(Severity.INFO) == "info"


# ---------------------------------------------------------------------------
# SafetyMonitor.get_violations
# ---------------------------------------------------------------------------


class TestGetViolations:
    def test_empty_initially(self, monitor):
        assert monitor.get_violations() == []

    def test_returns_alert_objects(self, monitor):
        alert = SafetyAlert(timestamp=1.0, severity=Severity.WARNING, constraint_name="test")
        monitor.record_step(np.zeros(2), np.zeros(2), {"violation": alert})
        violations = monitor.get_violations()
        assert len(violations) == 1
        assert violations[0].constraint_name == "test"

    def test_violation_from_dict(self, monitor):
        v_dict = {
            "timestamp": 42.0,
            "severity": "critical",
            "constraint_name": "collision",
            "details": {"distance": 0.01},
        }
        monitor.record_step(np.zeros(2), np.zeros(2), {"violation": v_dict})
        violations = monitor.get_violations()
        assert len(violations) == 1
        assert violations[0].severity == Severity.CRITICAL
        assert violations[0].constraint_name == "collision"
        assert violations[0].details == {"distance": 0.01}
        assert violations[0].timestamp == 42.0

    def test_violation_dict_defaults(self, monitor):
        """Dict missing keys should use defaults."""
        monitor.record_step(np.zeros(2), np.zeros(2), {"violation": {}})
        violations = monitor.get_violations()
        assert len(violations) == 1
        assert violations[0].severity == Severity.WARNING
        assert violations[0].constraint_name == "unknown"
        assert violations[0].details == {}

    def test_returns_copy(self, monitor):
        alert = SafetyAlert(timestamp=1.0, severity=Severity.INFO, constraint_name="x")
        monitor.record_step(np.zeros(2), np.zeros(2), {"violation": alert})
        v1 = monitor.get_violations()
        v2 = monitor.get_violations()
        assert v1 is not v2

    def test_multiple_violations(self, monitor):
        for i in range(5):
            alert = SafetyAlert(
                timestamp=float(i), severity=Severity.WARNING, constraint_name=f"c{i}"
            )
            monitor.record_step(np.zeros(2), np.zeros(2), {"violation": alert})
        assert len(monitor.get_violations()) == 5


# ---------------------------------------------------------------------------
# SafetyMonitor.get_statistics
# ---------------------------------------------------------------------------


class TestGetStatistics:
    def test_empty_stats(self, monitor):
        stats = monitor.get_statistics()
        assert stats["total_steps"] == 0
        assert stats["num_violations"] == 0
        assert stats["violation_rate"] == 0.0
        assert stats["shield_interventions"] == 0
        assert stats["shield_intervention_rate"] == 0.0
        # No speed or obstacle stats when empty
        assert "mean_speed" not in stats
        assert "min_obstacle_distance" not in stats

    def test_step_counting(self, monitor):
        for _ in range(10):
            monitor.record_step(np.zeros(2), np.array([1.0, 0.0]))
        stats = monitor.get_statistics()
        assert stats["total_steps"] == 10

    def test_speed_tracking(self, monitor):
        monitor.record_step(np.zeros(2), np.array([3.0, 4.0]))  # speed = 5.0
        monitor.record_step(np.zeros(2), np.array([0.0, 0.0]))  # speed = 0.0
        stats = monitor.get_statistics()
        assert stats["mean_speed"] == pytest.approx(2.5)
        assert stats["max_speed"] == pytest.approx(5.0)

    def test_obstacle_distance(self, monitor):
        monitor.record_step(np.zeros(2), np.zeros(2), {"min_obstacle_dist": 0.5})
        monitor.record_step(np.zeros(2), np.zeros(2), {"min_obstacle_dist": 1.5})
        monitor.record_step(np.zeros(2), np.zeros(2), {"min_obstacle_dist": 1.0})
        stats = monitor.get_statistics()
        assert stats["min_obstacle_distance"] == pytest.approx(0.5)
        assert stats["mean_obstacle_distance"] == pytest.approx(1.0)

    def test_shield_intervention_rate(self, monitor):
        monitor.record_step(np.zeros(2), np.zeros(2), {"shield_intervened": True})
        monitor.record_step(np.zeros(2), np.zeros(2), {"shield_intervened": False})
        monitor.record_step(np.zeros(2), np.zeros(2))
        monitor.record_step(np.zeros(2), np.zeros(2), {"shield_intervened": True})
        stats = monitor.get_statistics()
        assert stats["shield_interventions"] == 2
        assert stats["shield_intervention_rate"] == pytest.approx(0.5)

    def test_violation_rate(self, monitor):
        for i in range(4):
            info = {}
            if i % 2 == 0:
                info["violation"] = SafetyAlert(
                    timestamp=float(i), severity=Severity.WARNING, constraint_name="c"
                )
            monitor.record_step(np.zeros(2), np.zeros(2), info or None)
        stats = monitor.get_statistics()
        assert stats["num_violations"] == 2
        assert stats["violation_rate"] == pytest.approx(0.5)

    def test_severity_breakdown(self, monitor):
        severities = [Severity.INFO, Severity.WARNING, Severity.WARNING, Severity.CRITICAL]
        for i, sev in enumerate(severities):
            alert = SafetyAlert(timestamp=float(i), severity=sev, constraint_name="c")
            monitor.record_step(np.zeros(2), np.zeros(2), {"violation": alert})
        stats = monitor.get_statistics()
        assert stats["violations_info"] == 1
        assert stats["violations_warning"] == 2
        assert stats["violations_critical"] == 1

    def test_severity_breakdown_empty(self, monitor):
        """All severity counts should be 0 with no violations."""
        monitor.record_step(np.zeros(2), np.zeros(2))
        stats = monitor.get_statistics()
        assert stats["violations_info"] == 0
        assert stats["violations_warning"] == 0
        assert stats["violations_critical"] == 0


# ---------------------------------------------------------------------------
# SafetyMonitor.reset
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_everything(self, monitor):
        alert = SafetyAlert(timestamp=1.0, severity=Severity.CRITICAL, constraint_name="c")
        monitor.record_step(
            np.zeros(2),
            np.array([1.0, 0.0]),
            {
                "violation": alert,
                "shield_intervened": True,
                "min_obstacle_dist": 0.3,
            },
        )
        monitor.reset()

        stats = monitor.get_statistics()
        assert stats["total_steps"] == 0
        assert stats["num_violations"] == 0
        assert stats["shield_interventions"] == 0
        assert "mean_speed" not in stats
        assert "min_obstacle_distance" not in stats
        assert monitor.get_violations() == []

    def test_record_after_reset(self, monitor):
        monitor.record_step(np.zeros(2), np.array([1.0, 0.0]))
        monitor.reset()
        monitor.record_step(np.zeros(2), np.array([2.0, 0.0]))
        stats = monitor.get_statistics()
        assert stats["total_steps"] == 1
        assert stats["mean_speed"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# SafetyLogger
# ---------------------------------------------------------------------------


class TestSafetyLogger:
    def test_log_alert_info(self, logger, caplog):
        alert = SafetyAlert(timestamp=1.0, severity=Severity.INFO, constraint_name="speed_check")
        with caplog.at_level(logging.DEBUG, logger="navirl.safety"):
            logger.log_alert(alert)
        assert "speed_check" in caplog.text
        assert "INFO" in caplog.text

    def test_log_alert_warning(self, logger, caplog):
        alert = SafetyAlert(timestamp=1.0, severity=Severity.WARNING, constraint_name="distance")
        with caplog.at_level(logging.DEBUG, logger="navirl.safety"):
            logger.log_alert(alert)
        assert "distance" in caplog.text
        assert "WARNING" in caplog.text

    def test_log_alert_critical(self, logger, caplog):
        alert = SafetyAlert(
            timestamp=1.0,
            severity=Severity.CRITICAL,
            constraint_name="collision",
            details={"agent": "robot_1"},
        )
        with caplog.at_level(logging.DEBUG, logger="navirl.safety"):
            logger.log_alert(alert)
        assert "collision" in caplog.text
        assert "CRITICAL" in caplog.text

    def test_log_statistics(self, logger, caplog):
        stats = {"total_steps": 100, "num_violations": 3}
        with caplog.at_level(logging.DEBUG, logger="navirl.safety"):
            logger.log_statistics(stats)
        assert "Safety statistics" in caplog.text
        assert "100" in caplog.text

    def test_custom_logger_name(self):
        custom = SafetyLogger(name="custom.safety")
        assert custom._logger.name == "custom.safety"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_1d_action_speed(self, monitor):
        """Action with single element should give speed 0."""
        monitor.record_step(np.zeros(2), np.array([5.0]))
        stats = monitor.get_statistics()
        assert stats["mean_speed"] == pytest.approx(0.0)

    def test_3d_action_uses_first_two(self, monitor):
        """Speed should be norm of first two action components."""
        monitor.record_step(np.zeros(2), np.array([3.0, 4.0, 99.0]))
        stats = monitor.get_statistics()
        assert stats["mean_speed"] == pytest.approx(5.0)

    def test_none_info_handled(self, monitor):
        monitor.record_step(np.zeros(2), np.zeros(2), None)
        assert monitor.get_statistics()["total_steps"] == 1

    def test_no_violation_key(self, monitor):
        monitor.record_step(np.zeros(2), np.zeros(2), {"other_key": 42})
        assert len(monitor.get_violations()) == 0
