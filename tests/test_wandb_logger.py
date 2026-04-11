"""Tests for navirl/logging/wandb_logger.py."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Build a fake wandb module so the real import succeeds with mocks
# ---------------------------------------------------------------------------


def _make_mock_wandb() -> types.ModuleType:
    """Create a mock ``wandb`` module with the attributes the logger needs."""
    mod = types.ModuleType("wandb")

    # AlertLevel enum-like
    alert_level = types.SimpleNamespace(INFO="info", WARN="warn", ERROR="error")
    mod.AlertLevel = alert_level  # type: ignore[attr-defined]

    # Classes
    mod.Histogram = MagicMock(name="wandb.Histogram")  # type: ignore[attr-defined]
    mod.Image = MagicMock(name="wandb.Image")  # type: ignore[attr-defined]
    mod.Table = MagicMock(name="wandb.Table")  # type: ignore[attr-defined]
    mod.Artifact = MagicMock(name="wandb.Artifact")  # type: ignore[attr-defined]

    # Functions
    mod.plot_table = MagicMock(name="wandb.plot_table")  # type: ignore[attr-defined]
    mod.init = MagicMock(name="wandb.init")  # type: ignore[attr-defined]
    mod.sweep = MagicMock(name="wandb.sweep", return_value="sweep-id-123")  # type: ignore[attr-defined]
    mod.alert = MagicMock(name="wandb.alert")  # type: ignore[attr-defined]
    mod.watch = MagicMock(name="wandb.watch")  # type: ignore[attr-defined]

    return mod


_mock_wandb = _make_mock_wandb()

# Inject mock wandb into sys.modules BEFORE importing the logger module,
# then remove it so other test modules that check for wandb availability
# are not affected.
_orig_wandb = sys.modules.get("wandb")
sys.modules["wandb"] = _mock_wandb

# Now import the logger -- it will see wandb as available
from navirl.logging.wandb_logger import (
    AlertManager,
    SweepConfig,
    WandbLogger,
    create_wandb_logger,
    is_wandb_available,
)

# Restore original sys.modules state to avoid polluting other tests
if _orig_wandb is None:
    sys.modules.pop("wandb", None)
else:
    sys.modules["wandb"] = _orig_wandb


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_mock_run() -> MagicMock:
    """Return a mock W&B run object."""
    run = MagicMock(name="wandb_run")
    run.id = "run-abc123"
    run.get_url.return_value = "https://wandb.ai/test/run-abc123"
    run.tags = ("baseline",)
    run.config = {}
    run.summary = {}
    return run


@pytest.fixture()
def mock_run():
    return _make_mock_run()


@pytest.fixture()
def enabled_logger(mock_run):
    """Return a WandbLogger that believes it is enabled, with mocked wandb."""
    with (
        patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
        patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
    ):
        _mock_wandb.init.return_value = mock_run
        lg = WandbLogger(project="test", run_name="unit", enabled=True)
    # Ensure subsequent calls also see the mock
    with (
        patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
    ):
        yield lg


@pytest.fixture()
def disabled_logger():
    """Return a disabled WandbLogger."""
    with (
        patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
        patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
    ):
        lg = WandbLogger(project="test", enabled=False)
    yield lg


# ---------------------------------------------------------------------------
# is_wandb_available
# ---------------------------------------------------------------------------


class TestIsWandbAvailable:
    def test_returns_true_when_available(self):
        with patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True):
            assert is_wandb_available() is True

    def test_returns_false_when_unavailable(self):
        with patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", False):
            assert is_wandb_available() is False


# ---------------------------------------------------------------------------
# SweepConfig
# ---------------------------------------------------------------------------


class TestSweepConfig:
    def test_build_minimal(self):
        cfg = SweepConfig().build()
        assert cfg["method"] == "random"
        assert cfg["metric"] == {"name": "val_loss", "goal": "minimize"}
        assert cfg["parameters"] == {}
        assert "name" not in cfg
        assert "early_terminate" not in cfg

    def test_build_with_name(self):
        cfg = SweepConfig(name="my_sweep").build()
        assert cfg["name"] == "my_sweep"

    def test_add_uniform(self):
        cfg = SweepConfig().add_uniform("lr", 1e-5, 1e-2).build()
        assert cfg["parameters"]["lr"] == {
            "distribution": "uniform",
            "min": 1e-5,
            "max": 1e-2,
        }

    def test_add_log_uniform(self):
        cfg = SweepConfig().add_log_uniform("lr", 1e-6, 1e-1).build()
        assert cfg["parameters"]["lr"]["distribution"] == "log_uniform_values"

    def test_add_categorical(self):
        cfg = SweepConfig().add_categorical("opt", ["adam", "sgd"]).build()
        assert cfg["parameters"]["opt"] == {"values": ["adam", "sgd"]}

    def test_add_int_uniform(self):
        cfg = SweepConfig().add_int_uniform("layers", 1, 8).build()
        p = cfg["parameters"]["layers"]
        assert p["distribution"] == "int_uniform"
        assert p["min"] == 1
        assert p["max"] == 8

    def test_add_constant(self):
        cfg = SweepConfig().add_constant("seed", 42).build()
        assert cfg["parameters"]["seed"] == {"value": 42}

    def test_set_early_terminate(self):
        cfg = SweepConfig().set_early_terminate(min_iter=5, eta=2, s=1).build()
        et = cfg["early_terminate"]
        assert et["type"] == "hyperband"
        assert et["min_iter"] == 5
        assert et["eta"] == 2
        assert et["s"] == 1

    def test_fluent_chaining(self):
        sc = SweepConfig("bayes", "eval/reward", "maximize")
        result = (
            sc.add_uniform("lr", 1e-5, 1e-2)
            .add_categorical("batch_size", [32, 64])
            .add_constant("seed", 0)
            .set_early_terminate()
        )
        # All chained methods return the same instance
        assert result is sc
        cfg = result.build()
        assert len(cfg["parameters"]) == 3
        assert "early_terminate" in cfg

    def test_custom_method_and_metric(self):
        cfg = SweepConfig("grid", "loss", "minimize").build()
        assert cfg["method"] == "grid"
        assert cfg["metric"]["name"] == "loss"
        assert cfg["metric"]["goal"] == "minimize"


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------


class TestAlertManager:
    def test_send_info(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.send("title", "body", level="INFO")
            _mock_wandb.alert.assert_called_once_with(
                title="title",
                text="body",
                level=_mock_wandb.AlertLevel.INFO,
                wait_duration=0.0,
            )

    def test_send_warn(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.send("w", "warn text", level="WARN", wait_duration=5.0)
            _mock_wandb.alert.assert_called_once()
            call_kwargs = _mock_wandb.alert.call_args.kwargs
            assert call_kwargs["level"] == _mock_wandb.AlertLevel.WARN
            assert call_kwargs["wait_duration"] == 5.0

    def test_send_error(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.send("e", "err", level="ERROR")
            assert _mock_wandb.alert.call_args.kwargs["level"] == _mock_wandb.AlertLevel.ERROR

    def test_send_disabled_is_noop(self):
        lg = MagicMock()
        lg.enabled = False
        am = AlertManager(lg)
        with patch("navirl.logging.wandb_logger.wandb", _mock_wandb):
            _mock_wandb.alert.reset_mock()
            am.send("t", "b")
            _mock_wandb.alert.assert_not_called()

    def test_on_metric_threshold_above_triggered(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.on_metric_threshold("loss", 10.0, 5.0, direction="above")
            _mock_wandb.alert.assert_called_once()
            assert "loss" in _mock_wandb.alert.call_args.kwargs["title"]

    def test_on_metric_threshold_above_not_triggered(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.on_metric_threshold("loss", 3.0, 5.0, direction="above")
            _mock_wandb.alert.assert_not_called()

    def test_on_metric_threshold_below_triggered(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.on_metric_threshold("acc", 0.1, 0.5, direction="below")
            _mock_wandb.alert.assert_called_once()

    def test_on_metric_threshold_below_not_triggered(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="t", enabled=True)
            _mock_wandb.alert.reset_mock()
            lg.alerts.on_metric_threshold("acc", 0.9, 0.5, direction="below")
            _mock_wandb.alert.assert_not_called()


# ---------------------------------------------------------------------------
# WandbLogger
# ---------------------------------------------------------------------------


class TestWandbLoggerConstruction:
    def test_enabled_construction(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            assert lg.enabled is True
            assert lg.run is mock_run

    def test_disabled_construction(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = WandbLogger(project="p", enabled=False)
            assert lg.enabled is False
            assert lg.run is None

    def test_raises_import_error_when_wandb_unavailable_and_not_disabled(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", False),
            pytest.raises(ImportError, match="wandb is not installed"),
        ):
            WandbLogger(project="p", enabled=True, mode="online")

    def test_no_error_when_mode_disabled_and_wandb_unavailable(self):
        with patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", False):
            lg = WandbLogger(project="p", enabled=True, mode="disabled")
            assert lg.enabled is False


class TestWandbLoggerProperties:
    def test_run_id(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            assert lg.run_id == "run-abc123"

    def test_run_id_none_when_disabled(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = WandbLogger(project="p", enabled=False)
            assert lg.run_id is None

    def test_run_url(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            assert lg.run_url == "https://wandb.ai/test/run-abc123"

    def test_run_url_none_when_disabled(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = WandbLogger(project="p", enabled=False)
            assert lg.run_url is None

    def test_alerts_property(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            assert isinstance(lg.alerts, AlertManager)

    def test_is_closed_initially_false(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            assert lg.is_closed is False


class TestWandbLoggerContextManager:
    def test_context_manager_returns_self_and_finishes(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            with lg as ctx:
                assert ctx is lg
                assert lg.is_closed is False
            assert lg.is_closed is True
            mock_run.finish.assert_called_once()


class TestWandbLoggerConfig:
    def test_update_config(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.config = MagicMock()
            lg = WandbLogger(project="p", enabled=True)
            lg.update_config({"lr": 0.001})
            mock_run.config.update.assert_called_with({"lr": 0.001})

    def test_update_config_disabled_is_noop(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = WandbLogger(project="p", enabled=False)
            # Should not raise
            lg.update_config({"lr": 0.001})

    def test_set_config(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.config = {}
            lg = WandbLogger(project="p", enabled=True)
            lg.set_config({"a": 1, "b": 2})
            assert mock_run.config["a"] == 1
            assert mock_run.config["b"] == 2


class TestWandbLoggerLog:
    def test_log_with_step(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log({"loss": 0.5}, step=10)
            mock_run.log.assert_called_with({"loss": 0.5}, step=10, commit=True)

    def test_log_auto_step_increments(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log({"a": 1})  # step=0
            lg.log({"b": 2})  # step=1
            calls = mock_run.log.call_args_list
            assert calls[0][1]["step"] == 0
            assert calls[1][1]["step"] == 1

    def test_log_commit_false_does_not_increment(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log({"a": 1}, commit=False)  # step=0, no increment
            lg.log({"b": 2})  # step=0 still
            calls = mock_run.log.call_args_list
            assert calls[0][1]["step"] == 0
            assert calls[1][1]["step"] == 0

    def test_log_disabled_is_noop(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = WandbLogger(project="p", enabled=False)
            lg.log({"x": 1})  # should not raise

    def test_log_scalar(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_scalar("metric", 3.14, step=5)
            mock_run.log.assert_called_with({"metric": 3.14}, step=5, commit=True)

    def test_log_scalars_with_prefix(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_scalars({"a": 1.0, "b": 2.0}, step=0, prefix="train")
            call_data = mock_run.log.call_args[0][0]
            assert "train/a" in call_data
            assert "train/b" in call_data

    def test_log_scalars_without_prefix(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_scalars({"x": 5.0}, step=0)
            call_data = mock_run.log.call_args[0][0]
            assert "x" in call_data


class TestWandbLoggerTrainingHelpers:
    def test_log_training_step_basic(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_training_step(step=1, loss=0.5)
            call_data = mock_run.log.call_args[0][0]
            assert call_data["train/loss"] == 0.5
            assert "train/lr" not in call_data
            assert "train/grad_norm" not in call_data

    def test_log_training_step_with_optional_params(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_training_step(step=2, loss=0.3, lr=1e-4, grad_norm=0.1, extra={"entropy": 0.9})
            call_data = mock_run.log.call_args[0][0]
            assert call_data["train/loss"] == 0.3
            assert call_data["train/lr"] == 1e-4
            assert call_data["train/grad_norm"] == 0.1
            assert call_data["train/entropy"] == 0.9

    def test_log_evaluation(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_evaluation(step=10, metrics={"reward": 5.0, "success": 0.8})
            call_data = mock_run.log.call_args[0][0]
            assert "eval/reward" in call_data
            assert "eval/success" in call_data

    def test_log_episode_basic(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_episode(episode=5, reward=10.0, length=100)
            call_data = mock_run.log.call_args[0][0]
            assert call_data["episode/reward"] == 10.0
            assert call_data["episode/length"] == 100
            assert "episode/success" not in call_data

    def test_log_episode_with_success_and_extra(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.log_episode(episode=5, reward=10.0, length=100, success=True, extra={"col": 3.0})
            call_data = mock_run.log.call_args[0][0]
            assert call_data["episode/success"] == 1.0
            assert call_data["episode/col"] == 3.0


class TestWandbLoggerHistogram:
    def test_log_histogram(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Histogram.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            lg.log_histogram("h", [1.0, 2.0, 3.0], step=0, num_bins=32)
            _mock_wandb.Histogram.assert_called_once()
            call_kwargs = _mock_wandb.Histogram.call_args
            assert call_kwargs[1]["num_bins"] == 32

    def test_log_reward_distribution(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Histogram.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            lg.log_reward_distribution(step=0, rewards=np.array([1.0, 2.0]))
            _mock_wandb.Histogram.assert_called_once()


class TestWandbLoggerTable:
    def test_log_table(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Table.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            lg.log_table("t", ["a", "b"], [[1, 2], [3, 4]], step=0)
            _mock_wandb.Table.assert_called_once_with(columns=["a", "b"], data=[[1, 2], [3, 4]])

    def test_log_trajectory_table(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Table.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            positions = np.array([[0.0, 0.0], [3.0, 4.0]])
            velocities = np.array([[1.0, 0.0], [0.6, 0.8]])
            rewards = np.array([1.0, 2.0])
            lg.log_trajectory_table(
                step=0,
                agent_id=1,
                positions=positions,
                velocities=velocities,
                rewards=rewards,
            )
            _mock_wandb.Table.assert_called_once()
            call_kwargs = _mock_wandb.Table.call_args[1]
            cols = call_kwargs["columns"]
            assert "speed" in cols
            assert "reward" in cols
            rows = call_kwargs["data"]
            assert len(rows) == 2
            # Check speed computation for second row: sqrt(0.6^2 + 0.8^2) = 1.0
            speed_idx = cols.index("speed")
            assert rows[1][speed_idx] == pytest.approx(1.0)

    def test_log_trajectory_table_positions_only(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Table.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            positions = np.array([[1.0, 2.0]])
            lg.log_trajectory_table(step=0, agent_id=0, positions=positions)
            call_kwargs = _mock_wandb.Table.call_args[1]
            cols = call_kwargs["columns"]
            assert cols == ["timestep", "agent_id", "x", "y"]


class TestWandbLoggerImages:
    def test_log_image(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Image.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            img = np.zeros((64, 64, 3), dtype=np.uint8)
            lg.log_image("obs", img, step=0, caption="frame 0")
            _mock_wandb.Image.assert_called_once_with(img, caption="frame 0")

    def test_log_images_with_captions(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Image.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            imgs = [np.zeros((8, 8, 3)) for _ in range(3)]
            lg.log_images("batch", imgs, step=0, captions=["a", "b", "c"])
            assert _mock_wandb.Image.call_count == 3

    def test_log_images_without_captions(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Image.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            imgs = [np.zeros((8, 8, 3))]
            lg.log_images("batch", imgs, step=0)
            _mock_wandb.Image.assert_called_once_with(imgs[0], caption=None)


class TestWandbLoggerArtifacts:
    def test_log_artifact_file(self, mock_run, tmp_path):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            art = MagicMock()
            _mock_wandb.Artifact.return_value = art
            lg = WandbLogger(project="p", enabled=True)

            f = tmp_path / "data.csv"
            f.write_text("a,b\n1,2\n")
            lg.log_artifact(str(f), name="mydata", type="dataset")
            _mock_wandb.Artifact.assert_called()
            art.add_file.assert_called_once_with(str(f))
            mock_run.log_artifact.assert_called_once()

    def test_log_artifact_directory(self, mock_run, tmp_path):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            art = MagicMock()
            _mock_wandb.Artifact.return_value = art
            lg = WandbLogger(project="p", enabled=True)

            d = tmp_path / "mydir"
            d.mkdir()
            (d / "f.txt").write_text("hi")
            lg.log_artifact(str(d), name="dir_art")
            art.add_dir.assert_called_once_with(str(d))

    def test_log_model_checkpoint_adds_latest_alias(self, mock_run, tmp_path):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            art = MagicMock()
            _mock_wandb.Artifact.return_value = art
            lg = WandbLogger(project="p", enabled=True)

            f = tmp_path / "model.pt"
            f.write_text("weights")
            lg.log_model_checkpoint(str(f), step=100)
            call_kwargs = mock_run.log_artifact.call_args[1]
            assert "latest" in call_kwargs["aliases"]

    def test_log_model_checkpoint_preserves_existing_aliases(self, mock_run, tmp_path):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            art = MagicMock()
            _mock_wandb.Artifact.return_value = art
            lg = WandbLogger(project="p", enabled=True)

            f = tmp_path / "model.pt"
            f.write_text("weights")
            lg.log_model_checkpoint(str(f), aliases=["best"])
            call_kwargs = mock_run.log_artifact.call_args[1]
            assert "best" in call_kwargs["aliases"]
            assert "latest" in call_kwargs["aliases"]

    def test_use_artifact(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            art_mock = MagicMock()
            art_mock.download.return_value = "/tmp/artifacts/model"
            mock_run.use_artifact.return_value = art_mock
            lg = WandbLogger(project="p", enabled=True)

            result = lg.use_artifact("model", type="model")
            mock_run.use_artifact.assert_called_with("model:latest", type="model")
            assert result == Path("/tmp/artifacts/model")

    def test_use_artifact_with_version(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            art_mock = MagicMock()
            art_mock.download.return_value = "/tmp/art"
            mock_run.use_artifact.return_value = art_mock
            lg = WandbLogger(project="p", enabled=True)

            lg.use_artifact("model:v3")
            mock_run.use_artifact.assert_called_with("model:v3", type=None)

    def test_use_artifact_disabled_returns_none(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = WandbLogger(project="p", enabled=False)
            assert lg.use_artifact("x") is None


class TestWandbLoggerSummary:
    def test_set_summary(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.summary = {}
            lg = WandbLogger(project="p", enabled=True)
            lg.set_summary("best_reward", 99.0)
            assert mock_run.summary["best_reward"] == 99.0

    def test_set_summaries(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.summary = {}
            lg = WandbLogger(project="p", enabled=True)
            lg.set_summaries({"a": 1, "b": 2})
            assert mock_run.summary["a"] == 1
            assert mock_run.summary["b"] == 2


class TestWandbLoggerTags:
    def test_add_tags(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.tags = ("existing",)
            lg = WandbLogger(project="p", enabled=True)
            lg.add_tags(["new1", "new2"])
            assert set(mock_run.tags) == {"existing", "new1", "new2"}

    def test_remove_tags(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.tags = ("a", "b", "c")
            lg = WandbLogger(project="p", enabled=True)
            lg.remove_tags(["b"])
            assert "b" not in set(mock_run.tags)
            assert "a" in set(mock_run.tags)


class TestWandbLoggerSweep:
    def test_create_sweep_with_dict(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.sweep.reset_mock()
            _mock_wandb.sweep.return_value = "sweep-id"
            result = WandbLogger.create_sweep({"method": "random"}, project="p")
            assert result == "sweep-id"
            _mock_wandb.sweep.assert_called_once()

    def test_create_sweep_with_sweep_config(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.sweep.reset_mock()
            _mock_wandb.sweep.return_value = "sweep-id-2"
            sc = SweepConfig("bayes", "loss", "minimize").add_uniform("lr", 0.0, 1.0)
            result = WandbLogger.create_sweep(sc, project="p", entity="team")
            assert result == "sweep-id-2"
            call_kwargs = _mock_wandb.sweep.call_args[1]
            assert call_kwargs["project"] == "p"
            assert call_kwargs["entity"] == "team"
            assert call_kwargs["sweep"]["method"] == "bayes"

    def test_create_sweep_unavailable_returns_none(self):
        with patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", False):
            result = WandbLogger.create_sweep({"method": "random"})
            assert result is None


class TestWandbLoggerCustomCharts:
    def test_define_and_log_custom_chart(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Table.reset_mock()
            _mock_wandb.plot_table.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            lg.define_custom_chart("my_chart", {"$schema": "vega-lite"})
            lg.log_custom_chart("my_chart", [{"x": 1, "y": 2}], step=0)
            _mock_wandb.plot_table.assert_called_once()

    def test_log_custom_chart_without_definition_falls_back_to_table(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.Table.reset_mock()
            _mock_wandb.plot_table.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            lg.log_custom_chart("undefined_chart", [{"a": 1}], step=0)
            _mock_wandb.Table.assert_called_once()
            _mock_wandb.plot_table.assert_not_called()

    def test_log_custom_chart_empty_data_is_noop(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            mock_run.log.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            lg.log_custom_chart("chart", [], step=0)
            mock_run.log.assert_not_called()


class TestWandbLoggerContextManagers:
    def test_train_step_context(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            with lg.train_step_context(step=5) as m:
                m["loss"] = 0.1
            call_data = mock_run.log.call_args[0][0]
            assert "train/loss" in call_data
            assert "train/step_time" in call_data
            assert call_data["train/loss"] == 0.1

    def test_eval_context(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            with lg.eval_context(step=10) as m:
                m["reward"] = 42.0
            call_data = mock_run.log.call_args[0][0]
            assert "eval/reward" in call_data
            assert "eval/eval_time" in call_data


class TestWandbLoggerWatchModel:
    def test_watch_model(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            _mock_wandb.watch.reset_mock()
            lg = WandbLogger(project="p", enabled=True)
            fake_model = MagicMock()
            lg.watch_model(fake_model, log="gradients", log_freq=50, log_graph=True)
            _mock_wandb.watch.assert_called_once_with(
                fake_model,
                log="gradients",
                log_freq=50,
                log_graph=True,
            )

    def test_watch_model_disabled_is_noop(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.watch.reset_mock()
            lg = WandbLogger(project="p", enabled=False)
            lg.watch_model(MagicMock())
            _mock_wandb.watch.assert_not_called()


class TestWandbLoggerFinish:
    def test_finish(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.finish(exit_code=0, quiet=True)
            mock_run.finish.assert_called_once_with(exit_code=0, quiet=True)
            assert lg.is_closed is True

    def test_double_finish_is_safe(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.finish()
            lg.finish()  # second call should be no-op
            mock_run.finish.assert_called_once()

    def test_finish_without_args(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = WandbLogger(project="p", enabled=True)
            lg.finish()
            mock_run.finish.assert_called_once_with()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


class TestCreateWandbLogger:
    def test_creates_enabled_logger(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.return_value = mock_run
            lg = create_wandb_logger(project="test", enabled=True)
            assert lg.enabled is True

    def test_falls_back_to_disabled_when_unavailable(self):
        with patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", False):
            with pytest.warns(UserWarning, match="wandb is not installed"):
                lg = create_wandb_logger(project="test", enabled=True)
            assert lg.enabled is False

    def test_disabled_explicitly(self):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            lg = create_wandb_logger(project="test", enabled=False, mode="disabled")
            assert lg.enabled is False

    def test_mode_defaults_to_disabled_when_unavailable(self):
        with patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", False):
            with pytest.warns(UserWarning):
                lg = create_wandb_logger(project="test", enabled=True)
            # Should not raise ImportError because mode defaults to "disabled"
            assert lg.enabled is False


class TestWandbLoggerInitKwargs:
    """Verify that optional init kwargs (entity, dir, resume, run_id) are passed through."""

    def test_entity_passed(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.reset_mock()
            _mock_wandb.init.return_value = mock_run
            WandbLogger(project="p", entity="myteam", enabled=True)
            call_kwargs = _mock_wandb.init.call_args[1]
            assert call_kwargs["entity"] == "myteam"

    def test_dir_passed(self, mock_run, tmp_path):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.reset_mock()
            _mock_wandb.init.return_value = mock_run
            WandbLogger(project="p", dir=tmp_path, enabled=True)
            call_kwargs = _mock_wandb.init.call_args[1]
            assert call_kwargs["dir"] == str(tmp_path)

    def test_resume_and_run_id_passed(self, mock_run):
        with (
            patch("navirl.logging.wandb_logger._WANDB_AVAILABLE", True),
            patch("navirl.logging.wandb_logger.wandb", _mock_wandb),
        ):
            _mock_wandb.init.reset_mock()
            _mock_wandb.init.return_value = mock_run
            WandbLogger(project="p", resume="must", run_id="abc", enabled=True)
            call_kwargs = _mock_wandb.init.call_args[1]
            assert call_kwargs["resume"] == "must"
            assert call_kwargs["id"] == "abc"
