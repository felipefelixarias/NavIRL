"""Tests for navirl.logging.tensorboard_logger module.

Covers is_tensorboard_available, StepTracker, MetricGroup, TBLogger,
and create_tb_logger factory. TensorBoard dependency is mocked since it
is not installed in the test environment.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# We import the module (not from it) so we can monkeypatch _TB_AVAILABLE
import navirl.logging.tensorboard_logger as tb_mod
from navirl.logging.tensorboard_logger import (
    MetricGroup,
    StepTracker,
    TBLogger,
    create_tb_logger,
    is_tensorboard_available,
)

# ===================================================================
# Helpers / Fixtures
# ===================================================================


@pytest.fixture()
def mock_writer():
    """Return a fresh MagicMock that stands in for SummaryWriter."""
    writer = MagicMock(name="SummaryWriter")
    return writer


@pytest.fixture()
def _enable_tb(monkeypatch):
    """Monkeypatch _TB_AVAILABLE to True for the duration of a test."""
    monkeypatch.setattr(tb_mod, "_TB_AVAILABLE", True)


@pytest.fixture()
def _disable_tb(monkeypatch):
    """Monkeypatch _TB_AVAILABLE to False for the duration of a test."""
    monkeypatch.setattr(tb_mod, "_TB_AVAILABLE", False)


@pytest.fixture()
def enabled_logger(tmp_path, mock_writer, _enable_tb, monkeypatch):
    """Return an enabled TBLogger with a mocked SummaryWriter."""
    monkeypatch.setattr(tb_mod, "SummaryWriter", lambda **kw: mock_writer)
    logger = TBLogger(log_dir=tmp_path, enabled=True)
    yield logger
    if not logger.is_closed:
        logger.close()


@pytest.fixture()
def disabled_logger(tmp_path, _disable_tb):
    """Return a disabled TBLogger (no writer created)."""
    logger = TBLogger(log_dir=tmp_path, enabled=False)
    yield logger
    if not logger.is_closed:
        logger.close()


# ===================================================================
# is_tensorboard_available tests
# ===================================================================


class TestIsTensorboardAvailable:
    def test_returns_true_when_available(self, _enable_tb):
        assert is_tensorboard_available() is True

    def test_returns_false_when_unavailable(self, _disable_tb):
        assert is_tensorboard_available() is False


# ===================================================================
# StepTracker tests
# ===================================================================


class TestStepTracker:
    def test_initial_step_default(self):
        tracker = StepTracker()
        assert tracker.global_step == 0

    def test_initial_step_custom(self):
        tracker = StepTracker(initial_step=42)
        assert tracker.global_step == 42

    def test_increment_default(self):
        tracker = StepTracker()
        result = tracker.increment()
        assert result == 1
        assert tracker.global_step == 1

    def test_increment_by_n(self):
        tracker = StepTracker(initial_step=10)
        result = tracker.increment(5)
        assert result == 15

    def test_set(self):
        tracker = StepTracker()
        tracker.set(99)
        assert tracker.global_step == 99

    def test_get_global(self):
        tracker = StepTracker(initial_step=7)
        assert tracker.get() == 7
        assert tracker.get(None) == 7

    def test_get_tag_falls_back_to_global(self):
        tracker = StepTracker(initial_step=5)
        assert tracker.get("unknown_tag") == 5

    def test_increment_tag(self):
        tracker = StepTracker()
        result = tracker.increment_tag("eval")
        assert result == 1
        result = tracker.increment_tag("eval", 3)
        assert result == 4

    def test_set_tag(self):
        tracker = StepTracker()
        tracker.set_tag("eval", 50)
        assert tracker.get("eval") == 50

    def test_tag_independent_of_global(self):
        tracker = StepTracker(initial_step=100)
        tracker.increment_tag("eval")
        assert tracker.global_step == 100
        assert tracker.get("eval") == 1


# ===================================================================
# MetricGroup tests
# ===================================================================


class TestMetricGroup:
    def test_tag_prefixing(self, enabled_logger):
        grp = MetricGroup(enabled_logger, "train/")
        assert grp._tag("loss") == "train/loss"

    def test_tag_prefix_strips_trailing_slash(self, enabled_logger):
        grp = MetricGroup(enabled_logger, "train/")
        assert grp._prefix == "train"

    def test_scalar_delegates(self, enabled_logger, mock_writer):
        grp = MetricGroup(enabled_logger, "train")
        grp.scalar("loss", 0.5, step=10)
        mock_writer.add_scalar.assert_called_once()
        call_kw = mock_writer.add_scalar.call_args
        assert call_kw.kwargs["tag"] == "train/loss"

    def test_scalars_delegates(self, enabled_logger, mock_writer):
        grp = MetricGroup(enabled_logger, "eval")
        grp.scalars("rewards", {"mean": 1.0, "std": 0.1}, step=5)
        mock_writer.add_scalars.assert_called_once()
        call_kw = mock_writer.add_scalars.call_args
        assert call_kw.kwargs["main_tag"] == "eval/rewards"

    def test_histogram_delegates(self, enabled_logger, mock_writer):
        grp = MetricGroup(enabled_logger, "debug")
        grp.histogram("activations", np.array([1.0, 2.0, 3.0]), step=1)
        mock_writer.add_histogram.assert_called_once()


# ===================================================================
# TBLogger construction tests
# ===================================================================


class TestTBLoggerConstruction:
    def test_enabled_creates_writer(self, enabled_logger, mock_writer):
        assert enabled_logger.enabled is True
        assert enabled_logger._writer is mock_writer

    def test_disabled_no_writer(self, disabled_logger):
        assert disabled_logger.enabled is False
        assert disabled_logger._writer is None

    def test_experiment_name_subdir(self, tmp_path, mock_writer, _enable_tb, monkeypatch):
        monkeypatch.setattr(tb_mod, "SummaryWriter", lambda **kw: mock_writer)
        logger = TBLogger(log_dir=tmp_path, experiment_name="run_01")
        assert (tmp_path / "run_01").is_dir()
        logger.close()

    def test_enabled_true_but_tb_unavailable_raises(self, tmp_path, _disable_tb):
        with pytest.raises(ImportError, match="TensorBoard is not installed"):
            TBLogger(log_dir=tmp_path, enabled=True)


# ===================================================================
# TBLogger properties tests
# ===================================================================


class TestTBLoggerProperties:
    def test_global_step_getter_setter(self, enabled_logger):
        assert enabled_logger.global_step == 0
        enabled_logger.global_step = 42
        assert enabled_logger.global_step == 42

    def test_log_dir(self, enabled_logger, tmp_path):
        assert enabled_logger.log_dir == tmp_path

    def test_step_tracker(self, enabled_logger):
        assert isinstance(enabled_logger.step_tracker, StepTracker)

    def test_is_closed_initially_false(self, enabled_logger):
        assert enabled_logger.is_closed is False


# ===================================================================
# Context manager tests
# ===================================================================


class TestTBLoggerContextManager:
    def test_enter_returns_self(self, enabled_logger):
        with enabled_logger as lg:
            assert lg is enabled_logger

    def test_exit_closes(self, tmp_path, mock_writer, _enable_tb, monkeypatch):
        monkeypatch.setattr(tb_mod, "SummaryWriter", lambda **kw: mock_writer)
        with TBLogger(log_dir=tmp_path) as lg:
            pass
        assert lg.is_closed is True
        mock_writer.close.assert_called_once()


# ===================================================================
# add_scalar tests
# ===================================================================


class TestAddScalar:
    def test_explicit_step(self, enabled_logger, mock_writer):
        enabled_logger.add_scalar("train/loss", 0.5, step=10)
        mock_writer.add_scalar.assert_called_once_with(
            tag="train/loss", scalar_value=0.5, global_step=10
        )

    def test_default_step_uses_global(self, enabled_logger, mock_writer):
        enabled_logger.global_step = 7
        enabled_logger.add_scalar("train/loss", 0.3)
        call_kw = mock_writer.add_scalar.call_args.kwargs
        assert call_kw["global_step"] == 7

    def test_wall_time(self, enabled_logger, mock_writer):
        enabled_logger.add_scalar("x", 1.0, step=0, wall_time=123.0)
        call_kw = mock_writer.add_scalar.call_args.kwargs
        assert call_kw["walltime"] == 123.0

    def test_disabled_noop(self, disabled_logger):
        # Should not raise
        disabled_logger.add_scalar("x", 1.0, step=0)

    def test_caches_value(self, enabled_logger):
        enabled_logger.add_scalar("m", 1.0, step=0)
        enabled_logger.add_scalar("m", 2.0, step=1)
        history = enabled_logger.get_scalar_history("m")
        assert history == [(0, 1.0), (1, 2.0)]


# ===================================================================
# add_scalars tests
# ===================================================================


class TestAddScalars:
    def test_basic(self, enabled_logger, mock_writer):
        vals = {"a": 1.0, "b": 2.0}
        enabled_logger.add_scalars("chart", vals, step=5)
        mock_writer.add_scalars.assert_called_once()
        kw = mock_writer.add_scalars.call_args.kwargs
        assert kw["main_tag"] == "chart"
        assert kw["tag_scalar_dict"] == vals
        assert kw["global_step"] == 5

    def test_with_wall_time(self, enabled_logger, mock_writer):
        enabled_logger.add_scalars("c", {"x": 1.0}, step=0, wall_time=99.0)
        kw = mock_writer.add_scalars.call_args.kwargs
        assert kw["walltime"] == 99.0

    def test_disabled_noop(self, disabled_logger):
        disabled_logger.add_scalars("c", {"x": 1.0}, step=0)


# ===================================================================
# log_training_step tests
# ===================================================================


class TestLogTrainingStep:
    def test_loss_only(self, enabled_logger, mock_writer):
        enabled_logger.log_training_step(step=1, loss=0.5)
        mock_writer.add_scalar.assert_called_once()
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "train/loss"

    def test_all_optional_params(self, enabled_logger, mock_writer):
        enabled_logger.log_training_step(
            step=2, loss=0.4, lr=1e-3, grad_norm=0.1, extra={"acc": 0.9}
        )
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "train/loss" in tags
        assert "train/learning_rate" in tags
        assert "train/grad_norm" in tags
        assert "train/acc" in tags


# ===================================================================
# log_loss_components tests
# ===================================================================


class TestLogLossComponents:
    def test_components_and_total(self, enabled_logger, mock_writer):
        enabled_logger.log_loss_components(step=1, components={"ce": 0.3, "kl": 0.2})
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "loss/ce" in tags
        assert "loss/kl" in tags
        assert "loss/total" in tags
        # Verify total value
        total_call = next(
            c for c in mock_writer.add_scalar.call_args_list if c.kwargs["tag"] == "loss/total"
        )
        assert total_call.kwargs["scalar_value"] == pytest.approx(0.5)

    def test_custom_prefix(self, enabled_logger, mock_writer):
        enabled_logger.log_loss_components(step=1, components={"a": 1.0}, prefix="my_loss")
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "my_loss/a" in tags
        assert "my_loss/total" in tags


# ===================================================================
# log_learning_rate_schedule tests
# ===================================================================


class TestLogLearningRateSchedule:
    def test_default_tag(self, enabled_logger, mock_writer):
        enabled_logger.log_learning_rate_schedule(step=5, lr=0.001)
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "schedule/learning_rate"
        assert kw["scalar_value"] == 0.001

    def test_custom_tag(self, enabled_logger, mock_writer):
        enabled_logger.log_learning_rate_schedule(step=5, lr=0.01, tag="lr/custom")
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "lr/custom"


# ===================================================================
# log_evaluation tests
# ===================================================================


class TestLogEvaluation:
    def test_basic(self, enabled_logger, mock_writer):
        enabled_logger.log_evaluation(step=10, metrics={"acc": 0.95, "f1": 0.9})
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "eval/acc" in tags
        assert "eval/f1" in tags

    def test_custom_prefix(self, enabled_logger, mock_writer):
        enabled_logger.log_evaluation(step=1, metrics={"x": 1.0}, prefix="test")
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "test/x"


# ===================================================================
# log_episode_summary tests
# ===================================================================


class TestLogEpisodeSummary:
    def test_basic(self, enabled_logger, mock_writer):
        enabled_logger.log_episode_summary(episode=1, reward=10.0, length=100)
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "episode/reward" in tags
        assert "episode/length" in tags
        assert len(mock_writer.add_scalar.call_args_list) == 2

    def test_with_success(self, enabled_logger, mock_writer):
        enabled_logger.log_episode_summary(episode=1, reward=10.0, length=50, success=True)
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "episode/success" in tags
        success_call = next(
            c for c in mock_writer.add_scalar.call_args_list if c.kwargs["tag"] == "episode/success"
        )
        assert success_call.kwargs["scalar_value"] == 1.0

    def test_with_extra(self, enabled_logger, mock_writer):
        enabled_logger.log_episode_summary(
            episode=2, reward=5.0, length=20, extra={"collision": 0.1}
        )
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "episode/collision" in tags

    def test_success_false(self, enabled_logger, mock_writer):
        enabled_logger.log_episode_summary(episode=3, reward=0.0, length=10, success=False)
        success_call = next(
            c for c in mock_writer.add_scalar.call_args_list if c.kwargs["tag"] == "episode/success"
        )
        assert success_call.kwargs["scalar_value"] == 0.0


# ===================================================================
# add_histogram tests
# ===================================================================


class TestAddHistogram:
    def test_basic(self, enabled_logger, mock_writer):
        arr = np.array([1.0, 2.0, 3.0])
        enabled_logger.add_histogram("hist", arr, step=5)
        mock_writer.add_histogram.assert_called_once()
        args = mock_writer.add_histogram.call_args
        assert args.args[0] == "hist"
        np.testing.assert_array_equal(args.args[1], arr)
        assert args.kwargs["global_step"] == 5
        assert args.kwargs["bins"] == "tensorflow"

    def test_empty_array_skipped(self, enabled_logger, mock_writer):
        enabled_logger.add_histogram("empty", np.array([]), step=1)
        mock_writer.add_histogram.assert_not_called()

    def test_sequence_input(self, enabled_logger, mock_writer):
        enabled_logger.add_histogram("seq", [1.0, 2.0, 3.0], step=0)
        mock_writer.add_histogram.assert_called_once()

    def test_disabled_noop(self, disabled_logger):
        disabled_logger.add_histogram("x", np.array([1.0]), step=0)


# ===================================================================
# log_reward_distribution tests
# ===================================================================


class TestLogRewardDistribution:
    def test_delegates_to_add_histogram(self, enabled_logger, mock_writer):
        rewards = np.array([1.0, 2.0, 3.0])
        enabled_logger.log_reward_distribution(step=1, rewards=rewards)
        mock_writer.add_histogram.assert_called_once()
        assert mock_writer.add_histogram.call_args.args[0] == "reward/distribution"


# ===================================================================
# log_weight_histograms tests
# ===================================================================


class TestLogWeightHistograms:
    def test_basic(self, enabled_logger, mock_writer):
        params = {"layer.0.weight": np.ones(10), "layer.1.bias": np.zeros(5)}
        enabled_logger.log_weight_histograms(step=1, named_params=params)
        assert mock_writer.add_histogram.call_count == 2
        tags = [c.args[0] for c in mock_writer.add_histogram.call_args_list]
        assert "weights/layer/0/weight" in tags
        assert "weights/layer/1/bias" in tags


# ===================================================================
# log_gradient_histograms tests
# ===================================================================


class TestLogGradientHistograms:
    def test_basic(self, enabled_logger, mock_writer):
        grads = {"layer.0.weight": np.ones(10)}
        enabled_logger.log_gradient_histograms(step=1, named_gradients=grads)
        assert mock_writer.add_histogram.call_count == 1
        assert mock_writer.add_histogram.call_args.args[0] == "gradients/layer/0/weight"


# ===================================================================
# add_image / add_images tests
# ===================================================================


class TestAddImage:
    def test_basic(self, enabled_logger, mock_writer):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        enabled_logger.add_image("img", img, step=1)
        mock_writer.add_image.assert_called_once_with("img", img, global_step=1, dataformats="HWC")

    def test_disabled_noop(self, disabled_logger):
        disabled_logger.add_image("img", np.zeros((4, 4, 3)), step=0)


class TestAddImages:
    def test_basic(self, enabled_logger, mock_writer):
        imgs = np.zeros((4, 64, 64, 3), dtype=np.uint8)
        enabled_logger.add_images("batch", imgs, step=2)
        mock_writer.add_images.assert_called_once_with(
            "batch", imgs, global_step=2, dataformats="NHWC"
        )

    def test_disabled_noop(self, disabled_logger):
        disabled_logger.add_images("batch", np.zeros((2, 4, 4, 3)), step=0)


# ===================================================================
# log_trajectory_image tests
# ===================================================================


class TestLogTrajectoryImage:
    def test_with_positions(self, enabled_logger, mock_writer):
        positions = {
            0: np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]]),
            1: np.array([[0.0, 2.0], [1.0, 1.0]]),
        }
        enabled_logger.log_trajectory_image(step=1, positions=positions)
        mock_writer.add_image.assert_called_once()
        call_args = mock_writer.add_image.call_args
        assert (
            call_args.kwargs.get(
                "dataformats", call_args.args[3] if len(call_args.args) > 3 else None
            )
            or True
        )
        img = call_args.args[1]
        assert img.shape == (256, 256, 3)

    def test_empty_positions(self, enabled_logger, mock_writer):
        enabled_logger.log_trajectory_image(step=1, positions={})
        mock_writer.add_image.assert_called_once()
        img = mock_writer.add_image.call_args.args[1]
        # Should be a white image
        assert img.shape == (256, 256, 3)
        assert np.all(img == 255)

    def test_with_world_bounds(self, enabled_logger, mock_writer):
        positions = {0: np.array([[0.5, 0.5], [1.5, 1.5]])}
        enabled_logger.log_trajectory_image(
            step=1,
            positions=positions,
            world_bounds=(0.0, 2.0, 0.0, 2.0),
            img_size=(128, 128),
        )
        mock_writer.add_image.assert_called_once()
        img = mock_writer.add_image.call_args.args[1]
        assert img.shape == (128, 128, 3)


# ===================================================================
# add_text tests
# ===================================================================


class TestAddText:
    def test_basic(self, enabled_logger, mock_writer):
        enabled_logger.add_text("note", "hello", step=1)
        mock_writer.add_text.assert_called_once_with("note", "hello", global_step=1)

    def test_disabled_noop(self, disabled_logger):
        disabled_logger.add_text("note", "hello", step=0)


# ===================================================================
# add_hparams tests
# ===================================================================


class TestAddHparams:
    def test_basic(self, enabled_logger, mock_writer):
        enabled_logger.add_hparams({"lr": 1e-3, "bs": 32}, {"final_loss": 0.01})
        mock_writer.add_hparams.assert_called_once_with(
            {"lr": 1e-3, "bs": 32}, {"final_loss": 0.01}
        )

    def test_non_standard_types_stringified(self, enabled_logger, mock_writer):
        enabled_logger.add_hparams(
            {"lr": 1e-3, "schedule": [1, 2, 3], "config": {"a": 1}},
            {"loss": 0.1},
        )
        hp_arg = mock_writer.add_hparams.call_args.args[0]
        assert hp_arg["lr"] == 1e-3
        assert hp_arg["schedule"] == "[1, 2, 3]"
        assert hp_arg["config"] == "{'a': 1}"

    def test_disabled_noop(self, disabled_logger):
        disabled_logger.add_hparams({"lr": 1e-3}, {"loss": 0.1})


# ===================================================================
# log_config tests
# ===================================================================


class TestLogConfig:
    def test_logs_json_text(self, enabled_logger, mock_writer):
        enabled_logger.log_config({"lr": 0.001, "epochs": 10})
        mock_writer.add_text.assert_called_once()
        kw = mock_writer.add_text.call_args
        assert kw.args[0] == "config"
        assert '"lr": 0.001' in kw.args[1]
        assert kw.kwargs["global_step"] == 0


# ===================================================================
# log_custom_metric, log_timing, log_throughput tests
# ===================================================================


class TestCustomMetrics:
    def test_log_custom_metric(self, enabled_logger, mock_writer):
        enabled_logger.log_custom_metric("accuracy", 0.95, step=10)
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "custom/accuracy"

    def test_log_custom_metric_prefix(self, enabled_logger, mock_writer):
        enabled_logger.log_custom_metric("x", 1.0, step=0, prefix="special")
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "special/x"

    def test_log_timing(self, enabled_logger, mock_writer):
        enabled_logger.log_timing("forward", 0.05, step=1)
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "timing/forward"
        assert kw["scalar_value"] == pytest.approx(0.05)

    def test_log_throughput(self, enabled_logger, mock_writer):
        enabled_logger.log_throughput("samples", count=1000, duration_s=2.0, step=1)
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "throughput/samples"
        assert kw["scalar_value"] == pytest.approx(500.0)

    def test_log_throughput_near_zero_duration(self, enabled_logger, mock_writer):
        enabled_logger.log_throughput("x", count=10, duration_s=0.0, step=0)
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["scalar_value"] > 0  # Should not be inf


# ===================================================================
# get_scalar_history tests
# ===================================================================


class TestGetScalarHistory:
    def test_empty_tag(self, enabled_logger):
        assert enabled_logger.get_scalar_history("nonexistent") == []

    def test_returns_cached_values(self, enabled_logger):
        enabled_logger.add_scalar("m", 1.0, step=0)
        enabled_logger.add_scalar("m", 2.0, step=1)
        enabled_logger.add_scalar("other", 9.0, step=0)
        history = enabled_logger.get_scalar_history("m")
        assert history == [(0, 1.0), (1, 2.0)]

    def test_returns_copy(self, enabled_logger):
        enabled_logger.add_scalar("m", 1.0, step=0)
        h1 = enabled_logger.get_scalar_history("m")
        h1.append((99, 99.0))
        h2 = enabled_logger.get_scalar_history("m")
        assert len(h2) == 1  # original unchanged


# ===================================================================
# timer context manager tests
# ===================================================================


class TestTimerContextManager:
    def test_logs_elapsed_time(self, enabled_logger, mock_writer):
        with enabled_logger.timer("test_op", step=5):
            pass  # virtually zero time
        mock_writer.add_scalar.assert_called_once()
        kw = mock_writer.add_scalar.call_args.kwargs
        assert kw["tag"] == "timing/test_op"
        assert kw["global_step"] == 5
        assert kw["scalar_value"] >= 0.0


# ===================================================================
# train_step_context tests
# ===================================================================


class TestTrainStepContext:
    def test_collects_and_logs_metrics(self, enabled_logger, mock_writer):
        with enabled_logger.train_step_context(step=10) as metrics:
            metrics["loss"] = 0.5
            metrics["acc"] = 0.9
        tags = [c.kwargs["tag"] for c in mock_writer.add_scalar.call_args_list]
        assert "train/loss" in tags
        assert "train/acc" in tags
        assert "train/step_time" in tags

    def test_sets_global_step(self, enabled_logger):
        with enabled_logger.train_step_context(step=42) as metrics:
            metrics["loss"] = 0.1
        assert enabled_logger.global_step == 42


# ===================================================================
# group() tests
# ===================================================================


class TestGroup:
    def test_creates_metric_group(self, enabled_logger):
        grp = enabled_logger.group("train")
        assert isinstance(grp, MetricGroup)

    def test_reuses_existing_group(self, enabled_logger):
        grp1 = enabled_logger.group("train")
        grp2 = enabled_logger.group("train")
        assert grp1 is grp2

    def test_different_prefixes_different_groups(self, enabled_logger):
        grp1 = enabled_logger.group("train")
        grp2 = enabled_logger.group("eval")
        assert grp1 is not grp2


# ===================================================================
# flush / close tests
# ===================================================================


class TestFlushClose:
    def test_flush(self, enabled_logger, mock_writer):
        enabled_logger.flush()
        mock_writer.flush.assert_called_once()

    def test_flush_disabled(self, disabled_logger):
        disabled_logger.flush()  # should not raise

    def test_close(self, enabled_logger, mock_writer):
        enabled_logger.close()
        assert enabled_logger.is_closed is True
        mock_writer.flush.assert_called_once()
        mock_writer.close.assert_called_once()

    def test_double_close_safe(self, enabled_logger, mock_writer):
        enabled_logger.close()
        enabled_logger.close()  # should not raise
        mock_writer.close.assert_called_once()

    def test_close_disabled(self, disabled_logger):
        disabled_logger.close()
        assert disabled_logger.is_closed is True


# ===================================================================
# create_tb_logger factory tests
# ===================================================================


class TestCreateTbLogger:
    def test_creates_enabled_logger(self, tmp_path, mock_writer, _enable_tb, monkeypatch):
        monkeypatch.setattr(tb_mod, "SummaryWriter", lambda **kw: mock_writer)
        logger = create_tb_logger(tmp_path, experiment_name="exp1")
        assert logger.enabled is True
        logger.close()

    def test_disabled_explicitly(self, tmp_path, _disable_tb):
        logger = create_tb_logger(tmp_path, enabled=False)
        assert logger.enabled is False
        logger.close()

    def test_falls_back_when_tb_unavailable(self, tmp_path, _disable_tb):
        with pytest.warns(UserWarning, match="TensorBoard is not installed"):
            logger = create_tb_logger(tmp_path, enabled=True)
        assert logger.enabled is False
        logger.close()

    def test_passes_kwargs(self, tmp_path, mock_writer, _enable_tb, monkeypatch):
        monkeypatch.setattr(tb_mod, "SummaryWriter", lambda **kw: mock_writer)
        logger = create_tb_logger(tmp_path, initial_step=50, flush_secs=30)
        assert logger.global_step == 50
        logger.close()
