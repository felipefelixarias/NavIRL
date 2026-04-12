"""Tests for navirl.agents.base — HyperParameters, MetricsLogger, RunningMeanStd, CheckpointMeta.

These are pure-Python / NumPy components that work without PyTorch.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.agents.base import (
    _CHECKPOINT_VERSION,
    CheckpointMeta,
    HyperParameters,
    MetricsLogger,
    RunningMeanStd,
)

# ---------------------------------------------------------------------------
# HyperParameters
# ---------------------------------------------------------------------------


class TestHyperParameters:
    """Test the dict-like hyperparameter base class."""

    def _make_params(self):
        from dataclasses import dataclass

        @dataclass
        class TestParams(HyperParameters):
            lr: float = 0.001
            batch_size: int = 32
            hidden_dim: int = 128
            gamma: float = 0.99

        return TestParams

    def test_to_dict(self):
        Cls = self._make_params()
        p = Cls(lr=0.01, batch_size=64)
        d = p.to_dict()
        assert d["lr"] == 0.01
        assert d["batch_size"] == 64
        assert d["hidden_dim"] == 128

    def test_from_dict_filters_unknown_keys(self):
        Cls = self._make_params()
        p = Cls.from_dict({"lr": 0.1, "unknown_key": "ignored", "batch_size": 16})
        assert p.lr == 0.1
        assert p.batch_size == 16

    def test_from_dict_empty(self):
        Cls = self._make_params()
        p = Cls.from_dict({})
        assert p.lr == 0.001  # default

    def test_getitem(self):
        Cls = self._make_params()
        p = Cls()
        assert p["lr"] == 0.001
        assert p["batch_size"] == 32

    def test_setitem(self):
        Cls = self._make_params()
        p = Cls()
        p["lr"] = 0.05
        assert p.lr == 0.05

    def test_setitem_invalid_key(self):
        Cls = self._make_params()
        p = Cls()
        with pytest.raises(KeyError, match="not a valid hyperparameter"):
            p["nonexistent"] = 42

    def test_contains(self):
        Cls = self._make_params()
        p = Cls()
        assert "lr" in p
        assert "nonexistent" not in p

    def test_get_default(self):
        Cls = self._make_params()
        p = Cls()
        assert p.get("lr") == 0.001
        assert p.get("missing", "fallback") == "fallback"

    def test_update(self):
        Cls = self._make_params()
        p = Cls()
        p.update({"lr": 0.99, "batch_size": 256, "unknown": True})
        assert p.lr == 0.99
        assert p.batch_size == 256
        # unknown key is silently ignored
        assert not hasattr(p, "unknown")

    def test_roundtrip_dict(self):
        Cls = self._make_params()
        original = Cls(lr=0.005, batch_size=128, hidden_dim=256, gamma=0.95)
        d = original.to_dict()
        restored = Cls.from_dict(d)
        assert restored.lr == original.lr
        assert restored.batch_size == original.batch_size
        assert restored.hidden_dim == original.hidden_dim
        assert restored.gamma == original.gamma


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    def test_record_and_dump(self):
        ml = MetricsLogger()
        ml.record("loss", 1.0)
        ml.record("loss", 2.0)
        ml.record("loss", 3.0)
        summary = ml.dump()
        assert summary["loss"] == pytest.approx(2.0)

    def test_dump_clears_buffer(self):
        ml = MetricsLogger()
        ml.record("x", 10.0)
        ml.dump()
        summary = ml.dump()
        assert summary == {}

    def test_record_dict(self):
        ml = MetricsLogger()
        ml.record_dict({"a": 1.0, "b": 2.0})
        ml.record_dict({"a": 3.0, "b": 4.0})
        summary = ml.dump()
        assert summary["a"] == pytest.approx(2.0)
        assert summary["b"] == pytest.approx(3.0)

    def test_step_property(self):
        ml = MetricsLogger()
        assert ml.step == 0
        ml.step = 42
        assert ml.step == 42

    def test_callback_invoked(self):
        received = []

        def cb(metrics, step):
            received.append((metrics, step))

        ml = MetricsLogger(callback=cb)
        ml.record("loss", 5.0)
        ml.dump(step=10)
        assert len(received) == 1
        assert received[0][1] == 10
        assert received[0][0]["loss"] == pytest.approx(5.0)

    def test_callback_not_invoked_on_empty(self):
        received = []

        def cb(metrics, step):
            received.append(metrics)

        ml = MetricsLogger(callback=cb)
        ml.dump()
        assert len(received) == 0

    def test_set_callback(self):
        received = []
        ml = MetricsLogger()
        ml.set_callback(lambda m, s: received.append(m))
        ml.record("x", 1.0)
        ml.dump()
        assert len(received) == 1

    def test_multiple_keys(self):
        ml = MetricsLogger()
        ml.record("loss", 1.0)
        ml.record("accuracy", 0.9)
        ml.record("loss", 3.0)
        summary = ml.dump()
        assert summary["loss"] == pytest.approx(2.0)
        assert summary["accuracy"] == pytest.approx(0.9)

    def test_dump_uses_internal_step_when_none(self):
        received = []

        def cb(metrics, step):
            received.append(step)

        ml = MetricsLogger(callback=cb)
        ml.step = 7
        ml.record("x", 1.0)
        ml.dump()
        assert received[0] == 7


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------


class TestRunningMeanStd:
    def test_initial_state(self):
        rms = RunningMeanStd(shape=(3,))
        np.testing.assert_array_equal(rms.mean, np.zeros(3))
        np.testing.assert_array_equal(rms.var, np.ones(3))

    def test_single_update(self):
        rms = RunningMeanStd(shape=(2,))
        rms.update(np.array([[1.0, 2.0]]))
        # After one sample, mean should be close to the sample value
        # (modulated by the initial epsilon count)
        assert rms.mean[0] > 0.0
        assert rms.mean[1] > 0.0

    def test_batch_update_convergence(self):
        rms = RunningMeanStd(shape=(1,))
        rng = np.random.RandomState(42)
        data = rng.randn(2000, 1) * 3.0 + 5.0
        rms.update(data)
        assert rms.mean[0] == pytest.approx(5.0, abs=0.5)
        assert rms.var[0] == pytest.approx(9.0, abs=2.0)

    def test_incremental_updates(self):
        rms = RunningMeanStd(shape=(1,))
        data = np.random.randn(500, 1) * 2.0 + 3.0
        # Update in small batches
        for i in range(0, 500, 50):
            rms.update(data[i : i + 50])
        assert rms.mean[0] == pytest.approx(3.0, abs=0.5)
        assert rms.var[0] == pytest.approx(4.0, abs=1.5)

    def test_normalize(self):
        rms = RunningMeanStd(shape=(2,))
        rms.mean = np.array([5.0, 10.0])
        rms.var = np.array([4.0, 16.0])
        rms.count = 100.0

        x = np.array([7.0, 14.0])
        normalized = rms.normalize(x)
        # (7 - 5) / sqrt(4) = 1.0, (14 - 10) / sqrt(16) = 1.0
        np.testing.assert_array_almost_equal(normalized, [1.0, 1.0], decimal=4)

    def test_normalize_clipping(self):
        rms = RunningMeanStd(shape=(1,))
        rms.mean = np.array([0.0])
        rms.var = np.array([1.0])
        rms.count = 100.0

        x = np.array([100.0])
        normalized = rms.normalize(x, clip=5.0)
        assert normalized[0] == pytest.approx(5.0)

    def test_denormalize_inverse(self):
        rms = RunningMeanStd(shape=(2,))
        rms.mean = np.array([5.0, 10.0])
        rms.var = np.array([4.0, 16.0])
        rms.count = 100.0

        x = np.array([7.0, 14.0])
        normalized = rms.normalize(x, clip=100.0)
        restored = rms.denormalize(normalized)
        np.testing.assert_array_almost_equal(restored, x, decimal=4)

    def test_state_dict_roundtrip(self):
        rms = RunningMeanStd(shape=(3,))
        rms.update(np.random.randn(100, 3))

        state = rms.state_dict()
        rms2 = RunningMeanStd(shape=(3,))
        rms2.load_state_dict(state)

        np.testing.assert_array_equal(rms.mean, rms2.mean)
        np.testing.assert_array_equal(rms.var, rms2.var)
        assert rms.count == rms2.count

    def test_scalar_shape(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.array([5.0]))
        assert rms.mean.shape == ()

    def test_multidim(self):
        rms = RunningMeanStd(shape=(2, 3))
        data = np.random.randn(50, 2, 3)
        rms.update(data)
        assert rms.mean.shape == (2, 3)
        assert rms.var.shape == (2, 3)


# ---------------------------------------------------------------------------
# CheckpointMeta
# ---------------------------------------------------------------------------


class TestCheckpointMeta:
    def test_default_values(self):
        meta = CheckpointMeta()
        assert meta.agent_class == ""
        assert meta.checkpoint_version == _CHECKPOINT_VERSION
        assert meta.total_steps == 0
        assert meta.extra == {}

    def test_custom_values(self):
        meta = CheckpointMeta(
            agent_class="PPOAgent",
            total_steps=1000,
            total_episodes=50,
            wall_time=123.4,
            hyperparameters={"lr": 0.001},
            extra={"custom_key": "value"},
        )
        assert meta.agent_class == "PPOAgent"
        assert meta.total_steps == 1000
        assert meta.hyperparameters["lr"] == 0.001
        assert meta.extra["custom_key"] == "value"
