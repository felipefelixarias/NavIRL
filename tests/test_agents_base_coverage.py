"""Tests for navirl/agents/base.py — HyperParameters, MetricsLogger,
RunningMeanStd, CheckpointMeta, and BaseAgent non-torch functionality.
"""

from __future__ import annotations

import numpy as np
import pytest

from navirl.agents.base import (
    CheckpointMeta,
    HyperParameters,
    MetricsLogger,
    RunningMeanStd,
    _CHECKPOINT_VERSION,
)


# ---------------------------------------------------------------------------
# HyperParameters
# ---------------------------------------------------------------------------

from dataclasses import dataclass


@dataclass
class DummyHP(HyperParameters):
    lr: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99


class TestHyperParameters:
    def test_to_dict(self):
        hp = DummyHP(lr=0.01)
        d = hp.to_dict()
        assert d["lr"] == 0.01
        assert d["batch_size"] == 64

    def test_from_dict(self):
        hp = DummyHP.from_dict({"lr": 0.05, "batch_size": 128, "unknown_key": "ignored"})
        assert hp.lr == 0.05
        assert hp.batch_size == 128

    def test_getitem(self):
        hp = DummyHP()
        assert hp["lr"] == 1e-3
        assert hp["gamma"] == 0.99

    def test_setitem(self):
        hp = DummyHP()
        hp["lr"] = 0.1
        assert hp.lr == 0.1

    def test_setitem_invalid_key(self):
        hp = DummyHP()
        with pytest.raises(KeyError, match="not a valid hyperparameter"):
            hp["nonexistent"] = 42

    def test_contains(self):
        hp = DummyHP()
        assert "lr" in hp
        assert "nonexistent" not in hp

    def test_get(self):
        hp = DummyHP()
        assert hp.get("lr") == 1e-3
        assert hp.get("missing", 42) == 42

    def test_update(self):
        hp = DummyHP()
        hp.update({"lr": 0.5, "gamma": 0.95, "nonexistent": "ignored"})
        assert hp.lr == 0.5
        assert hp.gamma == 0.95

    def test_round_trip(self):
        hp = DummyHP(lr=0.01, batch_size=32, gamma=0.9)
        d = hp.to_dict()
        hp2 = DummyHP.from_dict(d)
        assert hp == hp2


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    def test_record_and_dump(self):
        ml = MetricsLogger()
        ml.record("loss", 1.0)
        ml.record("loss", 3.0)
        summary = ml.dump()
        assert summary["loss"] == pytest.approx(2.0)

    def test_dump_clears_buffer(self):
        ml = MetricsLogger()
        ml.record("loss", 1.0)
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

    def test_callback_invoked(self):
        calls = []
        def cb(metrics, step):
            calls.append((metrics, step))

        ml = MetricsLogger(callback=cb)
        ml.record("x", 5.0)
        ml.dump(step=10)
        assert len(calls) == 1
        assert calls[0][1] == 10
        assert calls[0][0]["x"] == pytest.approx(5.0)

    def test_callback_not_called_on_empty(self):
        calls = []
        def cb(metrics, step):
            calls.append(1)

        ml = MetricsLogger(callback=cb)
        ml.dump()
        assert len(calls) == 0

    def test_set_callback(self):
        calls = []
        def cb(metrics, step):
            calls.append(1)

        ml = MetricsLogger()
        ml.record("x", 1.0)
        ml.set_callback(cb)
        ml.dump()
        assert len(calls) == 1

    def test_step_property(self):
        ml = MetricsLogger()
        assert ml.step == 0
        ml.step = 42
        assert ml.step == 42

    def test_dump_uses_internal_step_by_default(self):
        calls = []
        def cb(metrics, step):
            calls.append(step)

        ml = MetricsLogger(callback=cb)
        ml.step = 100
        ml.record("x", 1.0)
        ml.dump()
        assert calls[0] == 100

    def test_multiple_keys(self):
        ml = MetricsLogger()
        for i in range(10):
            ml.record("loss", float(i))
            ml.record("reward", float(i * 2))
        summary = ml.dump()
        assert "loss" in summary
        assert "reward" in summary
        assert summary["loss"] == pytest.approx(4.5)  # mean(0..9)
        assert summary["reward"] == pytest.approx(9.0)  # mean(0,2,4,...18)


# ---------------------------------------------------------------------------
# RunningMeanStd
# ---------------------------------------------------------------------------


class TestRunningMeanStd:
    def test_initial_state(self):
        rms = RunningMeanStd(shape=(3,))
        np.testing.assert_allclose(rms.mean, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(rms.var, [1.0, 1.0, 1.0])

    def test_update_single_sample(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.array([5.0]))
        # After one sample, mean should be near 5.0
        assert rms.mean == pytest.approx(5.0, abs=0.01)

    def test_update_batch(self):
        rms = RunningMeanStd(shape=())
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        rms.update(data)
        assert rms.mean == pytest.approx(3.0, abs=0.1)

    def test_incremental_matches_batch(self):
        data = np.random.default_rng(42).normal(5.0, 2.0, size=(100,))

        rms_batch = RunningMeanStd(shape=())
        rms_batch.update(data)

        rms_inc = RunningMeanStd(shape=())
        for chunk in np.array_split(data, 10):
            rms_inc.update(chunk)

        np.testing.assert_allclose(rms_batch.mean, rms_inc.mean, atol=0.1)
        np.testing.assert_allclose(rms_batch.var, rms_inc.var, atol=0.5)

    def test_normalize(self):
        rms = RunningMeanStd(shape=())
        data = np.random.default_rng(42).normal(10.0, 3.0, size=(1000,))
        rms.update(data)
        normalized = rms.normalize(np.array([10.0]))
        # Mean input should normalize to ~0
        assert abs(normalized[0]) < 0.5

    def test_normalize_clipping(self):
        rms = RunningMeanStd(shape=())
        rms.update(np.array([0.0, 0.0, 0.0]))
        # Very extreme value
        result = rms.normalize(np.array([1000.0]), clip=5.0)
        assert result[0] <= 5.0

    def test_denormalize_inverts(self):
        rms = RunningMeanStd(shape=(2,))
        data = np.random.default_rng(42).normal([5.0, -3.0], [2.0, 1.0], size=(500, 2))
        rms.update(data)
        x = np.array([5.0, -3.0])
        normalized = rms.normalize(x, clip=100.0)
        recovered = rms.denormalize(normalized)
        np.testing.assert_allclose(recovered, x, atol=0.5)

    def test_state_dict_and_load(self):
        rms = RunningMeanStd(shape=(3,))
        data = np.random.default_rng(42).normal(size=(50, 3))
        rms.update(data)

        state = rms.state_dict()
        rms2 = RunningMeanStd(shape=(3,))
        rms2.load_state_dict(state)

        np.testing.assert_allclose(rms.mean, rms2.mean)
        np.testing.assert_allclose(rms.var, rms2.var)
        assert rms.count == pytest.approx(rms2.count)

    def test_multidim_shape(self):
        rms = RunningMeanStd(shape=(2, 3))
        data = np.random.default_rng(42).normal(size=(20, 2, 3))
        rms.update(data)
        assert rms.mean.shape == (2, 3)
        assert rms.var.shape == (2, 3)


# ---------------------------------------------------------------------------
# CheckpointMeta
# ---------------------------------------------------------------------------


class TestCheckpointMeta:
    def test_defaults(self):
        meta = CheckpointMeta()
        assert meta.agent_class == ""
        assert meta.checkpoint_version == _CHECKPOINT_VERSION
        assert meta.total_steps == 0
        assert meta.total_episodes == 0
        assert meta.hyperparameters == {}
        assert meta.extra == {}

    def test_custom_values(self):
        meta = CheckpointMeta(
            agent_class="PPOAgent",
            total_steps=5000,
            total_episodes=100,
            hyperparameters={"lr": 0.001},
            extra={"notes": "test"},
        )
        assert meta.agent_class == "PPOAgent"
        assert meta.total_steps == 5000
        assert meta.hyperparameters["lr"] == 0.001
        assert meta.extra["notes"] == "test"


# ---------------------------------------------------------------------------
# BaseAgent — registry and non-torch-dependent functionality
# ---------------------------------------------------------------------------


class TestBaseAgentRegistry:
    def test_registered_agents_returns_list(self):
        from navirl.agents.base import BaseAgent
        agents = BaseAgent.registered_agents()
        assert isinstance(agents, list)

    def test_make_unknown_agent_raises(self):
        from navirl.agents.base import BaseAgent
        with pytest.raises(ValueError, match="Unknown agent"):
            BaseAgent.make("NonExistentAgent9999", DummyHP(), None, None)
