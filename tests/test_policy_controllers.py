"""Tests for policy-based controllers (Issue #6).

Tests both PolicyHumanController and PolicyRobotController instantiation,
observation building, and plugin registration.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.core.types import Action, AgentState
from navirl.models.learned_policy import (
    PolicyRobotController,
    _build_observation,
    _build_robot_observation,
)


# ---------------------------------------------------------------------------
# Observation builder tests
# ---------------------------------------------------------------------------


def _make_state(agent_id, x, y, vx=0.0, vy=0.0, gx=5.0, gy=5.0, kind="human"):
    return AgentState(
        agent_id=agent_id,
        kind=kind,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=gx,
        goal_y=gy,
        radius=0.3,
        max_speed=1.5,
    )


class TestBuildObservation:
    """Tests for _build_observation (human) and _build_robot_observation."""

    def test_human_obs_shape(self):
        agent = _make_state(0, 1.0, 2.0)
        obs = _build_observation(agent, [], max_neighbours=6)
        assert obs.shape == (6 + 6 * 5,)
        assert obs.dtype == np.float32

    def test_robot_obs_shape(self):
        robot = _make_state(0, 1.0, 2.0, kind="robot")
        obs = _build_robot_observation(robot, [], max_neighbours=6)
        assert obs.shape == (6 + 6 * 5,)
        assert obs.dtype == np.float32

    def test_goal_relative(self):
        agent = _make_state(0, 1.0, 2.0, gx=4.0, gy=6.0)
        obs = _build_observation(agent, [])
        assert obs[0] == pytest.approx(3.0)  # dx_goal = 4 - 1
        assert obs[1] == pytest.approx(4.0)  # dy_goal = 6 - 2

    def test_robot_goal_relative(self):
        robot = _make_state(0, 1.0, 2.0, gx=4.0, gy=6.0, kind="robot")
        obs = _build_robot_observation(robot, [])
        assert obs[0] == pytest.approx(3.0)
        assert obs[1] == pytest.approx(4.0)

    def test_neighbours_sorted_by_distance(self):
        agent = _make_state(0, 0.0, 0.0)
        far = _make_state(1, 10.0, 0.0)
        near = _make_state(2, 1.0, 0.0)
        obs = _build_observation(agent, [far, near], max_neighbours=2)
        # First neighbour slot should be the near one
        base = 6  # own_dim
        assert obs[base + 0] == pytest.approx(1.0)  # dx to near
        assert obs[base + 5 + 0] == pytest.approx(10.0)  # dx to far

    def test_robot_neighbours_sorted_by_distance(self):
        robot = _make_state(0, 0.0, 0.0, kind="robot")
        far = _make_state(1, 10.0, 0.0)
        near = _make_state(2, 1.0, 0.0)
        obs = _build_robot_observation(robot, [far, near], max_neighbours=2)
        base = 6
        assert obs[base + 0] == pytest.approx(1.0)
        assert obs[base + 5 + 0] == pytest.approx(10.0)

    def test_zero_pad_when_fewer_neighbours(self):
        agent = _make_state(0, 0.0, 0.0)
        obs = _build_observation(agent, [], max_neighbours=4)
        # All neighbour slots should be zero
        assert np.all(obs[6:] == 0.0)


# ---------------------------------------------------------------------------
# PolicyRobotController instantiation tests
# ---------------------------------------------------------------------------


class TestPolicyRobotController:
    """Tests for PolicyRobotController that don't require PyTorch."""

    def test_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path"):
            PolicyRobotController(cfg={})

    def test_stores_config(self):
        ctrl = PolicyRobotController(
            cfg={"model_path": "/tmp/fake_model.pt", "device": "cpu", "max_neighbours": 4}
        )
        assert ctrl.model_path.name == "fake_model.pt"
        assert ctrl.device_str == "cpu"
        assert ctrl.max_neighbours == 4

    def test_reset_stores_state(self):
        ctrl = PolicyRobotController(cfg={"model_path": "/tmp/fake_model.pt"})
        ctrl.reset(robot_id=0, start=(1.0, 2.0), goal=(8.0, 9.0), backend=None)
        assert ctrl.robot_id == 0
        assert ctrl.start == (1.0, 2.0)
        assert ctrl.goal == (8.0, 9.0)

    def test_reset_validates_robot_id(self):
        ctrl = PolicyRobotController(cfg={"model_path": "/tmp/fake_model.pt"})
        with pytest.raises(ValueError):
            ctrl.reset(robot_id=-1, start=(0, 0), goal=(1, 1), backend=None)

    def test_reset_validates_positions(self):
        ctrl = PolicyRobotController(cfg={"model_path": "/tmp/fake_model.pt"})
        with pytest.raises(ValueError):
            ctrl.reset(robot_id=0, start="bad", goal=(1, 1), backend=None)


# ---------------------------------------------------------------------------
# Plugin registration tests
# ---------------------------------------------------------------------------


class TestPluginRegistration:
    """Test that policy controllers are properly registered."""

    def test_human_policy_registered(self):
        from navirl.core.registry import get_human_controller
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        factory = get_human_controller("policy")
        assert factory is not None

    def test_robot_policy_registered(self):
        from navirl.core.registry import get_robot_controller
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        factory = get_robot_controller("policy")
        assert factory is not None

    def test_human_policy_factory_returns_correct_type(self):
        """Verify the 'policy' human controller factory creates a
        PolicyHumanController, not an ORCAHumanController (regression test)."""
        from navirl.core.registry import get_human_controller
        from navirl.models.learned_policy import PolicyHumanController
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        factory = get_human_controller("policy")
        ctrl = factory({"model_path": "/tmp/fake_model.pt"})
        assert isinstance(ctrl, PolicyHumanController)

    def test_robot_policy_factory_returns_correct_type(self):
        from navirl.core.registry import get_robot_controller
        from navirl.models.learned_policy import PolicyRobotController
        from navirl.plugins import register_default_plugins

        register_default_plugins()
        factory = get_robot_controller("policy")
        ctrl = factory({"model_path": "/tmp/fake_model.pt"})
        assert isinstance(ctrl, PolicyRobotController)
