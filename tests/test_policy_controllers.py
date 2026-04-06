"""Tests for policy-controller hooks (ROADMAP #6).

Tests cover:
- PolicyHumanController cfg-dict interface and error handling
- PolicyRobotController cfg-dict interface and error handling
- Plugin registration of 'policy' type for both human and robot controllers
- Observation builders produce correct shapes
- Scenario validation accepts 'policy' controller type
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from navirl.core.types import Action, AgentState
from navirl.models.learned_policy import PolicyHumanController, _build_observation
from navirl.models.learned_robot_policy import (
    PolicyRobotController,
    _build_robot_observation,
)


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _make_state(
    agent_id: int,
    kind: str = "human",
    x: float = 0.0,
    y: float = 0.0,
    vx: float = 0.0,
    vy: float = 0.0,
    goal_x: float = 5.0,
    goal_y: float = 5.0,
    radius: float = 0.18,
    max_speed: float = 0.8,
) -> AgentState:
    return AgentState(
        agent_id=agent_id,
        kind=kind,
        x=x,
        y=y,
        vx=vx,
        vy=vy,
        goal_x=goal_x,
        goal_y=goal_y,
        radius=radius,
        max_speed=max_speed,
    )


def _noop_emit(event_type, agent_id, payload):
    pass


# ---------------------------------------------------------------------------
#  Observation builder tests
# ---------------------------------------------------------------------------

class TestHumanObservationBuilder:
    def test_shape_no_neighbours(self):
        agent = _make_state(0)
        obs = _build_observation(agent, [], max_neighbours=6)
        assert obs.shape == (6 + 6 * 5,)
        assert obs.dtype == np.float32

    def test_shape_with_neighbours(self):
        agent = _make_state(0)
        neighbours = [_make_state(i, x=float(i)) for i in range(1, 4)]
        obs = _build_observation(agent, neighbours, max_neighbours=6)
        assert obs.shape == (6 + 6 * 5,)

    def test_ego_encoding(self):
        agent = _make_state(0, x=1.0, y=2.0, vx=0.3, vy=0.4, goal_x=4.0, goal_y=6.0, radius=0.2)
        obs = _build_observation(agent, [], max_neighbours=4)
        assert obs[0] == pytest.approx(3.0)   # dx_goal
        assert obs[1] == pytest.approx(4.0)   # dy_goal
        assert obs[2] == pytest.approx(0.3)   # vx
        assert obs[3] == pytest.approx(0.4)   # vy
        assert obs[4] == pytest.approx(0.5)   # speed
        assert obs[5] == pytest.approx(0.2)   # radius

    def test_neighbours_sorted_by_distance(self):
        agent = _make_state(0, x=0.0, y=0.0)
        far = _make_state(1, x=10.0, y=0.0, radius=0.3)
        near = _make_state(2, x=1.0, y=0.0, radius=0.15)
        obs = _build_observation(agent, [far, near], max_neighbours=6)
        # First neighbour slot should be 'near' (dx=1.0)
        assert obs[6] == pytest.approx(1.0)
        assert obs[6 + 4] == pytest.approx(0.15)  # near.radius
        # Second neighbour slot should be 'far' (dx=10.0)
        assert obs[6 + 5] == pytest.approx(10.0)

    def test_excess_neighbours_truncated(self):
        agent = _make_state(0)
        neighbours = [_make_state(i, x=float(i)) for i in range(1, 20)]
        obs = _build_observation(agent, neighbours, max_neighbours=3)
        assert obs.shape == (6 + 3 * 5,)

    def test_zero_padded_when_fewer_neighbours(self):
        agent = _make_state(0)
        obs = _build_observation(agent, [_make_state(1, x=2.0)], max_neighbours=4)
        # Slots 2 and 3 should be zero
        assert obs[6 + 2 * 5: 6 + 4 * 5].sum() == 0.0


class TestRobotObservationBuilder:
    def test_shape_no_neighbours(self):
        robot = _make_state(0, kind="robot")
        obs = _build_robot_observation(robot, [], max_neighbours=6)
        assert obs.shape == (6 + 6 * 6,)
        assert obs.dtype == np.float32

    def test_includes_kind_flag(self):
        robot = _make_state(0, kind="robot", x=0.0, y=0.0)
        human = _make_state(1, kind="human", x=1.0, y=0.0)
        other_robot = _make_state(2, kind="robot", x=2.0, y=0.0)
        obs = _build_robot_observation(robot, [human, other_robot], max_neighbours=6)
        # First neighbour (human) has is_human=1.0
        assert obs[6 + 5] == pytest.approx(1.0)
        # Second neighbour (robot) has is_human=0.0
        assert obs[6 + 6 + 5] == pytest.approx(0.0)

    def test_ego_encoding(self):
        robot = _make_state(0, kind="robot", x=1.0, y=2.0, vx=0.1, vy=0.2, goal_x=3.0, goal_y=4.0, radius=0.2)
        obs = _build_robot_observation(robot, [], max_neighbours=4)
        assert obs[0] == pytest.approx(2.0)   # dx_goal
        assert obs[1] == pytest.approx(2.0)   # dy_goal
        assert obs[2] == pytest.approx(0.1)
        assert obs[3] == pytest.approx(0.2)
        assert obs[4] == pytest.approx(math.hypot(0.1, 0.2))
        assert obs[5] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
#  PolicyHumanController unit tests (no torch needed)
# ---------------------------------------------------------------------------

def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


class TestPolicyHumanControllerInit:
    def test_direct_args(self):
        ctrl = PolicyHumanController(model_path="/tmp/fake.pt", device="cpu", max_neighbours=4)
        assert str(ctrl.model_path) == "/tmp/fake.pt"
        assert ctrl.device_str == "cpu"
        assert ctrl.max_neighbours == 4

    def test_cfg_dict(self):
        ctrl = PolicyHumanController(cfg={
            "model_path": "/tmp/model_dir",
            "device": "cuda:0",
            "max_neighbours": 8,
        })
        assert str(ctrl.model_path) == "/tmp/model_dir"
        assert ctrl.device_str == "cuda:0"
        assert ctrl.max_neighbours == 8

    def test_cfg_overrides_direct_args(self):
        ctrl = PolicyHumanController(
            model_path="/tmp/old.pt",
            device="cpu",
            cfg={"model_path": "/tmp/new.pt", "device": "cuda"},
        )
        assert str(ctrl.model_path) == "/tmp/new.pt"
        assert ctrl.device_str == "cuda"

    def test_reset(self):
        ctrl = PolicyHumanController(cfg={"model_path": "/tmp/fake.pt"})
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (1.0, 1.0)},
            goals={1: (5.0, 5.0), 2: (6.0, 6.0)},
        )
        assert ctrl.human_ids == [1, 2]
        assert ctrl.goals[1] == (5.0, 5.0)

    @pytest.mark.skipif(
        not _torch_available(), reason="PyTorch not installed"
    )
    def test_missing_model_path_raises(self):
        ctrl = PolicyHumanController(cfg={"model_path": "/tmp/nonexistent_model.pt"})
        ctrl.reset([1], {1: (0.0, 0.0)}, {1: (5.0, 5.0)})
        with pytest.raises(FileNotFoundError):
            ctrl.step(0, 0.0, 0.1, {1: _make_state(1)}, 99, _noop_emit)


# ---------------------------------------------------------------------------
#  PolicyRobotController unit tests (no torch needed)
# ---------------------------------------------------------------------------

class TestPolicyRobotControllerInit:
    def test_cfg_defaults(self):
        ctrl = PolicyRobotController(cfg={})
        assert ctrl.device_str == "cpu"
        assert ctrl.max_neighbours == 6
        assert ctrl.goal_tolerance == pytest.approx(0.2)
        assert ctrl.max_speed == pytest.approx(0.8)

    def test_cfg_custom(self):
        ctrl = PolicyRobotController(cfg={
            "model_path": "/tmp/robot_policy.pt",
            "device": "cuda:1",
            "max_neighbours": 10,
            "goal_tolerance": 0.3,
            "max_speed": 1.2,
        })
        assert ctrl.model_path_str == "/tmp/robot_policy.pt"
        assert ctrl.device_str == "cuda:1"
        assert ctrl.max_neighbours == 10
        assert ctrl.goal_tolerance == pytest.approx(0.3)
        assert ctrl.max_speed == pytest.approx(1.2)

    def test_reset(self):
        ctrl = PolicyRobotController(cfg={"model_path": "/tmp/fake.pt"})
        ctrl.reset(robot_id=0, start=(1.0, 2.0), goal=(5.0, 6.0), backend=None)
        assert ctrl.robot_id == 0
        assert ctrl.start == (1.0, 2.0)
        assert ctrl.goal == (5.0, 6.0)

    def test_goal_reached_returns_done(self):
        """When the robot is within goal_tolerance, step returns DONE without needing a model."""
        ctrl = PolicyRobotController(cfg={"model_path": "/tmp/fake.pt", "goal_tolerance": 1.0})
        ctrl.reset(robot_id=0, start=(0.0, 0.0), goal=(0.5, 0.5), backend=None)
        robot_state = _make_state(0, kind="robot", x=0.5, y=0.5, goal_x=0.5, goal_y=0.5)
        action = ctrl.step(0, 0.0, 0.1, {0: robot_state}, _noop_emit)
        assert action.behavior == "DONE"
        assert action.pref_vx == 0.0
        assert action.pref_vy == 0.0


# ---------------------------------------------------------------------------
#  Plugin registration tests
# ---------------------------------------------------------------------------

class TestPolicyPluginRegistration:
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

    def test_human_policy_creates_correct_type(self):
        from navirl.core.registry import get_human_controller
        from navirl.plugins import register_default_plugins
        register_default_plugins()
        factory = get_human_controller("policy")
        ctrl = factory({"model_path": "/tmp/fake.pt"})
        assert isinstance(ctrl, PolicyHumanController)

    def test_robot_policy_creates_correct_type(self):
        from navirl.core.registry import get_robot_controller
        from navirl.plugins import register_default_plugins
        register_default_plugins()
        factory = get_robot_controller("policy")
        ctrl = factory({"model_path": "/tmp/fake.pt"})
        assert isinstance(ctrl, PolicyRobotController)


# ---------------------------------------------------------------------------
#  Scenario validation accepts policy type
# ---------------------------------------------------------------------------

class TestScenarioValidationPolicyType:
    def test_human_policy_type_accepted(self):
        from navirl.scenarios.validate import _validate_humans
        errors: list[str] = []
        humans = {
            "count": 3,
            "controller": {"type": "policy", "params": {"model_path": "/tmp/m.pt"}},
            "starts": [[0, 0], [1, 1], [2, 2]],
            "goals": [[5, 5], [6, 6], [7, 7]],
        }
        _validate_humans(humans, errors)
        assert not any("controller.type" in e for e in errors)

    def test_robot_policy_type_accepted(self):
        from navirl.scenarios.validate import _validate_robot
        errors: list[str] = []
        robot = {
            "controller": {"type": "policy", "params": {"model_path": "/tmp/m.pt"}},
            "start": [0, 0],
            "goal": [5, 5],
        }
        _validate_robot(robot, errors)
        assert not any("controller.type" in e for e in errors)

    def test_robot_social_astar_accepted(self):
        from navirl.scenarios.validate import _validate_robot
        errors: list[str] = []
        robot = {
            "controller": {"type": "social_astar", "params": {}},
            "start": [0, 0],
            "goal": [5, 5],
        }
        _validate_robot(robot, errors)
        assert not any("controller.type" in e for e in errors)
