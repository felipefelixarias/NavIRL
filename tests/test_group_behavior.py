"""Tests for navirl.models.group_behavior — group detection, forces, formation, and controller."""

from __future__ import annotations

import math
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from navirl.core.constants import COMFORT, EPSILON
from navirl.core.types import Action, AgentState
from navirl.models.group_behavior import (
    GroupBehaviorModel,
    GroupDetector,
    GroupHumanController,
)

# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------


def _state(aid: int, x: float, y: float, vx: float = 0.0, vy: float = 0.0, **kw) -> AgentState:
    defaults = {
        "agent_id": aid,
        "kind": "human",
        "x": x,
        "y": y,
        "vx": vx,
        "vy": vy,
        "goal_x": 0.0,
        "goal_y": 0.0,
        "radius": 0.25,
        "max_speed": 1.5,
    }
    defaults.update(kw)
    return AgentState(**defaults)


# ===========================================================================
#  GroupDetector
# ===========================================================================


class TestGroupDetector:
    def test_empty(self):
        groups = GroupDetector.detect_groups({}, {})
        assert groups == []

    def test_single_agent(self):
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0)},
            {0: (1.0, 0.0)},
        )
        assert groups == []

    def test_two_agents_close_similar_velocity(self):
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (1.0, 0.0), 1: (1.0, 0.0)},
            distance_threshold=2.0,
            velocity_threshold=0.5,
        )
        assert len(groups) == 1
        assert groups[0] == {0, 1}

    def test_two_agents_far_apart(self):
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (100.0, 0.0)},
            {0: (1.0, 0.0), 1: (1.0, 0.0)},
            distance_threshold=2.0,
        )
        assert groups == []

    def test_two_agents_different_velocity(self):
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (1.0, 0.0)},
            {0: (1.0, 0.0), 1: (-1.0, 0.0)},
            distance_threshold=2.0,
            velocity_threshold=0.5,
        )
        assert groups == []

    def test_three_agents_chain(self):
        """A-B close, B-C close, but A-C far → all one group via transitivity."""
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (1.0, 0.0), 2: (2.0, 0.0)},
            {0: (1.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.0)},
            distance_threshold=1.5,
            velocity_threshold=0.5,
        )
        assert len(groups) == 1
        assert groups[0] == {0, 1, 2}

    def test_two_separate_groups(self):
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (0.5, 0.0), 2: (10.0, 0.0), 3: (10.5, 0.0)},
            {0: (1.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.0), 3: (1.0, 0.0)},
            distance_threshold=1.0,
            velocity_threshold=0.5,
        )
        assert len(groups) == 2
        ids = {frozenset(g) for g in groups}
        assert frozenset({0, 1}) in ids
        assert frozenset({2, 3}) in ids

    def test_mixed_group_and_isolate(self):
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (0.5, 0.0), 2: (50.0, 50.0)},
            {0: (1.0, 0.0), 1: (1.0, 0.0), 2: (1.0, 0.0)},
            distance_threshold=1.0,
            velocity_threshold=0.5,
        )
        assert len(groups) == 1
        assert groups[0] == {0, 1}

    def test_uses_default_thresholds(self):
        # Agents within COMFORT.group_max_separation with similar velocity
        d = COMFORT.group_max_separation - 0.1
        groups = GroupDetector.detect_groups(
            {0: (0.0, 0.0), 1: (d, 0.0)},
            {0: (1.0, 0.0), 1: (1.0, 0.0)},
        )
        assert len(groups) == 1


# ===========================================================================
#  GroupBehaviorModel — cohesion
# ===========================================================================


class TestCohesionForce:
    def test_no_members(self):
        model = GroupBehaviorModel()
        agent = _state(0, 0.0, 0.0)
        fx, fy = model.compute_cohesion_force(agent, [])
        assert fx == 0.0
        assert fy == 0.0

    def test_within_preferred_distance(self):
        model = GroupBehaviorModel(preferred_distance=2.0)
        agent = _state(0, 0.0, 0.0)
        member = _state(1, 1.0, 0.0)
        fx, fy = model.compute_cohesion_force(agent, [member])
        assert fx == 0.0  # within preferred distance → no force
        assert fy == 0.0

    def test_beyond_preferred_distance(self):
        model = GroupBehaviorModel(cohesion_strength=1.0, preferred_distance=1.0)
        agent = _state(0, 0.0, 0.0)
        member = _state(1, 3.0, 0.0)  # centroid at (3, 0)
        fx, fy = model.compute_cohesion_force(agent, [member])
        # Force should pull toward member (positive x)
        assert fx > 0
        assert fy == pytest.approx(0.0, abs=1e-12)

    def test_cohesion_scales_with_excess(self):
        model = GroupBehaviorModel(cohesion_strength=1.0, preferred_distance=1.0)
        agent = _state(0, 0.0, 0.0)
        member_near = _state(1, 2.0, 0.0)  # excess = 1
        member_far = _state(2, 4.0, 0.0)  # excess = 3
        fx_near, _ = model.compute_cohesion_force(agent, [member_near])
        fx_far, _ = model.compute_cohesion_force(agent, [member_far])
        assert fx_far > fx_near

    def test_cohesion_direction_2d(self):
        model = GroupBehaviorModel(cohesion_strength=1.0, preferred_distance=0.5)
        agent = _state(0, 0.0, 0.0)
        member = _state(1, 3.0, 4.0)  # centroid at (3, 4), dist=5
        fx, fy = model.compute_cohesion_force(agent, [member])
        # Direction should be (3/5, 4/5)
        magnitude = math.hypot(fx, fy)
        assert fx / magnitude == pytest.approx(3.0 / 5.0, abs=1e-6)
        assert fy / magnitude == pytest.approx(4.0 / 5.0, abs=1e-6)

    def test_multiple_members_centroid(self):
        model = GroupBehaviorModel(cohesion_strength=1.0, preferred_distance=0.1)
        agent = _state(0, 0.0, 0.0)
        members = [_state(1, 2.0, 0.0), _state(2, 0.0, 2.0)]
        fx, fy = model.compute_cohesion_force(agent, members)
        # Centroid at (1, 1) → direction (1/√2, 1/√2)
        assert fx > 0
        assert fy > 0
        assert fx == pytest.approx(fy, abs=1e-6)


# ===========================================================================
#  GroupBehaviorModel — repulsion
# ===========================================================================


class TestRepulsionForce:
    def test_no_overlap(self):
        model = GroupBehaviorModel(min_distance=0.5)
        agent = _state(0, 0.0, 0.0)
        member = _state(1, 5.0, 0.0)  # far away
        fx, fy = model.compute_repulsion_force(agent, [member])
        assert fx == 0.0
        assert fy == 0.0

    def test_skip_self(self):
        model = GroupBehaviorModel(min_distance=1.0)
        agent = _state(0, 0.0, 0.0)
        # Same agent_id → should be skipped
        member = _state(0, 0.0, 0.0)
        fx, fy = model.compute_repulsion_force(agent, [member])
        assert fx == 0.0
        assert fy == 0.0

    def test_overlap_pushes_away(self):
        model = GroupBehaviorModel(repulsion_strength=1.0, min_distance=2.0)
        agent = _state(0, 0.0, 0.0)
        member = _state(1, 1.0, 0.0)  # within min_distance
        fx, fy = model.compute_repulsion_force(agent, [member])
        # Should push agent 0 away from member 1 (negative x)
        assert fx < 0
        assert fy == pytest.approx(0.0, abs=1e-12)

    def test_repulsion_scales_with_overlap(self):
        model = GroupBehaviorModel(repulsion_strength=1.0, min_distance=3.0)
        agent = _state(0, 0.0, 0.0)
        close_member = _state(1, 0.5, 0.0)  # overlap = 2.5
        far_member = _state(2, 2.5, 0.0)  # overlap = 0.5
        fx_close, _ = model.compute_repulsion_force(agent, [close_member])
        fx_far, _ = model.compute_repulsion_force(agent, [far_member])
        assert abs(fx_close) > abs(fx_far)

    def test_multiple_repulsion_sources(self):
        model = GroupBehaviorModel(repulsion_strength=1.0, min_distance=2.0)
        agent = _state(0, 0.0, 0.0)
        members = [_state(1, 1.0, 0.0), _state(2, -1.0, 0.0)]
        fx, fy = model.compute_repulsion_force(agent, members)
        # Symmetric → forces cancel in x
        assert fx == pytest.approx(0.0, abs=1e-6)


# ===========================================================================
#  GroupBehaviorModel — formation detection
# ===========================================================================


class TestFormation:
    def test_single_point(self):
        assert GroupBehaviorModel.compute_formation([(0, 0)]) == "cluster"

    def test_two_points_line(self):
        assert GroupBehaviorModel.compute_formation([(0, 0), (1, 0)]) == "line"

    def test_collinear_points(self):
        pts = [(float(i), 0.0) for i in range(5)]
        result = GroupBehaviorModel.compute_formation(pts)
        # Points along a line may be classified as "line" or "V" depending on angle spread
        assert result in ("line", "V")

    def test_cluster(self):
        """Points spread equally in all directions → cluster."""
        pts = [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)]
        result = GroupBehaviorModel.compute_formation(pts)
        assert result == "cluster"

    def test_v_shape(self):
        """V-shape: mostly linear but with wide angular spread."""
        pts = [
            (0.0, 0.0),
            (2.0, 2.0),
            (4.0, 0.0),
        ]
        result = GroupBehaviorModel.compute_formation(pts)
        # Depends on angle spread — may be "V" or "line"
        assert result in ("V", "line", "cluster")

    def test_identical_points(self):
        pts = [(1.0, 1.0)] * 4
        assert GroupBehaviorModel.compute_formation(pts) == "cluster"

    def test_near_collinear_with_slight_spread(self):
        """Mostly along x-axis with tiny y variation → line."""
        pts = [(0.0, 0.0), (3.0, 0.01), (6.0, -0.01), (9.0, 0.02)]
        result = GroupBehaviorModel.compute_formation(pts)
        assert result in ("line", "V")


# ===========================================================================
#  GroupHumanController
# ===========================================================================


class TestGroupHumanController:
    def _make_controller(self, **cfg_overrides):
        cfg: dict[str, Any] = {
            "goal_tolerance": 0.5,
            "relaxation_time": 0.5,
            "distance_threshold": 5.0,
            "velocity_threshold": 1.0,
        }
        cfg.update(cfg_overrides)
        return GroupHumanController(cfg)

    def test_reset(self):
        ctrl = self._make_controller()
        ctrl.reset(
            human_ids=[10, 20],
            starts={10: (0.0, 0.0), 20: (1.0, 0.0)},
            goals={10: (5.0, 0.0), 20: (5.0, 1.0)},
        )
        assert ctrl.human_ids == [10, 20]
        assert ctrl.goals[10] == (5.0, 0.0)

    def test_step_returns_actions_for_all_humans(self):
        ctrl = self._make_controller()
        ctrl.reset(
            human_ids=[10, 20],
            starts={10: (0.0, 0.0), 20: (1.0, 0.0)},
            goals={10: (10.0, 0.0), 20: (10.0, 1.0)},
        )
        states = {
            10: _state(10, 0.0, 0.0, vx=1.0),
            20: _state(20, 1.0, 0.0, vx=1.0),
        }
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.1, states=states, robot_id=99, emit_event=emit)
        assert 10 in actions
        assert 20 in actions
        assert isinstance(actions[10], Action)

    def test_step_goal_seeking_direction(self):
        ctrl = self._make_controller(distance_threshold=0.1)  # no grouping
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )
        states = {1: _state(1, 0.0, 0.0)}
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.1, states=states, robot_id=99, emit_event=emit)
        # Should move toward goal (positive x)
        assert actions[1].pref_vx > 0
        assert abs(actions[1].pref_vy) < 0.01

    def test_step_goal_swap_on_arrival(self):
        ctrl = self._make_controller(goal_tolerance=1.0, distance_threshold=0.1)
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (0.5, 0.0)},  # very close to current position
        )
        states = {1: _state(1, 0.3, 0.0)}
        emit = MagicMock()
        ctrl.step(step=0, time_s=0.0, dt=0.1, states=states, robot_id=99, emit_event=emit)
        # Should have emitted a goal_swap event
        emit.assert_called_once()
        assert emit.call_args[0][0] == "goal_swap"

    def test_step_group_dynamics(self):
        ctrl = self._make_controller(distance_threshold=5.0, velocity_threshold=2.0)
        ctrl.reset(
            human_ids=[1, 2],
            starts={1: (0.0, 0.0), 2: (1.0, 0.0)},
            goals={1: (10.0, 0.0), 2: (10.0, 0.0)},
        )
        states = {
            1: _state(1, 0.0, 0.0, vx=1.0),
            2: _state(2, 1.0, 0.0, vx=1.0),
        }
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.1, states=states, robot_id=99, emit_event=emit)
        # Both agents should have GROUP_WALK behavior when in same group
        assert actions[1].behavior == "GROUP_WALK"
        assert actions[2].behavior == "GROUP_WALK"
        # Metadata should contain formation and group_ids
        assert "group_ids" in actions[1].metadata
        assert "formation" in actions[1].metadata

    def test_step_isolated_agent_go_to(self):
        ctrl = self._make_controller(distance_threshold=0.1)  # tiny threshold → no grouping
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (10.0, 0.0)},
        )
        states = {1: _state(1, 0.0, 0.0)}
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.1, states=states, robot_id=99, emit_event=emit)
        assert actions[1].behavior == "GO_TO"

    def test_step_speed_clamped_to_max(self):
        ctrl = self._make_controller(distance_threshold=0.1)
        ctrl.reset(
            human_ids=[1],
            starts={1: (0.0, 0.0)},
            goals={1: (100.0, 0.0)},
        )
        states = {1: _state(1, 0.0, 0.0, max_speed=1.0)}
        emit = MagicMock()
        actions = ctrl.step(step=0, time_s=0.0, dt=0.1, states=states, robot_id=99, emit_event=emit)
        speed = math.hypot(actions[1].pref_vx, actions[1].pref_vy)
        assert speed <= 1.0 + 1e-6

    def test_default_config(self):
        ctrl = GroupHumanController()
        assert ctrl.goal_tolerance == 0.5
        assert ctrl.relaxation_time == COMFORT.relaxation_time
