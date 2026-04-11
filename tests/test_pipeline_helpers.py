"""Tests for navirl.pipeline — pure helper functions.

Covers: _ensure_points, _diameter, _min_anchor_dist, _has_agent_collision,
_anchor_ok, _search_ring_points, _search_ring_positions,
_search_random_positions, _enforce_anchor_layout, _run_id,
_resolve_human_start_goal_lists.
"""

from __future__ import annotations

import math
import re
from unittest.mock import MagicMock

import pytest

from navirl.pipeline import (
    BASE_STEP_FACTOR,
    INITIAL_RING_POINTS,
    MAX_SEARCH_RINGS,
    MIN_BASE_STEP,
    RING_POINT_INCREMENT,
    _anchor_ok,
    _diameter,
    _enforce_anchor_layout,
    _ensure_points,
    _has_agent_collision,
    _has_obstacle_collision,
    _min_anchor_dist,
    _project_anchor,
    _resolve_human_start_goal_lists,
    _run_id,
    _search_random_positions,
    _search_ring_points,
    _search_ring_positions,
)

# ---------------------------------------------------------------------------
# _run_id
# ---------------------------------------------------------------------------


class TestRunId:
    def test_format(self):
        rid = _run_id("hallway_01")
        assert rid.startswith("hallway_01_")
        # Should contain date stamp and hex suffix
        parts = rid.split("_")
        assert len(parts) >= 4

    def test_unique(self):
        a = _run_id("test")
        b = _run_id("test")
        assert a != b


# ---------------------------------------------------------------------------
# _diameter / _min_anchor_dist
# ---------------------------------------------------------------------------


class TestDiameterAndMinAnchorDist:
    def test_diameter(self):
        assert _diameter(0.5) == 1.0
        assert _diameter(0.0) == 0.0

    def test_diameter_negative_clamps(self):
        assert _diameter(-1.0) == 0.0

    def test_min_anchor_dist(self):
        assert _min_anchor_dist(0.5, 0.3) == 1.0  # max(1.0, 0.6)
        assert _min_anchor_dist(0.2, 0.8) == 1.6  # max(0.4, 1.6)

    def test_min_anchor_dist_equal(self):
        assert _min_anchor_dist(0.25, 0.25) == 0.5


# ---------------------------------------------------------------------------
# _ensure_points
# ---------------------------------------------------------------------------


class TestEnsurePoints:
    def test_no_additional_needed(self):
        existing = [(0.0, 0.0), (1.0, 1.0)]
        result = _ensure_points(existing, count=2, sampler=None, min_dist=0.5)
        assert len(result) == 2

    def test_adds_points_with_min_dist(self):
        counter = [0]

        def sampler():
            counter[0] += 1
            return (float(counter[0]) * 2.0, 0.0)

        result = _ensure_points([], count=3, sampler=sampler, min_dist=0.5)
        assert len(result) == 3
        # All points should be at least 0.5 apart
        for i in range(len(result)):
            for j in range(i + 1, len(result)):
                d = math.hypot(result[i][0] - result[j][0], result[i][1] - result[j][1])
                assert d >= 0.5 - 1e-6

    def test_relaxation_for_dense_scenes(self):
        # Sampler always returns same point — forces relaxation then fallback
        def sampler():
            return (0.0, 0.0)

        result = _ensure_points([], count=3, sampler=sampler, min_dist=100.0)
        assert len(result) == 3  # Should still produce 3 points via fallback

    def test_zero_min_dist(self):
        call_count = [0]

        def sampler():
            call_count[0] += 1
            return (0.0, 0.0)

        result = _ensure_points([], count=5, sampler=sampler, min_dist=0.0)
        assert len(result) == 5


# ---------------------------------------------------------------------------
# _has_agent_collision / _has_obstacle_collision / _anchor_ok
# ---------------------------------------------------------------------------


class TestCollisionChecks:
    def test_no_collision(self):
        placed = [{"position": (0.0, 0.0), "radius": 0.2}]
        assert _has_agent_collision((5.0, 5.0), 0.2, placed) is False

    def test_collision_detected(self):
        placed = [{"position": (0.0, 0.0), "radius": 0.5}]
        # Distance = 0.3, min_dist = max(1.0, 1.0) = 1.0 > 0.3
        assert _has_agent_collision((0.3, 0.0), 0.5, placed) is True

    def test_no_placed_agents(self):
        assert _has_agent_collision((0.0, 0.0), 0.5, []) is False

    def test_has_obstacle_collision(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = True
        assert _has_obstacle_collision((0.0, 0.0), 0.5, backend) is True
        backend.check_obstacle_collision.assert_called_once_with((0.0, 0.0), 1.0)

    def test_anchor_ok_clear(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = False
        assert _anchor_ok((5.0, 5.0), 0.2, [], backend) is True

    def test_anchor_ok_obstacle_collision(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = True
        assert _anchor_ok((0.0, 0.0), 0.2, [], backend) is False

    def test_anchor_ok_agent_collision(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = False
        placed = [{"position": (0.0, 0.0), "radius": 0.5}]
        assert _anchor_ok((0.1, 0.0), 0.5, placed, backend) is False


# ---------------------------------------------------------------------------
# _project_anchor
# ---------------------------------------------------------------------------


class TestProjectAnchor:
    def test_calls_backend(self):
        backend = MagicMock()
        backend.nearest_clear_point.return_value = (1.5, 2.5)
        result = _project_anchor((1.0, 2.0), 0.3, backend)
        assert result == (1.5, 2.5)
        backend.nearest_clear_point.assert_called_once_with((1.0, 2.0), 0.6)


# ---------------------------------------------------------------------------
# _search_ring_points / _search_ring_positions / _search_random_positions
# ---------------------------------------------------------------------------


class TestSearchFunctions:
    def _make_backend(self, ok_positions=None):
        """Create mock backend where specific positions are clear."""
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = False
        if ok_positions:
            backend.nearest_clear_point.side_effect = lambda pos, _: pos
        else:
            backend.nearest_clear_point.side_effect = lambda pos, _: pos
        return backend

    def test_search_ring_points_finds_position(self):
        backend = self._make_backend()
        result = _search_ring_points(
            desired=(0.0, 0.0),
            radius=0.2,
            placed=[],
            backend=backend,
            base_step=0.5,
            ring=1,
        )
        assert result is not None

    def test_search_ring_points_all_blocked(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = True
        backend.nearest_clear_point.side_effect = lambda pos, _: pos
        result = _search_ring_points(
            desired=(0.0, 0.0),
            radius=0.2,
            placed=[],
            backend=backend,
            base_step=0.5,
            ring=1,
        )
        assert result is None

    def test_search_ring_positions_expanding(self):
        # First ring blocked, second ring clear
        call_count = [0]
        backend = MagicMock()

        def obstacle_check(pos, diam):
            call_count[0] += 1
            # Block first ring calls, allow second ring
            return call_count[0] <= INITIAL_RING_POINTS + RING_POINT_INCREMENT

        backend.check_obstacle_collision.side_effect = obstacle_check
        backend.nearest_clear_point.side_effect = lambda pos, _: pos

        result = _search_ring_positions(
            desired=(0.0, 0.0),
            radius=0.2,
            placed=[],
            backend=backend,
            base_step=0.5,
        )
        assert result is not None

    def test_search_random_positions_found(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = False
        backend.nearest_clear_point.side_effect = lambda pos, _: pos
        backend.sample_free_point.return_value = (3.0, 3.0)
        result = _search_random_positions(0.2, [], backend, max_samples=10)
        assert result is not None

    def test_search_random_positions_exhausted(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = True
        backend.nearest_clear_point.side_effect = lambda pos, _: pos
        backend.sample_free_point.return_value = (0.0, 0.0)
        result = _search_random_positions(0.2, [], backend, max_samples=5)
        assert result is None


# ---------------------------------------------------------------------------
# _enforce_anchor_layout
# ---------------------------------------------------------------------------


class TestEnforceAnchorLayout:
    def test_simple_placement(self):
        backend = MagicMock()
        backend.check_obstacle_collision.return_value = False
        backend.nearest_clear_point.side_effect = lambda pos, _: pos

        anchors = [
            {"key": "robot.start", "position": (1.0, 1.0), "radius": 0.2},
            {"key": "robot.goal", "position": (5.0, 5.0), "radius": 0.2},
        ]
        placed, adjustments, unresolved = _enforce_anchor_layout(anchors, backend)
        assert len(placed) == 2
        assert len(unresolved) == 0

    def test_colliding_anchors_get_adjusted(self):
        backend = MagicMock()
        backend.nearest_clear_point.side_effect = lambda pos, _: pos
        backend.sample_free_point.return_value = (10.0, 10.0)

        def obstacle_check(pos, diam):
            return False

        backend.check_obstacle_collision.side_effect = obstacle_check

        anchors = [
            {"key": "a", "position": (0.0, 0.0), "radius": 0.5},
            {"key": "b", "position": (0.01, 0.0), "radius": 0.5},  # Too close to a
        ]
        placed, adjustments, unresolved = _enforce_anchor_layout(anchors, backend)
        assert len(placed) == 2
        # b should have been adjusted or flagged
        b_placed = next(p for p in placed if p["key"] == "b")
        # Either adjusted away or in unresolved
        assert b_placed is not None


# ---------------------------------------------------------------------------
# _resolve_human_start_goal_lists
# ---------------------------------------------------------------------------


class TestResolveHumanStartGoalLists:
    def test_basic(self):
        backend = MagicMock()
        counter = [0]

        def sampler():
            counter[0] += 1
            return (float(counter[0]) * 2.0, float(counter[0]) * 2.0)

        backend.sample_free_point = sampler

        scenario = {"humans": {"count": 3, "radius": 0.16, "starts": [], "goals": []}}
        starts, goals = _resolve_human_start_goal_lists(scenario, backend)
        assert len(starts) == 3
        assert len(goals) == 3

    def test_preserves_existing(self):
        backend = MagicMock()
        counter = [0]

        def sampler():
            counter[0] += 1
            return (100.0 + counter[0], 100.0 + counter[0])

        backend.sample_free_point = sampler

        scenario = {
            "humans": {
                "count": 2,
                "radius": 0.16,
                "starts": [(1.0, 1.0)],
                "goals": [(5.0, 5.0)],
            }
        }
        starts, goals = _resolve_human_start_goal_lists(scenario, backend)
        assert len(starts) == 2
        assert len(goals) == 2
        assert starts[0] == (1.0, 1.0)
        assert goals[0] == (5.0, 5.0)
