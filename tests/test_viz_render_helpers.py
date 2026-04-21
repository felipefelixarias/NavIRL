"""Tests for pure helper functions in navirl.viz.render.

Covers _world_to_px, _load_rows, _door_token_by_step, _arrow_endpoint,
_agent_palette, and EnvironmentRenderer (without requiring a real backend).
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pytest

from navirl.viz.render import (
    EnvironmentRenderer,
    _agent_palette,
    _arrow_endpoint,
    _door_token_by_step,
    _load_rows,
    _world_to_px,
)

# ---------------------------------------------------------------------------
# _world_to_px
# ---------------------------------------------------------------------------


class TestWorldToPx:
    def test_origin_maps_to_center(self):
        shape = (100, 200)  # h=100, w=200
        px, py = _world_to_px(0.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0)
        assert px == 100  # w/2
        assert py == 50   # h/2

    def test_scale_factor(self):
        shape = (100, 100)
        px1, py1 = _world_to_px(1.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0)
        px2, py2 = _world_to_px(1.0, 0.0, shape, scale=2.0, pixels_per_meter=10.0)
        assert px2 == pytest.approx(px1 * 2, abs=1)

    def test_offset_applied(self):
        shape = (100, 100)
        px1, py1 = _world_to_px(0.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0)
        px2, py2 = _world_to_px(
            0.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0,
            row_offset=10, col_offset=20,
        )
        assert px2 == px1 - 20
        assert py2 == py1 - 10

    def test_positive_world_x_increases_col(self):
        shape = (100, 100)
        px_zero, _ = _world_to_px(0.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0)
        px_pos, _ = _world_to_px(1.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0)
        assert px_pos > px_zero

    def test_positive_world_y_increases_row(self):
        shape = (100, 100)
        _, py_zero = _world_to_px(0.0, 0.0, shape, scale=1.0, pixels_per_meter=10.0)
        _, py_pos = _world_to_px(0.0, 1.0, shape, scale=1.0, pixels_per_meter=10.0)
        assert py_pos > py_zero


# ---------------------------------------------------------------------------
# _load_rows
# ---------------------------------------------------------------------------


class TestLoadRows:
    def test_loads_jsonl(self, tmp_path):
        p = tmp_path / "state.jsonl"
        rows = [{"step": 0, "x": 1.0}, {"step": 1, "x": 2.0}]
        p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        loaded = _load_rows(p)
        assert len(loaded) == 2
        assert loaded[0]["step"] == 0
        assert loaded[1]["x"] == 2.0

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "state.jsonl"
        p.write_text('{"step": 0}\n\n\n{"step": 1}\n')
        loaded = _load_rows(p)
        assert len(loaded) == 2

    def test_empty_file(self, tmp_path):
        p = tmp_path / "state.jsonl"
        p.write_text("")
        assert _load_rows(p) == []


# ---------------------------------------------------------------------------
# _door_token_by_step
# ---------------------------------------------------------------------------


class TestDoorTokenByStep:
    def test_nonexistent_file(self, tmp_path):
        result = _door_token_by_step(tmp_path / "no_such_file.jsonl")
        assert result == {}

    def test_acquire_and_release(self, tmp_path):
        p = tmp_path / "events.jsonl"
        events = [
            {"step": 0, "event_type": "door_token_acquire", "agent_id": 1},
            {"step": 5, "event_type": "door_token_release", "agent_id": 1},
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = _door_token_by_step(p)
        assert result[0] == 1
        assert result[5] is None

    def test_multiple_agents(self, tmp_path):
        p = tmp_path / "events.jsonl"
        events = [
            {"step": 0, "event_type": "door_token_acquire", "agent_id": 1},
            {"step": 3, "event_type": "door_token_release", "agent_id": 1},
            {"step": 4, "event_type": "door_token_acquire", "agent_id": 2},
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = _door_token_by_step(p)
        assert result[0] == 1
        assert result[3] is None
        assert result[4] == 2

    def test_skips_blank_lines(self, tmp_path):
        p = tmp_path / "events.jsonl"
        p.write_text('{"step": 0, "event_type": "door_token_acquire", "agent_id": 1}\n\n')
        result = _door_token_by_step(p)
        assert result[0] == 1

    def test_non_token_events_dont_change_holder(self, tmp_path):
        """Non-token events still record the current token_holder at that step."""
        p = tmp_path / "events.jsonl"
        events = [
            {"step": 0, "event_type": "door_token_acquire", "agent_id": 1},
            {"step": 1, "event_type": "collision", "agent_id": 99},
            {"step": 2, "event_type": "door_token_release", "agent_id": 1},
        ]
        p.write_text("\n".join(json.dumps(e) for e in events) + "\n")
        result = _door_token_by_step(p)
        assert result[0] == 1
        # collision event at step 1 records current holder (still 1)
        assert result[1] == 1
        assert result[2] is None


# ---------------------------------------------------------------------------
# _arrow_endpoint
# ---------------------------------------------------------------------------


class TestArrowEndpoint:
    def test_moving_agent_arrow(self):
        x, y = 0.0, 0.0
        vx, vy = 1.0, 0.0
        ex, ey, heading = _arrow_endpoint(x, y, vx, vy, None)
        assert ex > x  # Arrow points in +x direction
        assert abs(ey - y) < 1e-6
        assert heading is not None
        assert abs(heading[0] - 1.0) < 1e-6

    def test_stationary_agent_with_last_heading(self):
        x, y = 0.0, 0.0
        vx, vy = 0.0, 0.0
        last_heading = (0.0, 1.0)
        ex, ey, heading = _arrow_endpoint(x, y, vx, vy, last_heading)
        assert ey > y  # Arrow points in +y using last heading
        assert heading == last_heading

    def test_stationary_agent_no_heading(self):
        x, y = 1.0, 2.0
        ex, ey, heading = _arrow_endpoint(x, y, 0.0, 0.0, None)
        assert ex == x
        assert ey == y
        assert heading is None

    def test_min_arrow_length(self):
        x, y = 0.0, 0.0
        # Very slow speed
        vx, vy = 0.001, 0.0
        ex, ey, heading = _arrow_endpoint(x, y, vx, vy, None)
        length = math.hypot(ex - x, ey - y)
        assert length >= 0.2  # min_len_m default

    def test_max_arrow_length_capped(self):
        x, y = 0.0, 0.0
        vx, vy = 100.0, 0.0
        ex, ey, heading = _arrow_endpoint(x, y, vx, vy, None)
        length = math.hypot(ex - x, ey - y)
        assert length <= 0.52 + 0.01  # capped at 0.52

    def test_diagonal_direction(self):
        x, y = 0.0, 0.0
        vx, vy = 1.0, 1.0
        ex, ey, heading = _arrow_endpoint(x, y, vx, vy, None)
        assert ex > 0 and ey > 0
        # Should be roughly equal displacement
        assert abs(ex - ey) < 0.01


# ---------------------------------------------------------------------------
# _agent_palette
# ---------------------------------------------------------------------------


class TestAgentPalette:
    def test_robot_palette(self):
        edge, fill, highlight = _agent_palette("robot", "GO_TO")
        assert len(edge) == 3
        assert len(fill) == 3
        assert len(highlight) == 3

    def test_yielding_human(self):
        edge, fill, highlight = _agent_palette("human", "YIELDING")
        # Should differ from regular human
        edge2, fill2, highlight2 = _agent_palette("human", "GO_TO")
        assert fill != fill2

    def test_regular_human(self):
        edge, fill, highlight = _agent_palette("human", "GO_TO")
        assert all(isinstance(c, int) for c in fill)

    def test_robot_palette_ignores_behavior(self):
        p1 = _agent_palette("robot", "GO_TO")
        p2 = _agent_palette("robot", "YIELDING")
        p3 = _agent_palette("robot", "STOP")
        assert p1 == p2 == p3


# ---------------------------------------------------------------------------
# EnvironmentRenderer
# ---------------------------------------------------------------------------


class TestEnvironmentRenderer:
    def test_construction(self):
        renderer = EnvironmentRenderer()
        # cv2 should be available in test env
        assert isinstance(renderer._cv2_available, bool)

    def test_prepare_map_image_none_backend(self):
        renderer = EnvironmentRenderer()
        result = renderer.prepare_map_image(None)
        assert result is None

    def test_prepare_map_image_grayscale_to_rgb(self):
        renderer = EnvironmentRenderer()

        class FakeBackend:
            def map_image(self):
                return np.ones((10, 10), dtype=np.uint8) * 128

        img = renderer.prepare_map_image(FakeBackend())
        assert img is not None
        assert img.ndim == 3
        assert img.shape == (10, 10, 3)
        assert img.dtype == np.uint8

    def test_prepare_map_image_already_rgb(self):
        renderer = EnvironmentRenderer()

        class FakeBackend:
            def map_image(self):
                return np.ones((10, 10, 3), dtype=np.uint8) * 128

        img = renderer.prepare_map_image(FakeBackend(), grayscale_to_rgb=True)
        assert img.shape == (10, 10, 3)

    def test_prepare_map_image_no_grayscale_conversion(self):
        renderer = EnvironmentRenderer()

        class FakeBackend:
            def map_image(self):
                return np.ones((10, 10), dtype=np.uint8) * 128

        img = renderer.prepare_map_image(FakeBackend(), grayscale_to_rgb=False)
        assert img.ndim == 2

    def test_prepare_map_image_returns_none_for_none_image(self):
        renderer = EnvironmentRenderer()

        class FakeBackend:
            def map_image(self):
                return None

        result = renderer.prepare_map_image(FakeBackend())
        assert result is None

    def test_draw_agent_circle_with_backend(self):
        renderer = EnvironmentRenderer()
        if not renderer._cv2_available:
            pytest.skip("cv2 not available")

        img = np.zeros((50, 50, 3), dtype=np.uint8)

        class FakeBackend:
            def world_to_map(self, pos):
                return (25, 25)

        renderer.draw_agent_circle(img, (0.0, 0.0), FakeBackend())
        # Some pixels should be non-zero after drawing
        assert img.sum() > 0

    def test_draw_goal_circle(self):
        renderer = EnvironmentRenderer()
        if not renderer._cv2_available:
            pytest.skip("cv2 not available")

        img = np.zeros((50, 50, 3), dtype=np.uint8)

        class FakeBackend:
            def world_to_map(self, pos):
                return (25, 25)

        renderer.draw_goal_circle(img, (1.0, 1.0), FakeBackend())
        assert img.sum() > 0

    def test_draw_humans(self):
        renderer = EnvironmentRenderer()
        if not renderer._cv2_available:
            pytest.skip("cv2 not available")

        img = np.zeros((50, 50, 3), dtype=np.uint8)

        class FakeBackend:
            def get_position(self, hid):
                return (0.0, 0.0)

            def world_to_map(self, pos):
                return (25, 25)

        renderer.draw_humans(img, [1, 2], FakeBackend())
        assert img.sum() > 0
