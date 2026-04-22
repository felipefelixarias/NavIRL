"""Additional coverage tests for navirl.sensors.camera.

Targets branches that the top-level sensor suite leaves uncovered:
- CameraSensor top-down wall-segment rendering via Bresenham _draw_line.
- CameraSensor perspective mode with visible, behind, and out-of-FOV agents.
- CameraSensor list-style agents (non-dict) and custom colour field.
- CameraSensor._draw_line Bresenham single-pixel and off-canvas clipping.
- DepthSensor with wall segments, circular obstacles, dict agents, and
  array-style agents.
- DepthSensor min_range clipping.
"""

from __future__ import annotations

import numpy as np

from navirl.sensors.camera import (
    CameraConfig,
    CameraSensor,
    DepthSensor,
    DepthSensorConfig,
)


def _world_state(
    robot_pos=(0.0, 0.0),
    robot_heading=0.0,
    agents=None,
    obstacles_segments=None,
    obstacles_circles=None,
):
    ws = {
        "robot_pos": np.array(robot_pos, dtype=np.float64),
        "robot_heading": robot_heading,
        "robot_vel": np.array([0.0, 0.0], dtype=np.float64),
    }
    if agents is not None:
        ws["agents"] = agents
    if obstacles_segments is not None:
        ws["obstacles_segments"] = np.array(obstacles_segments, dtype=np.float64)
    if obstacles_circles is not None:
        ws["obstacles_circles"] = obstacles_circles
    return ws


# ---------------------------------------------------------------------------
# CameraSensor._draw_line -- Bresenham primitive
# ---------------------------------------------------------------------------


class TestDrawLineBresenham:
    def test_single_pixel_line(self):
        """x0==x1 and y0==y1 should paint exactly one pixel and terminate."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        CameraSensor._draw_line(img, 2, 2, 2, 2, color=[7, 8, 9])
        assert img[2, 2].tolist() == [7, 8, 9]
        # No other pixel touched.
        painted = np.sum(img.sum(axis=-1) > 0)
        assert painted == 1

    def test_diagonal_line_in_bounds(self):
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        CameraSensor._draw_line(img, 0, 0, 9, 9, color=[10, 20, 30])
        diag = [img[i, i].tolist() for i in range(10)]
        assert all(px == [10, 20, 30] for px in diag)

    def test_line_clipped_to_canvas(self):
        """Pixels outside the image must be dropped silently."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        # Start inside, finish well outside.
        CameraSensor._draw_line(img, 2, 2, 20, 2, color=[255, 255, 255])
        # The in-bounds portion (cols 2..4) should be painted.
        assert img[2, 2].tolist() == [255, 255, 255]
        assert img[2, 4].tolist() == [255, 255, 255]
        # Nothing should leak beyond the canvas (it would have crashed if it did).

    def test_reverse_direction_line(self):
        """sx/sy stepping must handle x0>x1 and y0>y1."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        CameraSensor._draw_line(img, 8, 8, 1, 1, color=[50, 60, 70])
        # Endpoints are both inside and painted.
        assert img[8, 8].tolist() == [50, 60, 70]
        assert img[1, 1].tolist() == [50, 60, 70]


# ---------------------------------------------------------------------------
# CameraSensor top-down wall segments
# ---------------------------------------------------------------------------


class TestTopDownWithSegments:
    def test_wall_segment_renders_line(self):
        """obstacles_segments trigger _draw_line calls."""
        cam = CameraSensor(
            CameraConfig(resolution_x=64, resolution_y=64, render_mode="top_down")
        )
        ws_empty = _world_state(robot_pos=(0.0, 0.0))
        segments = [[[-2.0, 2.0], [2.0, 2.0]]]  # wall in front of robot
        ws_wall = _world_state(robot_pos=(0.0, 0.0), obstacles_segments=segments)

        img_empty = cam.observe(ws_empty)
        img_wall = cam.observe(ws_wall)
        assert not np.array_equal(img_empty, img_wall)


# ---------------------------------------------------------------------------
# CameraSensor perspective mode
# ---------------------------------------------------------------------------


class TestPerspectiveAgents:
    def _cam(self):
        return CameraSensor(
            CameraConfig(
                resolution_x=128,
                resolution_y=64,
                render_mode="perspective",
                fov_horizontal=np.pi / 2.0,  # 90 deg -> wide enough to catch center
                max_depth=20.0,
            )
        )

    def test_visible_agent_renders(self):
        """Agent in front of robot within FOV should paint pixels."""
        cam = self._cam()
        # Agent 3m in +x direction, robot heading=0 (also +x).
        agents = [{"pos": np.array([3.0, 0.0]), "radius": 0.4}]
        ws = _world_state(robot_pos=(0.0, 0.0), robot_heading=0.0, agents=agents)

        empty_ws = _world_state(robot_pos=(0.0, 0.0), robot_heading=0.0)
        img_empty = cam.observe(empty_ws)
        img_agent = cam.observe(ws)
        assert not np.array_equal(img_empty, img_agent)

    def test_agent_behind_robot_skipped(self):
        """cam_x < EPSILON branch -- behind the robot -> no render."""
        cam = self._cam()
        agents = [{"pos": np.array([-3.0, 0.0]), "radius": 0.4}]
        ws = _world_state(robot_pos=(0.0, 0.0), robot_heading=0.0, agents=agents)
        img_agent = cam.observe(ws)
        img_empty = cam.observe(_world_state(robot_pos=(0.0, 0.0)))
        np.testing.assert_array_equal(img_agent, img_empty)

    def test_agent_outside_fov_skipped(self):
        """An agent beyond half-FOV angle is skipped."""
        cam = self._cam()  # half_fov = 45 deg
        # 1m forward, 5m to the side -> angle ~79 deg -> outside.
        agents = [{"pos": np.array([1.0, 5.0]), "radius": 0.4}]
        ws = _world_state(robot_pos=(0.0, 0.0), robot_heading=0.0, agents=agents)
        img_agent = cam.observe(ws)
        img_empty = cam.observe(_world_state(robot_pos=(0.0, 0.0)))
        np.testing.assert_array_equal(img_agent, img_empty)

    def test_perspective_list_agent_format(self):
        """Perspective mode also accepts (x, y, radius) tuples/arrays."""
        cam = self._cam()
        agents = [np.array([2.0, 0.0, 0.3])]
        ws = _world_state(robot_pos=(0.0, 0.0), robot_heading=0.0, agents=agents)
        img = cam.observe(ws)
        # Something rendered (non-background pixels exist).
        assert (img != 200).any()

    def test_rotated_heading_changes_projection(self):
        """Agent position constant, heading changes -> image differs."""
        cam = self._cam()
        agents = [{"pos": np.array([3.0, 0.0]), "radius": 0.4}]
        ws0 = _world_state(robot_pos=(0.0, 0.0), robot_heading=0.0, agents=agents)
        ws_back = _world_state(robot_pos=(0.0, 0.0), robot_heading=np.pi, agents=agents)
        img0 = cam.observe(ws0)
        img_back = cam.observe(ws_back)
        # With heading=pi, agent is now behind -> skipped, image stays background.
        assert not np.array_equal(img0, img_back)


# ---------------------------------------------------------------------------
# CameraSensor top-down with list-style agents and custom colour
# ---------------------------------------------------------------------------


class TestTopDownListAgents:
    def test_list_agent_format(self):
        """List-style agents [x, y, radius] take the non-dict isinstance branch."""
        cam = CameraSensor(
            CameraConfig(resolution_x=64, resolution_y=64, render_mode="top_down")
        )
        # Agent offset from robot so the centre-drawn robot does not fully cover it.
        agents = [np.array([2.0, 0.0, 0.5])]
        ws_agent = _world_state(robot_pos=(0.0, 0.0), agents=agents)
        ws_empty = _world_state(robot_pos=(0.0, 0.0))
        assert not np.array_equal(cam.observe(ws_agent), cam.observe(ws_empty))

    def test_dict_agent_custom_colour_renders(self):
        """A dict agent with a colour field should produce that colour in the image."""
        cam = CameraSensor(
            CameraConfig(resolution_x=64, resolution_y=64, render_mode="top_down")
        )
        # Offset from centre so robot doesn't paint over it.
        agents = [{"pos": np.array([2.0, 0.0]), "radius": 0.5, "color": [10, 20, 30]}]
        ws = _world_state(robot_pos=(0.0, 0.0), agents=agents)
        img = cam.observe(ws)
        matches = np.all(img == [10, 20, 30], axis=-1)
        assert matches.any()


# ---------------------------------------------------------------------------
# DepthSensor branches
# ---------------------------------------------------------------------------


class TestDepthSensorBranches:
    def test_depth_with_wall_segments(self):
        ds = DepthSensor(DepthSensorConfig(resolution=8, max_range=20.0, noise_std=0.0))
        segments = [[[5.0, -10.0], [5.0, 10.0]]]  # wall at x=5
        ws = _world_state(
            robot_pos=(0.0, 0.0), robot_heading=0.0, obstacles_segments=segments
        )
        ranges = ds.observe(ws)
        # Centre ray hits wall at distance ~5.
        assert np.min(ranges) < 6.0
        assert np.min(ranges) > 4.0

    def test_depth_with_circle_and_segment(self):
        ds = DepthSensor(DepthSensorConfig(resolution=16, max_range=20.0, noise_std=0.0))
        ws = _world_state(
            robot_pos=(0.0, 0.0),
            obstacles_segments=[[[8.0, -5.0], [8.0, 5.0]]],
            obstacles_circles={
                "centres": np.array([[3.0, 0.0]]),
                "radii": np.array([0.5]),
            },
        )
        ranges = ds.observe(ws)
        # Circle is closer than wall -> min should be near 2.5.
        assert np.min(ranges) < 3.5

    def test_depth_with_dict_agents(self):
        ds = DepthSensor(DepthSensorConfig(resolution=16, max_range=20.0, noise_std=0.0))
        agents = [{"pos": np.array([4.0, 0.0]), "radius": 0.3}]
        ws = _world_state(robot_pos=(0.0, 0.0), agents=agents)
        ranges = ds.observe(ws)
        assert np.min(ranges) < 20.0

    def test_depth_with_array_style_agents(self):
        """agents as a (N, 3) array of [x, y, radius] triggers the non-dict branch."""
        ds = DepthSensor(DepthSensorConfig(resolution=16, max_range=20.0, noise_std=0.0))
        agents = np.array([[4.0, 0.0, 0.4]])  # shape (1, 3)
        ws = _world_state(robot_pos=(0.0, 0.0), agents=list(agents))
        # Convert to list of rows so agents[0] is an ndarray (not dict).
        ws["agents"] = [np.asarray(row) for row in agents]
        ranges = ds.observe(ws)
        assert np.min(ranges) < 20.0

    def test_depth_clipped_to_min_range(self):
        """A very close obstacle still returns >= min_range after clipping."""
        ds = DepthSensor(
            DepthSensorConfig(
                resolution=8, min_range=0.5, max_range=20.0, noise_std=0.0
            )
        )
        # Centre the robot inside a circle so raw intersection would be negative/tiny.
        ws = _world_state(
            robot_pos=(0.0, 0.0),
            obstacles_circles={
                "centres": np.array([[0.1, 0.0]]),
                "radii": np.array([0.05]),
            },
        )
        ranges = ds.observe(ws)
        assert np.all(ranges >= 0.5 - 1e-9)
