"""End-to-end tests for navirl.viz.render.render_trace.

Exercises the full render_trace pipeline with synthetic state logs and
scenarios using the builtin doorway map. Covers stylized background rendering,
trail/halo/arrow drawing, label/HUD overlays, video output, frame downsampling,
door-token highlighting, and crop-to-free behavior.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from navirl.viz.render import (
    _stylized_background,
    render_trace,
)


def _write_scenario(bundle_dir: Path, render_cfg: dict | None = None) -> Path:
    """Build a minimal scenario.yaml using a builtin map."""
    scenario = {
        "scene": {
            "map": {
                "source": "builtin",
                "id": "doorway",
                "pixels_per_meter": 100.0,
            }
        },
        "_meta": {
            "source_path": str(bundle_dir / "scenario.yaml"),
        },
    }
    if render_cfg:
        scenario["render"] = render_cfg
    path = bundle_dir / "scenario.yaml"
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(scenario, f)
    return path


def _write_state_log(bundle_dir: Path, n_frames: int = 4) -> Path:
    """Write a simple state.jsonl with one robot and one human moving in +x."""
    rows = []
    for step in range(n_frames):
        rows.append(
            {
                "step": step,
                "time_s": step * 0.1,
                "agents": [
                    {
                        "id": 0,
                        "kind": "robot",
                        "behavior": "GO_TO",
                        "x": -0.5 + step * 0.05,
                        "y": 0.0,
                        "vx": 0.5,
                        "vy": 0.0,
                        "radius": 0.2,
                    },
                    {
                        "id": 1,
                        "kind": "human",
                        "behavior": "GO_TO",
                        "x": 0.5 - step * 0.05,
                        "y": 0.1,
                        "vx": -0.4,
                        "vy": 0.0,
                        "radius": 0.18,
                    },
                ],
            }
        )
    path = bundle_dir / "state.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


def _write_events(bundle_dir: Path) -> Path:
    """Write events.jsonl exercising door-token acquire/release."""
    events = [
        {"step": 1, "event_type": "door_token_acquire", "agent_id": 0},
        {"step": 3, "event_type": "door_token_release", "agent_id": 0},
    ]
    path = bundle_dir / "events.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    return path


# ---------------------------------------------------------------------------
# _stylized_background
# ---------------------------------------------------------------------------


class TestStylizedBackground:
    def test_returns_uint8_rgb_array(self):
        map_img = np.zeros((20, 30), dtype=np.uint8)
        map_img[5:15, 5:25] = 1
        bg = _stylized_background(map_img, scale=2.0)
        assert bg.dtype == np.uint8
        assert bg.shape == (40, 60, 3)

    def test_scale_resizes_canvas(self):
        map_img = np.ones((10, 10), dtype=np.uint8)
        bg_1x = _stylized_background(map_img, scale=1.0)
        bg_3x = _stylized_background(map_img, scale=3.0)
        assert bg_3x.shape[0] == 30
        assert bg_3x.shape[1] == 30
        assert bg_1x.shape[0] == 10

    def test_min_dim_one(self):
        map_img = np.ones((1, 1), dtype=np.uint8)
        bg = _stylized_background(map_img, scale=0.5)
        # Output should be at least 1x1 even at sub-pixel scale.
        assert bg.shape[0] >= 1
        assert bg.shape[1] >= 1

    def test_values_in_valid_range(self):
        map_img = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        bg = _stylized_background(map_img, scale=4.0)
        assert bg.min() >= 0
        assert bg.max() <= 255

    def test_free_and_obstacle_distinct(self):
        # A canvas with a clear free/obstacle split should have visibly
        # different mean colors in those regions.
        m = np.zeros((40, 40), dtype=np.uint8)
        m[:, 20:] = 1  # right half free, left half obstacle
        bg = _stylized_background(m, scale=1.0)
        left = bg[:, :20].mean()
        right = bg[:, 20:].mean()
        assert right > left


# ---------------------------------------------------------------------------
# render_trace — end-to-end
# ---------------------------------------------------------------------------


class TestRenderTrace:
    def test_renders_frames_and_diagnostics(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        state_path = _write_state_log(bundle_dir, n_frames=3)

        out_dir = bundle_dir / "frames"
        result = render_trace(state_path, out_dir, fps=10, video=False)

        assert result["frame_count"] == 3
        assert len(result["frame_paths"]) == 3
        for fp in result["frame_paths"]:
            assert Path(fp).exists()
            assert Path(fp).stat().st_size > 0
        assert result["video_path"] is None

        # Diagnostics file exists and has expected keys
        diag_path = Path(result["render_diagnostics_path"])
        assert diag_path.exists()
        with diag_path.open() as f:
            diag = json.load(f)
        assert diag["frame_count"] == 3
        assert diag["map_id"] == "doorway"
        assert diag["total_agents_drawn"] == 6  # 2 agents x 3 frames
        assert diag["style_version"] == "v3_cinematic_glow"
        assert diag["video_fps"] is None
        assert diag["labels_enabled"] is False
        assert diag["hud_enabled"] is False

    def test_empty_state_raises(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        state_path = bundle_dir / "state.jsonl"
        state_path.write_text("")  # empty
        with pytest.raises(ValueError, match="No frames"):
            render_trace(state_path, bundle_dir / "frames", video=False)

    def test_video_output(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        state_path = _write_state_log(bundle_dir, n_frames=2)

        out_dir = bundle_dir / "frames"
        result = render_trace(state_path, out_dir, fps=8, video=True)

        assert result["video_path"] is not None
        video_path = Path(result["video_path"])
        assert video_path.exists()
        assert video_path.suffix == ".mp4"
        # Video file should be non-empty
        assert video_path.stat().st_size > 0

        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        assert diag["video_fps"] is not None
        assert diag["video_fps"] >= 1

    def test_max_frames_subsamples(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        _write_state_log(bundle_dir, n_frames=20)

        out_dir = bundle_dir / "frames"
        result = render_trace(bundle_dir / "state.jsonl", out_dir, video=False, max_frames=5)
        assert result["frame_count"] == 5

    def test_max_frames_no_op_when_below(self, tmp_path):
        # max_frames > available rows should keep all rows.
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        _write_state_log(bundle_dir, n_frames=2)

        result = render_trace(
            bundle_dir / "state.jsonl",
            bundle_dir / "frames",
            video=False,
            max_frames=100,
        )
        assert result["frame_count"] == 2

    def test_labels_and_hud_enabled(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(
            bundle_dir,
            render_cfg={"show_labels": True, "show_hud": True},
        )
        _write_state_log(bundle_dir, n_frames=2)
        _write_events(bundle_dir)  # adds door token holder for HUD line

        result = render_trace(bundle_dir / "state.jsonl", bundle_dir / "frames", video=False)
        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        assert diag["labels_enabled"] is True
        assert diag["hud_enabled"] is True
        # 2 frames * 2 agents = 4 label texts; HUD adds at least 2 per frame.
        assert diag["total_text_elements"] >= 4

    def test_clears_stale_frames(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        _write_state_log(bundle_dir, n_frames=2)

        out_dir = bundle_dir / "frames"
        out_dir.mkdir()
        # Pre-existing stale frame and video that should be cleared.
        (out_dir / "frame_9999.png").write_bytes(b"stale")
        (out_dir / "video.mp4").write_bytes(b"stale")
        (out_dir / "render_diagnostics.json").write_text("{}")

        render_trace(bundle_dir / "state.jsonl", out_dir, video=False)

        # Stale frame_9999.png should be removed.
        assert not (out_dir / "frame_9999.png").exists()
        # New diagnostics should not be the placeholder anymore.
        diag = json.loads((out_dir / "render_diagnostics.json").read_text())
        assert "frame_count" in diag

    def test_door_token_highlight_diagnostic(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        _write_state_log(bundle_dir, n_frames=4)
        _write_events(bundle_dir)

        result = render_trace(bundle_dir / "state.jsonl", bundle_dir / "frames", video=False)
        # Should still draw all agents; existence of events file should
        # only affect token highlighting (drawn pixels), not frame counts.
        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        assert diag["frame_count"] == 4

    def test_crop_disabled(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir, render_cfg={"crop_to_free": False})
        _write_state_log(bundle_dir, n_frames=1)

        result = render_trace(bundle_dir / "state.jsonl", bundle_dir / "frames", video=False)
        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        assert diag["crop_to_free"] is False
        # When crop is disabled, the crop window is the full map.
        assert diag["crop_row0_px"] == 0
        assert diag["crop_col0_px"] == 0

    def test_max_canvas_dim_caps_scale(self, tmp_path):
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        # Force a large pixel_scale but a tight canvas cap so the effective
        # scale gets adjusted downward.
        _write_scenario(
            bundle_dir,
            render_cfg={"pixel_scale": 10.0, "max_canvas_dim": 100},
        )
        _write_state_log(bundle_dir, n_frames=1)

        result = render_trace(bundle_dir / "state.jsonl", bundle_dir / "frames", video=False)
        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        assert diag["pixel_scale_requested"] == 10.0
        assert diag["pixel_scale_effective"] < 10.0
        assert max(diag["canvas_width_px"], diag["canvas_height_px"]) <= 100 + 5

    def test_trail_length_short(self, tmp_path):
        # With a short trail length, only the last few segments should be
        # retained per agent — diagnostics should still be self-consistent.
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir, render_cfg={"trail_length": 2})
        _write_state_log(bundle_dir, n_frames=5)

        result = render_trace(bundle_dir / "state.jsonl", bundle_dir / "frames", video=False)
        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        # Trail segments at most: 2 agents * (2-1) per frame * 5 frames = 10
        # In practice fewer because frame 0 has no segments.
        assert diag["total_trail_segments"] <= 2 * 1 * 5
        assert diag["frame_count"] == 5

    def test_arrow_drawn_for_moving_agents(self, tmp_path):
        # Both agents move in every frame, so arrows should be drawn each step
        # for each agent.
        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        _write_state_log(bundle_dir, n_frames=3)

        result = render_trace(bundle_dir / "state.jsonl", bundle_dir / "frames", video=False)
        with Path(result["render_diagnostics_path"]).open() as f:
            diag = json.load(f)
        # 2 agents * 3 frames = 6 arrows drawn (all moving).
        assert diag["total_arrows_drawn"] == 6


class TestReplayLog:
    def test_replay_log_delegates_to_render_trace(self, tmp_path):
        from navirl.viz.viewer import replay_log

        bundle_dir = tmp_path / "bundle"
        bundle_dir.mkdir()
        _write_scenario(bundle_dir)
        _write_state_log(bundle_dir, n_frames=2)

        result = replay_log(
            bundle_dir / "state.jsonl",
            bundle_dir / "frames",
            fps=8,
            video=False,
            max_frames=None,
        )
        assert result["frame_count"] == 2
        assert result["video_path"] is None
