from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import yaml

from navirl.core.maps import load_map_info


def _world_to_px(
    x: float,
    y: float,
    shape: tuple[int, int],
    scale: float,
    pixels_per_meter: float,
    row_offset: int = 0,
    col_offset: int = 0,
) -> tuple[int, int]:
    h, w = shape
    row = int(round(y * pixels_per_meter + h / 2.0)) - int(row_offset)
    col = int(round(x * pixels_per_meter + w / 2.0)) - int(col_offset)
    return int(round(col * scale)), int(round(row * scale))


def _load_rows(state_path: Path) -> list[dict]:
    rows = []
    with state_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _door_token_by_step(events_path: Path) -> dict[int, int | None]:
    if not events_path.exists():
        return {}

    holder_by_step: dict[int, int | None] = {}
    token_holder: int | None = None

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ev = json.loads(line)
            step = int(ev["step"])
            et = ev["event_type"]
            if et == "door_token_acquire":
                token_holder = ev["agent_id"]
            elif et == "door_token_release":
                token_holder = None
            holder_by_step[step] = token_holder

    return holder_by_step


def _arrow_endpoint(
    x: float,
    y: float,
    vx: float,
    vy: float,
    last_heading: tuple[float, float] | None,
    min_len_m: float = 0.2,
) -> tuple[float, float, tuple[float, float] | None]:
    speed = math.hypot(vx, vy)
    if speed > 1e-5:
        ux, uy = vx / speed, vy / speed
    elif last_heading is not None:
        ux, uy = last_heading
    else:
        return x, y, last_heading

    length = max(min_len_m, min(0.52, speed * 1.05))
    return x + ux * length, y + uy * length, (ux, uy)


def _stylized_background(map_img: np.ndarray, scale: float) -> np.ndarray:
    h, w = map_img.shape[:2]
    out_h = max(1, int(round(h * scale)))
    out_w = max(1, int(round(w * scale)))
    expanded = cv2.resize(map_img.astype(np.uint8), (out_w, out_h), interpolation=cv2.INTER_NEAREST)
    free = expanded > 0

    h, w = expanded.shape
    yy = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    xx = np.linspace(0.0, 1.0, w, dtype=np.float32)[None, :]

    base = np.zeros((h, w, 3), dtype=np.float32)

    # Warm-cool gradient for traversable space.
    free_b = 235.0 + 7.0 * (1.0 - yy) + 5.0 * np.sin(5.0 * xx)
    free_g = 232.0 + 8.0 * (1.0 - yy)
    free_r = 223.0 + 12.0 * yy

    # Deep slate-blue obstacle tone with subtle variation.
    obst_b = 44.0 + 8.0 * yy
    obst_g = 40.0 + 4.0 * (1.0 - yy)
    obst_r = 33.0 + 5.0 * yy

    base[..., 0] = np.where(free, free_b, obst_b)
    base[..., 1] = np.where(free, free_g, obst_g)
    base[..., 2] = np.where(free, free_r, obst_r)

    rng = np.random.default_rng(17)
    noise = rng.normal(0.0, 3.0, size=(h, w, 1)).astype(np.float32)
    base += noise

    # Soft vignette to focus center action.
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    yy_i, xx_i = np.indices((h, w), dtype=np.float32)
    r2 = ((yy_i - cy) / max(1.0, h * 0.55)) ** 2 + ((xx_i - cx) / max(1.0, w * 0.55)) ** 2
    vignette = 1.0 - 0.2 * np.clip(r2, 0.0, 1.0)
    base *= vignette[..., None]

    return np.clip(base, 0, 255).astype(np.uint8)


def _agent_palette(
    kind: str, behavior: str
) -> tuple[tuple[int, int, int], tuple[int, int, int], tuple[int, int, int]]:
    if kind == "robot":
        return (46, 56, 72), (57, 168, 244), (245, 234, 198)

    if behavior == "YIELDING":
        return (40, 50, 64), (90, 180, 244), (213, 232, 246)

    return (38, 52, 44), (86, 198, 126), (214, 245, 218)


def render_trace(
    state_path: str | Path,
    out_dir: str | Path,
    fps: int = 12,
    video: bool = False,
    max_frames: int | None = None,
) -> dict:
    """Render cinematic debug overlays from a state log."""

    state_path = Path(state_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for stale in out_dir.glob("frame_*.png"):
        stale.unlink(missing_ok=True)
    (out_dir / "video.mp4").unlink(missing_ok=True)
    (out_dir / "render_diagnostics.json").unlink(missing_ok=True)

    bundle_dir = state_path.parent
    scenario_path = bundle_dir / "scenario.yaml"
    events_path = bundle_dir / "events.jsonl"

    with scenario_path.open("r", encoding="utf-8") as f:
        scenario = yaml.safe_load(f)

    rows = _load_rows(state_path)
    if not rows:
        raise ValueError(f"No frames found in {state_path}")

    render_cfg = dict(scenario.get("render", {}))
    trail_len = int(render_cfg.get("trail_length", 64))
    show_labels = bool(render_cfg.get("show_labels", False))
    show_hud = bool(render_cfg.get("show_hud", False))
    playback_speed = float(render_cfg.get("playback_speed", 1.85))

    token_by_step = _door_token_by_step(events_path)
    source_path = Path(scenario.get("_meta", {}).get("source_path", "."))
    map_info = load_map_info(scene_cfg=scenario["scene"], base_dir=source_path.parent)
    map_img_full = map_info.binary_map
    ppm = float(map_info.pixels_per_meter)
    full_h, full_w = map_img_full.shape[:2]

    crop_to_free = bool(render_cfg.get("crop_to_free", True))
    crop_margin_m = float(render_cfg.get("crop_margin_m", 0.6))
    crop_margin_px = int(round(max(0.0, crop_margin_m) * ppm))
    row0 = 0
    col0 = 0
    row1 = full_h
    col1 = full_w
    if crop_to_free:
        free_rc = np.argwhere(map_img_full > 0)
        if free_rc.size > 0:
            min_r = int(np.min(free_rc[:, 0]))
            max_r = int(np.max(free_rc[:, 0]))
            min_c = int(np.min(free_rc[:, 1]))
            max_c = int(np.max(free_rc[:, 1]))
            row0 = max(0, min_r - crop_margin_px)
            row1 = min(full_h, max_r + crop_margin_px + 1)
            col0 = max(0, min_c - crop_margin_px)
            col1 = min(full_w, max_c + crop_margin_px + 1)

    map_img = map_img_full[row0:row1, col0:col1]

    scale = float(render_cfg.get("pixel_scale", 2.0))
    max_canvas_dim = int(render_cfg.get("max_canvas_dim", 1800))
    if max_canvas_dim > 0:
        max_scaled = max(map_img.shape[0] * scale, map_img.shape[1] * scale)
        if max_scaled > max_canvas_dim:
            scale *= float(max_canvas_dim / max_scaled)
    base_bg = _stylized_background(map_img, scale)

    if max_frames is not None and len(rows) > max_frames:
        idxs = np.linspace(0, len(rows) - 1, max_frames).astype(int).tolist()
        rows_to_render = [rows[i] for i in idxs]
    else:
        rows_to_render = rows

    frame_paths: list[str] = []
    video_path = None
    writer = None

    if video:
        video_path = out_dir / "video.mp4"
        video_fps = max(1, int(round(float(fps) * max(0.5, playback_speed))))
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            video_fps,
            (base_bg.shape[1], base_bg.shape[0]),
        )
    else:
        video_fps = None

    history: dict[int, list[tuple[int, int]]] = {}
    heading_cache: dict[int, tuple[float, float]] = {}

    total_agents_drawn = 0
    total_arrows_drawn = 0
    total_trail_segments = 0
    total_text_elements = 0

    for idx, row in enumerate(rows_to_render):
        canvas = base_bg.copy()
        trail_glow = np.zeros_like(canvas)
        trail_core = np.zeros_like(canvas)
        halo_layer = np.zeros_like(canvas)

        step = int(row["step"])
        token_holder = token_by_step.get(step)

        # Update position history.
        for agent in row["agents"]:
            aid = int(agent["id"])
            x, y = float(agent["x"]), float(agent["y"])
            px, py = _world_to_px(
                x, y, map_img_full.shape, scale, ppm, row_offset=row0, col_offset=col0
            )
            history.setdefault(aid, []).append((px, py))
            if len(history[aid]) > trail_len:
                history[aid] = history[aid][-trail_len:]

        # Draw trajectory trails with glow pass.
        for agent in row["agents"]:
            aid = int(agent["id"])
            kind = str(agent["kind"])
            pts = history.get(aid, [])
            if len(pts) < 2:
                continue

            glow_color = (88, 189, 255) if kind == "robot" else (112, 228, 142)
            core_color = (53, 161, 240) if kind == "robot" else (76, 198, 117)
            base_thick = 3 if kind == "robot" else 2

            for i in range(1, len(pts)):
                age = i / max(1, len(pts) - 1)
                glow_t = max(2, int(round(base_thick + 5 * age)))
                core_t = max(1, int(round(base_thick * (0.75 + 0.4 * age))))
                cv2.line(trail_glow, pts[i - 1], pts[i], glow_color, glow_t, cv2.LINE_AA)
                cv2.line(trail_core, pts[i - 1], pts[i], core_color, core_t, cv2.LINE_AA)
                total_trail_segments += 1

        trail_glow = cv2.GaussianBlur(trail_glow, (0, 0), sigmaX=2.6, sigmaY=2.6)
        canvas = cv2.addWeighted(canvas, 1.0, trail_glow, 0.34, 0.0)
        canvas = cv2.addWeighted(canvas, 1.0, trail_core, 0.9, 0.0)

        # Agent halos and body.
        for agent in row["agents"]:
            aid = int(agent["id"])
            kind = str(agent["kind"])
            behavior = str(agent.get("behavior", ""))
            x, y = float(agent["x"]), float(agent["y"])
            radius = float(agent.get("radius", 0.18))

            px, py = _world_to_px(
                x, y, map_img_full.shape, scale, ppm, row_offset=row0, col_offset=col0
            )
            rad_px = max(3, int(round(radius * ppm * scale)))

            edge_color, fill_color, highlight_color = _agent_palette(kind, behavior)

            halo_radius = int(round(rad_px * 2.6))
            halo_color = (112, 208, 255) if kind == "robot" else (109, 228, 148)
            cv2.circle(
                halo_layer, (px, py), halo_radius, halo_color, thickness=-1, lineType=cv2.LINE_AA
            )

            # Soft shadow for depth.
            cv2.circle(
                canvas, (px + 2, py + 2), rad_px, (22, 22, 28), thickness=-1, lineType=cv2.LINE_AA
            )
            cv2.circle(canvas, (px, py), rad_px, fill_color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.circle(canvas, (px, py), rad_px, edge_color, thickness=2, lineType=cv2.LINE_AA)
            if kind != "robot":
                cv2.circle(
                    canvas,
                    (px - max(1, rad_px // 3), py - max(1, rad_px // 3)),
                    max(1, rad_px // 3),
                    highlight_color,
                    thickness=-1,
                    lineType=cv2.LINE_AA,
                )

        halo_layer = cv2.GaussianBlur(halo_layer, (0, 0), sigmaX=5.5, sigmaY=5.5)
        canvas = cv2.addWeighted(canvas, 1.0, halo_layer, 0.2, 0.0)

        # Draw arrows and optional labels.
        for agent in row["agents"]:
            aid = int(agent["id"])
            kind = str(agent["kind"])
            x, y = float(agent["x"]), float(agent["y"])
            vx, vy = float(agent["vx"]), float(agent["vy"])

            px, py = _world_to_px(
                x, y, map_img_full.shape, scale, ppm, row_offset=row0, col_offset=col0
            )
            last_heading = heading_cache.get(aid)
            tx, ty, new_heading = _arrow_endpoint(x, y, vx, vy, last_heading)
            if new_heading is not None:
                heading_cache[aid] = new_heading
                apx, apy = _world_to_px(
                    tx, ty, map_img_full.shape, scale, ppm, row_offset=row0, col_offset=col0
                )
                arrow_core = (28, 30, 34)
                arrow_highlight = (56, 187, 255) if kind == "robot" else (72, 212, 128)
                cv2.arrowedLine(
                    canvas, (px, py), (apx, apy), arrow_core, 3, cv2.LINE_AA, tipLength=0.25
                )
                cv2.arrowedLine(
                    canvas, (px, py), (apx, apy), arrow_highlight, 2, cv2.LINE_AA, tipLength=0.27
                )
                total_arrows_drawn += 1

            if token_holder is not None and aid == int(token_holder):
                cv2.circle(
                    canvas,
                    (px, py),
                    max(6, int(round(0.34 * ppm * scale))),
                    (70, 204, 255),
                    2,
                    cv2.LINE_AA,
                )

            if show_labels:
                cv2.putText(
                    canvas,
                    f"{aid}",
                    (px + 4, py - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (24, 24, 28),
                    1,
                    cv2.LINE_AA,
                )
                total_text_elements += 1

            total_agents_drawn += 1

        if show_hud:
            cv2.putText(
                canvas,
                f"NavIRL | t={row['time_s']:.2f}s",
                (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (28, 28, 32),
                1,
                cv2.LINE_AA,
            )
            total_text_elements += 1
            if token_holder is not None:
                cv2.putText(
                    canvas,
                    f"door token: {token_holder}",
                    (10, 42),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (28, 28, 32),
                    1,
                    cv2.LINE_AA,
                )
                total_text_elements += 1
            cv2.putText(
                canvas,
                f"map: {map_info.map_id} | {map_info.pixels_per_meter:.1f}px/m",
                (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (28, 28, 32),
                1,
                cv2.LINE_AA,
            )
            total_text_elements += 1

        frame_path = out_dir / f"frame_{idx:04d}.png"
        cv2.imwrite(str(frame_path), canvas)
        frame_paths.append(str(frame_path))

        if writer is not None:
            writer.write(canvas)

    if writer is not None:
        writer.release()

    diagnostics = {
        "style_version": "v3_cinematic_glow",
        "map_id": map_info.map_id,
        "map_source": map_info.source,
        "pixels_per_meter": float(map_info.pixels_per_meter),
        "meters_per_pixel": float(map_info.meters_per_pixel),
        "pixel_scale_requested": float(render_cfg.get("pixel_scale", 2.0)),
        "pixel_scale_effective": float(scale),
        "canvas_width_px": int(base_bg.shape[1]),
        "canvas_height_px": int(base_bg.shape[0]),
        "crop_row0_px": int(row0),
        "crop_row1_px": int(row1),
        "crop_col0_px": int(col0),
        "crop_col1_px": int(col1),
        "crop_to_free": bool(crop_to_free),
        "frame_count": len(frame_paths),
        "total_agents_drawn": int(total_agents_drawn),
        "total_arrows_drawn": int(total_arrows_drawn),
        "total_trail_segments": int(total_trail_segments),
        "total_text_elements": int(total_text_elements),
        "avg_agents_per_frame": float(total_agents_drawn / max(1, len(frame_paths))),
        "avg_arrows_per_frame": float(total_arrows_drawn / max(1, len(frame_paths))),
        "avg_trail_segments_per_frame": float(total_trail_segments / max(1, len(frame_paths))),
        "labels_enabled": bool(show_labels),
        "hud_enabled": bool(show_hud),
        "playback_speed": float(playback_speed),
        "video_fps": int(video_fps) if video_fps is not None else None,
    }
    diag_path = out_dir / "render_diagnostics.json"
    with diag_path.open("w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)

    return {
        "frames_dir": str(out_dir),
        "frame_count": len(frame_paths),
        "frame_paths": frame_paths,
        "video_path": str(video_path) if video_path is not None else None,
        "render_diagnostics_path": str(diag_path),
    }


class EnvironmentRenderer:
    """Shared rendering utilities for environment visualization."""

    def __init__(self):
        self._cv2_available = self._check_cv2_availability()

    @staticmethod
    def _check_cv2_availability() -> bool:
        """Check if OpenCV is available."""
        try:
            import cv2

            return True
        except ImportError:
            return False

    def prepare_map_image(self, backend, *, grayscale_to_rgb: bool = True) -> np.ndarray | None:
        """Prepare base map image for rendering.

        Parameters
        ----------
        backend : Backend
            Environment backend that provides map_image() method.
        grayscale_to_rgb : bool, default True
            Convert grayscale images to RGB by stacking channels.

        Returns
        -------
        np.ndarray | None
            Prepared map image as uint8 array, or None if unavailable.
        """
        if backend is None:
            return None
        img = backend.map_image()
        if img is None:
            return None
        img = np.asarray(img, dtype=np.uint8)
        if grayscale_to_rgb and img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        return img

    def draw_agent_circle(
        self,
        img: np.ndarray,
        position: tuple[float, float],
        backend,
        *,
        radius: int = 5,
        color: tuple[int, int, int] = (31, 119, 180),
    ) -> None:
        """Draw a colored circle for an agent on the image.

        Parameters
        ----------
        img : np.ndarray
            Image to draw on (modified in place).
        position : tuple[float, float]
            World coordinates (x, y) of the agent.
        backend : Backend
            Backend to convert world coordinates to map pixels.
        radius : int, default 5
            Circle radius in pixels.
        color : tuple[int, int, int], default (31, 119, 180)
            RGB color for the circle.
        """
        if not self._cv2_available:
            return

        try:
            import cv2

            px = backend.world_to_map(position)
            # OpenCV uses (x, y) = (col, row) format
            cv2.circle(img, (px[1], px[0]), radius, color, -1)
        except (ImportError, Exception):
            # Silently handle any drawing errors
            pass

    def draw_goal_circle(
        self,
        img: np.ndarray,
        goal_position: tuple[float, float],
        backend,
        *,
        radius: int = 5,
        color: tuple[int, int, int] = (214, 39, 40),
    ) -> None:
        """Draw a colored circle for a goal on the image.

        Parameters
        ----------
        img : np.ndarray
            Image to draw on (modified in place).
        goal_position : tuple[float, float]
            World coordinates (x, y) of the goal.
        backend : Backend
            Backend to convert world coordinates to map pixels.
        radius : int, default 5
            Circle radius in pixels.
        color : tuple[int, int, int], default (214, 39, 40)
            RGB color for the circle.
        """
        if not self._cv2_available:
            return

        try:
            import cv2

            g_px = backend.world_to_map((float(goal_position[0]), float(goal_position[1])))
            cv2.circle(img, (g_px[1], g_px[0]), radius, color, -1)
        except (ImportError, Exception):
            pass

    def draw_humans(
        self,
        img: np.ndarray,
        human_ids: list[int],
        backend,
        *,
        radius: int = 4,
        color: tuple[int, int, int] = (255, 127, 14),
    ) -> None:
        """Draw circles for all humans on the image.

        Parameters
        ----------
        img : np.ndarray
            Image to draw on (modified in place).
        human_ids : list[int]
            List of human agent IDs to draw.
        backend : Backend
            Backend to get positions and convert coordinates.
        radius : int, default 4
            Circle radius in pixels.
        color : tuple[int, int, int], default (255, 127, 14)
            RGB color for human circles.
        """
        if not self._cv2_available:
            return

        try:
            import cv2

            for hid in human_ids:
                hp = backend.get_position(hid)
                hp_px = backend.world_to_map(hp)
                cv2.circle(img, (hp_px[1], hp_px[0]), radius, color, -1)
        except (ImportError, Exception):
            pass

    def show_image(self, img: np.ndarray, window_name: str = "NavIRL Environment") -> None:
        """Display image in an OpenCV window for human rendering mode.

        Parameters
        ----------
        img : np.ndarray
            Image to display.
        window_name : str, default "NavIRL Environment"
            Name of the display window.
        """
        if not self._cv2_available:
            return

        try:
            import cv2

            cv2.imshow(window_name, img)
            cv2.waitKey(1)
        except ImportError:
            pass


# Create a shared instance for convenience
env_renderer = EnvironmentRenderer()
