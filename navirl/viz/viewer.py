from __future__ import annotations

from pathlib import Path

from navirl.viz.render import render_trace


def replay_log(
    state_path: str | Path,
    out_dir: str | Path,
    fps: int = 12,
    video: bool = True,
    max_frames: int | None = None,
) -> dict:
    """Replay a state log by rendering overlays to frames/video."""

    return render_trace(
        state_path=state_path,
        out_dir=out_dir,
        fps=fps,
        video=video,
        max_frames=max_frames,
    )
