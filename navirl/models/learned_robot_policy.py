"""Wrappers for neural-network robot navigation policies.

Provides ``PolicyRobotController``, which loads one or more trained PyTorch
models and uses them to compute robot actions from local observations.
Ensemble inference (averaging over multiple checkpoints) is supported for
improved robustness.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from navirl.core.constants import EPSILON
from navirl.core.types import Action, AgentState
from navirl.robots.base import EventSink, RobotController

__all__ = ["PolicyRobotController"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Observation helpers
# ---------------------------------------------------------------------------


def _build_robot_observation(
    robot: AgentState,
    neighbours: list[AgentState],
    max_neighbours: int = 6,
) -> np.ndarray:
    """Construct a flat observation vector for the robot.

    Layout (robot ego):
        [dx_goal, dy_goal, vx, vy, speed, radius]
    followed by up to *max_neighbours* nearest-neighbour blocks:
        [dx, dy, dvx, dvy, radius_other, is_human]
    Unoccupied slots are zero-padded.
    """
    own_dim = 6
    neigh_dim = 6
    obs = np.zeros(own_dim + max_neighbours * neigh_dim, dtype=np.float32)

    obs[0] = robot.goal_x - robot.x
    obs[1] = robot.goal_y - robot.y
    obs[2] = robot.vx
    obs[3] = robot.vy
    obs[4] = math.hypot(robot.vx, robot.vy)
    obs[5] = robot.radius

    # Sort neighbours by distance.
    dists: list[tuple[float, AgentState]] = []
    for n in neighbours:
        d = math.hypot(n.x - robot.x, n.y - robot.y)
        dists.append((d, n))
    dists.sort(key=lambda t: t[0])

    for idx, (_, n) in enumerate(dists[:max_neighbours]):
        base = own_dim + idx * neigh_dim
        obs[base + 0] = n.x - robot.x
        obs[base + 1] = n.y - robot.y
        obs[base + 2] = n.vx - robot.vx
        obs[base + 3] = n.vy - robot.vy
        obs[base + 4] = n.radius
        obs[base + 5] = 1.0 if n.kind == "human" else 0.0

    return obs


# ---------------------------------------------------------------------------
#  PolicyRobotController
# ---------------------------------------------------------------------------


class PolicyRobotController(RobotController):
    """Robot controller that delegates action selection to a trained model.

    Parameters (via cfg dict)
    -------------------------
    model_path : str
        Path to a saved PyTorch model (``*.pt`` / ``*.pth``) or a
        directory containing multiple checkpoints for ensemble inference.
    device : str
        PyTorch device string (``'cpu'``, ``'cuda'``, ``'cuda:0'``, ...).
    max_neighbours : int
        Maximum number of nearest neighbours included in each
        observation vector.
    goal_tolerance : float
        Distance threshold below which the robot is considered to have
        reached its goal.
    max_speed : float
        Maximum allowed speed; policy outputs are clamped to this.
    """

    def __init__(self, cfg: dict | None = None) -> None:
        cfg = cfg or {}
        self.cfg = cfg
        self.model_path_str = cfg.get("model_path", "")
        self.device_str = str(cfg.get("device", "cpu"))
        self.max_neighbours = int(cfg.get("max_neighbours", 6))
        self.goal_tolerance = float(cfg.get("goal_tolerance", 0.2))
        self.max_speed = float(cfg.get("max_speed", 0.8))

        # Lazy-loaded models (deferred so that import does not require torch).
        self._models: list[Any] = []
        self._device: Any = None
        self._loaded = False

        self.robot_id = -1
        self.start = (0.0, 0.0)
        self.goal = (0.0, 0.0)
        self.backend: Any = None

    # -- lazy model loading -------------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load PyTorch model(s) on first use."""
        if self._loaded:
            return

        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PolicyRobotController requires PyTorch.  Install it with: pip install torch"
            ) from exc

        model_path = Path(self.model_path_str)
        if not self.model_path_str:
            raise FileNotFoundError(
                "PolicyRobotController requires 'model_path' in config "
                "pointing to a .pt/.pth file or directory of checkpoints."
            )

        self._device = torch.device(self.device_str)

        paths: list[Path] = []
        if model_path.is_dir():
            paths = sorted(model_path.glob("*.pt")) + sorted(model_path.glob("*.pth"))
            if not paths:
                raise FileNotFoundError(f"No .pt/.pth files found in {model_path}")
        else:
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            paths = [model_path]

        for p in paths:
            model = torch.jit.load(str(p), map_location=self._device)  # type: ignore[attr-defined]
            model.eval()
            self._models.append(model)
            logger.info("Loaded robot policy model: %s", p)

        self._loaded = True

    # -- RobotController interface ------------------------------------------

    def reset(
        self,
        robot_id: int,
        start: tuple[float, float],
        goal: tuple[float, float],
        backend,
    ) -> None:
        self.robot_id = robot_id
        self.start = start
        self.goal = goal
        self.backend = backend

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        emit_event: EventSink,
    ) -> Action:
        st = states[self.robot_id]
        dist_goal = math.hypot(self.goal[0] - st.x, self.goal[1] - st.y)
        if dist_goal <= self.goal_tolerance:
            return Action(pref_vx=0.0, pref_vy=0.0, behavior="DONE")

        import torch  # type: ignore[import-untyped]

        self._ensure_loaded()

        # Build observation.
        neighbours = [s for aid, s in states.items() if aid != self.robot_id]
        obs = _build_robot_observation(st, neighbours, self.max_neighbours)
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self._device).unsqueeze(0)

        # Ensemble forward pass.
        vx_sum, vy_sum = 0.0, 0.0
        with torch.no_grad():
            for model in self._models:
                output = model(obs_tensor)  # expected shape (1, 2)
                out_np = output.cpu().numpy().flatten()
                vx_sum += float(out_np[0])
                vy_sum += float(out_np[1])

        n_models = len(self._models)
        pvx = vx_sum / n_models
        pvy = vy_sum / n_models

        # Clamp to max speed.
        speed = math.hypot(pvx, pvy)
        effective_max = min(st.max_speed, self.max_speed)
        if speed > effective_max and speed > EPSILON:
            scale = effective_max / speed
            pvx *= scale
            pvy *= scale

        return Action(
            pref_vx=pvx,
            pref_vy=pvy,
            behavior="LEARNED",
            metadata={
                "ensemble_size": n_models,
                "raw_speed": speed,
            },
        )
