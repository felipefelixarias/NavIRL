"""Wrappers for neural-network pedestrian policies.

Provides ``PolicyHumanController``, which loads one or more trained PyTorch
models and uses them to compute pedestrian actions from local observations.
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
from navirl.humans.base import EventSink, HumanController

__all__ = ["PolicyHumanController"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
#  Observation helpers
# ---------------------------------------------------------------------------


def _build_observation(
    agent: AgentState,
    neighbours: list[AgentState],
    max_neighbours: int = 6,
) -> np.ndarray:
    """Construct a flat observation vector for a single agent.

    Layout (per agent):
        [dx_goal, dy_goal, vx, vy, speed, radius]
    followed by up to *max_neighbours* nearest-neighbour blocks:
        [dx, dy, dvx, dvy, radius_other]
    Unoccupied slots are zero-padded.
    """
    own_dim = 6
    neigh_dim = 5
    obs = np.zeros(own_dim + max_neighbours * neigh_dim, dtype=np.float32)

    obs[0] = agent.goal_x - agent.x
    obs[1] = agent.goal_y - agent.y
    obs[2] = agent.vx
    obs[3] = agent.vy
    obs[4] = math.hypot(agent.vx, agent.vy)
    obs[5] = agent.radius

    # Sort neighbours by distance.
    dists: list[tuple[float, AgentState]] = []
    for n in neighbours:
        d = math.hypot(n.x - agent.x, n.y - agent.y)
        dists.append((d, n))
    dists.sort(key=lambda t: t[0])

    for idx, (_, n) in enumerate(dists[:max_neighbours]):
        base = own_dim + idx * neigh_dim
        obs[base + 0] = n.x - agent.x
        obs[base + 1] = n.y - agent.y
        obs[base + 2] = n.vx - agent.vx
        obs[base + 3] = n.vy - agent.vy
        obs[base + 4] = n.radius

    return obs


# ---------------------------------------------------------------------------
#  PolicyHumanController
# ---------------------------------------------------------------------------


class PolicyHumanController(HumanController):
    """Human controller that delegates action selection to a trained model.

    Parameters
    ----------
    model_path:
        Path to a saved PyTorch model (``*.pt`` / ``*.pth``) or a
        directory containing multiple checkpoints for ensemble inference.
    device:
        PyTorch device string (``'cpu'``, ``'cuda'``, ``'cuda:0'``, ...).
    max_neighbours:
        Maximum number of nearest neighbours included in each
        observation vector.
    """

    def __init__(
        self,
        model_path: str | Path,
        device: str = "cpu",
        max_neighbours: int = 6,
    ) -> None:
        self.model_path = Path(model_path)
        self.device_str = device
        self.max_neighbours = max_neighbours

        # Lazy-loaded models (deferred so that import does not require torch).
        self._models: list[Any] = []
        self._device: Any = None
        self._loaded = False

        self.human_ids: list[int] = []
        self.starts: dict[int, tuple[float, float]] = {}
        self.goals: dict[int, tuple[float, float]] = {}
        self.backend: Any = None

    # -- lazy model loading ---------------------------------------------

    def _ensure_loaded(self) -> None:
        """Load PyTorch model(s) on first use."""
        if self._loaded:
            return

        try:
            import torch  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PolicyHumanController requires PyTorch.  Install it with: pip install torch"
            ) from exc

        self._device = torch.device(self.device_str)

        paths: list[Path] = []
        if self.model_path.is_dir():
            paths = sorted(self.model_path.glob("*.pt")) + sorted(self.model_path.glob("*.pth"))
            if not paths:
                raise FileNotFoundError(f"No .pt/.pth files found in {self.model_path}")
        else:
            paths = [self.model_path]

        for p in paths:
            model = torch.jit.load(str(p), map_location=self._device)  # type: ignore[attr-defined]
            model.eval()
            self._models.append(model)
            logger.info("Loaded policy model: %s", p)

        self._loaded = True

    # -- HumanController interface --------------------------------------

    def reset(
        self,
        human_ids: list[int],
        starts: dict[int, tuple[float, float]],
        goals: dict[int, tuple[float, float]],
        backend=None,
    ) -> None:
        self.human_ids = list(human_ids)
        self.starts = dict(starts)
        self.goals = dict(goals)
        self.backend = backend

    def step(
        self,
        step: int,
        time_s: float,
        dt: float,
        states: dict[int, AgentState],
        robot_id: int,
        emit_event: EventSink,
    ) -> dict[int, Action]:
        import torch  # type: ignore[import-untyped]

        self._ensure_loaded()

        actions: dict[int, Action] = {}

        for hid in self.human_ids:
            agent = states[hid]

            # Check goal arrival and swap.
            gx, gy = self.goals[hid]
            dist_to_goal = math.hypot(gx - agent.x, gy - agent.y)
            if dist_to_goal < 0.5:
                prev = self.goals[hid]
                self.goals[hid] = self.starts[hid]
                self.starts[hid] = prev
                emit_event(
                    "goal_swap",
                    hid,
                    {
                        "new_goal": list(self.goals[hid]),
                        "new_start": list(self.starts[hid]),
                    },
                )

            # Build observation.
            neighbours = [s for aid, s in states.items() if aid != hid]
            obs = _build_observation(agent, neighbours, self.max_neighbours)
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
            if speed > agent.max_speed and speed > EPSILON:
                scale = agent.max_speed / speed
                pvx *= scale
                pvy *= scale

            actions[hid] = Action(
                pref_vx=pvx,
                pref_vy=pvy,
                behavior="LEARNED",
                metadata={
                    "ensemble_size": n_models,
                    "raw_speed": speed,
                },
            )

        return actions
