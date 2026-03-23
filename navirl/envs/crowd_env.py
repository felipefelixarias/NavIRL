"""Crowd navigation Gymnasium environment with configurable density.

Extends :class:`NavEnv` to support randomised crowd sizes, density-based
spawning, and additional crowd-specific reward terms (e.g., penalties for
disrupting pedestrian flow or entering high-density zones).

Exports
-------
CrowdNavEnv     -- crowd navigation environment
CrowdNavConfig  -- extended configuration dataclass
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Tuple, Union

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as _exc:
    raise ImportError(
        "NavIRL Gymnasium environments require the 'gymnasium' package.  "
        "Install it with:  pip install gymnasium"
    ) from _exc

from navirl.core.constants import (
    EPSILON,
    LOS,
    PROXEMICS,
    REWARD,
    SIM,
)
from navirl.envs.base_env import NavEnv, NavEnvConfig

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CrowdNavConfig(NavEnvConfig):
    """Configuration for :class:`CrowdNavEnv`.

    Adds crowd-density and human-policy knobs on top of :class:`NavEnvConfig`.

    Parameters
    ----------
    num_humans_range : tuple[int, int]
        Each episode randomly samples the number of humans from this range
        (inclusive on both ends).  Overrides *num_humans* from the base config.
    human_policy : ``"orca"`` | ``"sfm"`` | ``"random"``
        Motion model used for the background pedestrians.
    crowd_density_target : float or None
        If set (ped/m^2), the environment automatically computes how many
        humans to spawn so that the walkable area reaches this density,
        ignoring *num_humans_range*.
    density_reward_weight : float
        Multiplier on the crowd-specific density penalty term.
    flow_disruption_weight : float
        Multiplier on the reward penalty for disrupting pedestrian flow
        (measured as the change in average human speed caused by the robot).
    personal_space_weight : float
        Extra multiplier on personal-space invasion penalties beyond the
        base intimate-zone penalty.
    """

    num_humans_range: Tuple[int, int] = (3, 15)
    human_policy: Literal["orca", "sfm", "random"] = "orca"
    crowd_density_target: Optional[float] = None

    # Crowd-specific reward weights
    density_reward_weight: float = 0.5
    flow_disruption_weight: float = 0.3
    personal_space_weight: float = 0.5


# ---------------------------------------------------------------------------
#  Environment
# ---------------------------------------------------------------------------


class CrowdNavEnv(NavEnv):
    """Crowd navigation with configurable density and flow-aware rewards.

    On each :meth:`reset`, the number and placement of pedestrians is
    randomised.  When *crowd_density_target* is set the environment
    automatically computes how many agents are needed to approximate the
    requested density in the walkable area.
    """

    def __init__(self, config: CrowdNavConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = CrowdNavConfig(**kwargs)
        # Ensure the base class stores a CrowdNavConfig
        super().__init__(config=config)
        self._crowd_config: CrowdNavConfig = config  # typed alias

        # Per-episode bookkeeping
        self._prev_avg_human_speed: float = 0.0
        self._walkable_area: float = 1.0  # m^2, estimated on reset

    # -----------------------------------------------------------------
    #  Reset
    # -----------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        # Determine number of humans for this episode
        num_humans = self._compute_num_humans(seed)

        # Temporarily patch the base config so NavEnv._spawn_humans uses it
        original_num = self.config.num_humans
        self.config.num_humans = num_humans  # type: ignore[misc]

        obs, info = super().reset(seed=seed, options=options)

        self.config.num_humans = original_num  # type: ignore[misc]

        # Estimate walkable area for density calculations
        self._estimate_walkable_area()

        # Capture baseline human speed
        self._prev_avg_human_speed = self._average_human_speed()

        info["num_humans"] = len(self._human_ids)
        info["walkable_area_m2"] = self._walkable_area
        info["crowd_density"] = len(self._human_ids) / max(self._walkable_area, 1.0)
        return obs, info

    # -----------------------------------------------------------------
    #  Reward (extends base)
    # -----------------------------------------------------------------

    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        reward, terminated, info = super()._compute_reward()

        if terminated:
            return reward, terminated, info

        cfg = self._crowd_config
        assert self._backend is not None

        rx, ry = self._backend.get_position(self._robot_id)

        # --- Density penalty: penalise being in high-density zones ---
        local_density = self._local_density(rx, ry, radius=3.0)
        if local_density > LOS.C_max_density:
            density_penalty = -cfg.density_reward_weight * (
                local_density - LOS.C_max_density
            )
            reward += density_penalty
            info["density_penalty"] = density_penalty
        info["local_density"] = local_density

        # --- Flow disruption: penalise slowing pedestrians down ---
        avg_speed = self._average_human_speed()
        speed_drop = max(0.0, self._prev_avg_human_speed - avg_speed)
        flow_penalty = -cfg.flow_disruption_weight * speed_drop
        reward += flow_penalty
        info["flow_penalty"] = flow_penalty
        self._prev_avg_human_speed = avg_speed

        # --- Personal space: extra penalty for personal-zone intrusion ---
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            d = math.hypot(rx - hx, ry - hy)
            if PROXEMICS.intimate.outer <= d < PROXEMICS.personal.outer:
                reward += -cfg.personal_space_weight * (
                    1.0 - (d - PROXEMICS.intimate.outer)
                    / (PROXEMICS.personal.outer - PROXEMICS.intimate.outer)
                )
                info["personal_space_intrusion"] = True

        return reward, terminated, info

    # -----------------------------------------------------------------
    #  Human stepping (policy variants)
    # -----------------------------------------------------------------

    def _step_humans(self) -> None:
        policy = self._crowd_config.human_policy
        if policy == "orca":
            super()._step_humans()
        elif policy == "sfm":
            # Fallback to ORCA goal-seeking (SFM backend not yet wired)
            super()._step_humans()
        elif policy == "random":
            self._step_humans_random()
        else:
            super()._step_humans()

    def _step_humans_random(self) -> None:
        """Humans take random preferred velocities each step."""
        assert self._backend is not None
        cfg = self.config
        for hid in self._human_ids:
            angle = float(self._rng.uniform(0, 2 * math.pi))
            speed = float(self._rng.uniform(0, cfg.human_speed_range[1]))
            vx = speed * math.cos(angle)
            vy = speed * math.sin(angle)
            self._backend.set_preferred_velocity(hid, (vx, vy))

    # -----------------------------------------------------------------
    #  Helpers
    # -----------------------------------------------------------------

    def _compute_num_humans(self, seed: Optional[int]) -> int:
        """Decide how many humans to spawn this episode."""
        rng = np.random.default_rng(seed)
        cfg = self._crowd_config

        if cfg.crowd_density_target is not None:
            # Estimate walkable area from world size (rough)
            area = cfg.world_width * cfg.world_height * 0.7  # ~70% walkable
            n = max(1, int(round(cfg.crowd_density_target * area)))
            return n

        lo, hi = cfg.num_humans_range
        return int(rng.integers(lo, hi + 1))

    def _estimate_walkable_area(self) -> None:
        """Estimate the walkable area from the map image."""
        if self._backend is None:
            self._walkable_area = self.config.world_width * self.config.world_height
            return
        try:
            img = self._backend.map_image()
            if img is not None:
                arr = np.asarray(img)
                free_pixels = int(np.sum(arr > 128)) if arr.ndim == 2 else arr.size // 3
                meta = self._backend.map_metadata()
                ppm = meta.get("pixels_per_meter", 10.0)
                self._walkable_area = max(1.0, free_pixels / (ppm * ppm))
                return
        except Exception:
            pass
        self._walkable_area = self.config.world_width * self.config.world_height

    def _local_density(self, x: float, y: float, radius: float = 3.0) -> float:
        """Count pedestrians within *radius* metres and return density (ped/m^2)."""
        count = 0
        assert self._backend is not None
        for hid in self._human_ids:
            hx, hy = self._backend.get_position(hid)
            if math.hypot(x - hx, y - hy) <= radius:
                count += 1
        area = math.pi * radius * radius
        return count / area

    def _average_human_speed(self) -> float:
        """Mean speed of all humans."""
        if not self._human_ids or self._backend is None:
            return 0.0
        total = 0.0
        for hid in self._human_ids:
            vx, vy = self._backend.get_velocity(hid)
            total += math.hypot(vx, vy)
        return total / len(self._human_ids)


__all__ = ["CrowdNavEnv", "CrowdNavConfig"]
