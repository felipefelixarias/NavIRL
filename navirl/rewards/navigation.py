"""
NavIRL Navigation Rewards
=========================

Reward functions that drive the agent toward goal-directed navigation while
penalising undesirable locomotion behaviours (collisions, jerk, stopping,
boundary violations).

Classes
-------
GoalReward
    Distance-based goal reward with sparse, dense and shaped modes.
PathFollowingReward
    Reward for staying close to a reference path.
TimePenaltyReward
    Constant per-step penalty encouraging faster navigation.
CollisionPenalty
    Penalty when the agent collides with obstacles or pedestrians.
ProgressReward
    Reward proportional to reduction in goal distance.
VelocityReward
    Penalise stopping or excessively slow motion.
SmoothnessReward
    Penalise jerky, abrupt changes in velocity or heading.
BoundaryPenalty
    Penalty for approaching or crossing environment boundaries.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from navirl.rewards.base import RewardFunction, State, Action

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def _norm(v: np.ndarray) -> float:
    """L2 norm of a vector."""
    return float(np.linalg.norm(np.asarray(v, dtype=np.float64)))


# ---------------------------------------------------------------------------
# GoalReward
# ---------------------------------------------------------------------------


class GoalReward(RewardFunction):
    """Distance-based goal reward supporting three modes.

    Modes
    -----
    ``"sparse"``
        Returns *success_reward* when the agent is within *threshold* of
        the goal, else 0.
    ``"dense"``
        Returns a continuous signal inversely proportional to goal distance.
    ``"shaped"``
        Returns the *reduction* in goal distance between consecutive steps
        (equivalent to potential-based shaping with :math:`\\Phi = -d`).

    Parameters
    ----------
    mode : str
        One of ``"sparse"``, ``"dense"``, ``"shaped"``.
    threshold : float
        Distance (metres) at which the goal is considered reached.
    success_reward : float
        Reward given on goal arrival (always added in every mode when
        threshold is crossed).
    dense_scale : float
        Multiplier for the dense reward signal.
    max_distance : float
        Expected maximum goal distance; used to normalise the dense signal
        into roughly [-1, 0].
    shaped_scale : float
        Multiplier for the shaped (progress) signal.
    """

    VALID_MODES = ("sparse", "dense", "shaped")

    def __init__(
        self,
        mode: str = "shaped",
        *,
        threshold: float = 0.3,
        success_reward: float = 10.0,
        dense_scale: float = 1.0,
        max_distance: float = 20.0,
        shaped_scale: float = 1.0,
        name: str | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}, got {mode!r}")
        super().__init__(name=name or "GoalReward")
        self._mode = mode
        self._threshold = threshold
        self._success_reward = success_reward
        self._dense_scale = dense_scale
        self._max_distance = max(max_distance, 1e-6)
        self._shaped_scale = shaped_scale
        self._prev_distance: float | None = None

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute the goal reward.

        Parameters
        ----------
        state : State
            Must contain ``position`` and ``goal`` (each array-like shape ``(2,)``).
        action : Action
            Unused.
        next_state : State
            Must contain ``position`` and ``goal``.
        info : dict, optional
            Extra environment info.

        Returns
        -------
        float
            Scalar reward value.
        """
        pos = np.asarray(next_state["position"], dtype=np.float64)
        goal = np.asarray(next_state["goal"], dtype=np.float64)
        d = float(np.linalg.norm(pos - goal))

        reward = 0.0

        # Arrival bonus (all modes)
        if d <= self._threshold:
            reward += self._success_reward

        if self._mode == "sparse":
            self._prev_distance = d
            return reward

        if self._mode == "dense":
            # Normalised negative distance in roughly [-1, 0]
            reward += self._dense_scale * (1.0 - d / self._max_distance)
            self._prev_distance = d
            return reward

        # shaped mode
        if self._prev_distance is not None:
            progress = self._prev_distance - d
            reward += self._shaped_scale * progress
        self._prev_distance = d
        return reward

    def reset(self) -> None:
        """Clear the cached previous distance."""
        self._prev_distance = None


# ---------------------------------------------------------------------------
# PathFollowingReward
# ---------------------------------------------------------------------------


class PathFollowingReward(RewardFunction):
    """Reward for staying close to a reference path.

    The reference path is a sequence of 2-D waypoints.  At each step the
    reward is based on the minimum distance from the agent to any path
    segment, encouraging the agent to stay on-track.

    Parameters
    ----------
    path : Sequence of array-like
        Ordered waypoints ``[(x0, y0), (x1, y1), ...]``.
    tolerance : float
        Distance within which the agent receives full reward.
    falloff : float
        Controls how quickly reward drops beyond *tolerance*.
        ``reward = exp(-falloff * max(0, dist - tolerance))``.
    scale : float
        Reward multiplier.
    advance_bonus : float
        Additional reward for advancing along the path (reaching new segments).
    """

    def __init__(
        self,
        path: Sequence[Any] | None = None,
        *,
        tolerance: float = 0.5,
        falloff: float = 2.0,
        scale: float = 1.0,
        advance_bonus: float = 0.1,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "PathFollowingReward")
        self._tolerance = tolerance
        self._falloff = falloff
        self._scale = scale
        self._advance_bonus = advance_bonus
        self._path: np.ndarray | None = None
        self._furthest_segment: int = 0
        if path is not None:
            self.set_path(path)

    def set_path(self, path: Sequence[Any]) -> None:
        """Set or replace the reference path.

        Parameters
        ----------
        path : sequence of array-like
            At least 2 waypoints.
        """
        arr = np.asarray(path, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError("path must be shape (N, 2)")
        if arr.shape[0] < 2:
            raise ValueError("path must have at least 2 waypoints")
        self._path = arr
        self._furthest_segment = 0

    @staticmethod
    def _point_segment_distance(
        p: np.ndarray, a: np.ndarray, b: np.ndarray
    ) -> float:
        """Return the minimum distance from point *p* to segment *a-b*.

        Parameters
        ----------
        p : np.ndarray
            Query point, shape ``(2,)``.
        a, b : np.ndarray
            Segment endpoints, each shape ``(2,)``.

        Returns
        -------
        float
            Minimum Euclidean distance.
        """
        ab = b - a
        ab_sq = float(np.dot(ab, ab))
        if ab_sq < 1e-12:
            return float(np.linalg.norm(p - a))
        t = float(np.clip(np.dot(p - a, ab) / ab_sq, 0.0, 1.0))
        proj = a + t * ab
        return float(np.linalg.norm(p - proj))

    def _closest_segment(self, pos: np.ndarray) -> Tuple[int, float]:
        """Find the closest path segment and distance.

        Parameters
        ----------
        pos : np.ndarray
            Agent position, shape ``(2,)``.

        Returns
        -------
        tuple[int, float]
            ``(segment_index, distance)``
        """
        assert self._path is not None
        best_idx = 0
        best_dist = float("inf")
        for i in range(len(self._path) - 1):
            d = self._point_segment_distance(pos, self._path[i], self._path[i + 1])
            if d < best_dist:
                best_dist = d
                best_idx = i
        return best_idx, best_dist

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute path-following reward.

        Parameters
        ----------
        state, action, next_state, info
            Standard transition tuple.  ``next_state`` must contain
            ``position``.

        Returns
        -------
        float
            Reward based on proximity to path plus any advance bonus.
        """
        if self._path is None:
            return 0.0
        pos = np.asarray(next_state["position"], dtype=np.float64)
        seg_idx, dist = self._closest_segment(pos)

        # Proximity reward
        excess = max(0.0, dist - self._tolerance)
        proximity = math.exp(-self._falloff * excess)
        reward = self._scale * proximity

        # Advance bonus
        if seg_idx > self._furthest_segment:
            reward += self._advance_bonus * (seg_idx - self._furthest_segment)
            self._furthest_segment = seg_idx

        return reward

    def reset(self) -> None:
        """Reset segment progress tracking."""
        self._furthest_segment = 0


# ---------------------------------------------------------------------------
# TimePenaltyReward
# ---------------------------------------------------------------------------


class TimePenaltyReward(RewardFunction):
    """Constant per-step penalty that encourages faster navigation.

    Parameters
    ----------
    penalty : float
        Negative value subtracted each step.  A positive value is negated
        automatically for convenience.
    use_dt : bool
        If ``True``, scale the penalty by ``state["dt"]`` so that it is
        invariant to the simulation time-step.
    max_cumulative : float or None
        If set, stop accumulating penalty after this absolute value is
        reached within an episode.
    """

    def __init__(
        self,
        penalty: float = -0.01,
        *,
        use_dt: bool = False,
        max_cumulative: float | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "TimePenaltyReward")
        self._penalty = -abs(penalty)
        self._use_dt = use_dt
        self._max_cumulative = max_cumulative
        self._cumulative: float = 0.0

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Return a constant negative reward each step.

        Parameters
        ----------
        state, action, next_state, info
            Standard transition.  If *use_dt* is ``True``, ``state`` must
            contain ``dt``.

        Returns
        -------
        float
            The (possibly dt-scaled) time penalty.
        """
        if self._max_cumulative is not None and abs(self._cumulative) >= self._max_cumulative:
            return 0.0

        p = self._penalty
        if self._use_dt:
            dt = float(state.get("dt", 0.1))
            p *= dt

        self._cumulative += p
        return p

    def reset(self) -> None:
        """Reset cumulative penalty counter."""
        self._cumulative = 0.0


# ---------------------------------------------------------------------------
# CollisionPenalty
# ---------------------------------------------------------------------------


class CollisionPenalty(RewardFunction):
    """Penalty when the agent collides with obstacles or pedestrians.

    Detection is based on distance thresholds rather than physics engine
    callbacks, making it simulator-agnostic.

    Parameters
    ----------
    agent_radius : float
        Radius of the controlled agent (metres).
    obstacle_penalty : float
        Penalty per obstacle collision.
    pedestrian_penalty : float
        Penalty per pedestrian collision.
    pedestrian_radius : float
        Default radius for pedestrians (used when not provided per-entity).
    obstacle_margin : float
        Additional clearance beyond agent radius for obstacles.
    pedestrian_margin : float
        Additional clearance beyond combined radii for pedestrians.
    cumulative : bool
        If ``True``, sum penalties for *all* collisions; if ``False``,
        return the single worst penalty.
    """

    def __init__(
        self,
        agent_radius: float = 0.2,
        *,
        obstacle_penalty: float = -5.0,
        pedestrian_penalty: float = -10.0,
        pedestrian_radius: float = 0.18,
        obstacle_margin: float = 0.0,
        pedestrian_margin: float = 0.0,
        cumulative: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "CollisionPenalty")
        self._agent_radius = agent_radius
        self._obstacle_penalty = obstacle_penalty
        self._pedestrian_penalty = pedestrian_penalty
        self._pedestrian_radius = pedestrian_radius
        self._obstacle_margin = obstacle_margin
        self._pedestrian_margin = pedestrian_margin
        self._cumulative = cumulative
        self._last_n_collisions: int = 0

    def _check_pedestrian_collisions(self, pos: np.ndarray, state: State) -> float:
        """Check for collisions with all pedestrians in the state.

        Parameters
        ----------
        pos : np.ndarray
            Agent position shape ``(2,)``.
        state : State
            Must contain ``pedestrians`` list.

        Returns
        -------
        float
            Total or worst pedestrian collision penalty.
        """
        peds = state.get("pedestrians", [])
        if not peds:
            return 0.0

        penalty = 0.0
        for ped in peds:
            ped_pos = np.asarray(ped["position"], dtype=np.float64)
            ped_r = float(ped.get("radius", self._pedestrian_radius))
            dist = float(np.linalg.norm(pos - ped_pos))
            threshold = self._agent_radius + ped_r + self._pedestrian_margin
            if dist < threshold:
                self._last_n_collisions += 1
                if self._cumulative:
                    penalty += self._pedestrian_penalty
                else:
                    penalty = min(penalty, self._pedestrian_penalty)
        return penalty

    def _check_obstacle_collisions(self, pos: np.ndarray, state: State) -> float:
        """Check for collisions with obstacles.

        Obstacles can be provided as:
        * list of dicts with ``position`` and ``radius``.
        * ndarray of shape ``(N, 2)`` or ``(N, 4)`` (line segments).

        Parameters
        ----------
        pos : np.ndarray
            Agent position shape ``(2,)``.
        state : State
            May contain ``obstacles`` key.

        Returns
        -------
        float
            Total or worst obstacle collision penalty.
        """
        obs = state.get("obstacles", [])
        if obs is None or (isinstance(obs, (list, tuple)) and len(obs) == 0):
            return 0.0

        penalty = 0.0
        threshold = self._agent_radius + self._obstacle_margin

        if isinstance(obs, np.ndarray):
            if obs.ndim == 2 and obs.shape[1] == 2:
                # Point obstacles
                dists = np.linalg.norm(obs - pos[np.newaxis, :], axis=1)
                n_hits = int(np.sum(dists < threshold))
                self._last_n_collisions += n_hits
                if self._cumulative:
                    penalty += n_hits * self._obstacle_penalty
                elif n_hits > 0:
                    penalty = min(penalty, self._obstacle_penalty)
            elif obs.ndim == 2 and obs.shape[1] == 4:
                # Line segments (x1, y1, x2, y2)
                for seg in obs:
                    a = seg[:2]
                    b = seg[2:]
                    d = PathFollowingReward._point_segment_distance(pos, a, b)
                    if d < threshold:
                        self._last_n_collisions += 1
                        if self._cumulative:
                            penalty += self._obstacle_penalty
                        else:
                            penalty = min(penalty, self._obstacle_penalty)
        else:
            for ob in obs:
                if isinstance(ob, dict):
                    ob_pos = np.asarray(ob["position"], dtype=np.float64)
                    ob_r = float(ob.get("radius", 0.0))
                    dist = float(np.linalg.norm(pos - ob_pos))
                    if dist < threshold + ob_r:
                        self._last_n_collisions += 1
                        if self._cumulative:
                            penalty += self._obstacle_penalty
                        else:
                            penalty = min(penalty, self._obstacle_penalty)
        return penalty

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute collision penalty.

        Parameters
        ----------
        state, action, next_state, info
            Standard transition.

        Returns
        -------
        float
            Non-positive penalty (0.0 when no collision).
        """
        self._last_n_collisions = 0
        pos = np.asarray(next_state["position"], dtype=np.float64)
        total = 0.0
        total += self._check_pedestrian_collisions(pos, next_state)
        total += self._check_obstacle_collisions(pos, next_state)
        return total

    def get_info(self) -> Dict[str, Any]:
        """Return collision count from the last call."""
        return {"n_collisions": self._last_n_collisions}


# ---------------------------------------------------------------------------
# ProgressReward
# ---------------------------------------------------------------------------


class ProgressReward(RewardFunction):
    """Reward proportional to reduction in goal distance.

    Unlike ``GoalReward(mode="shaped")``, this class offers additional
    options such as asymmetric scaling (punish regression more than
    rewarding progress) and distance-dependent gain.

    Parameters
    ----------
    scale : float
        Base multiplier applied to the raw progress value.
    regression_scale : float
        Multiplier applied when distance *increases* (regression).
        Set higher than *scale* to penalise going backward more heavily.
    distance_gain : bool
        If ``True``, scale the reward by ``1 / (1 + d_goal)`` so that
        progress near the goal is worth more.
    max_reward : float
        Clamp the per-step reward to ``[-max_reward, max_reward]``.
    """

    def __init__(
        self,
        scale: float = 1.0,
        *,
        regression_scale: float | None = None,
        distance_gain: bool = False,
        max_reward: float = 5.0,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "ProgressReward")
        self._scale = scale
        self._regression_scale = regression_scale if regression_scale is not None else scale
        self._distance_gain = distance_gain
        self._max_reward = max_reward
        self._prev_dist: float | None = None

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute progress reward.

        Parameters
        ----------
        state : State
            Must contain ``position`` and ``goal``.
        action : Action
            Unused.
        next_state : State
            Must contain ``position`` and ``goal``.
        info : dict, optional
            Not used.

        Returns
        -------
        float
            Positive for progress, negative for regression.
        """
        pos = np.asarray(next_state["position"], dtype=np.float64)
        goal = np.asarray(next_state["goal"], dtype=np.float64)
        d = float(np.linalg.norm(pos - goal))

        if self._prev_dist is None:
            self._prev_dist = d
            return 0.0

        delta = self._prev_dist - d  # positive = closer
        self._prev_dist = d

        if delta >= 0.0:
            reward = self._scale * delta
        else:
            reward = self._regression_scale * delta

        if self._distance_gain:
            reward *= 1.0 / (1.0 + d)

        return float(np.clip(reward, -self._max_reward, self._max_reward))

    def reset(self) -> None:
        """Reset cached previous distance."""
        self._prev_dist = None


# ---------------------------------------------------------------------------
# VelocityReward
# ---------------------------------------------------------------------------


class VelocityReward(RewardFunction):
    """Penalise stopping or reward maintaining a target speed.

    Parameters
    ----------
    target_speed : float
        Desired agent speed (m/s).  The reward peaks at this speed.
    tolerance : float
        Speed deviation within which reward is maximal.
    penalty_weight : float
        Weight applied to speed deviation beyond tolerance.
    stop_threshold : float
        Speed below which the agent is considered stopped.
    stop_penalty : float
        Extra penalty for being stopped.
    mode : str
        ``"penalty_only"`` -- only penalise deviation.
        ``"reward_match"`` -- reward speed near target, penalise deviation.
    """

    VALID_MODES = ("penalty_only", "reward_match")

    def __init__(
        self,
        target_speed: float = 0.8,
        *,
        tolerance: float = 0.1,
        penalty_weight: float = 0.5,
        stop_threshold: float = 0.05,
        stop_penalty: float = -0.5,
        mode: str = "penalty_only",
        name: str | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        super().__init__(name=name or "VelocityReward")
        self._target_speed = target_speed
        self._tolerance = tolerance
        self._penalty_weight = penalty_weight
        self._stop_threshold = stop_threshold
        self._stop_penalty = stop_penalty
        self._mode = mode

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute velocity reward/penalty.

        Parameters
        ----------
        next_state : State
            Must contain ``velocity`` of shape ``(2,)`` or a scalar ``speed``.

        Returns
        -------
        float
            Reward or penalty based on speed.
        """
        if "speed" in next_state:
            speed = float(next_state["speed"])
        else:
            vel = np.asarray(next_state["velocity"], dtype=np.float64)
            speed = float(np.linalg.norm(vel))

        # Stopped penalty
        if speed < self._stop_threshold:
            return self._stop_penalty

        deviation = abs(speed - self._target_speed)
        excess = max(0.0, deviation - self._tolerance)

        if self._mode == "penalty_only":
            return -self._penalty_weight * excess
        else:
            # reward_match: Gaussian-like reward centred on target
            return math.exp(-self._penalty_weight * excess * excess)


# ---------------------------------------------------------------------------
# SmoothnessReward
# ---------------------------------------------------------------------------


class SmoothnessReward(RewardFunction):
    """Penalise jerky, abrupt changes in velocity or heading.

    Computes a penalty based on the magnitude of acceleration (velocity
    change) and heading change between consecutive steps.

    Parameters
    ----------
    accel_weight : float
        Penalty weight for acceleration magnitude.
    heading_weight : float
        Penalty weight for heading change.
    max_accel : float
        Acceleration values above this threshold are penalised quadratically.
    max_heading_change : float
        Heading changes (radians) above this are penalised quadratically.
    linear : bool
        If ``True``, use linear (L1) penalty instead of quadratic (L2).
    """

    def __init__(
        self,
        *,
        accel_weight: float = 0.5,
        heading_weight: float = 0.3,
        max_accel: float = 2.0,
        max_heading_change: float = 0.5,
        linear: bool = False,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name or "SmoothnessReward")
        self._accel_weight = accel_weight
        self._heading_weight = heading_weight
        self._max_accel = max_accel
        self._max_heading_change = max_heading_change
        self._linear = linear
        self._prev_velocity: np.ndarray | None = None
        self._prev_heading: float | None = None

    @staticmethod
    def _angle_diff(a: float, b: float) -> float:
        """Signed angular difference wrapped to [-pi, pi].

        Parameters
        ----------
        a, b : float
            Angles in radians.

        Returns
        -------
        float
            Wrapped difference ``a - b``.
        """
        d = a - b
        return float((d + math.pi) % (2 * math.pi) - math.pi)

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute smoothness penalty.

        Parameters
        ----------
        state, next_state : State
            Must contain ``velocity`` (shape ``(2,)``).
            Optionally ``heading`` (float, radians).

        Returns
        -------
        float
            Non-positive penalty.
        """
        vel = np.asarray(next_state["velocity"], dtype=np.float64)
        dt = float(next_state.get("dt", state.get("dt", 0.1)))
        penalty = 0.0

        # Acceleration penalty
        if self._prev_velocity is not None:
            accel_vec = (vel - self._prev_velocity) / max(dt, 1e-6)
            accel_mag = float(np.linalg.norm(accel_vec))
            excess = max(0.0, accel_mag - self._max_accel)
            if self._linear:
                penalty -= self._accel_weight * excess
            else:
                penalty -= self._accel_weight * excess * excess

        # Heading penalty
        heading = next_state.get("heading")
        if heading is not None and self._prev_heading is not None:
            h_change = abs(self._angle_diff(float(heading), self._prev_heading))
            excess_h = max(0.0, h_change - self._max_heading_change)
            if self._linear:
                penalty -= self._heading_weight * excess_h
            else:
                penalty -= self._heading_weight * excess_h * excess_h

        self._prev_velocity = vel.copy()
        if "heading" in next_state:
            self._prev_heading = float(next_state["heading"])

        return penalty

    def reset(self) -> None:
        """Clear cached previous velocity and heading."""
        self._prev_velocity = None
        self._prev_heading = None


# ---------------------------------------------------------------------------
# BoundaryPenalty
# ---------------------------------------------------------------------------


class BoundaryPenalty(RewardFunction):
    """Penalty for approaching or crossing environment boundaries.

    Supports rectangular bounds defined by ``(x_min, y_min, x_max, y_max)``
    and/or circular bounds defined by ``(cx, cy, radius)``.

    Parameters
    ----------
    x_min, x_max, y_min, y_max : float or None
        Rectangular boundary limits.  ``None`` means unbounded in that
        direction.
    center : tuple[float, float] or None
        Centre of a circular boundary.
    radius : float or None
        Radius of the circular boundary.
    margin : float
        Distance from the boundary at which penalty starts.
    penalty_scale : float
        Multiplier for the penalty signal.
    hard_penalty : float
        Fixed penalty applied when the agent is *outside* the boundary.
    mode : str
        ``"linear"`` or ``"quadratic"`` falloff within the margin.
    """

    VALID_MODES = ("linear", "quadratic")

    def __init__(
        self,
        *,
        x_min: float | None = None,
        x_max: float | None = None,
        y_min: float | None = None,
        y_max: float | None = None,
        center: Tuple[float, float] | None = None,
        radius: float | None = None,
        margin: float = 0.5,
        penalty_scale: float = 1.0,
        hard_penalty: float = -5.0,
        mode: str = "linear",
        name: str | None = None,
    ) -> None:
        if mode not in self.VALID_MODES:
            raise ValueError(f"mode must be one of {self.VALID_MODES}")
        super().__init__(name=name or "BoundaryPenalty")
        self._x_min = x_min
        self._x_max = x_max
        self._y_min = y_min
        self._y_max = y_max
        self._center = np.asarray(center, dtype=np.float64) if center is not None else None
        self._radius = radius
        self._margin = margin
        self._penalty_scale = penalty_scale
        self._hard_penalty = hard_penalty
        self._mode = mode

    def _margin_penalty(self, distance_inside: float) -> float:
        """Compute penalty based on how close agent is to boundary.

        Parameters
        ----------
        distance_inside : float
            Distance from the agent to the boundary, positive when inside.

        Returns
        -------
        float
            Non-positive penalty.
        """
        if distance_inside < 0:
            return self._hard_penalty
        if distance_inside >= self._margin:
            return 0.0
        frac = 1.0 - distance_inside / self._margin
        if self._mode == "quadratic":
            return -self._penalty_scale * frac * frac
        return -self._penalty_scale * frac

    def compute(
        self,
        state: State,
        action: Action,
        next_state: State,
        *,
        info: Dict[str, Any] | None = None,
    ) -> float:
        """Compute boundary penalty.

        Parameters
        ----------
        next_state : State
            Must contain ``position`` of shape ``(2,)``.

        Returns
        -------
        float
            Non-positive penalty; 0.0 if well inside all boundaries.
        """
        pos = np.asarray(next_state["position"], dtype=np.float64)
        x, y = float(pos[0]), float(pos[1])
        worst = 0.0

        # Rectangular bounds
        if self._x_min is not None:
            worst = min(worst, self._margin_penalty(x - self._x_min))
        if self._x_max is not None:
            worst = min(worst, self._margin_penalty(self._x_max - x))
        if self._y_min is not None:
            worst = min(worst, self._margin_penalty(y - self._y_min))
        if self._y_max is not None:
            worst = min(worst, self._margin_penalty(self._y_max - y))

        # Circular bound
        if self._center is not None and self._radius is not None:
            dist_from_center = float(np.linalg.norm(pos - self._center))
            dist_inside = self._radius - dist_from_center
            worst = min(worst, self._margin_penalty(dist_inside))

        return worst
