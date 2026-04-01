from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from navirl.prediction.base import PredictionResult, TrajectoryPredictor


class PedestrianIntent(Enum):
    """Discrete pedestrian intent categories."""

    CROSSING = "crossing"
    WAITING = "waiting"
    TURNING_LEFT = "turning_left"
    TURNING_RIGHT = "turning_right"
    STOPPING = "stopping"
    WALKING_STRAIGHT = "walking_straight"
    UNKNOWN = "unknown"


@dataclass
class _CandidateGoal:
    position: np.ndarray  # (2,)
    probability: float = 0.0


class GoalConditionedPredictor(TrajectoryPredictor):
    """Predict trajectories conditioned on estimated goals.

    Pipeline:
    1. Estimate a set of candidate goal positions from the trajectory history
       and optional scene context (e.g. exits, points of interest).
    2. For each candidate goal, plan a smooth path from the current position.
    3. Weight each path by the estimated goal probability.
    """

    def __init__(
        self,
        horizon: int = 12,
        dt: float = 0.4,
        num_goals: int = 5,
        num_samples_per_goal: int = 4,
        goal_noise_std: float = 0.3,
        velocity_smoothing: float = 0.8,
    ) -> None:
        self.horizon = horizon
        self.dt = dt
        self.num_goals = num_goals
        self.num_samples_per_goal = num_samples_per_goal
        self.goal_noise_std = goal_noise_std
        self.velocity_smoothing = velocity_smoothing

    # ------------------------------------------------------------------
    # Goal estimation
    # ------------------------------------------------------------------

    def _estimate_goals(
        self,
        observed: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> list[_CandidateGoal]:
        """Estimate candidate goal positions and their probabilities.

        Uses a combination of velocity extrapolation and scene-provided
        candidate goals.
        """
        context = context or {}
        goals: list[_CandidateGoal] = []

        # 1. Velocity-based extrapolation at several horizons.
        if observed.shape[0] >= 2:
            velocity = observed[-1] - observed[-2]
            speed = float(np.linalg.norm(velocity))
            if speed > 1e-4:
                direction = velocity / speed
                for scale in [1.0, 1.5, 2.0]:
                    goal_pos = observed[-1] + direction * speed * self.horizon * self.dt * scale
                    goals.append(_CandidateGoal(position=goal_pos))

        # 2. Scene-provided candidate goals (points of interest, exits, etc.).
        if "candidate_goals" in context:
            for gp in context["candidate_goals"]:
                goals.append(_CandidateGoal(position=np.asarray(gp, dtype=np.float64)))

        # Ensure we have at least one goal.
        if not goals:
            goals.append(_CandidateGoal(position=observed[-1].copy()))

        # Trim or pad to num_goals.
        goals = goals[: self.num_goals]

        # Compute probabilities proportional to inverse distance from
        # extrapolated position (closer to the heading direction is more
        # probable).
        if observed.shape[0] >= 2:
            extrap = observed[-1] + (observed[-1] - observed[-2]) * self.horizon
            dists = np.array([float(np.linalg.norm(g.position - extrap)) for g in goals])
            dists = dists + 1e-6
            inv_dists = 1.0 / dists
            probs = inv_dists / inv_dists.sum()
        else:
            probs = np.ones(len(goals)) / len(goals)

        for i, g in enumerate(goals):
            g.probability = float(probs[i])

        return goals

    # ------------------------------------------------------------------
    # Path planning to a goal
    # ------------------------------------------------------------------

    def _plan_path_to_goal(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        observed: np.ndarray,
    ) -> np.ndarray:
        """Plan a smooth trajectory from *start* toward *goal*.

        Uses cubic interpolation between the current state (position +
        velocity) and the goal.  Returns shape ``(horizon, 2)``.
        """
        if observed.shape[0] >= 2:
            vel = (observed[-1] - observed[-2]) / self.dt
        else:
            vel = np.zeros(2)

        # Parameterize time 0..1 over the prediction horizon.
        ts = np.linspace(0, 1, self.horizon)
        # Hermite interpolation: p(t) = (2t^3 - 3t^2 + 1)*p0 + (t^3 - 2t^2 + t)*m0
        #                              + (-2t^3 + 3t^2)*p1 + (t^3 - t^2)*m1
        p0 = start
        p1 = goal
        m0 = vel * self.horizon * self.dt * self.velocity_smoothing
        m1 = np.zeros(2)  # Assume zero velocity at goal.

        traj = np.zeros((self.horizon, 2))
        for i, t in enumerate(ts):
            h00 = 2 * t**3 - 3 * t**2 + 1
            h10 = t**3 - 2 * t**2 + t
            h01 = -2 * t**3 + 3 * t**2
            h11 = t**3 - t**2
            traj[i] = h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1

        return traj

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        observed_trajectory: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> PredictionResult:
        goals = self._estimate_goals(observed_trajectory, context)

        all_trajs: list[np.ndarray] = []
        all_probs: list[float] = []

        for goal in goals:
            for _ in range(self.num_samples_per_goal):
                noisy_goal = goal.position + np.random.randn(2) * self.goal_noise_std
                traj = self._plan_path_to_goal(
                    observed_trajectory[-1], noisy_goal, observed_trajectory
                )
                all_trajs.append(traj)
                all_probs.append(goal.probability / self.num_samples_per_goal)

        trajectories = np.stack(all_trajs, axis=0)
        probabilities = np.array(all_probs)
        probabilities /= probabilities.sum()
        timestamps = np.arange(1, self.horizon + 1) * self.dt

        return PredictionResult(
            trajectories=trajectories,
            probabilities=probabilities,
            timestamps=timestamps,
        )


class IntentPredictor:
    """Classifies pedestrian intent from trajectory history.

    Uses simple heuristics based on velocity, curvature, and
    acceleration to assign one of the :class:`PedestrianIntent`
    categories.
    """

    def __init__(
        self,
        stop_speed_threshold: float = 0.1,
        turn_curvature_threshold: float = 0.3,
        crossing_lateral_threshold: float = 0.5,
        dt: float = 0.4,
    ) -> None:
        self.stop_speed_threshold = stop_speed_threshold
        self.turn_curvature_threshold = turn_curvature_threshold
        self.crossing_lateral_threshold = crossing_lateral_threshold
        self.dt = dt

    def classify(
        self,
        observed_trajectory: np.ndarray,
        context: dict[str, Any] | None = None,
    ) -> tuple[PedestrianIntent, dict[str, float]]:
        """Classify the pedestrian's current intent.

        Args:
            observed_trajectory: Observed positions ``(T_obs, 2)``.
            context: Optional scene information (e.g. road geometry).

        Returns:
            Tuple of the most likely intent and a dictionary mapping each
            intent to its estimated probability.
        """
        if observed_trajectory.shape[0] < 3:
            return PedestrianIntent.UNKNOWN, {PedestrianIntent.UNKNOWN.value: 1.0}

        velocities = np.diff(observed_trajectory, axis=0) / self.dt
        speeds = np.linalg.norm(velocities, axis=1)
        recent_speed = float(speeds[-1])

        # Curvature approximation via cross product of consecutive velocity vectors.
        curvatures: list[float] = []
        for i in range(1, velocities.shape[0]):
            v0 = velocities[i - 1]
            v1 = velocities[i]
            cross = float(v0[0] * v1[1] - v0[1] * v1[0])
            norm_prod = float(np.linalg.norm(v0) * np.linalg.norm(v1)) + 1e-8
            curvatures.append(cross / norm_prod)
        mean_curvature = float(np.mean(curvatures)) if curvatures else 0.0

        # Acceleration.
        accels = np.diff(speeds) / self.dt
        recent_accel = float(accels[-1]) if accels.size > 0 else 0.0

        # Build probability distribution over intents.
        scores: dict[str, float] = {}

        # Stopping / waiting.
        if recent_speed < self.stop_speed_threshold:
            if recent_accel < -0.1:
                scores[PedestrianIntent.STOPPING.value] = 3.0
            else:
                scores[PedestrianIntent.WAITING.value] = 3.0
        else:
            scores[PedestrianIntent.STOPPING.value] = 0.1
            scores[PedestrianIntent.WAITING.value] = 0.1

        # Turning.
        if abs(mean_curvature) > self.turn_curvature_threshold:
            if mean_curvature > 0:
                scores[PedestrianIntent.TURNING_LEFT.value] = 3.0
            else:
                scores[PedestrianIntent.TURNING_RIGHT.value] = 3.0
        else:
            scores[PedestrianIntent.TURNING_LEFT.value] = 0.1
            scores[PedestrianIntent.TURNING_RIGHT.value] = 0.1

        # Crossing (lateral motion relative to dominant direction).
        context = context or {}
        road_direction = context.get("road_direction", np.array([1.0, 0.0]))
        road_direction = np.asarray(road_direction, dtype=np.float64)
        rd_norm = np.linalg.norm(road_direction)
        if rd_norm > 1e-8:
            road_direction = road_direction / rd_norm
        lateral = np.array([-road_direction[1], road_direction[0]])
        lateral_speed = abs(float(np.dot(velocities[-1], lateral)))
        if lateral_speed > self.crossing_lateral_threshold:
            scores[PedestrianIntent.CROSSING.value] = 3.0
        else:
            scores[PedestrianIntent.CROSSING.value] = 0.2

        # Straight walking (default).
        if (
            recent_speed >= self.stop_speed_threshold
            and abs(mean_curvature) <= self.turn_curvature_threshold
        ):
            scores[PedestrianIntent.WALKING_STRAIGHT.value] = 2.0
        else:
            scores[PedestrianIntent.WALKING_STRAIGHT.value] = 0.2

        # Normalise to probabilities.
        total = sum(scores.values())
        probs = {k: v / total for k, v in scores.items()}

        best_intent = PedestrianIntent(max(probs, key=probs.get))  # type: ignore[arg-type]
        return best_intent, probs
