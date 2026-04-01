"""Comprehensive trajectory evaluation metrics for pedestrian navigation.

Provides a rich set of quantitative metrics used in the pedestrian navigation
literature including displacement errors, collision statistics, path quality,
social compliance, comfort, and safety indicators.  All heavy computation is
done with NumPy.  Each metric function operates on :class:`Trajectory` objects
from ``navirl.data.trajectory`` and returns scalar or structured results.
Statistical helpers (bootstrap CIs, paired tests) are included so that every
metric can be reported with proper uncertainty quantification.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from navirl.data.trajectory import Trajectory

# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class MetricResult:
    """Container for a single metric evaluation.

    Attributes:
        name: Human-readable metric name.
        value: Point estimate of the metric.
        unit: Physical unit string (e.g. ``"m"``, ``"m/s"``).
        ci_lower: Lower bound of the confidence interval (if computed).
        ci_upper: Upper bound of the confidence interval (if computed).
        confidence_level: Confidence level used for the CI (e.g. 0.95).
        sample_size: Number of samples the metric was computed over.
        raw_values: Per-sample raw values (optional).
    """

    name: str = ""
    value: float = 0.0
    unit: str = ""
    ci_lower: float = float("nan")
    ci_upper: float = float("nan")
    confidence_level: float = 0.95
    sample_size: int = 0
    raw_values: np.ndarray | None = None

    def __repr__(self) -> str:
        ci = ""
        if not np.isnan(self.ci_lower):
            ci = f" CI=[{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        return f"MetricResult({self.name}={self.value:.4f} {self.unit}{ci}, n={self.sample_size})"


@dataclass
class MetricSummary:
    """Aggregated summary over multiple :class:`MetricResult` instances.

    Attributes:
        results: Mapping from metric name to its :class:`MetricResult`.
        model_name: Optional model identifier.
        dataset_name: Optional dataset identifier.
    """

    results: dict[str, MetricResult] = field(default_factory=dict)
    model_name: str = ""
    dataset_name: str = ""

    def add(self, result: MetricResult) -> None:
        """Register a :class:`MetricResult`."""
        self.results[result.name] = result

    def to_dict(self) -> dict[str, float]:
        """Return a flat ``{metric_name: value}`` dictionary."""
        return {k: v.value for k, v in self.results.items()}

    def to_table_row(self, precision: int = 4) -> str:
        """Format as a single row suitable for a Markdown or LaTeX table."""
        parts = [self.model_name or "model"]
        for r in self.results.values():
            parts.append(f"{r.value:.{precision}f}")
        return " | ".join(parts)


# ---------------------------------------------------------------------------
# Bootstrap helper
# ---------------------------------------------------------------------------

def _bootstrap_ci(
    values: np.ndarray,
    statistic: str = "mean",
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a statistic.

    Parameters:
        values: 1-D array of sample values.
        statistic: ``"mean"`` or ``"median"``.
        confidence: Confidence level in (0, 1).
        n_bootstrap: Number of bootstrap resamples.
        rng: Optional NumPy random generator for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    if rng is None:
        rng = np.random.default_rng()
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))

    stat_fn = np.mean if statistic == "mean" else np.median
    point = float(stat_fn(values))

    boot_stats = np.empty(n_bootstrap, dtype=np.float64)
    for i in range(n_bootstrap):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = stat_fn(sample)

    alpha = 1.0 - confidence
    lo = float(np.percentile(boot_stats, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_stats, 100.0 * (1.0 - alpha / 2.0)))
    return point, lo, hi


# ---------------------------------------------------------------------------
# Paired permutation test
# ---------------------------------------------------------------------------

def paired_permutation_test(
    values_a: np.ndarray,
    values_b: np.ndarray,
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
) -> float:
    """Two-sided paired permutation test for difference in means.

    Parameters:
        values_a: Per-sample metric values for system A.
        values_b: Per-sample metric values for system B.
        n_permutations: Number of random permutations.
        rng: Optional random generator.

    Returns:
        Approximate p-value.
    """
    if rng is None:
        rng = np.random.default_rng()
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    assert len(a) == len(b), "Paired test requires equal-length arrays."
    diffs = a - b
    observed = np.abs(np.mean(diffs))
    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        if np.abs(np.mean(diffs * signs)) >= observed:
            count += 1
    return count / n_permutations


# ---------------------------------------------------------------------------
# Displacement error metrics
# ---------------------------------------------------------------------------

def average_displacement_error(
    predicted: Trajectory,
    ground_truth: Trajectory,
) -> float:
    """Average Displacement Error (ADE) between predicted and ground-truth trajectories.

    ADE is the mean L2 distance over all time steps.  If the trajectories
    differ in length the shorter one is used.

    Parameters:
        predicted: Predicted trajectory.
        ground_truth: Ground-truth trajectory.

    Returns:
        ADE in metres.
    """
    n = min(len(predicted), len(ground_truth))
    if n == 0:
        return float("nan")
    dists = np.linalg.norm(
        predicted.positions[:n] - ground_truth.positions[:n], axis=1
    )
    return float(np.mean(dists))


def final_displacement_error(
    predicted: Trajectory,
    ground_truth: Trajectory,
) -> float:
    """Final Displacement Error (FDE) -- L2 distance at the last time step.

    Parameters:
        predicted: Predicted trajectory.
        ground_truth: Ground-truth trajectory.

    Returns:
        FDE in metres.
    """
    if len(predicted) == 0 or len(ground_truth) == 0:
        return float("nan")
    n = min(len(predicted), len(ground_truth))
    return float(
        np.linalg.norm(predicted.positions[n - 1] - ground_truth.positions[n - 1])
    )


def ade_fde_batch(
    predicted_list: Sequence[Trajectory],
    ground_truth_list: Sequence[Trajectory],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
) -> tuple[MetricResult, MetricResult]:
    """Compute ADE and FDE over a batch with bootstrap confidence intervals.

    Parameters:
        predicted_list: Sequence of predicted trajectories.
        ground_truth_list: Corresponding ground-truth trajectories.
        confidence: Confidence level for the CI.
        n_bootstrap: Number of bootstrap resamples.

    Returns:
        Tuple of (ADE MetricResult, FDE MetricResult).
    """
    assert len(predicted_list) == len(ground_truth_list)
    ade_vals = np.array(
        [average_displacement_error(p, g) for p, g in zip(predicted_list, ground_truth_list, strict=False)]
    )
    fde_vals = np.array(
        [final_displacement_error(p, g) for p, g in zip(predicted_list, ground_truth_list, strict=False)]
    )

    ade_pt, ade_lo, ade_hi = _bootstrap_ci(ade_vals, confidence=confidence, n_bootstrap=n_bootstrap)
    fde_pt, fde_lo, fde_hi = _bootstrap_ci(fde_vals, confidence=confidence, n_bootstrap=n_bootstrap)

    return (
        MetricResult("ADE", ade_pt, "m", ade_lo, ade_hi, confidence, len(ade_vals), ade_vals),
        MetricResult("FDE", fde_pt, "m", fde_lo, fde_hi, confidence, len(fde_vals), fde_vals),
    )


def displacement_error_at_horizon(
    predicted: Trajectory,
    ground_truth: Trajectory,
    horizons: Sequence[float] = (1.0, 2.0, 3.0, 4.0),
) -> dict[float, float]:
    """Displacement error evaluated at specific prediction horizons (seconds).

    Parameters:
        predicted: Predicted trajectory.
        ground_truth: Ground-truth trajectory.
        horizons: Sequence of time horizons in seconds.

    Returns:
        Mapping from horizon (s) to displacement error (m).
    """
    results: dict[float, float] = {}
    t0 = ground_truth.timestamps[0]
    for h in horizons:
        t_target = t0 + h
        idx_gt = np.searchsorted(ground_truth.timestamps, t_target)
        idx_pr = np.searchsorted(predicted.timestamps, t_target)
        if idx_gt >= len(ground_truth) or idx_pr >= len(predicted):
            results[h] = float("nan")
        else:
            results[h] = float(
                np.linalg.norm(predicted.positions[idx_pr] - ground_truth.positions[idx_gt])
            )
    return results


# ---------------------------------------------------------------------------
# Collision and safety metrics
# ---------------------------------------------------------------------------

def collision_rate(
    agent_trajectory: Trajectory,
    other_trajectories: Sequence[Trajectory],
    collision_radius: float = 0.5,
) -> float:
    """Fraction of time steps where the agent is within *collision_radius* of any other agent.

    Parameters:
        agent_trajectory: The ego-agent trajectory.
        other_trajectories: Trajectories of surrounding agents.
        collision_radius: Distance threshold in metres.

    Returns:
        Collision rate in [0, 1].
    """
    if len(agent_trajectory) == 0 or len(other_trajectories) == 0:
        return 0.0
    collisions = np.zeros(len(agent_trajectory), dtype=bool)
    for other in other_trajectories:
        n = min(len(agent_trajectory), len(other))
        dists = np.linalg.norm(
            agent_trajectory.positions[:n] - other.positions[:n], axis=1
        )
        collisions[:n] |= dists < collision_radius
    return float(np.mean(collisions))


def collision_count(
    agent_trajectory: Trajectory,
    other_trajectories: Sequence[Trajectory],
    collision_radius: float = 0.5,
    min_gap_steps: int = 5,
) -> int:
    """Count distinct collision events (consecutive collisions = one event).

    Parameters:
        agent_trajectory: The ego-agent trajectory.
        other_trajectories: Trajectories of surrounding agents.
        collision_radius: Distance threshold in metres.
        min_gap_steps: Minimum gap between events to count as separate.

    Returns:
        Number of distinct collision events.
    """
    if len(agent_trajectory) == 0:
        return 0
    collisions = np.zeros(len(agent_trajectory), dtype=bool)
    for other in other_trajectories:
        n = min(len(agent_trajectory), len(other))
        dists = np.linalg.norm(
            agent_trajectory.positions[:n] - other.positions[:n], axis=1
        )
        collisions[:n] |= dists < collision_radius

    # Count distinct events
    count = 0
    in_collision = False
    gap = 0
    for c in collisions:
        if c:
            if not in_collision:
                count += 1
                in_collision = True
            gap = 0
        else:
            gap += 1
            if gap >= min_gap_steps:
                in_collision = False
    return count


def time_to_collision(
    agent_trajectory: Trajectory,
    other_trajectories: Sequence[Trajectory],
    collision_radius: float = 0.5,
) -> np.ndarray:
    """Time-to-collision (TTC) at each time step, assuming constant velocities.

    For each time step, TTC is the minimum time until the agent would collide
    with any neighbour if both maintained their current velocities.  If no
    collision is predicted, the value is ``inf``.

    Parameters:
        agent_trajectory: Ego-agent trajectory.
        other_trajectories: Neighbour trajectories.
        collision_radius: Combined collision radius.

    Returns:
        1-D array of shape ``(T,)`` with TTC values in seconds.
    """
    T = len(agent_trajectory)
    ttc = np.full(T, np.inf)
    if T < 2:
        return ttc

    # Compute agent velocities via finite differences
    dt_arr = np.diff(agent_trajectory.timestamps)
    dt_arr[dt_arr == 0] = 1e-6
    agent_vel = np.diff(agent_trajectory.positions, axis=0) / dt_arr[:, None]
    # Pad last velocity
    agent_vel = np.vstack([agent_vel, agent_vel[-1:]])

    for other in other_trajectories:
        n = min(T, len(other))
        if n < 2:
            continue
        dt_o = np.diff(other.timestamps[:n])
        dt_o[dt_o == 0] = 1e-6
        other_vel = np.diff(other.positions[:n], axis=0) / dt_o[:, None]
        other_vel = np.vstack([other_vel, other_vel[-1:]])

        rel_pos = agent_trajectory.positions[:n] - other.positions[:n]
        rel_vel = agent_vel[:n] - other_vel

        # Solve ||rel_pos + t * rel_vel||^2 = collision_radius^2
        a = np.sum(rel_vel ** 2, axis=1)
        b = 2.0 * np.sum(rel_pos * rel_vel, axis=1)
        c = np.sum(rel_pos ** 2, axis=1) - collision_radius ** 2

        discriminant = b ** 2 - 4.0 * a * c
        valid = (discriminant >= 0) & (a > 1e-12)

        t_enter = np.full(n, np.inf)
        sqrt_disc = np.sqrt(np.maximum(discriminant[valid], 0.0))
        t1 = (-b[valid] - sqrt_disc) / (2.0 * a[valid])
        t2 = (-b[valid] + sqrt_disc) / (2.0 * a[valid])

        # Take earliest positive root
        t_min = np.where((t1 > 0) & (t1 < t2), t1, t2)
        t_min = np.where(t_min > 0, t_min, np.inf)
        t_enter[valid] = t_min

        ttc[:n] = np.minimum(ttc[:n], t_enter)

    return ttc


def minimum_separation_distance(
    agent_trajectory: Trajectory,
    other_trajectories: Sequence[Trajectory],
) -> float:
    """Minimum distance between the ego-agent and any neighbour over all time steps.

    Parameters:
        agent_trajectory: Ego-agent trajectory.
        other_trajectories: Neighbour trajectories.

    Returns:
        Minimum separation distance in metres.
    """
    min_dist = np.inf
    for other in other_trajectories:
        n = min(len(agent_trajectory), len(other))
        if n == 0:
            continue
        dists = np.linalg.norm(
            agent_trajectory.positions[:n] - other.positions[:n], axis=1
        )
        min_dist = min(min_dist, float(np.min(dists)))
    return float(min_dist)


def mean_minimum_distance(
    agent_trajectory: Trajectory,
    other_trajectories: Sequence[Trajectory],
) -> float:
    """Average over time of the distance to the nearest neighbour.

    Parameters:
        agent_trajectory: Ego-agent trajectory.
        other_trajectories: Neighbour trajectories.

    Returns:
        Mean minimum distance in metres.
    """
    T = len(agent_trajectory)
    if T == 0 or len(other_trajectories) == 0:
        return float("inf")
    nearest = np.full(T, np.inf)
    for other in other_trajectories:
        n = min(T, len(other))
        dists = np.linalg.norm(
            agent_trajectory.positions[:n] - other.positions[:n], axis=1
        )
        nearest[:n] = np.minimum(nearest[:n], dists)
    return float(np.mean(nearest[nearest < np.inf]))


# ---------------------------------------------------------------------------
# Path quality metrics
# ---------------------------------------------------------------------------

def path_efficiency_ratio(
    trajectory: Trajectory,
    goal: np.ndarray | None = None,
) -> float:
    """Ratio of straight-line distance to actual path length.

    A value of 1.0 means the agent took the shortest possible path.

    Parameters:
        trajectory: Agent trajectory.
        goal: Optional explicit goal; defaults to the last position.

    Returns:
        Efficiency ratio in (0, 1].
    """
    if len(trajectory) < 2:
        return 1.0
    start = trajectory.positions[0]
    end = goal if goal is not None else trajectory.positions[-1]
    straight = float(np.linalg.norm(np.asarray(end) - start))
    diffs = np.diff(trajectory.positions, axis=0)
    actual = float(np.sum(np.linalg.norm(diffs, axis=1)))
    if actual < 1e-9:
        return 1.0
    return min(straight / actual, 1.0)


def path_irregularity(
    trajectory: Trajectory,
) -> float:
    """Measure of how much the heading direction oscillates along the path.

    Computed as the mean absolute change in heading angle (radians) per step.
    Lower values indicate smoother paths.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        Mean heading change in radians per step.
    """
    if len(trajectory) < 3:
        return 0.0
    diffs = np.diff(trajectory.positions, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    heading_changes = np.diff(headings)
    # Wrap to [-pi, pi]
    heading_changes = (heading_changes + np.pi) % (2 * np.pi) - np.pi
    return float(np.mean(np.abs(heading_changes)))


def path_curvature(
    trajectory: Trajectory,
) -> np.ndarray:
    """Instantaneous curvature at each interior point of the trajectory.

    Curvature is computed using the Menger formula for discrete points.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        1-D array of curvature values (1/m) at each interior point.
    """
    pos = trajectory.positions
    if len(pos) < 3:
        return np.array([], dtype=np.float64)

    p0 = pos[:-2]
    p1 = pos[1:-1]
    p2 = pos[2:]

    a = np.linalg.norm(p1 - p0, axis=1)
    b = np.linalg.norm(p2 - p1, axis=1)
    c = np.linalg.norm(p2 - p0, axis=1)

    # Triangle area via cross product magnitude
    cross = np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p1[:, 1] - p0[:, 1]) * (p2[:, 0] - p0[:, 0])
    )
    denom = a * b * c
    denom = np.where(denom < 1e-12, 1e-12, denom)
    curvature = 2.0 * cross / denom
    return curvature


# ---------------------------------------------------------------------------
# Comfort metrics
# ---------------------------------------------------------------------------

def speed_profile(trajectory: Trajectory) -> np.ndarray:
    """Compute speed at each time step via finite differences.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        1-D array of speeds (m/s).
    """
    if len(trajectory) < 2:
        return np.array([0.0])
    dt = np.diff(trajectory.timestamps)
    dt[dt == 0] = 1e-6
    vel = np.diff(trajectory.positions, axis=0) / dt[:, None]
    speeds = np.linalg.norm(vel, axis=1)
    return np.concatenate([speeds, speeds[-1:]])


def acceleration_profile(trajectory: Trajectory) -> np.ndarray:
    """Compute acceleration magnitude at each time step.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        1-D array of acceleration magnitudes (m/s^2).
    """
    speeds = speed_profile(trajectory)
    if len(speeds) < 2:
        return np.array([0.0])
    dt = np.diff(trajectory.timestamps)
    dt[dt == 0] = 1e-6
    accel = np.abs(np.diff(speeds)) / dt
    return np.concatenate([accel, accel[-1:]])


def jerk_metric(trajectory: Trajectory) -> float:
    """Mean absolute jerk (rate of change of acceleration).

    Lower jerk indicates smoother, more comfortable motion.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        Mean jerk in m/s^3.
    """
    accel = acceleration_profile(trajectory)
    if len(accel) < 2:
        return 0.0
    dt = np.diff(trajectory.timestamps)
    dt[dt == 0] = 1e-6
    # Use the minimum length
    n = min(len(accel) - 1, len(dt))
    jerk_vals = np.abs(np.diff(accel[:n + 1])) / dt[:n]
    return float(np.mean(jerk_vals)) if len(jerk_vals) > 0 else 0.0


def comfort_score(
    trajectory: Trajectory,
    max_speed: float = 2.0,
    max_accel: float = 1.5,
    max_jerk: float = 3.0,
) -> float:
    """Composite comfort score in [0, 1] combining speed, acceleration, and jerk.

    A score of 1.0 indicates perfectly comfortable motion; 0.0 indicates
    severe discomfort.

    Parameters:
        trajectory: Agent trajectory.
        max_speed: Maximum comfortable speed (m/s).
        max_accel: Maximum comfortable acceleration (m/s^2).
        max_jerk: Maximum comfortable jerk (m/s^3).

    Returns:
        Comfort score in [0, 1].
    """
    speeds = speed_profile(trajectory)
    accels = acceleration_profile(trajectory)
    j = jerk_metric(trajectory)

    speed_penalty = float(np.mean(np.clip(speeds / max_speed, 0, 2) ** 2))
    accel_penalty = float(np.mean(np.clip(accels / max_accel, 0, 2) ** 2))
    jerk_penalty = min((j / max_jerk) ** 2, 4.0)

    # Weighted combination
    penalty = 0.3 * speed_penalty + 0.4 * accel_penalty + 0.3 * jerk_penalty
    return float(np.clip(1.0 - penalty / 4.0, 0.0, 1.0))


def energy_expenditure(trajectory: Trajectory, mass: float = 70.0) -> float:
    """Approximate kinetic energy expenditure along the trajectory.

    Computes the cumulative work done by acceleration:
    ``sum(m * |a| * |v| * dt)``.

    Parameters:
        trajectory: Agent trajectory.
        mass: Agent mass in kg.

    Returns:
        Energy in Joules.
    """
    if len(trajectory) < 3:
        return 0.0
    speeds = speed_profile(trajectory)
    accels = acceleration_profile(trajectory)
    dt = np.diff(trajectory.timestamps)
    n = min(len(speeds) - 1, len(accels) - 1, len(dt))
    power = mass * accels[:n] * speeds[:n]
    return float(np.sum(power * dt[:n]))


# ---------------------------------------------------------------------------
# Goal achievement
# ---------------------------------------------------------------------------

def goal_achievement_rate(
    trajectories: Sequence[Trajectory],
    goals: Sequence[np.ndarray],
    threshold: float = 1.0,
) -> MetricResult:
    """Fraction of trajectories that reach their goal within *threshold*.

    Parameters:
        trajectories: List of agent trajectories.
        goals: Corresponding goal positions.
        threshold: Distance threshold in metres.

    Returns:
        :class:`MetricResult` with rate and bootstrap CI.
    """
    assert len(trajectories) == len(goals)
    reached = np.array([
        float(np.linalg.norm(t.positions[-1] - np.asarray(g)) <= threshold)
        if len(t) > 0 else 0.0
        for t, g in zip(trajectories, goals, strict=False)
    ])
    pt, lo, hi = _bootstrap_ci(reached, statistic="mean")
    return MetricResult("goal_achievement_rate", pt, "", lo, hi, 0.95, len(reached), reached)


def time_to_goal(
    trajectory: Trajectory,
    goal: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Time elapsed until the agent first comes within *threshold* of *goal*.

    Parameters:
        trajectory: Agent trajectory.
        goal: Goal position ``(x, y)``.
        threshold: Distance threshold in metres.

    Returns:
        Time in seconds, or ``inf`` if the goal is never reached.
    """
    goal = np.asarray(goal, dtype=np.float64)
    dists = np.linalg.norm(trajectory.positions - goal, axis=1)
    reached = np.where(dists <= threshold)[0]
    if len(reached) == 0:
        return float("inf")
    return float(trajectory.timestamps[reached[0]] - trajectory.timestamps[0])


# ---------------------------------------------------------------------------
# Social compliance score (composite)
# ---------------------------------------------------------------------------

def social_compliance_score(
    agent_trajectory: Trajectory,
    other_trajectories: Sequence[Trajectory],
    personal_space: float = 0.8,
    collision_radius: float = 0.5,
) -> float:
    """Composite social compliance score in [0, 1].

    Combines personal-space violations, collision rate, and path smoothness
    into a single score where 1.0 = fully compliant.

    Parameters:
        agent_trajectory: Ego-agent trajectory.
        other_trajectories: Neighbour trajectories.
        personal_space: Personal-space radius (m).
        collision_radius: Collision radius (m).

    Returns:
        Social compliance score.
    """
    # Personal space violation rate
    ps_violations = 0.0
    T = len(agent_trajectory)
    if T > 0 and len(other_trajectories) > 0:
        violation_steps = np.zeros(T, dtype=bool)
        for other in other_trajectories:
            n = min(T, len(other))
            dists = np.linalg.norm(
                agent_trajectory.positions[:n] - other.positions[:n], axis=1
            )
            violation_steps[:n] |= dists < personal_space
        ps_violations = float(np.mean(violation_steps))

    coll = collision_rate(agent_trajectory, other_trajectories, collision_radius)
    irreg = path_irregularity(agent_trajectory)

    # Weighted penalties
    penalty = (
        0.4 * ps_violations
        + 0.4 * coll
        + 0.2 * min(irreg / np.pi, 1.0)
    )
    return float(np.clip(1.0 - penalty, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Aggregation and full evaluation
# ---------------------------------------------------------------------------

class TrajectoryEvaluator:
    """High-level evaluator that computes all metrics for a set of trajectories.

    Parameters:
        collision_radius: Distance threshold for collision detection.
        personal_space: Personal-space radius for social compliance.
        goal_threshold: Distance threshold for goal achievement.
        confidence: Confidence level for bootstrap CIs.
        n_bootstrap: Number of bootstrap resamples.
    """

    def __init__(
        self,
        collision_radius: float = 0.5,
        personal_space: float = 0.8,
        goal_threshold: float = 1.0,
        confidence: float = 0.95,
        n_bootstrap: int = 1000,
    ) -> None:
        self.collision_radius = collision_radius
        self.personal_space = personal_space
        self.goal_threshold = goal_threshold
        self.confidence = confidence
        self.n_bootstrap = n_bootstrap

    def evaluate_single(
        self,
        predicted: Trajectory,
        ground_truth: Trajectory,
        neighbours: Sequence[Trajectory] | None = None,
        goal: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Evaluate a single predicted trajectory against ground truth.

        Parameters:
            predicted: Predicted trajectory.
            ground_truth: Ground-truth trajectory.
            neighbours: Optional neighbour trajectories for social metrics.
            goal: Optional goal position.

        Returns:
            Dictionary of metric name to value.
        """
        results: dict[str, float] = {}
        results["ADE"] = average_displacement_error(predicted, ground_truth)
        results["FDE"] = final_displacement_error(predicted, ground_truth)
        results["path_efficiency"] = path_efficiency_ratio(predicted, goal)
        results["path_irregularity"] = path_irregularity(predicted)
        results["comfort_score"] = comfort_score(predicted)
        results["jerk"] = jerk_metric(predicted)
        results["energy"] = energy_expenditure(predicted)

        if neighbours is not None:
            results["collision_rate"] = collision_rate(
                predicted, neighbours, self.collision_radius
            )
            results["collision_count"] = float(
                collision_count(predicted, neighbours, self.collision_radius)
            )
            results["min_separation"] = minimum_separation_distance(predicted, neighbours)
            results["mean_min_distance"] = mean_minimum_distance(predicted, neighbours)
            results["social_compliance"] = social_compliance_score(
                predicted, neighbours, self.personal_space, self.collision_radius
            )
            ttc = time_to_collision(predicted, neighbours, self.collision_radius)
            results["mean_ttc"] = float(np.mean(ttc[ttc < np.inf])) if np.any(ttc < np.inf) else float("inf")
            results["min_ttc"] = float(np.min(ttc))

        if goal is not None:
            results["time_to_goal"] = time_to_goal(predicted, goal, self.goal_threshold)
            reached = float(np.linalg.norm(predicted.positions[-1] - np.asarray(goal)) <= self.goal_threshold)
            results["goal_reached"] = reached

        return results

    def evaluate_batch(
        self,
        predicted_list: Sequence[Trajectory],
        ground_truth_list: Sequence[Trajectory],
        neighbours_list: Sequence[Sequence[Trajectory]] | None = None,
        goals: Sequence[np.ndarray] | None = None,
        model_name: str = "",
        dataset_name: str = "",
    ) -> MetricSummary:
        """Evaluate a batch of trajectories and return aggregated results with CIs.

        Parameters:
            predicted_list: Predicted trajectories.
            ground_truth_list: Ground-truth trajectories.
            neighbours_list: Optional per-sample neighbour trajectories.
            goals: Optional per-sample goal positions.
            model_name: Model identifier for reporting.
            dataset_name: Dataset identifier for reporting.

        Returns:
            :class:`MetricSummary` containing all computed metrics.
        """
        n = len(predicted_list)
        assert n == len(ground_truth_list)

        # Collect per-sample metrics
        all_metrics: dict[str, list[float]] = {}
        for i in range(n):
            nbrs = neighbours_list[i] if neighbours_list is not None else None
            g = goals[i] if goals is not None else None
            single = self.evaluate_single(
                predicted_list[i], ground_truth_list[i], nbrs, g
            )
            for k, v in single.items():
                all_metrics.setdefault(k, []).append(v)

        summary = MetricSummary(model_name=model_name, dataset_name=dataset_name)
        for metric_name, values in all_metrics.items():
            arr = np.array(values, dtype=np.float64)
            # Filter out nans and infs for aggregation
            finite = arr[np.isfinite(arr)]
            if len(finite) == 0:
                pt, lo, hi = float("nan"), float("nan"), float("nan")
            else:
                pt, lo, hi = _bootstrap_ci(
                    finite, confidence=self.confidence, n_bootstrap=self.n_bootstrap
                )
            summary.add(MetricResult(metric_name, pt, "", lo, hi, self.confidence, len(finite), arr))

        return summary

    def compare(
        self,
        summary_a: MetricSummary,
        summary_b: MetricSummary,
        n_permutations: int = 10000,
    ) -> dict[str, float]:
        """Run paired permutation tests between two model summaries.

        Parameters:
            summary_a: First model's evaluation summary.
            summary_b: Second model's evaluation summary.
            n_permutations: Number of permutations for the test.

        Returns:
            Dictionary mapping metric name to p-value.
        """
        p_values: dict[str, float] = {}
        common_metrics = set(summary_a.results.keys()) & set(summary_b.results.keys())
        for metric in common_metrics:
            raw_a = summary_a.results[metric].raw_values
            raw_b = summary_b.results[metric].raw_values
            if raw_a is None or raw_b is None:
                continue
            # Use minimum common length
            n = min(len(raw_a), len(raw_b))
            fa = raw_a[:n]
            fb = raw_b[:n]
            # Filter both to finite
            mask = np.isfinite(fa) & np.isfinite(fb)
            if np.sum(mask) < 2:
                p_values[metric] = float("nan")
            else:
                p_values[metric] = paired_permutation_test(fa[mask], fb[mask], n_permutations)
        return p_values
