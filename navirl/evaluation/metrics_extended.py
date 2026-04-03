"""Extended navigation metrics beyond the standard ones in ``navirl.metrics``."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from navirl.data.trajectory import Trajectory


def time_to_goal(trajectory: Trajectory, goal: np.ndarray, threshold: float = 0.5) -> float:
    """Time elapsed until the agent first reaches within *threshold* of *goal*.

    Parameters:
        trajectory: Agent trajectory.
        goal: ``(x, y)`` goal position.
        threshold: Distance (metres) considered as having reached the goal.

    Returns:
        Time in seconds, or ``float('inf')`` if the goal is never reached.
    """
    goal = np.asarray(goal, dtype=np.float64)
    dists = np.linalg.norm(trajectory.positions - goal, axis=1)
    reached = np.where(dists <= threshold)[0]
    if len(reached) == 0:
        return float("inf")
    return float(trajectory.timestamps[reached[0]] - trajectory.timestamps[0])


def path_length(trajectory: Trajectory) -> float:
    """Total Euclidean path length of the trajectory.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        Cumulative distance travelled in metres.
    """
    if len(trajectory) < 2:
        return 0.0
    diffs = np.diff(trajectory.positions, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def path_efficiency(trajectory: Trajectory, optimal_length: float) -> float:
    """Ratio of optimal path length to actual path length.

    Parameters:
        trajectory: Agent trajectory.
        optimal_length: Shortest possible path length.

    Returns:
        Efficiency ratio in ``(0, 1]``; 1.0 means perfectly efficient.
    """
    actual = path_length(trajectory)
    if actual <= 0:
        return 0.0
    return min(optimal_length / actual, 1.0)


def jerk_metric(trajectory: Trajectory) -> float:
    """Integrated squared jerk (derivative of acceleration) as a smoothness measure.

    Lower values indicate smoother trajectories.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        Mean squared jerk magnitude.
    """
    if len(trajectory) < 4:
        return 0.0
    dt_arr = np.diff(trajectory.timestamps)
    dt = float(np.mean(dt_arr)) if len(dt_arr) > 0 else 1.0
    if dt <= 0:
        return 0.0
    vel = np.diff(trajectory.positions, axis=0) / dt
    acc = np.diff(vel, axis=0) / dt
    jerk = np.diff(acc, axis=0) / dt
    return float(np.mean(np.sum(jerk**2, axis=1)))


def social_force_integral(
    trajectory: Trajectory,
    pedestrian_trajectories: Sequence[Trajectory],
    *,
    sigma: float = 0.6,
    amplitude: float = 2.1,
) -> float:
    """Integrate the repulsive social force experienced along the trajectory.

    Uses a simplified exponential repulsion model with vectorized computation for
    performance on large pedestrian populations.

    Parameters:
        trajectory: Ego agent trajectory.
        pedestrian_trajectories: Other pedestrian trajectories.
        sigma: Characteristic length scale of the social force (metres).
        amplitude: Force amplitude.

    Returns:
        Total integrated social force magnitude.
    """
    if not pedestrian_trajectories or len(trajectory) < 2:
        return 0.0

    # Pre-filter empty trajectories
    valid_peds = [ped for ped in pedestrian_trajectories if len(ped) > 0]
    if not valid_peds:
        return 0.0

    dt_arr = np.diff(trajectory.timestamps)
    n_timesteps = len(trajectory)
    n_pedestrians = len(valid_peds)

    # Vectorized approach: compute all pedestrian positions at ego timestamps
    # Shape: (n_timesteps, n_pedestrians, 2)
    ped_positions = np.full((n_timesteps, n_pedestrians, 2), np.nan)

    for ped_idx, ped in enumerate(valid_peds):
        for t_idx, ts in enumerate(trajectory.timestamps):
            # Find closest timestamp using binary search (O(log m) vs O(m))
            p_idx = np.searchsorted(ped.timestamps, ts)
            p_idx = min(p_idx, len(ped.timestamps) - 1)

            # Check if previous timestamp is closer
            if p_idx > 0 and abs(ped.timestamps[p_idx - 1] - ts) < abs(ped.timestamps[p_idx] - ts):
                p_idx = p_idx - 1

            ped_positions[t_idx, ped_idx] = ped.positions[p_idx]

    # Vectorized distance computation
    # ego_positions shape: (n_timesteps, 1, 2) to enable broadcasting
    ego_positions = trajectory.positions[:, np.newaxis, :]

    # Compute all distances at once: shape (n_timesteps, n_pedestrians)
    # Using broadcasting: (n_timesteps, 1, 2) - (n_timesteps, n_pedestrians, 2)
    position_diffs = ego_positions - ped_positions
    distances = np.linalg.norm(position_diffs, axis=2)

    # Apply social force model vectorially, avoiding division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        forces = np.where(distances > 0, amplitude * np.exp(-distances / sigma), 0.0)

    # Sum forces across all pedestrians for each timestep
    force_per_timestep = np.nansum(forces, axis=1)  # shape: (n_timesteps,)

    # Integrate over time using appropriate dt values
    total_force = 0.0
    for t_idx in range(n_timesteps):
        if t_idx < len(dt_arr):
            total_force += force_per_timestep[t_idx] * dt_arr[t_idx]
        elif len(dt_arr) > 0:
            # Use last dt for the final timestep
            total_force += force_per_timestep[t_idx] * dt_arr[-1]

    return float(total_force)


def personal_space_violations(
    trajectory: Trajectory,
    pedestrians: Sequence[Trajectory],
    threshold: float = 0.5,
) -> int:
    """Count the number of timesteps where the agent violates personal space.

    Parameters:
        trajectory: Ego agent trajectory.
        pedestrians: Pedestrian trajectories.
        threshold: Personal space radius in metres.

    Returns:
        Number of timesteps with at least one violation.
    """
    violations = 0
    for t_idx in range(len(trajectory)):
        ego_pos = trajectory.positions[t_idx]
        ts = trajectory.timestamps[t_idx]
        for ped in pedestrians:
            if len(ped) == 0:
                continue
            # Use binary search instead of linear search for O(log m) vs O(m) performance
            p_idx = np.searchsorted(ped.timestamps, ts)
            # Handle edge case: if ts is beyond the last timestamp
            p_idx = min(p_idx, len(ped.timestamps) - 1)
            # Check if previous timestamp is closer
            if p_idx > 0 and abs(ped.timestamps[p_idx - 1] - ts) < abs(ped.timestamps[p_idx] - ts):
                p_idx = p_idx - 1
            dist = float(np.linalg.norm(ego_pos - ped.positions[p_idx]))
            if dist < threshold:
                violations += 1
                break  # One violation per timestep is enough.
    return violations


def minimum_separation_distance(
    trajectory: Trajectory,
    pedestrians: Sequence[Trajectory],
) -> float:
    """Minimum distance between the ego and any pedestrian over the full trajectory.

    Parameters:
        trajectory: Ego agent trajectory.
        pedestrians: Pedestrian trajectories.

    Returns:
        Minimum distance in metres, or ``float('inf')`` if no pedestrians.
    """
    if not pedestrians:
        return float("inf")
    min_dist = float("inf")
    for t_idx in range(len(trajectory)):
        ego_pos = trajectory.positions[t_idx]
        ts = trajectory.timestamps[t_idx]
        for ped in pedestrians:
            if len(ped) == 0:
                continue
            # Use binary search instead of linear search for O(log m) vs O(m) performance
            p_idx = np.searchsorted(ped.timestamps, ts)
            # Handle edge case: if ts is beyond the last timestamp
            p_idx = min(p_idx, len(ped.timestamps) - 1)
            # Check if previous timestamp is closer
            if p_idx > 0 and abs(ped.timestamps[p_idx - 1] - ts) < abs(ped.timestamps[p_idx] - ts):
                p_idx = p_idx - 1
            dist = float(np.linalg.norm(ego_pos - ped.positions[p_idx]))
            if dist < min_dist:
                min_dist = dist
    return min_dist


def topological_complexity(
    trajectory: Trajectory,
    obstacles: np.ndarray,
    *,
    proximity_threshold: float = 1.0,
) -> int:
    """Estimate topological complexity as the number of distinct obstacle passes.

    Counts transitions where the agent moves from being far from all obstacles to
    being close (within *proximity_threshold*), indicating a navigation decision.

    Parameters:
        trajectory: Ego agent trajectory.
        obstacles: Array of shape ``(M, 2)`` with obstacle centre positions.
        proximity_threshold: Distance threshold for being "near" an obstacle.

    Returns:
        Number of obstacle interaction transitions.
    """
    obstacles = np.asarray(obstacles, dtype=np.float64)
    if len(obstacles) == 0 or len(trajectory) < 2:
        return 0

    was_near = False
    transitions = 0
    for t_idx in range(len(trajectory)):
        ego_pos = trajectory.positions[t_idx]
        dists = np.linalg.norm(obstacles - ego_pos, axis=1)
        is_near = bool(np.any(dists < proximity_threshold))
        if is_near and not was_near:
            transitions += 1
        was_near = is_near
    return transitions


def heading_change_rate(trajectory: Trajectory) -> float:
    """Average absolute heading change per second.

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        Mean heading change rate in radians per second.
    """
    if len(trajectory) < 3:
        return 0.0
    diffs = np.diff(trajectory.positions, axis=0)
    headings = np.arctan2(diffs[:, 1], diffs[:, 0])
    dheading = np.diff(headings)
    # Wrap to [-pi, pi]
    dheading = (dheading + np.pi) % (2 * np.pi) - np.pi
    dt_total = trajectory.duration
    if dt_total <= 0:
        return 0.0
    return float(np.sum(np.abs(dheading)) / dt_total)


def velocity_smoothness(trajectory: Trajectory) -> float:
    """Standard deviation of speed over the trajectory (lower = smoother).

    Parameters:
        trajectory: Agent trajectory.

    Returns:
        Standard deviation of the scalar speed.
    """
    if len(trajectory) < 2:
        return 0.0
    dt = float(np.mean(np.diff(trajectory.timestamps)))
    if dt <= 0:
        return 0.0
    vel = np.diff(trajectory.positions, axis=0) / dt
    speed = np.linalg.norm(vel, axis=1)
    return float(np.std(speed))


def collision_rate(events: Sequence[dict[str, Any]]) -> float:
    """Fraction of events that are collisions.

    Parameters:
        events: List of event dicts; collisions have ``"type" == "collision"``.

    Returns:
        Collision rate in ``[0, 1]``.
    """
    if not events:
        return 0.0
    n_collisions = sum(1 for e in events if e.get("type") == "collision")
    return n_collisions / len(events)


def success_rate(episodes: Sequence[dict[str, Any]]) -> float:
    """Fraction of episodes where the agent reached the goal.

    Parameters:
        episodes: List of episode summary dicts with a ``"success"`` boolean key.

    Returns:
        Success rate in ``[0, 1]``.
    """
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.get("success", False)) / len(episodes)


def timeout_rate(episodes: Sequence[dict[str, Any]]) -> float:
    """Fraction of episodes that ended due to timeout.

    Parameters:
        episodes: List of episode summary dicts with a ``"timeout"`` boolean key.

    Returns:
        Timeout rate in ``[0, 1]``.
    """
    if not episodes:
        return 0.0
    return sum(1 for ep in episodes if ep.get("timeout", False)) / len(episodes)


def comfort_score(
    trajectory: Trajectory,
    pedestrians: Sequence[Trajectory],
    *,
    personal_space: float = 0.5,
    jerk_weight: float = 0.3,
    space_weight: float = 0.5,
    heading_weight: float = 0.2,
) -> float:
    """Composite comfort metric combining jerk, personal space, and heading change.

    Returns a score in ``[0, 1]`` where 1.0 is maximally comfortable.

    Parameters:
        trajectory: Ego agent trajectory.
        pedestrians: Pedestrian trajectories.
        personal_space: Radius for personal space violations.
        jerk_weight: Weight for jerk component.
        space_weight: Weight for personal space component.
        heading_weight: Weight for heading change component.

    Returns:
        Comfort score in ``[0, 1]``.
    """
    total_weight = jerk_weight + space_weight + heading_weight
    if total_weight <= 0:
        return 1.0

    # Jerk component: transform to [0, 1] via sigmoid-like mapping
    j = jerk_metric(trajectory)
    jerk_score = 1.0 / (1.0 + j)

    # Personal space component: fraction of timesteps without violation
    n_violations = personal_space_violations(trajectory, pedestrians, personal_space)
    n_steps = max(len(trajectory), 1)
    space_score = 1.0 - (n_violations / n_steps)

    # Heading change component
    hcr = heading_change_rate(trajectory)
    heading_score = 1.0 / (1.0 + hcr)

    score = (
        jerk_weight * jerk_score + space_weight * space_score + heading_weight * heading_score
    ) / total_weight
    return float(np.clip(score, 0.0, 1.0))
