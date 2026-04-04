"""Core simulation pipeline for NavIRL.

Provides the main execution pipeline for running single scenarios and
batch experiments, including scenario loading, agent orchestration,
logging, metrics collection, and result rendering.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import uuid
from datetime import UTC, datetime
from pathlib import Path

from navirl.core.registry import get_backend, get_human_controller, get_robot_controller
from navirl.core.seeds import set_global_seed
from navirl.core.types import AgentState, EpisodeLog, EventRecord
from navirl.logging.episode_log import EpisodeLogger
from navirl.metrics.standard import StandardMetrics
from navirl.plugins import register_default_plugins
from navirl.scenarios.load import load_scenario
from navirl.viz.render import render_trace

# ==============================================================================
# Pipeline Configuration Constants
# ==============================================================================

# Spatial sampling and placement constraints
MIN_CLEARANCE_FLOOR = 0.08  # Minimum clearance floor to prevent deadlocks (meters)
RELAXATION_FACTOR = 0.85  # Factor to reduce spacing when strict placement fails
HUMAN_SPACING_FACTOR = 2.4  # Human radius multiplier for minimum separation distance
MIN_ABSOLUTE_SPACING = 0.2  # Absolute minimum spacing between agents (meters)
BASE_STEP_FACTOR = 0.35  # Multiplier for local search step size around desired positions
MIN_BASE_STEP = 0.02  # Minimum absolute base step for local search (meters)

# Sampling limits and tolerances
MAX_SPATIAL_SAMPLES = 4000  # Maximum attempts for spatial search before fallback
ATTEMPTS_PER_POINT = 2500  # Sampling attempts per point when ensuring minimum distance
MAX_SEARCH_RINGS = 25  # Maximum radial rings for local placement search around desired position
POSITION_TOLERANCE = 1e-6  # Tolerance for detecting position adjustments (meters)

# Search pattern parameters
INITIAL_RING_POINTS = 12  # Base number of points in first search ring
RING_POINT_INCREMENT = 4  # Additional points per ring in spiral search


def _run_id(scenario_id: str) -> str:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{scenario_id}_{stamp}_{uuid.uuid4().hex[:8]}"


def _ensure_points(
    existing: list[tuple[float, float]],
    count: int,
    sampler,
    min_dist: float,
) -> list[tuple[float, float]]:
    pts = [tuple(map(float, p)) for p in existing]
    cur_min_dist = max(0.0, float(min_dist))
    min_floor = min(MIN_CLEARANCE_FLOOR, cur_min_dist) if cur_min_dist > 0.0 else 0.0
    attempts_per_point = ATTEMPTS_PER_POINT

    while len(pts) < count:
        found = None
        for _ in range(attempts_per_point):
            cand = tuple(map(float, sampler()))
            if all(math.hypot(cand[0] - p[0], cand[1] - p[1]) >= cur_min_dist for p in pts):
                found = cand
                break

        if found is not None:
            pts.append(found)
            continue

        # If strict spacing is infeasible for dense scenes, relax gradually.
        if cur_min_dist > min_floor:
            cur_min_dist = max(min_floor, cur_min_dist * RELAXATION_FACTOR)
            continue

        # Final fallback: keep moving by accepting a free sample point.
        pts.append(tuple(map(float, sampler())))
    return pts


def _resolve_human_start_goal_lists(
    scenario: dict, backend
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    humans = scenario["humans"]
    count = int(humans["count"])
    radius = float(humans["radius"])

    starts = _ensure_points(
        existing=humans.get("starts", []),
        count=count,
        sampler=backend.sample_free_point,
        min_dist=max(MIN_ABSOLUTE_SPACING, radius * HUMAN_SPACING_FACTOR),
    )
    goals = _ensure_points(
        existing=humans.get("goals", []),
        count=count,
        sampler=backend.sample_free_point,
        min_dist=max(MIN_ABSOLUTE_SPACING, radius * HUMAN_SPACING_FACTOR),
    )
    return starts, goals


def _diameter(radius: float) -> float:
    return max(0.0, 2.0 * float(radius))


def _min_anchor_dist(a_radius: float, b_radius: float) -> float:
    return max(_diameter(a_radius), _diameter(b_radius))


def _has_obstacle_collision(position: tuple[float, float], radius: float, backend) -> bool:
    """Check if position collides with obstacles.

    Parameters
    ----------
    position : tuple[float, float]
        Candidate position to check.
    radius : float
        Agent radius.
    backend : Any
        Backend providing collision detection.

    Returns
    -------
    bool
        True if position would cause an obstacle collision.
    """
    return backend.check_obstacle_collision(position, _diameter(radius))


def _has_agent_collision(
    position: tuple[float, float],
    radius: float,
    placed_agents: list[dict],
) -> bool:
    """Check if position collides with already-placed agents.

    Parameters
    ----------
    position : tuple[float, float]
        Candidate position to check.
    radius : float
        Agent radius.
    placed_agents : list[dict]
        List of already-placed agents with 'position' and 'radius' keys.

    Returns
    -------
    bool
        True if position would cause an agent-agent collision.
    """
    for agent in placed_agents:
        min_dist = _min_anchor_dist(radius, float(agent["radius"]))
        agent_pos = agent["position"]
        distance = math.hypot(position[0] - agent_pos[0], position[1] - agent_pos[1])
        if distance < min_dist:
            return True
    return False


def _anchor_ok(
    candidate: tuple[float, float],
    radius: float,
    placed: list[dict],
    backend,
) -> bool:
    """Check if anchor position is valid (no collisions with obstacles or other agents).

    Parameters
    ----------
    candidate : tuple[float, float]
        Candidate anchor position.
    radius : float
        Agent radius.
    placed : list[dict]
        List of already-placed agents.
    backend : Any
        Backend providing collision detection.

    Returns
    -------
    bool
        True if position is valid for placement.
    """
    return not (
        _has_obstacle_collision(candidate, radius, backend)
        or _has_agent_collision(candidate, radius, placed)
    )


def _project_anchor(candidate: tuple[float, float], radius: float, backend) -> tuple[float, float]:
    projected = backend.nearest_clear_point(candidate, _diameter(radius))
    return float(projected[0]), float(projected[1])


def _search_ring_positions(
    desired: tuple[float, float], radius: float, placed: list[dict], backend, base_step: float
) -> tuple[float, float] | None:
    """Search for valid anchor position using expanding ring pattern.

    Searches in expanding rings around the desired position to find a valid
    placement that doesn't collide with existing anchors or map obstacles.

    Args:
        desired: Desired (x, y) position coordinates.
        radius: Agent radius for collision detection.
        placed: List of already-placed anchor dictionaries.
        backend: Simulation backend for spatial queries.
        base_step: Base distance between rings in the search pattern.

    Returns:
        Valid (x, y) position if found, None otherwise.
    """
    for ring in range(1, MAX_SEARCH_RINGS + 1):
        position = _search_ring_points(desired, radius, placed, backend, base_step, ring)
        if position is not None:
            return position
    return None


def _search_ring_points(
    desired: tuple[float, float],
    radius: float,
    placed: list[dict],
    backend,
    base_step: float,
    ring: int,
) -> tuple[float, float] | None:
    """Search points around a specific ring for valid anchor position.

    Args:
        desired: Desired (x, y) position coordinates.
        radius: Agent radius for collision detection.
        placed: List of already-placed anchor dictionaries.
        backend: Simulation backend for spatial queries.
        base_step: Base distance between rings.
        ring: Ring number (1-based) to search.

    Returns:
        Valid (x, y) position if found, None otherwise.
    """
    dist = base_step * ring
    num_points = INITIAL_RING_POINTS + ring * RING_POINT_INCREMENT

    for i in range(num_points):
        angle = (2.0 * math.pi * i) / max(1, num_points)
        local = (
            desired[0] + dist * math.cos(angle),
            desired[1] + dist * math.sin(angle),
        )
        trial = _project_anchor(local, radius, backend)
        if _anchor_ok(trial, radius, placed, backend):
            return trial
    return None


def _search_random_positions(
    radius: float, placed: list[dict], backend, max_samples: int
) -> tuple[float, float] | None:
    """Search for valid anchor position using random sampling.

    Fallback search method that randomly samples free points when ring
    search fails to find a valid position.

    Args:
        radius: Agent radius for collision detection.
        placed: List of already-placed anchor dictionaries.
        backend: Simulation backend for spatial queries.
        max_samples: Maximum number of random samples to try.

    Returns:
        Valid (x, y) position if found, None otherwise.
    """
    for _ in range(max_samples):
        trial = _project_anchor(tuple(map(float, backend.sample_free_point())), radius, backend)
        if _anchor_ok(trial, radius, placed, backend):
            return trial
    return None


def _enforce_anchor_layout(
    anchors: list[dict],
    backend,
    max_samples: int = MAX_SPATIAL_SAMPLES,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Enforce spatial layout constraints for anchor points.

    Attempts to place anchors at desired positions while avoiding collisions
    with obstacles and other anchors. Uses ring-based search followed by
    random sampling as fallback strategies.

    Args:
        anchors: List of anchor specifications with position, radius, and key.
        backend: Simulation backend for spatial queries and collision detection.
        max_samples: Maximum random samples to try per anchor during fallback.

    Returns:
        Tuple of (placed, adjustments, unresolved) anchor lists.
    """
    placed: list[dict] = []
    adjustments: list[dict] = []
    unresolved: list[dict] = []

    for anchor in anchors:
        key = str(anchor["key"])
        radius = float(anchor["radius"])
        desired = (float(anchor["position"][0]), float(anchor["position"][1]))
        candidate = _project_anchor(desired, radius, backend)

        if not _anchor_ok(candidate, radius, placed, backend):
            found = None

            # Preserve scenario semantics when possible by searching nearby offsets first.
            base_step = max(MIN_BASE_STEP, _diameter(radius) * BASE_STEP_FACTOR)
            found = _search_ring_positions(desired, radius, placed, backend, base_step)

            # Fallback to random sampling if ring search fails
            if found is None:
                found = _search_random_positions(radius, placed, backend, max_samples)

            if found is None:
                fallback = _project_anchor(desired, radius, backend)
                unresolved.append(
                    {
                        "key": key,
                        "reason": "unable_to_place_with_diameter_constraints",
                        "requested": [float(desired[0]), float(desired[1])],
                        "fallback": [float(fallback[0]), float(fallback[1])],
                        "radius": float(radius),
                    }
                )
                candidate = fallback
            else:
                candidate = found

        if math.hypot(candidate[0] - desired[0], candidate[1] - desired[1]) > POSITION_TOLERANCE:
            adjustments.append(
                {
                    "key": key,
                    "from": [float(desired[0]), float(desired[1])],
                    "to": [float(candidate[0]), float(candidate[1])],
                    "radius": radius,
                }
            )

        placed.append(
            {
                "key": key,
                "position": candidate,
                "radius": radius,
            }
        )

    return placed, adjustments, unresolved


def _sanitize_starts_goals(
    scenario: dict,
    backend,
    human_ids: list[int],
    human_starts: dict[int, tuple[float, float]],
    human_goals: dict[int, tuple[float, float]],
    human_radius: float,
    robot_radius: float,
) -> dict:
    start_anchors: list[dict] = []
    for hid in human_ids:
        start_anchors.append(
            {
                "key": f"human.{hid}.start",
                "position": tuple(human_starts[hid]),
                "radius": human_radius,
            }
        )
    start_anchors.append(
        {
            "key": "robot.start",
            "position": tuple(scenario["robot"]["start"]),
            "radius": robot_radius,
        }
    )

    goal_anchors: list[dict] = []
    for hid in human_ids:
        goal_anchors.append(
            {
                "key": f"human.{hid}.goal",
                "position": tuple(human_goals[hid]),
                "radius": human_radius,
            }
        )
    goal_anchors.append(
        {"key": "robot.goal", "position": tuple(scenario["robot"]["goal"]), "radius": robot_radius}
    )

    placed_starts, adj_starts, unresolved_starts = _enforce_anchor_layout(
        start_anchors, backend=backend
    )
    placed_goals, adj_goals, unresolved_goals = _enforce_anchor_layout(
        goal_anchors, backend=backend
    )
    placed = placed_starts + placed_goals
    adjustments = adj_starts + adj_goals
    unresolved = unresolved_starts + unresolved_goals
    by_key = {rec["key"]: rec["position"] for rec in placed}

    scenario["robot"]["start"] = tuple(map(float, by_key["robot.start"]))
    scenario["robot"]["goal"] = tuple(map(float, by_key["robot.goal"]))
    for hid in human_ids:
        human_starts[hid] = tuple(map(float, by_key[f"human.{hid}.start"]))
        human_goals[hid] = tuple(map(float, by_key[f"human.{hid}.goal"]))

    scenario["humans"]["starts"] = [human_starts[hid] for hid in human_ids]
    scenario["humans"]["goals"] = [human_goals[hid] for hid in human_ids]
    scenario.setdefault("_meta", {})["anchor_adjustments"] = adjustments
    scenario.setdefault("_meta", {})["anchor_unresolved"] = unresolved

    return {
        "count": len(adjustments),
        "adjustments": adjustments,
        "unresolved_count": len(unresolved),
        "unresolved": unresolved,
    }


def _resample_human_starts_goals_for_retry(scenario: dict, backend) -> None:
    humans = scenario.get("humans", {})
    count = int(humans.get("count", 0))
    if count <= 0:
        return

    radius = float(humans.get("radius", 0.16))
    min_sep = max(0.25, _diameter(radius) * 1.05)
    starts = _ensure_points([], count, backend.sample_free_point, min_sep)

    goals: list[tuple[float, float]] = []
    min_start_goal = max(0.8, _diameter(radius) * 2.0)
    for idx in range(count):
        assigned = False
        for _ in range(5000):
            cand = tuple(map(float, backend.sample_free_point()))
            if math.hypot(cand[0] - starts[idx][0], cand[1] - starts[idx][1]) < min_start_goal:
                continue
            if any(math.hypot(cand[0] - g[0], cand[1] - g[1]) < min_sep for g in goals):
                continue
            goals.append(cand)
            assigned = True
            break
        if not assigned:
            # Deterministic fallback: keep sampling without the start-goal distance requirement.
            goals.append(tuple(map(float, backend.sample_free_point())))

    humans["starts"] = starts
    humans["goals"] = goals


def _bump_traversability_offset_for_retry(scenario: dict) -> float:
    eval_cfg = scenario.setdefault("evaluation", {})
    scene_cfg = scenario.setdefault("scene", {})
    orca_cfg = scene_cfg.setdefault("orca", {})

    step = float(eval_cfg.get("traversability_offset_step", 0.005))
    max_val = float(eval_cfg.get("traversability_offset_max", 0.04))
    cur = float(orca_cfg.get("wall_clearance_buffer_m", eval_cfg.get("wall_clearance_buffer", 0.0)))
    nxt = min(max_val, cur + max(0.0, step))

    orca_cfg["wall_clearance_buffer_m"] = float(nxt)
    eval_cfg["wall_clearance_buffer"] = float(nxt)
    return float(nxt)


def _human_goal_map(
    controller, fallback: dict[int, tuple[float, float]]
) -> dict[int, tuple[float, float]]:
    ctrl_goals = getattr(controller, "goals", None)
    if isinstance(ctrl_goals, dict):
        return {int(k): tuple(map(float, v)) for k, v in ctrl_goals.items()}
    return fallback


def run_scenario_dict(
    scenario: dict,
    out_root: str | Path,
    run_id: str | None = None,
    render_override: bool | None = None,
    video_override: bool | None = None,
) -> EpisodeLog:
    """Execute a scenario from a dictionary configuration.

    Parameters
    ----------
    scenario : dict
        Scenario configuration containing required fields: id, seed, humans, robot, scene, horizon
    out_root : str | Path
        Output directory for results
    run_id : str, optional
        Unique run identifier, generated if None
    render_override : bool, optional
        Override scenario render setting
    video_override : bool, optional
        Override scenario video setting

    Returns
    -------
    EpisodeLog
        Simulation results and metrics

    Raises
    ------
    ValueError
        If scenario is missing required fields or has invalid values
    TypeError
        If scenario is not a dictionary
    """
    # Validate input parameters
    if not isinstance(scenario, dict):
        raise TypeError(f"Scenario must be a dictionary, got {type(scenario).__name__}")

    if not scenario:
        raise ValueError("Scenario dictionary cannot be empty")

    # Validate required scenario fields
    required_fields = ["id", "seed", "humans", "robot", "scene", "horizon"]
    missing_fields = [field for field in required_fields if field not in scenario]
    if missing_fields:
        raise ValueError(f"Scenario missing required fields: {missing_fields}")

    register_default_plugins()

    # Safely extract and validate scenario_id
    scenario_id = scenario["id"]
    if not scenario_id or not isinstance(scenario_id, str):
        raise ValueError(f"Scenario 'id' must be a non-empty string, got: {scenario_id}")

    run_id = run_id or _run_id(scenario_id)

    try:
        out_root = Path(out_root)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid output root path: {e}") from e

    bundle_dir = out_root / run_id / "bundle"

    eval_cfg = scenario.setdefault("evaluation", {})
    if not isinstance(eval_cfg, dict):
        raise ValueError("Scenario 'evaluation' field must be a dictionary")

    # Safely parse evaluation configuration with validation
    try:
        retry_max_attempts = int(eval_cfg.get("deadlock_resample_attempts", 4))
        if retry_max_attempts < 0:
            raise ValueError("deadlock_resample_attempts must be non-negative")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid deadlock_resample_attempts: {e}") from e

    auto_resample_on_deadlock = bool(eval_cfg.get("resample_on_deadlock", True))
    auto_tune_traversability = bool(eval_cfg.get("auto_tune_traversability_offset", True))
    fail_on_deadlock = bool(eval_cfg.get("fail_on_deadlock", True))

    meta = scenario.setdefault("_meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Scenario '_meta' field must be a dictionary")

    try:
        retry_attempt = int(meta.get("deadlock_retry_attempt", 0))
        if retry_attempt < 0:
            raise ValueError("deadlock_retry_attempt must be non-negative")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid deadlock_retry_attempt: {e}") from e

    retry_history = list(meta.get("deadlock_retry_history", []))

    # Safely parse and validate seed
    try:
        seed = int(scenario["seed"])
        if seed < 0:
            raise ValueError("Seed must be non-negative")
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid seed value '{scenario['seed']}': {e}") from e

    attempt_seed = int(seed + retry_attempt * 9973)
    set_global_seed(attempt_seed)

    source_path = scenario.get("_meta", {}).get("source_path")
    base_dir = Path(source_path).parent if source_path else None
    backend_factory = get_backend(scenario["scene"]["backend"])
    backend = backend_factory(scenario["scene"], scenario["horizon"], base_dir=base_dir)

    if retry_attempt > 0 and auto_resample_on_deadlock:
        _resample_human_starts_goals_for_retry(scenario, backend)
        tuned_offset = None
        if auto_tune_traversability:
            tuned_offset = _bump_traversability_offset_for_retry(scenario)
        retry_history.append(
            {
                "attempt": retry_attempt,
                "reason": "deadlock_or_anchor_retry",
                "seed": attempt_seed,
                "tuned_wall_clearance_buffer_m": tuned_offset,
            }
        )
        meta["deadlock_retry_history"] = retry_history

    map_metadata = backend.map_metadata()
    if map_metadata:
        scenario.setdefault("scene", {}).setdefault("map", {})["resolved"] = map_metadata
        scenario.setdefault("_meta", {})["map"] = map_metadata

    robot_id = 0
    human_count = int(scenario["humans"]["count"])
    human_ids = [i + 1 for i in range(human_count)]

    human_radius = float(scenario["humans"]["radius"])
    human_max_speed = float(scenario["humans"]["max_speed"])
    robot_radius = float(scenario["robot"]["radius"])
    robot_max_speed = float(scenario["robot"]["max_speed"])

    starts, goals = _resolve_human_start_goal_lists(scenario, backend)
    human_starts = {hid: starts[idx] for idx, hid in enumerate(human_ids)}
    human_goals = {hid: goals[idx] for idx, hid in enumerate(human_ids)}
    anchor_adjust = _sanitize_starts_goals(
        scenario=scenario,
        backend=backend,
        human_ids=human_ids,
        human_starts=human_starts,
        human_goals=human_goals,
        human_radius=human_radius,
        robot_radius=robot_radius,
    )

    if anchor_adjust.get("unresolved_count", 0) > 0:
        if auto_resample_on_deadlock and retry_attempt < retry_max_attempts:
            retry_history.append(
                {
                    "attempt": retry_attempt,
                    "reason": "anchor_unresolved",
                    "unresolved_count": int(anchor_adjust.get("unresolved_count", 0)),
                }
            )
            meta["deadlock_retry_history"] = retry_history
            meta["deadlock_retry_attempt"] = retry_attempt + 1
            return run_scenario_dict(
                scenario=scenario,
                out_root=out_root,
                run_id=run_id,
                render_override=render_override,
                video_override=video_override,
            )
        raise ValueError(
            "Unable to satisfy start/goal diameter-clearance constraints after retries. "
            "This likely indicates a traversability/radius mismatch for the current map."
        )

    backend.add_agent(
        robot_id,
        tuple(scenario["robot"]["start"]),
        radius=robot_radius,
        max_speed=robot_max_speed,
        kind="robot",
    )
    for hid in human_ids:
        backend.add_agent(
            hid,
            human_starts[hid],
            radius=human_radius,
            max_speed=human_max_speed,
            kind="human",
        )

    human_ctrl_cfg = scenario["humans"]["controller"]
    robot_ctrl_cfg = scenario["robot"]["controller"]

    human_factory = get_human_controller(human_ctrl_cfg["type"])
    human_controller = human_factory(human_ctrl_cfg.get("params", {}), seed=attempt_seed)
    human_controller.reset(human_ids, human_starts, human_goals, backend=backend)

    robot_factory = get_robot_controller(robot_ctrl_cfg["type"])
    robot_controller = robot_factory(robot_ctrl_cfg.get("params", {}))
    robot_controller.reset(
        robot_id=robot_id,
        start=tuple(scenario["robot"]["start"]),
        goal=tuple(scenario["robot"]["goal"]),
        backend=backend,
    )

    dt = float(scenario["horizon"]["dt"])
    steps = int(scenario["horizon"]["steps"])

    logger = EpisodeLogger(bundle_dir)
    logger.write_resolved_scenario(scenario)

    current_step = 0
    events: list[EventRecord] = []

    def emit_event(event_type: str, agent_id: int | None, payload: dict) -> None:
        """Emit a simulation event and log it for analysis.

        Creates an EventRecord with current simulation state and writes it to both
        the in-memory events list and the episode logger.

        Args:
            event_type: Type identifier for the event (e.g., "collision", "goal_reached").
            agent_id: ID of the agent associated with this event, or None for global events.
            payload: Additional event-specific data as a dictionary.
        """
        ev = EventRecord(
            step=current_step,
            time_s=current_step * dt,
            event_type=event_type,
            agent_id=agent_id,
            payload=payload,
        )
        events.append(ev)
        logger.write_event(ev)

    def capture_states(behaviors: dict[int, str] | None = None) -> dict[int, AgentState]:
        """Capture current state of all agents in the simulation.

        Extracts position, velocity, goals, and behavior information for both robot
        and human agents from the simulation backend.

        Args:
            behaviors: Optional mapping from agent ID to behavior string. If not provided,
                      defaults to empty dict and standard behaviors are inferred.

        Returns:
            Dictionary mapping agent IDs to their current AgentState objects.
        """
        behaviors = behaviors or {}
        goals_now = _human_goal_map(human_controller, human_goals)
        out: dict[int, AgentState] = {}

        rx, ry = backend.get_position(robot_id)
        rvx, rvy = backend.get_velocity(robot_id)
        out[robot_id] = AgentState(
            agent_id=robot_id,
            kind="robot",
            x=rx,
            y=ry,
            vx=rvx,
            vy=rvy,
            goal_x=float(scenario["robot"]["goal"][0]),
            goal_y=float(scenario["robot"]["goal"][1]),
            radius=robot_radius,
            max_speed=robot_max_speed,
            behavior=behaviors.get(robot_id, "GO_TO"),
        )

        for hid in human_ids:
            hx, hy = backend.get_position(hid)
            hvx, hvy = backend.get_velocity(hid)
            gx, gy = goals_now.get(hid, human_goals[hid])
            out[hid] = AgentState(
                agent_id=hid,
                kind="human",
                x=hx,
                y=hy,
                vx=hvx,
                vy=hvy,
                goal_x=float(gx),
                goal_y=float(gy),
                radius=human_radius,
                max_speed=human_max_speed,
                behavior=behaviors.get(hid, "GO_TO"),
            )

        return out

    # initial frame
    init_states = capture_states({})
    logger.write_state(0, 0.0, [init_states[aid] for aid in sorted(init_states)])

    for step_idx in range(steps):
        current_step = step_idx
        time_s = step_idx * dt
        pre_states = capture_states({})

        human_actions = human_controller.step(
            step=step_idx,
            time_s=time_s,
            dt=dt,
            states=pre_states,
            robot_id=robot_id,
            emit_event=emit_event,
        )
        robot_action = robot_controller.step(
            step=step_idx,
            time_s=time_s,
            dt=dt,
            states=pre_states,
            emit_event=emit_event,
        )

        behaviors = {robot_id: robot_action.behavior}
        for hid, action in human_actions.items():
            behaviors[hid] = action.behavior

        backend.set_preferred_velocity(robot_id, (robot_action.pref_vx, robot_action.pref_vy))
        for hid in human_ids:
            action = human_actions.get(hid)
            if action is None:
                action = type(robot_action)(0.0, 0.0, behavior="WAIT")
            backend.set_preferred_velocity(hid, (action.pref_vx, action.pref_vy))

        backend.step()

        post_states = capture_states(behaviors)
        logger.write_state(
            step_idx + 1, (step_idx + 1) * dt, [post_states[aid] for aid in sorted(post_states)]
        )

    logger.close()

    state_path = bundle_dir / "state.jsonl"
    metrics = StandardMetrics().compute(state_path=state_path, scenario=scenario)
    deadlock_count = int(metrics.get("deadlock_count", 0))

    if deadlock_count > 0 and auto_resample_on_deadlock and retry_attempt < retry_max_attempts:
        retry_history.append(
            {
                "attempt": retry_attempt,
                "reason": "deadlock_detected",
                "deadlock_count": deadlock_count,
            }
        )
        meta["deadlock_retry_history"] = retry_history
        meta["deadlock_retry_attempt"] = retry_attempt + 1
        return run_scenario_dict(
            scenario=scenario,
            out_root=out_root,
            run_id=run_id,
            render_override=render_override,
            video_override=video_override,
        )
    if deadlock_count > 0 and fail_on_deadlock:
        raise ValueError(
            f"Deadlock remained after retry budget ({retry_max_attempts}) for "
            f"scenario '{scenario_id}'. Resample starts/goals, increase traversability "
            "offset budget, or reduce agent radius."
        )

    # Finalized attempt; keep only trace metadata and clear retry counter.
    meta["deadlock_retry_attempt"] = 0
    if retry_history:
        meta["deadlock_retry_history"] = retry_history

    summary = {
        "scenario_id": scenario_id,
        "run_id": run_id,
        "seed": seed,
        "attempt_seed": attempt_seed,
        "retry_attempt": int(retry_attempt),
        "retry_history": retry_history,
        "map": map_metadata,
        "anchor_adjustments": anchor_adjust,
        "metrics": metrics,
        "events": {
            "count": len(events),
        },
        "artifacts": {
            "scenario": str(bundle_dir / "scenario.yaml"),
            "state": str(bundle_dir / "state.jsonl"),
            "events": str(bundle_dir / "events.jsonl"),
        },
    }

    render_cfg = dict(scenario.get("render", {}))
    if render_override is not None:
        render_cfg["enabled"] = bool(render_override)
    if video_override is not None:
        render_cfg["video"] = bool(video_override)

    if render_cfg.get("enabled", True):
        render_info = render_trace(
            state_path=state_path,
            out_dir=bundle_dir / "frames",
            fps=int(render_cfg.get("fps", 12)),
            video=bool(render_cfg.get("video", False)),
        )
        summary["artifacts"].update(render_info)

    with (bundle_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    return EpisodeLog(
        run_id=run_id,
        bundle_dir=str(bundle_dir),
        scenario_path=str(bundle_dir / "scenario.yaml"),
        state_path=str(bundle_dir / "state.jsonl"),
        events_path=str(bundle_dir / "events.jsonl"),
        summary_path=str(bundle_dir / "summary.json"),
    )


def run_scenario_file(
    scenario_path: str | Path,
    out_root: str | Path,
    run_id: str | None = None,
    render_override: bool | None = None,
    video_override: bool | None = None,
) -> EpisodeLog:
    scenario = load_scenario(scenario_path)
    return run_scenario_dict(
        scenario=scenario,
        out_root=out_root,
        run_id=run_id,
        render_override=render_override,
        video_override=video_override,
    )


def expand_state_paths(inputs: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in inputs:
        p = Path(raw)
        if any(ch in raw for ch in ["*", "?", "["]):
            out.extend(sorted(Path().glob(raw)))
            continue

        if p.is_dir():
            # bundle dir or run dir
            cand = p / "state.jsonl"
            if cand.exists():
                out.append(cand)
                continue
            cand = p / "bundle" / "state.jsonl"
            if cand.exists():
                out.append(cand)
                continue
            out.extend(sorted(p.rglob("state.jsonl")))
            continue

        if p.is_file():
            out.append(p)
            continue

    # dedupe preserving order
    seen = set()
    deduped = []
    for p in out:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            deduped.append(Path(rp))
    return deduped


def _run_scenario_worker(task_data: tuple) -> EpisodeLog:
    """Worker function for parallel batch execution.

    Args:
        task_data: Tuple of (scenario_path, seed, out_root, render_override, video_override)

    Returns:
        EpisodeLog for the completed scenario run
    """
    scenario_path, seed, out_root, render_override, video_override = task_data

    scenario = load_scenario(scenario_path)
    scenario["seed"] = seed
    run_id = f"{scenario['id']}_seed{seed}"

    return run_scenario_dict(
        scenario=scenario,
        out_root=out_root,
        run_id=run_id,
        render_override=render_override,
        video_override=video_override,
    )


def run_batch(args: argparse.Namespace) -> list[EpisodeLog]:
    """Run multiple scenarios in batch mode with parallel execution support.

    Discovers all scenario YAML files in the specified directory and executes them
    across multiple random seeds. Supports both sequential and parallel execution.

    Args:
        args: Argparse namespace containing batch execution parameters:
            - scenarios: Directory path containing scenario YAML files
            - seeds: Comma-separated string of random seeds to use
            - out: Output directory for simulation results
            - render: Whether to enable rendering during simulation
            - video: Whether to record video output
            - parallel: Number of parallel processes (<=1 for sequential)

    Returns:
        List of EpisodeLog objects, one for each scenario-seed combination.

    Raises:
        FileNotFoundError: If no scenario YAML files are found in the directory.
    """
    scenario_dir = Path(args.scenarios)
    scenario_files = sorted(scenario_dir.rglob("*.yaml"))
    if not scenario_files:
        msg = f"No scenario YAML files found under {scenario_dir}"
        raise FileNotFoundError(msg)

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Prepare all tasks (scenario, seed combinations)
    tasks = []
    for scenario_path in scenario_files:
        for seed in seeds:
            tasks.append((scenario_path, seed, args.out, args.render, args.video))

    # Execute tasks in parallel or sequentially based on args.parallel
    if args.parallel <= 1:
        # Sequential execution (maintains backward compatibility)
        logs = [_run_scenario_worker(task) for task in tasks]
    else:
        # Parallel execution using multiprocessing
        num_processes = min(args.parallel, len(tasks), mp.cpu_count())
        with mp.Pool(processes=num_processes) as pool:
            logs = pool.map(_run_scenario_worker, tasks)

    return logs
