"""Full pedestrian state representation with history tracking and prediction.

Provides a rich state container for individual pedestrians that goes beyond
the minimal :class:`~navirl.core.types.AgentState` used by the simulation
loop.  Includes kinematic quantities, social attributes, activity labels,
comfort/stress indicators, and utilities for state history and linear
prediction.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Activity(enum.Enum):
    """Discrete activity labels for a pedestrian."""

    WALKING = "walking"
    STANDING = "standing"
    SITTING = "sitting"
    TALKING = "talking"
    RUNNING = "running"
    WAITING = "waiting"
    BROWSING_PHONE = "browsing_phone"


class PersonalityTag(enum.Enum):
    """Coarse personality archetype tags."""

    AGGRESSIVE = "aggressive"
    PASSIVE = "passive"
    DISTRACTED = "distracted"
    HURRIED = "hurried"
    NORMAL = "normal"
    ELDERLY = "elderly"
    CHILD = "child"


# ---------------------------------------------------------------------------
# Gaze direction helper
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class GazeDirection:
    """2-D gaze direction encoded as unit vector or angle.

    Parameters
    ----------
    angle_rad : float
        Gaze angle in radians (0 = positive-x, pi/2 = positive-y).
    """

    angle_rad: float = 0.0

    # -- derived properties --------------------------------------------------

    @property
    def unit_vector(self) -> tuple[float, float]:
        """Return the unit vector (ux, uy) for the current gaze angle."""
        return (math.cos(self.angle_rad), math.sin(self.angle_rad))

    def set_from_vector(self, vx: float, vy: float) -> None:
        """Set the gaze angle from an arbitrary direction vector.

        Parameters
        ----------
        vx, vy : float
            Direction vector (does not need to be unit-length).
        """
        if abs(vx) + abs(vy) > 1e-12:
            self.angle_rad = math.atan2(vy, vx)

    def angular_distance(self, other_angle: float) -> float:
        """Signed angular distance to *other_angle* wrapped to (-pi, pi].

        Parameters
        ----------
        other_angle : float
            Target angle in radians.
        """
        d = other_angle - self.angle_rad
        while d > math.pi:
            d -= 2.0 * math.pi
        while d <= -math.pi:
            d += 2.0 * math.pi
        return d


# ---------------------------------------------------------------------------
# Core pedestrian state
# ---------------------------------------------------------------------------


@dataclass
class PedestrianState:
    """Rich state representation for a single pedestrian.

    Attributes
    ----------
    pid : int
        Unique pedestrian identifier.
    position : numpy.ndarray
        2-D position ``[x, y]`` in world coordinates (metres).
    velocity : numpy.ndarray
        2-D velocity ``[vx, vy]`` (m/s).
    acceleration : numpy.ndarray
        2-D acceleration ``[ax, ay]`` (m/s^2).
    heading : float
        Body heading in radians.
    goal : numpy.ndarray
        Current 2-D goal position ``[gx, gy]``.
    intended_velocity : numpy.ndarray
        Desired (preferred) velocity before collision avoidance.
    max_speed : float
        Maximum walking speed (m/s).
    preferred_speed : float
        Comfortable cruising speed (m/s).
    radius : float
        Physical body radius (m).
    personal_space_radius : float
        Desired personal-space radius (m).  Typically larger than *radius*.
    group_id : int or None
        Identifier of the social group this pedestrian belongs to, or ``None``.
    personality : PersonalityTag
        Coarse personality archetype.
    activity : Activity
        Current discrete activity label.
    gaze : GazeDirection
        Current gaze direction.
    comfort_level : float
        Normalised comfort score in ``[0, 1]``.  1 = fully comfortable.
    stress_level : float
        Normalised stress score in ``[0, 1]``.  0 = no stress.
    metadata : dict
        Arbitrary key-value metadata for extensions.
    """

    pid: int = 0
    position: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    acceleration: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    heading: float = 0.0
    goal: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    intended_velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    max_speed: float = 1.5
    preferred_speed: float = 1.2
    radius: float = 0.3
    personal_space_radius: float = 0.6
    group_id: int | None = None
    personality: PersonalityTag = PersonalityTag.NORMAL
    activity: Activity = Activity.WALKING
    gaze: GazeDirection = field(default_factory=GazeDirection)
    comfort_level: float = 1.0
    stress_level: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- convenience properties ----------------------------------------------

    @property
    def speed(self) -> float:
        """Scalar speed (m/s)."""
        return float(np.linalg.norm(self.velocity))

    @property
    def x(self) -> float:
        """X coordinate shortcut."""
        return float(self.position[0])

    @property
    def y(self) -> float:
        """Y coordinate shortcut."""
        return float(self.position[1])

    @property
    def vx(self) -> float:
        """X velocity shortcut."""
        return float(self.velocity[0])

    @property
    def vy(self) -> float:
        """Y velocity shortcut."""
        return float(self.velocity[1])

    # -- state manipulation --------------------------------------------------

    def update_heading_from_velocity(self) -> None:
        """Set *heading* to match the current velocity direction.

        No-op when speed is near zero.
        """
        sp = self.speed
        if sp > 1e-8:
            self.heading = float(math.atan2(self.velocity[1], self.velocity[0]))

    def distance_to(self, other: PedestrianState) -> float:
        """Euclidean distance to *other* pedestrian (centre-to-centre).

        Parameters
        ----------
        other : PedestrianState
            The other pedestrian state.
        """
        return float(np.linalg.norm(self.position - other.position))

    def bearing_to(self, other: PedestrianState) -> float:
        """Bearing angle (radians) from this pedestrian to *other*.

        Parameters
        ----------
        other : PedestrianState
            The other pedestrian state.
        """
        diff = other.position - self.position
        return float(math.atan2(diff[1], diff[0]))

    def in_personal_space(self, other: PedestrianState) -> bool:
        """Return ``True`` if *other* intrudes into this pedestrian's personal space.

        Parameters
        ----------
        other : PedestrianState
            The other pedestrian state.
        """
        return self.distance_to(other) < self.personal_space_radius + other.radius

    def distance_to_goal(self) -> float:
        """Euclidean distance to the current goal."""
        return float(np.linalg.norm(self.position - self.goal))

    def clone(self) -> PedestrianState:
        """Return a deep copy of this state."""
        return PedestrianState(
            pid=self.pid,
            position=self.position.copy(),
            velocity=self.velocity.copy(),
            acceleration=self.acceleration.copy(),
            heading=self.heading,
            goal=self.goal.copy(),
            intended_velocity=self.intended_velocity.copy(),
            max_speed=self.max_speed,
            preferred_speed=self.preferred_speed,
            radius=self.radius,
            personal_space_radius=self.personal_space_radius,
            group_id=self.group_id,
            personality=self.personality,
            activity=self.activity,
            gaze=GazeDirection(angle_rad=self.gaze.angle_rad),
            comfort_level=self.comfort_level,
            stress_level=self.stress_level,
            metadata=dict(self.metadata),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise the state to a plain dictionary.

        Returns
        -------
        dict
            JSON-friendly dictionary with all public attributes.
        """
        return {
            "pid": self.pid,
            "x": self.x,
            "y": self.y,
            "vx": self.vx,
            "vy": self.vy,
            "ax": float(self.acceleration[0]),
            "ay": float(self.acceleration[1]),
            "heading": self.heading,
            "goal_x": float(self.goal[0]),
            "goal_y": float(self.goal[1]),
            "intended_vx": float(self.intended_velocity[0]),
            "intended_vy": float(self.intended_velocity[1]),
            "max_speed": self.max_speed,
            "preferred_speed": self.preferred_speed,
            "radius": self.radius,
            "personal_space_radius": self.personal_space_radius,
            "group_id": self.group_id,
            "personality": self.personality.value,
            "activity": self.activity.value,
            "gaze_angle": self.gaze.angle_rad,
            "comfort_level": self.comfort_level,
            "stress_level": self.stress_level,
            "speed": self.speed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PedestrianState:
        """Construct a :class:`PedestrianState` from a plain dictionary.

        Parameters
        ----------
        d : dict
            Dictionary produced by :meth:`to_dict` or compatible.
        """
        state = cls(
            pid=int(d.get("pid", 0)),
            position=np.array([d.get("x", 0.0), d.get("y", 0.0)], dtype=np.float64),
            velocity=np.array([d.get("vx", 0.0), d.get("vy", 0.0)], dtype=np.float64),
            acceleration=np.array([d.get("ax", 0.0), d.get("ay", 0.0)], dtype=np.float64),
            heading=float(d.get("heading", 0.0)),
            goal=np.array([d.get("goal_x", 0.0), d.get("goal_y", 0.0)], dtype=np.float64),
            intended_velocity=np.array(
                [d.get("intended_vx", 0.0), d.get("intended_vy", 0.0)], dtype=np.float64
            ),
            max_speed=float(d.get("max_speed", 1.5)),
            preferred_speed=float(d.get("preferred_speed", 1.2)),
            radius=float(d.get("radius", 0.3)),
            personal_space_radius=float(d.get("personal_space_radius", 0.6)),
            group_id=d.get("group_id"),
            comfort_level=float(d.get("comfort_level", 1.0)),
            stress_level=float(d.get("stress_level", 0.0)),
        )
        personality_str = d.get("personality", "normal")
        try:
            state.personality = PersonalityTag(personality_str)
        except ValueError:
            state.personality = PersonalityTag.NORMAL
        activity_str = d.get("activity", "walking")
        try:
            state.activity = Activity(activity_str)
        except ValueError:
            state.activity = Activity.WALKING
        state.gaze = GazeDirection(angle_rad=float(d.get("gaze_angle", 0.0)))
        return state


# ---------------------------------------------------------------------------
# State history tracker
# ---------------------------------------------------------------------------


class StateHistory:
    """Fixed-capacity ring-buffer that stores past :class:`PedestrianState` snapshots.

    Parameters
    ----------
    capacity : int
        Maximum number of snapshots to retain.
    """

    def __init__(self, capacity: int = 200) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._capacity: int = capacity
        self._buffer: list[PedestrianState] = []
        self._timestamps: list[float] = []

    # -- public API -----------------------------------------------------------

    @property
    def capacity(self) -> int:
        """Maximum number of entries the buffer can hold."""
        return self._capacity

    def __len__(self) -> int:
        return len(self._buffer)

    def record(self, state: PedestrianState, time_s: float) -> None:
        """Append a state snapshot.

        Parameters
        ----------
        state : PedestrianState
            The state to record (a clone is stored).
        time_s : float
            Simulation time at which the snapshot was taken.
        """
        if len(self._buffer) >= self._capacity:
            self._buffer.pop(0)
            self._timestamps.pop(0)
        self._buffer.append(state.clone())
        self._timestamps.append(time_s)

    def latest(self) -> PedestrianState | None:
        """Return the most recent state, or ``None`` if empty."""
        return self._buffer[-1] if self._buffer else None

    def at(self, index: int) -> tuple[PedestrianState, float]:
        """Return the *(state, timestamp)* pair at *index*.

        Parameters
        ----------
        index : int
            Index into the buffer (negative indices are supported).
        """
        return self._buffer[index], self._timestamps[index]

    def positions_array(self) -> np.ndarray:
        """Return an ``(N, 2)`` array of recorded positions.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, 2)`` float64 array.
        """
        if not self._buffer:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([s.position for s in self._buffer], dtype=np.float64)

    def velocities_array(self) -> np.ndarray:
        """Return an ``(N, 2)`` array of recorded velocities.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, 2)`` float64 array.
        """
        if not self._buffer:
            return np.empty((0, 2), dtype=np.float64)
        return np.array([s.velocity for s in self._buffer], dtype=np.float64)

    def timestamps_array(self) -> np.ndarray:
        """Return a 1-D array of timestamps.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)`` float64 array.
        """
        return np.array(self._timestamps, dtype=np.float64)

    def speed_array(self) -> np.ndarray:
        """Return a 1-D array of scalar speeds.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)`` float64 array.
        """
        vels = self.velocities_array()
        if vels.shape[0] == 0:
            return np.empty(0, dtype=np.float64)
        return np.linalg.norm(vels, axis=1)

    def clear(self) -> None:
        """Remove all recorded snapshots."""
        self._buffer.clear()
        self._timestamps.clear()

    def window(self, last_n: int) -> list[tuple[PedestrianState, float]]:
        """Return the last *last_n* entries as a list of ``(state, time)`` pairs.

        Parameters
        ----------
        last_n : int
            Number of most recent entries to return.
        """
        n = min(last_n, len(self._buffer))
        return [(self._buffer[-n + i], self._timestamps[-n + i]) for i in range(n)]

    def path_length(self) -> float:
        """Compute the cumulative path length from recorded positions.

        Returns
        -------
        float
            Total path length in metres.
        """
        pos = self.positions_array()
        if pos.shape[0] < 2:
            return 0.0
        diffs = np.diff(pos, axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def mean_speed(self) -> float:
        """Mean scalar speed across all recorded states.

        Returns
        -------
        float
            Mean speed in m/s.  Returns 0.0 if no history.
        """
        sp = self.speed_array()
        if sp.shape[0] == 0:
            return 0.0
        return float(np.mean(sp))

    def comfort_array(self) -> np.ndarray:
        """Return a 1-D array of comfort levels.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)`` float64 array.
        """
        if not self._buffer:
            return np.empty(0, dtype=np.float64)
        return np.array([s.comfort_level for s in self._buffer], dtype=np.float64)

    def stress_array(self) -> np.ndarray:
        """Return a 1-D array of stress levels.

        Returns
        -------
        numpy.ndarray
            Shape ``(N,)`` float64 array.
        """
        if not self._buffer:
            return np.empty(0, dtype=np.float64)
        return np.array([s.stress_level for s in self._buffer], dtype=np.float64)


# ---------------------------------------------------------------------------
# State prediction
# ---------------------------------------------------------------------------


class StatePredictor:
    """Linear-extrapolation predictor for pedestrian state.

    Uses position, velocity and (optionally) acceleration recorded in a
    :class:`StateHistory` to forecast future positions.

    Parameters
    ----------
    use_acceleration : bool
        If ``True``, apply constant-acceleration model; otherwise use
        constant-velocity.
    """

    def __init__(self, use_acceleration: bool = False) -> None:
        self.use_acceleration: bool = use_acceleration

    def predict_position(self, state: PedestrianState, dt: float) -> np.ndarray:
        """Predict position after *dt* seconds from the given *state*.

        Parameters
        ----------
        state : PedestrianState
            Current state.
        dt : float
            Look-ahead time horizon in seconds.

        Returns
        -------
        numpy.ndarray
            Predicted ``[x, y]`` position.
        """
        pos = state.position.copy()
        pos += state.velocity * dt
        if self.use_acceleration:
            pos += 0.5 * state.acceleration * dt * dt
        return pos

    def predict_trajectory(
        self,
        state: PedestrianState,
        horizon: float,
        step_dt: float,
    ) -> np.ndarray:
        """Predict a trajectory as a sequence of positions.

        Parameters
        ----------
        state : PedestrianState
            Current state.
        horizon : float
            Total prediction horizon in seconds.
        step_dt : float
            Time step between successive predicted positions.

        Returns
        -------
        numpy.ndarray
            Shape ``(K, 2)`` array of predicted positions where
            ``K = floor(horizon / step_dt) + 1``.
        """
        n_steps = max(1, int(math.floor(horizon / step_dt))) + 1
        traj = np.empty((n_steps, 2), dtype=np.float64)
        for i in range(n_steps):
            t = i * step_dt
            traj[i] = self.predict_position(state, t)
        return traj

    def predict_from_history(
        self,
        history: StateHistory,
        dt: float,
    ) -> np.ndarray | None:
        """Predict the next position using linear regression on recent history.

        Fits a linear model to the last positions in *history* and
        extrapolates by *dt*.  Falls back to constant-velocity if fewer
        than three history samples are available.

        Parameters
        ----------
        history : StateHistory
            Recorded state history.
        dt : float
            Look-ahead time in seconds.

        Returns
        -------
        numpy.ndarray or None
            Predicted ``[x, y]`` or ``None`` if the history is empty.
        """
        if len(history) == 0:
            return None

        if len(history) < 3:
            latest = history.latest()
            assert latest is not None
            return self.predict_position(latest, dt)

        ts = history.timestamps_array()
        pos = history.positions_array()

        # Use last 10 samples at most for the fit.
        n = min(len(ts), 10)
        ts_fit = ts[-n:]
        pos_fit = pos[-n:]

        # Normalise time so the fit is numerically stable.
        t0 = ts_fit[0]
        ts_norm = ts_fit - t0

        # Linear regression for x and y separately.
        predicted = np.empty(2, dtype=np.float64)
        t_predict = ts_norm[-1] + dt
        for dim in range(2):
            coeffs = np.polyfit(ts_norm, pos_fit[:, dim], deg=1)
            predicted[dim] = np.polyval(coeffs, t_predict)

        return predicted

    def collision_time(
        self,
        state_a: PedestrianState,
        state_b: PedestrianState,
        max_horizon: float = 10.0,
    ) -> float | None:
        """Estimate time to collision between two pedestrians (constant-velocity).

        Parameters
        ----------
        state_a, state_b : PedestrianState
            States of the two pedestrians.
        max_horizon : float
            Maximum look-ahead time (seconds).

        Returns
        -------
        float or None
            Estimated collision time, or ``None`` if no collision within the
            horizon.
        """
        dp = state_b.position - state_a.position
        dv = state_b.velocity - state_a.velocity
        combined_r = state_a.radius + state_b.radius

        a = float(np.dot(dv, dv))
        b = 2.0 * float(np.dot(dp, dv))
        c = float(np.dot(dp, dp)) - combined_r * combined_r

        if a < 1e-12:
            # Parallel velocities - check if already overlapping.
            return 0.0 if c <= 0.0 else None

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0.0:
            return None

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # We want the earliest positive collision time.
        for t in (t1, t2):
            if 0.0 <= t <= max_horizon:
                return t
        return None


# ---------------------------------------------------------------------------
# Batch utilities
# ---------------------------------------------------------------------------


def states_to_array(states: list[PedestrianState]) -> np.ndarray:
    """Pack a list of pedestrian states into a structured numpy array.

    Each row contains ``[pid, x, y, vx, vy, heading, speed, radius]``.

    Parameters
    ----------
    states : list[PedestrianState]
        States to pack.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 8)`` float64 array.
    """
    n = len(states)
    out = np.empty((n, 8), dtype=np.float64)
    for i, s in enumerate(states):
        out[i, 0] = s.pid
        out[i, 1] = s.position[0]
        out[i, 2] = s.position[1]
        out[i, 3] = s.velocity[0]
        out[i, 4] = s.velocity[1]
        out[i, 5] = s.heading
        out[i, 6] = s.speed
        out[i, 7] = s.radius
    return out


def pairwise_distances(states: list[PedestrianState]) -> np.ndarray:
    """Compute pairwise centre-to-centre distance matrix.

    Parameters
    ----------
    states : list[PedestrianState]
        List of pedestrian states.

    Returns
    -------
    numpy.ndarray
        Shape ``(N, N)`` symmetric distance matrix.
    """
    n = len(states)
    if n == 0:
        return np.empty((0, 0), dtype=np.float64)
    pos = np.array([s.position for s in states], dtype=np.float64)
    diff = pos[:, None, :] - pos[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=-1))


def filter_by_activity(states: list[PedestrianState], activity: Activity) -> list[PedestrianState]:
    """Return only states matching the given *activity*.

    Parameters
    ----------
    states : list[PedestrianState]
        Input states.
    activity : Activity
        Activity to filter on.

    Returns
    -------
    list[PedestrianState]
        Filtered list (references, not copies).
    """
    return [s for s in states if s.activity == activity]


def filter_by_group(states: list[PedestrianState], group_id: int) -> list[PedestrianState]:
    """Return only states belonging to *group_id*.

    Parameters
    ----------
    states : list[PedestrianState]
        Input states.
    group_id : int
        Group identifier.

    Returns
    -------
    list[PedestrianState]
        Filtered list (references, not copies).
    """
    return [s for s in states if s.group_id == group_id]


def compute_centroid(states: list[PedestrianState]) -> np.ndarray:
    """Compute the mean position (centroid) of a set of pedestrians.

    Parameters
    ----------
    states : list[PedestrianState]
        Non-empty list of states.

    Returns
    -------
    numpy.ndarray
        ``[cx, cy]`` centroid position.

    Raises
    ------
    ValueError
        If *states* is empty.
    """
    if not states:
        raise ValueError("Cannot compute centroid of empty list")
    pos = np.array([s.position for s in states], dtype=np.float64)
    return pos.mean(axis=0)
