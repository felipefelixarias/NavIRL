"""Event system for NavIRL simulation.

Implements a publish / subscribe :class:`EventBus` with typed events,
filtering, recording, and replay capabilities.
"""

from __future__ import annotations

import enum
import time
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import (
    Any,
)

# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(enum.Enum):
    """Built-in simulation event types."""

    COLLISION = "collision"
    GOAL_REACHED = "goal_reached"
    TIMEOUT = "timeout"
    ZONE_ENTER = "zone_enter"
    ZONE_EXIT = "zone_exit"
    ENTITY_ADDED = "entity_added"
    ENTITY_REMOVED = "entity_removed"
    SIMULATION_START = "simulation_start"
    SIMULATION_END = "simulation_end"
    EPISODE_START = "episode_start"
    EPISODE_END = "episode_end"
    STEP = "step"
    DOOR_OPEN = "door_open"
    DOOR_CLOSE = "door_close"
    WAYPOINT_REACHED = "waypoint_reached"
    CUSTOM = "custom"


# ---------------------------------------------------------------------------
# Event record
# ---------------------------------------------------------------------------


@dataclass
class EventRecord:
    """Immutable record of a single fired event.

    Parameters
    ----------
    event_type : EventType
        The type of event.
    sim_time : float
        Simulation time at which the event occurred.
    wall_time : float
        Wall-clock time at which the event occurred.
    source_id : int
        Entity id that originated the event (``-1`` for global events).
    target_id : int
        Entity id that is the target of the event (``-1`` if N/A).
    data : dict
        Arbitrary payload.
    """

    event_type: EventType
    sim_time: float = 0.0
    wall_time: float = 0.0
    source_id: int = -1
    target_id: int = -1
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dict."""
        return {
            "event_type": self.event_type.value,
            "sim_time": self.sim_time,
            "wall_time": self.wall_time,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "data": self.data,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> EventRecord:
        """Reconstruct from dict."""
        return cls(
            event_type=EventType(d["event_type"]),
            sim_time=d.get("sim_time", 0.0),
            wall_time=d.get("wall_time", 0.0),
            source_id=d.get("source_id", -1),
            target_id=d.get("target_id", -1),
            data=d.get("data", {}),
        )


# ---------------------------------------------------------------------------
# Subscription handle
# ---------------------------------------------------------------------------


@dataclass
class _Subscription:
    """Internal subscription entry."""

    callback: Callable[[EventRecord], None]
    event_type: EventType | None  # None = wildcard
    entity_filter: int | None  # filter by source or target id
    tag_filter: str | None  # filter by data["tag"]
    once: bool
    active: bool = True
    _id: int = 0


# ---------------------------------------------------------------------------
# EventFilter helper
# ---------------------------------------------------------------------------


class EventFilter:
    """Composable predicate for filtering :class:`EventRecord` instances.

    Filters can be combined with ``&`` (AND) and ``|`` (OR).
    """

    def __init__(
        self,
        event_types: set[EventType] | None = None,
        source_ids: set[int] | None = None,
        target_ids: set[int] | None = None,
        min_sim_time: float | None = None,
        max_sim_time: float | None = None,
        data_key: str | None = None,
        data_value: Any = None,
    ) -> None:
        self.event_types = event_types
        self.source_ids = source_ids
        self.target_ids = target_ids
        self.min_sim_time = min_sim_time
        self.max_sim_time = max_sim_time
        self.data_key = data_key
        self.data_value = data_value

    def matches(self, record: EventRecord) -> bool:
        """Return ``True`` if *record* passes this filter."""
        if self.event_types is not None and record.event_type not in self.event_types:
            return False
        if self.source_ids is not None and record.source_id not in self.source_ids:
            return False
        if self.target_ids is not None and record.target_id not in self.target_ids:
            return False
        if self.min_sim_time is not None and record.sim_time < self.min_sim_time:
            return False
        if self.max_sim_time is not None and record.sim_time > self.max_sim_time:
            return False
        if self.data_key is not None:
            val = record.data.get(self.data_key)
            if self.data_value is not None and val != self.data_value:
                return False
            if self.data_value is None and val is None:
                return False
        return True

    def __and__(self, other: EventFilter) -> _CombinedFilter:
        return _CombinedFilter([self, other], mode="and")

    def __or__(self, other: EventFilter) -> _CombinedFilter:
        return _CombinedFilter([self, other], mode="or")


class _CombinedFilter(EventFilter):
    """Logical combination of multiple filters."""

    def __init__(self, filters: list[EventFilter], mode: str = "and") -> None:
        super().__init__()
        self._filters = filters
        self._mode = mode

    def matches(self, record: EventRecord) -> bool:
        if self._mode == "and":
            return all(f.matches(record) for f in self._filters)
        return any(f.matches(record) for f in self._filters)


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------


class EventBus:
    """Publish / subscribe event bus for simulation events.

    Features:
    * Subscribe to specific event types or all events (wildcard).
    * Filter subscriptions by source / target entity id or data tags.
    * One-shot subscriptions.
    * Full event recording for later replay or analysis.
    * Event replay at original or modified speed.
    """

    def __init__(self, recording: bool = True) -> None:
        self._subs: list[_Subscription] = []
        self._sub_index: dict[EventType | None, list[_Subscription]] = defaultdict(list)
        self._next_sub_id: int = 0
        self._recording = recording
        self._history: list[EventRecord] = []
        self._paused: bool = False
        self._muted_types: set[EventType] = set()
        self._publish_count: int = 0
        self._replay_callbacks: list[Callable[[EventRecord], None]] = []

    # ------------------------------------------------------------------
    # Subscribe
    # ------------------------------------------------------------------

    def subscribe(
        self,
        callback: Callable[[EventRecord], None],
        event_type: EventType | None = None,
        entity_filter: int | None = None,
        tag_filter: str | None = None,
        once: bool = False,
    ) -> int:
        """Register a callback.

        Parameters
        ----------
        callback : callable
            Receives an :class:`EventRecord` when fired.
        event_type : EventType or None
            Subscribe to a specific type or all (``None``).
        entity_filter : int or None
            Only fire if *source_id* or *target_id* matches.
        tag_filter : str or None
            Only fire if ``data["tag"]`` matches.
        once : bool
            Auto-unsubscribe after first delivery.

        Returns
        -------
        int
            Subscription id (use with :meth:`unsubscribe`).
        """
        sid = self._next_sub_id
        self._next_sub_id += 1
        sub = _Subscription(
            callback=callback,
            event_type=event_type,
            entity_filter=entity_filter,
            tag_filter=tag_filter,
            once=once,
            _id=sid,
        )
        self._subs.append(sub)
        self._sub_index[event_type].append(sub)
        return sid

    def unsubscribe(self, sub_id: int) -> bool:
        """Remove a subscription by its id.  Returns ``True`` if found."""
        for sub in self._subs:
            if sub._id == sub_id:
                sub.active = False
                return True
        return False

    def unsubscribe_all(self) -> None:
        """Remove every subscription."""
        self._subs.clear()
        self._sub_index.clear()

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------

    def publish(
        self,
        event_type: EventType,
        sim_time: float = 0.0,
        source_id: int = -1,
        target_id: int = -1,
        data: dict[str, Any] | None = None,
    ) -> EventRecord:
        """Publish an event and deliver to matching subscribers.

        Returns the :class:`EventRecord`.
        """
        record = EventRecord(
            event_type=event_type,
            sim_time=sim_time,
            wall_time=time.monotonic(),
            source_id=source_id,
            target_id=target_id,
            data=data or {},
        )
        self._publish_count += 1

        if self._recording:
            self._history.append(record)

        if self._paused:
            return record

        if event_type in self._muted_types:
            return record

        self._deliver(record)
        return record

    def _deliver(self, record: EventRecord) -> None:
        """Deliver *record* to matching subscriptions."""
        # Gather matching subs: wildcard + type-specific
        candidates = list(self._sub_index.get(None, []))
        candidates.extend(self._sub_index.get(record.event_type, []))

        to_remove: list[_Subscription] = []
        for sub in candidates:
            if not sub.active:
                continue
            # Entity filter
            if sub.entity_filter is not None and (
                record.source_id != sub.entity_filter and record.target_id != sub.entity_filter
            ):
                continue
            # Tag filter
            if sub.tag_filter is not None and record.data.get("tag") != sub.tag_filter:
                continue
            sub.callback(record)
            if sub.once:
                sub.active = False
                to_remove.append(sub)

        # Lazy cleanup
        if to_remove:
            self._subs = [s for s in self._subs if s.active]
            for key in self._sub_index:
                self._sub_index[key] = [s for s in self._sub_index[key] if s.active]

    # ------------------------------------------------------------------
    # Convenience publishers
    # ------------------------------------------------------------------

    def emit_collision(
        self,
        sim_time: float,
        entity_a: int,
        entity_b: int,
        penetration: float = 0.0,
        normal: Sequence[float] | None = None,
    ) -> EventRecord:
        """Emit a collision event."""
        data: dict[str, Any] = {
            "penetration": penetration,
        }
        if normal is not None:
            data["normal"] = list(normal)
        return self.publish(
            EventType.COLLISION,
            sim_time=sim_time,
            source_id=entity_a,
            target_id=entity_b,
            data=data,
        )

    def emit_goal_reached(self, sim_time: float, entity_id: int, goal_id: int = -1) -> EventRecord:
        """Emit a goal-reached event."""
        return self.publish(
            EventType.GOAL_REACHED,
            sim_time=sim_time,
            source_id=entity_id,
            target_id=goal_id,
        )

    def emit_timeout(self, sim_time: float) -> EventRecord:
        """Emit a timeout event."""
        return self.publish(EventType.TIMEOUT, sim_time=sim_time)

    def emit_zone_enter(self, sim_time: float, entity_id: int, zone_id: int) -> EventRecord:
        """Emit a zone-enter event."""
        return self.publish(
            EventType.ZONE_ENTER,
            sim_time=sim_time,
            source_id=entity_id,
            target_id=zone_id,
        )

    def emit_zone_exit(self, sim_time: float, entity_id: int, zone_id: int) -> EventRecord:
        """Emit a zone-exit event."""
        return self.publish(
            EventType.ZONE_EXIT,
            sim_time=sim_time,
            source_id=entity_id,
            target_id=zone_id,
        )

    # ------------------------------------------------------------------
    # Mute / unmute
    # ------------------------------------------------------------------

    def mute(self, event_type: EventType) -> None:
        """Suppress delivery of *event_type* (still recorded)."""
        self._muted_types.add(event_type)

    def unmute(self, event_type: EventType) -> None:
        """Re-enable delivery of *event_type*."""
        self._muted_types.discard(event_type)

    def pause(self) -> None:
        """Pause all event delivery (recording continues)."""
        self._paused = True

    def resume(self) -> None:
        """Resume event delivery."""
        self._paused = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    @property
    def history(self) -> list[EventRecord]:
        """Full event history."""
        return self._history

    def clear_history(self) -> None:
        """Discard recorded events."""
        self._history.clear()

    def set_recording(self, enabled: bool) -> None:
        """Enable or disable event recording."""
        self._recording = enabled

    def filter_history(self, filt: EventFilter) -> list[EventRecord]:
        """Return history records matching *filt*."""
        return [r for r in self._history if filt.matches(r)]

    def history_by_type(self, event_type: EventType) -> list[EventRecord]:
        """Return history records of a given type."""
        return [r for r in self._history if r.event_type == event_type]

    def history_in_range(self, t_start: float, t_end: float) -> list[EventRecord]:
        """Return history records within a sim-time range."""
        return [r for r in self._history if t_start <= r.sim_time <= t_end]

    def history_for_entity(self, entity_id: int) -> list[EventRecord]:
        """Return history records involving *entity_id*."""
        return [r for r in self._history if r.source_id == entity_id or r.target_id == entity_id]

    def event_counts(self) -> dict[str, int]:
        """Return a dict mapping event type names to counts."""
        counts: dict[str, int] = {}
        for r in self._history:
            key = r.event_type.value
            counts[key] = counts.get(key, 0) + 1
        return counts

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def register_replay_callback(self, callback: Callable[[EventRecord], None]) -> None:
        """Register a callback invoked during replay."""
        self._replay_callbacks.append(callback)

    def replay(
        self,
        records: list[EventRecord] | None = None,
        speed: float = 1.0,
        deliver: bool = True,
    ) -> None:
        """Replay recorded events.

        Parameters
        ----------
        records : list or None
            Events to replay.  Defaults to the full history.
        speed : float
            Replay speed multiplier (1.0 = real wall-time spacing).
            Set to ``0`` for instant replay.
        deliver : bool
            If ``True`` events are delivered to current subscribers.
        """
        events = records if records is not None else list(self._history)
        if not events:
            return

        base_wall = events[0].wall_time
        replay_start = time.monotonic()

        for rec in events:
            # Timing
            if speed > 0:
                desired = (rec.wall_time - base_wall) / speed
                elapsed = time.monotonic() - replay_start
                wait = desired - elapsed
                if wait > 0:
                    time.sleep(wait)

            # Deliver
            if deliver:
                self._deliver(rec)

            for cb in self._replay_callbacks:
                cb(rec)

    def serialize_history(self) -> list[dict[str, Any]]:
        """Serialize event history to a list of dicts."""
        return [r.to_dict() for r in self._history]

    def load_history(self, data: list[dict[str, Any]]) -> None:
        """Load event history from serialized data."""
        self._history = [EventRecord.from_dict(d) for d in data]

    # ------------------------------------------------------------------
    # Stats & repr
    # ------------------------------------------------------------------

    @property
    def publish_count(self) -> int:
        return self._publish_count

    @property
    def subscriber_count(self) -> int:
        return sum(1 for s in self._subs if s.active)

    def stats(self) -> dict[str, Any]:
        """Return summary statistics."""
        return {
            "publish_count": self._publish_count,
            "subscriber_count": self.subscriber_count,
            "history_size": len(self._history),
            "muted_types": [t.value for t in self._muted_types],
            "event_counts": self.event_counts(),
        }

    def reset(self) -> None:
        """Clear subscriptions, history, and counters."""
        self._subs.clear()
        self._sub_index.clear()
        self._history.clear()
        self._publish_count = 0
        self._paused = False
        self._muted_types.clear()
        self._replay_callbacks.clear()

    def __repr__(self) -> str:
        return (
            f"EventBus(subscribers={self.subscriber_count}, "
            f"published={self._publish_count}, "
            f"history={len(self._history)})"
        )
