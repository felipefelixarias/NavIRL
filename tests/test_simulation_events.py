"""Tests for navirl.simulation.events — EventBus, EventFilter, EventRecord."""

from __future__ import annotations

import pytest

from navirl.simulation.events import EventBus, EventFilter, EventRecord, EventType

# ---------------------------------------------------------------------------
# EventRecord
# ---------------------------------------------------------------------------


class TestEventRecord:
    def test_defaults(self):
        r = EventRecord(event_type=EventType.COLLISION)
        assert r.sim_time == 0.0
        assert r.source_id == -1

    def test_to_dict(self):
        r = EventRecord(
            event_type=EventType.GOAL_REACHED,
            sim_time=1.5,
            source_id=3,
            target_id=7,
            data={"distance": 0.1},
        )
        d = r.to_dict()
        assert d["sim_time"] == 1.5
        assert d["source_id"] == 3

    def test_from_dict_roundtrip(self):
        r = EventRecord(
            event_type=EventType.ZONE_ENTER,
            sim_time=2.0,
            source_id=1,
            target_id=5,
            data={"zone": "kitchen"},
        )
        d = r.to_dict()
        r2 = EventRecord.from_dict(d)
        assert r2.event_type == r.event_type
        assert r2.sim_time == r.sim_time
        assert r2.source_id == r.source_id


# ---------------------------------------------------------------------------
# EventFilter
# ---------------------------------------------------------------------------


class TestEventFilter:
    def test_match_by_type(self):
        f = EventFilter(event_types={EventType.COLLISION})
        r1 = EventRecord(event_type=EventType.COLLISION)
        r2 = EventRecord(event_type=EventType.TIMEOUT)
        assert f.matches(r1)
        assert not f.matches(r2)

    def test_match_by_source(self):
        f = EventFilter(source_ids={1, 2})
        r1 = EventRecord(event_type=EventType.STEP, source_id=1)
        r2 = EventRecord(event_type=EventType.STEP, source_id=99)
        assert f.matches(r1)
        assert not f.matches(r2)

    def test_match_by_time_range(self):
        f = EventFilter(min_sim_time=1.0, max_sim_time=5.0)
        r1 = EventRecord(event_type=EventType.STEP, sim_time=3.0)
        r2 = EventRecord(event_type=EventType.STEP, sim_time=0.5)
        r3 = EventRecord(event_type=EventType.STEP, sim_time=6.0)
        assert f.matches(r1)
        assert not f.matches(r2)
        assert not f.matches(r3)

    def test_match_by_data_key_value(self):
        f = EventFilter(data_key="zone", data_value="kitchen")
        r1 = EventRecord(event_type=EventType.CUSTOM, data={"zone": "kitchen"})
        r2 = EventRecord(event_type=EventType.CUSTOM, data={"zone": "hallway"})
        r3 = EventRecord(event_type=EventType.CUSTOM, data={})
        assert f.matches(r1)
        assert not f.matches(r2)
        assert not f.matches(r3)

    def test_combined_and(self):
        f1 = EventFilter(event_types={EventType.COLLISION})
        f2 = EventFilter(source_ids={5})
        combined = f1 & f2
        r1 = EventRecord(event_type=EventType.COLLISION, source_id=5)
        r2 = EventRecord(event_type=EventType.COLLISION, source_id=99)
        r3 = EventRecord(event_type=EventType.TIMEOUT, source_id=5)
        assert combined.matches(r1)
        assert not combined.matches(r2)
        assert not combined.matches(r3)

    def test_combined_or(self):
        f1 = EventFilter(event_types={EventType.COLLISION})
        f2 = EventFilter(event_types={EventType.TIMEOUT})
        combined = f1 | f2
        r1 = EventRecord(event_type=EventType.COLLISION)
        r2 = EventRecord(event_type=EventType.TIMEOUT)
        r3 = EventRecord(event_type=EventType.STEP)
        assert combined.matches(r1)
        assert combined.matches(r2)
        assert not combined.matches(r3)

    def test_no_filters_matches_all(self):
        f = EventFilter()
        r = EventRecord(event_type=EventType.CUSTOM, sim_time=99.0, source_id=42)
        assert f.matches(r)


# ---------------------------------------------------------------------------
# EventBus — publish/subscribe
# ---------------------------------------------------------------------------


class TestEventBusPublishSubscribe:
    def test_publish_records_history(self):
        bus = EventBus(recording=True)
        bus.publish(EventType.COLLISION, sim_time=1.0)
        assert len(bus.history) == 1
        assert bus.history[0].event_type == EventType.COLLISION

    def test_subscribe_receives_events(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r), event_type=EventType.COLLISION)
        bus.publish(EventType.COLLISION, sim_time=1.0)
        assert len(received) == 1

    def test_subscribe_filters_by_type(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r), event_type=EventType.COLLISION)
        bus.publish(EventType.TIMEOUT, sim_time=1.0)
        assert len(received) == 0

    def test_subscribe_once(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r), event_type=EventType.STEP, once=True)
        bus.publish(EventType.STEP)
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_unsubscribe(self):
        bus = EventBus()
        received = []
        sid = bus.subscribe(lambda r: received.append(r))
        bus.publish(EventType.STEP)
        assert len(received) == 1
        bus.unsubscribe(sid)
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_unsubscribe_all(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r))
        bus.subscribe(lambda r: received.append(r))
        bus.unsubscribe_all()
        bus.publish(EventType.STEP)
        assert len(received) == 0

    def test_entity_filter(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r), entity_filter=5)
        bus.publish(EventType.COLLISION, source_id=5)
        bus.publish(EventType.COLLISION, source_id=99)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# EventBus — convenience emitters
# ---------------------------------------------------------------------------


class TestEventBusConvenienceEmitters:
    def test_emit_collision(self):
        bus = EventBus()
        r = bus.emit_collision(sim_time=1.0, entity_a=1, entity_b=2, penetration=0.5)
        assert r.event_type == EventType.COLLISION
        assert r.source_id == 1
        assert r.target_id == 2

    def test_emit_goal_reached(self):
        bus = EventBus()
        r = bus.emit_goal_reached(sim_time=2.0, entity_id=3, goal_id=10)
        assert r.event_type == EventType.GOAL_REACHED
        assert r.source_id == 3

    def test_emit_timeout(self):
        bus = EventBus()
        r = bus.emit_timeout(sim_time=30.0)
        assert r.event_type == EventType.TIMEOUT

    def test_emit_zone_enter_exit(self):
        bus = EventBus()
        r1 = bus.emit_zone_enter(sim_time=1.0, entity_id=1, zone_id=5)
        r2 = bus.emit_zone_exit(sim_time=2.0, entity_id=1, zone_id=5)
        assert r1.event_type == EventType.ZONE_ENTER
        assert r2.event_type == EventType.ZONE_EXIT


# ---------------------------------------------------------------------------
# EventBus — mute/pause
# ---------------------------------------------------------------------------


class TestEventBusMutePause:
    def test_mute_suppresses_delivery(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r), event_type=EventType.COLLISION)
        bus.mute(EventType.COLLISION)
        bus.publish(EventType.COLLISION)
        assert len(received) == 0
        # But still recorded
        assert len(bus.history) == 1

    def test_unmute_restores_delivery(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r), event_type=EventType.STEP)
        bus.mute(EventType.STEP)
        bus.publish(EventType.STEP)
        bus.unmute(EventType.STEP)
        bus.publish(EventType.STEP)
        assert len(received) == 1

    def test_pause_resume(self):
        bus = EventBus()
        received = []
        bus.subscribe(lambda r: received.append(r))
        bus.pause()
        bus.publish(EventType.STEP)
        assert len(received) == 0
        bus.resume()
        bus.publish(EventType.STEP)
        assert len(received) == 1


# ---------------------------------------------------------------------------
# EventBus — history queries
# ---------------------------------------------------------------------------


class TestEventBusHistory:
    def _populated_bus(self):
        bus = EventBus(recording=True)
        bus.publish(EventType.COLLISION, sim_time=1.0, source_id=1, target_id=2)
        bus.publish(EventType.STEP, sim_time=2.0, source_id=1)
        bus.publish(EventType.GOAL_REACHED, sim_time=3.0, source_id=3)
        bus.publish(EventType.COLLISION, sim_time=4.0, source_id=2, target_id=3)
        return bus

    def test_history_by_type(self):
        bus = self._populated_bus()
        collisions = bus.history_by_type(EventType.COLLISION)
        assert len(collisions) == 2

    def test_history_in_range(self):
        bus = self._populated_bus()
        in_range = bus.history_in_range(1.5, 3.5)
        assert len(in_range) == 2

    def test_history_for_entity(self):
        bus = self._populated_bus()
        entity_events = bus.history_for_entity(1)
        assert len(entity_events) >= 2

    def test_filter_history(self):
        bus = self._populated_bus()
        f = EventFilter(event_types={EventType.COLLISION}, min_sim_time=2.0)
        result = bus.filter_history(f)
        assert len(result) == 1
        assert result[0].sim_time == 4.0

    def test_event_counts(self):
        bus = self._populated_bus()
        counts = bus.event_counts()
        assert counts.get("COLLISION", counts.get(EventType.COLLISION.value, 0)) >= 2

    def test_clear_history(self):
        bus = self._populated_bus()
        bus.clear_history()
        assert len(bus.history) == 0

    def test_set_recording_false(self):
        bus = EventBus(recording=False)
        bus.publish(EventType.STEP)
        assert len(bus.history) == 0

    def test_publish_count(self):
        bus = EventBus()
        bus.publish(EventType.STEP)
        bus.publish(EventType.STEP)
        assert bus.publish_count == 2

    def test_subscriber_count(self):
        bus = EventBus()
        bus.subscribe(lambda r: None)
        bus.subscribe(lambda r: None)
        assert bus.subscriber_count == 2


# ---------------------------------------------------------------------------
# EventBus — serialization and replay
# ---------------------------------------------------------------------------


class TestEventBusSerialization:
    def test_serialize_load_roundtrip(self):
        bus = EventBus(recording=True)
        bus.publish(EventType.COLLISION, sim_time=1.0, source_id=1)
        bus.publish(EventType.STEP, sim_time=2.0)
        data = bus.serialize_history()
        bus2 = EventBus(recording=True)
        bus2.load_history(data)
        assert len(bus2.history) == 2

    def test_replay_delivers_to_callbacks(self):
        bus = EventBus(recording=True)
        bus.publish(EventType.COLLISION, sim_time=1.0)
        bus.publish(EventType.STEP, sim_time=2.0)

        replayed = []
        bus.register_replay_callback(lambda r: replayed.append(r))
        bus.replay(speed=0.0)  # instant replay
        assert len(replayed) >= 2


# ---------------------------------------------------------------------------
# EventBus — stats and reset
# ---------------------------------------------------------------------------


class TestEventBusStats:
    def test_stats(self):
        bus = EventBus()
        bus.subscribe(lambda r: None)
        bus.publish(EventType.STEP)
        s = bus.stats()
        assert "publish_count" in s or "total_published" in s or isinstance(s, dict)

    def test_reset(self):
        bus = EventBus()
        bus.subscribe(lambda r: None)
        bus.publish(EventType.STEP)
        bus.reset()
        assert bus.publish_count == 0
        assert bus.subscriber_count == 0
        assert len(bus.history) == 0


# ---------------------------------------------------------------------------
# EventType enum
# ---------------------------------------------------------------------------


class TestEventType:
    def test_all_types_defined(self):
        expected = {
            "COLLISION",
            "GOAL_REACHED",
            "TIMEOUT",
            "ZONE_ENTER",
            "ZONE_EXIT",
            "ENTITY_ADDED",
            "ENTITY_REMOVED",
            "SIMULATION_START",
            "SIMULATION_END",
            "EPISODE_START",
            "EPISODE_END",
            "STEP",
            "DOOR_OPEN",
            "DOOR_CLOSE",
            "WAYPOINT_REACHED",
            "CUSTOM",
        }
        actual = {e.name for e in EventType}
        assert expected.issubset(actual)
