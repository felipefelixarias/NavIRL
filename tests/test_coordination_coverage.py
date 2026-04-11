"""Tests for navirl/coordination/ — communication channels and MARL config.

Covers: BroadcastChannel buffer overflow, DirectChannel edge cases,
SharedMemory operations, MARLConfig, and message protocol details.
"""

from __future__ import annotations

import threading
import time

import pytest

from navirl.coordination.communication import (
    BroadcastChannel,
    DirectChannel,
    MessageProtocol,
    SharedMemory,
)
from navirl.coordination.marl import MARLConfig

# ---------------------------------------------------------------------------
# MARLConfig
# ---------------------------------------------------------------------------


class TestMARLConfig:
    def test_defaults(self):
        cfg = MARLConfig()
        assert cfg.algorithm == "mappo"
        assert cfg.centralized_critic is True
        assert cfg.shared_policy is True
        assert cfg.communication is False
        assert cfg.gamma == 0.99
        assert cfg.lr == 3e-4
        assert cfg.num_agents == 2

    def test_custom_values(self):
        cfg = MARLConfig(
            algorithm="qmix",
            centralized_critic=False,
            shared_policy=False,
            communication=True,
            gamma=0.95,
            lr=1e-3,
            num_agents=5,
        )
        assert cfg.algorithm == "qmix"
        assert cfg.centralized_critic is False
        assert cfg.shared_policy is False
        assert cfg.communication is True
        assert cfg.gamma == 0.95
        assert cfg.lr == 1e-3
        assert cfg.num_agents == 5


# ---------------------------------------------------------------------------
# MessageProtocol
# ---------------------------------------------------------------------------


class TestMessageProtocolExtended:
    def test_broadcast_message(self):
        msg = MessageProtocol(sender="a1", receiver=None, content="broadcast")
        assert msg.receiver is None

    def test_metadata_default_empty(self):
        msg = MessageProtocol(sender="a1", receiver="a2", content="hi")
        assert msg.metadata == {}

    def test_metadata_custom(self):
        msg = MessageProtocol(
            sender="a1", receiver="a2", content="hi", metadata={"priority": "high"}
        )
        assert msg.metadata["priority"] == "high"

    def test_timestamp_auto_set(self):
        before = time.time()
        msg = MessageProtocol(sender="a1", receiver="a2", content="x")
        after = time.time()
        assert before <= msg.timestamp <= after

    def test_custom_timestamp(self):
        msg = MessageProtocol(sender="a1", receiver="a2", content="x", timestamp=100.0)
        assert msg.timestamp == 100.0


# ---------------------------------------------------------------------------
# BroadcastChannel
# ---------------------------------------------------------------------------


class TestBroadcastChannelExtended:
    def test_buffer_overflow_trims(self):
        ch = BroadcastChannel(max_buffer_size=3)
        for i in range(5):
            ch.send(MessageProtocol(sender=f"a{i}", receiver=None, content=i))
        assert ch.size == 3
        msgs = ch.receive()
        assert len(msgs) == 3
        # Should keep the last 3 messages
        assert msgs[0].content == 2
        assert msgs[1].content == 3
        assert msgs[2].content == 4

    def test_clear(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        assert ch.size == 1
        ch.clear()
        assert ch.size == 0
        assert ch.receive() == []

    def test_receive_returns_copy(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        msgs1 = ch.receive()
        msgs2 = ch.receive()
        assert msgs1 == msgs2
        assert msgs1 is not msgs2  # Different list objects

    def test_receive_ignores_agent_id(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        msgs = ch.receive(agent_id="any_agent")
        assert len(msgs) == 1

    def test_thread_safety(self):
        ch = BroadcastChannel(max_buffer_size=1000)
        errors = []

        def writer(agent_id, count):
            try:
                for i in range(count):
                    ch.send(MessageProtocol(sender=agent_id, receiver=None, content=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"a{i}", 100)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert ch.size <= 1000

    def test_empty_channel(self):
        ch = BroadcastChannel()
        assert ch.size == 0
        assert ch.receive() == []


# ---------------------------------------------------------------------------
# DirectChannel
# ---------------------------------------------------------------------------


class TestDirectChannelExtended:
    def test_send_requires_receiver(self):
        ch = DirectChannel()
        with pytest.raises(ValueError, match="requires a specific receiver"):
            ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))

    def test_receive_clears_mailbox(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="msg1"))
        msgs = ch.receive("a2")
        assert len(msgs) == 1
        # Second receive should be empty
        assert ch.receive("a2") == []

    def test_peek_does_not_clear(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="msg1"))
        msgs = ch.peek("a2")
        assert len(msgs) == 1
        # Peek again — still there
        msgs2 = ch.peek("a2")
        assert len(msgs2) == 1
        # Receive clears
        msgs3 = ch.receive("a2")
        assert len(msgs3) == 1
        assert ch.peek("a2") == []

    def test_multiple_receivers(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="for_a2"))
        ch.send(MessageProtocol(sender="a1", receiver="a3", content="for_a3"))
        assert len(ch.receive("a2")) == 1
        assert len(ch.receive("a3")) == 1
        assert ch.receive("a4") == []

    def test_multiple_messages_same_receiver(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="msg1"))
        ch.send(MessageProtocol(sender="a3", receiver="a2", content="msg2"))
        msgs = ch.receive("a2")
        assert len(msgs) == 2
        assert msgs[0].content == "msg1"
        assert msgs[1].content == "msg2"

    def test_peek_nonexistent_agent(self):
        ch = DirectChannel()
        assert ch.peek("nonexistent") == []

    def test_thread_safety(self):
        ch = DirectChannel()
        errors = []

        def writer(sender, receiver, count):
            try:
                for i in range(count):
                    ch.send(
                        MessageProtocol(sender=sender, receiver=receiver, content=i)
                    )
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"s{i}", "target", 50))
            for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        msgs = ch.receive("target")
        assert len(msgs) == 250


# ---------------------------------------------------------------------------
# SharedMemory
# ---------------------------------------------------------------------------


class TestSharedMemoryExtended:
    def test_write_and_read(self):
        sm = SharedMemory()
        sm.write("key1", "value1")
        assert sm.read("key1") == "value1"

    def test_read_default(self):
        sm = SharedMemory()
        assert sm.read("missing") is None
        assert sm.read("missing", default=42) == 42

    def test_read_all(self):
        sm = SharedMemory()
        sm.write("a", 1)
        sm.write("b", 2)
        all_data = sm.read_all()
        assert all_data == {"a": 1, "b": 2}
        # Should be a copy
        all_data["c"] = 3
        assert sm.read("c") is None

    def test_clear(self):
        sm = SharedMemory()
        sm.write("a", 1)
        sm.write("b", 2)
        sm.clear()
        assert sm.keys == []
        assert sm.read("a") is None

    def test_keys_property(self):
        sm = SharedMemory()
        sm.write("x", 10)
        sm.write("y", 20)
        assert sorted(sm.keys) == ["x", "y"]

    def test_overwrite(self):
        sm = SharedMemory()
        sm.write("key", "old")
        sm.write("key", "new")
        assert sm.read("key") == "new"

    def test_complex_values(self):
        sm = SharedMemory()
        sm.write("list", [1, 2, 3])
        sm.write("dict", {"nested": True})
        assert sm.read("list") == [1, 2, 3]
        assert sm.read("dict") == {"nested": True}

    def test_thread_safety(self):
        sm = SharedMemory()
        errors = []

        def writer(key_prefix, count):
            try:
                for i in range(count):
                    sm.write(f"{key_prefix}_{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer, args=(f"t{i}", 50)) for i in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(sm.keys) == 250
