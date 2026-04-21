"""Tests for navirl.coordination.communication classical channels.

Covers BroadcastChannel, DirectChannel, SharedMemory, and MessageProtocol.
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

# ---------------------------------------------------------------------------
# MessageProtocol
# ---------------------------------------------------------------------------


class TestMessageProtocol:
    def test_basic_construction(self):
        msg = MessageProtocol(sender="a1", receiver="a2", content="hello")
        assert msg.sender == "a1"
        assert msg.receiver == "a2"
        assert msg.content == "hello"
        assert isinstance(msg.timestamp, float)
        assert msg.metadata == {}

    def test_broadcast_message_receiver_none(self):
        msg = MessageProtocol(sender="a1", receiver=None, content=[1, 2, 3])
        assert msg.receiver is None
        assert msg.content == [1, 2, 3]

    def test_metadata_dict(self):
        msg = MessageProtocol(
            sender="a1", receiver="a2", content="x", metadata={"priority": 5}
        )
        assert msg.metadata["priority"] == 5

    def test_custom_timestamp(self):
        ts = 1000.0
        msg = MessageProtocol(sender="a1", receiver=None, content="x", timestamp=ts)
        assert msg.timestamp == ts

    def test_default_metadata_independent(self):
        """Each instance should get its own metadata dict."""
        m1 = MessageProtocol(sender="a", receiver=None, content="x")
        m2 = MessageProtocol(sender="b", receiver=None, content="y")
        m1.metadata["key"] = "val"
        assert "key" not in m2.metadata


# ---------------------------------------------------------------------------
# BroadcastChannel
# ---------------------------------------------------------------------------


class TestBroadcastChannel:
    def test_send_and_receive(self):
        ch = BroadcastChannel()
        msg = MessageProtocol(sender="a1", receiver=None, content="hello")
        ch.send(msg)
        received = ch.receive()
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_receive_returns_copy(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        r1 = ch.receive()
        r2 = ch.receive()
        assert r1 is not r2
        assert len(r1) == len(r2) == 1

    def test_receive_agent_id_ignored(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        assert len(ch.receive(agent_id="a1")) == 1
        assert len(ch.receive(agent_id="a2")) == 1

    def test_size_property(self):
        ch = BroadcastChannel()
        assert ch.size == 0
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        assert ch.size == 1
        ch.send(MessageProtocol(sender="a2", receiver=None, content="y"))
        assert ch.size == 2

    def test_clear(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a1", receiver=None, content="x"))
        ch.clear()
        assert ch.size == 0
        assert ch.receive() == []

    def test_buffer_overflow_trims_oldest(self):
        ch = BroadcastChannel(max_buffer_size=3)
        for i in range(5):
            ch.send(MessageProtocol(sender="a1", receiver=None, content=i))
        assert ch.size == 3
        msgs = ch.receive()
        assert [m.content for m in msgs] == [2, 3, 4]

    def test_concurrent_sends(self):
        ch = BroadcastChannel(max_buffer_size=10000)
        num_threads = 4
        msgs_per_thread = 50
        barrier = threading.Barrier(num_threads)

        def sender(tid):
            barrier.wait()
            for i in range(msgs_per_thread):
                ch.send(
                    MessageProtocol(sender=f"t{tid}", receiver=None, content=(tid, i))
                )

        threads = [threading.Thread(target=sender, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert ch.size == num_threads * msgs_per_thread


# ---------------------------------------------------------------------------
# DirectChannel
# ---------------------------------------------------------------------------


class TestDirectChannel:
    def test_send_and_receive(self):
        ch = DirectChannel()
        msg = MessageProtocol(sender="a1", receiver="a2", content="hello")
        ch.send(msg)
        received = ch.receive("a2")
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_receive_clears_mailbox(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="x"))
        assert len(ch.receive("a2")) == 1
        assert len(ch.receive("a2")) == 0

    def test_receive_empty_mailbox(self):
        ch = DirectChannel()
        assert ch.receive("nonexistent") == []

    def test_peek_does_not_remove(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="x"))
        peeked = ch.peek("a2")
        assert len(peeked) == 1
        # Still there after peek
        assert len(ch.peek("a2")) == 1
        # receive should still get it
        assert len(ch.receive("a2")) == 1

    def test_peek_empty(self):
        ch = DirectChannel()
        assert ch.peek("a1") == []

    def test_messages_routed_correctly(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="for_a2"))
        ch.send(MessageProtocol(sender="a1", receiver="a3", content="for_a3"))
        ch.send(MessageProtocol(sender="a2", receiver="a3", content="also_for_a3"))

        a2_msgs = ch.receive("a2")
        a3_msgs = ch.receive("a3")
        assert len(a2_msgs) == 1
        assert a2_msgs[0].content == "for_a2"
        assert len(a3_msgs) == 2

    def test_send_requires_receiver(self):
        ch = DirectChannel()
        msg = MessageProtocol(sender="a1", receiver=None, content="x")
        with pytest.raises(ValueError, match="specific receiver"):
            ch.send(msg)

    def test_peek_returns_copy(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a1", receiver="a2", content="x"))
        p1 = ch.peek("a2")
        p2 = ch.peek("a2")
        assert p1 is not p2


# ---------------------------------------------------------------------------
# SharedMemory
# ---------------------------------------------------------------------------


class TestSharedMemory:
    def test_write_and_read(self):
        mem = SharedMemory()
        mem.write("key1", 42)
        assert mem.read("key1") == 42

    def test_read_default(self):
        mem = SharedMemory()
        assert mem.read("missing") is None
        assert mem.read("missing", "default") == "default"

    def test_overwrite(self):
        mem = SharedMemory()
        mem.write("k", 1)
        mem.write("k", 2)
        assert mem.read("k") == 2

    def test_read_all(self):
        mem = SharedMemory()
        mem.write("a", 1)
        mem.write("b", 2)
        snapshot = mem.read_all()
        assert snapshot == {"a": 1, "b": 2}

    def test_read_all_returns_copy(self):
        mem = SharedMemory()
        mem.write("a", 1)
        snapshot = mem.read_all()
        snapshot["a"] = 999
        assert mem.read("a") == 1

    def test_clear(self):
        mem = SharedMemory()
        mem.write("a", 1)
        mem.clear()
        assert mem.read("a") is None
        assert mem.keys == []

    def test_keys_property(self):
        mem = SharedMemory()
        assert mem.keys == []
        mem.write("x", 1)
        mem.write("y", 2)
        assert sorted(mem.keys) == ["x", "y"]

    def test_complex_values(self):
        mem = SharedMemory()
        mem.write("nested", {"a": [1, 2], "b": {"c": 3}})
        val = mem.read("nested")
        assert val["a"] == [1, 2]
        assert val["b"]["c"] == 3

    def test_concurrent_writes(self):
        mem = SharedMemory()
        barrier = threading.Barrier(4)

        def writer(tid):
            barrier.wait()
            for i in range(50):
                mem.write(f"t{tid}_{i}", i)

        threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(mem.keys) == 200
