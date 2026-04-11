"""Tests for navirl/coordination/communication.py — classical channels."""

from __future__ import annotations

import threading
import time

import numpy as np
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
    def test_create_message(self):
        msg = MessageProtocol(sender="agent_1", receiver="agent_2", content="hello")
        assert msg.sender == "agent_1"
        assert msg.receiver == "agent_2"
        assert msg.content == "hello"
        assert msg.timestamp > 0

    def test_broadcast_message(self):
        msg = MessageProtocol(sender="agent_1", receiver=None, content="announce")
        assert msg.receiver is None

    def test_metadata(self):
        msg = MessageProtocol(
            sender="a", receiver="b", content=42, metadata={"priority": "high"}
        )
        assert msg.metadata["priority"] == "high"

    def test_default_metadata_empty(self):
        msg = MessageProtocol(sender="a", receiver="b", content=None)
        assert msg.metadata == {}

    def test_timestamp_auto(self):
        t_before = time.time()
        msg = MessageProtocol(sender="a", receiver=None, content="x")
        t_after = time.time()
        assert t_before <= msg.timestamp <= t_after

    def test_content_arbitrary_type(self):
        msg = MessageProtocol(sender="a", receiver=None, content={"key": [1, 2, 3]})
        assert msg.content["key"] == [1, 2, 3]

    def test_content_numpy(self):
        data = np.array([1.0, 2.0, 3.0])
        msg = MessageProtocol(sender="a", receiver=None, content=data)
        np.testing.assert_allclose(msg.content, data)


# ---------------------------------------------------------------------------
# BroadcastChannel
# ---------------------------------------------------------------------------


class TestBroadcastChannel:
    def test_send_and_receive(self):
        ch = BroadcastChannel()
        msg = MessageProtocol(sender="a", receiver=None, content="hi")
        ch.send(msg)
        received = ch.receive()
        assert len(received) == 1
        assert received[0].content == "hi"

    def test_receive_returns_copy(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a", receiver=None, content="x"))
        r1 = ch.receive()
        r2 = ch.receive()
        assert r1 == r2
        assert r1 is not r2

    def test_multiple_messages(self):
        ch = BroadcastChannel()
        for i in range(5):
            ch.send(MessageProtocol(sender=f"a{i}", receiver=None, content=i))
        received = ch.receive()
        assert len(received) == 5

    def test_buffer_limit(self):
        ch = BroadcastChannel(max_buffer_size=3)
        for i in range(10):
            ch.send(MessageProtocol(sender="a", receiver=None, content=i))
        received = ch.receive()
        assert len(received) == 3
        # Should keep the most recent
        assert received[0].content == 7
        assert received[2].content == 9

    def test_clear(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a", receiver=None, content="x"))
        ch.clear()
        assert ch.receive() == []

    def test_size(self):
        ch = BroadcastChannel()
        assert ch.size == 0
        ch.send(MessageProtocol(sender="a", receiver=None, content="x"))
        assert ch.size == 1
        ch.send(MessageProtocol(sender="b", receiver=None, content="y"))
        assert ch.size == 2

    def test_agent_id_ignored(self):
        ch = BroadcastChannel()
        ch.send(MessageProtocol(sender="a", receiver=None, content="x"))
        # agent_id parameter is ignored for broadcast
        received = ch.receive(agent_id="b")
        assert len(received) == 1

    def test_thread_safety(self):
        ch = BroadcastChannel()
        errors = []

        def sender(n):
            try:
                for i in range(n):
                    ch.send(MessageProtocol(sender="t", receiver=None, content=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sender, args=(50,)) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert ch.size == 200


# ---------------------------------------------------------------------------
# DirectChannel
# ---------------------------------------------------------------------------


class TestDirectChannel:
    def test_send_and_receive(self):
        ch = DirectChannel()
        msg = MessageProtocol(sender="a", receiver="b", content="hello")
        ch.send(msg)
        received = ch.receive("b")
        assert len(received) == 1
        assert received[0].content == "hello"

    def test_receive_clears_mailbox(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a", receiver="b", content="x"))
        ch.receive("b")
        received = ch.receive("b")
        assert received == []

    def test_no_messages(self):
        ch = DirectChannel()
        assert ch.receive("nonexistent") == []

    def test_send_requires_receiver(self):
        ch = DirectChannel()
        msg = MessageProtocol(sender="a", receiver=None, content="x")
        with pytest.raises(ValueError, match="specific receiver"):
            ch.send(msg)

    def test_multiple_receivers(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a", receiver="b", content="for_b"))
        ch.send(MessageProtocol(sender="a", receiver="c", content="for_c"))
        assert len(ch.receive("b")) == 1
        assert len(ch.receive("c")) == 1

    def test_peek_does_not_remove(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a", receiver="b", content="msg"))
        peeked = ch.peek("b")
        assert len(peeked) == 1
        # Still available
        received = ch.receive("b")
        assert len(received) == 1

    def test_peek_empty(self):
        ch = DirectChannel()
        assert ch.peek("nonexistent") == []

    def test_peek_returns_copy(self):
        ch = DirectChannel()
        ch.send(MessageProtocol(sender="a", receiver="b", content="x"))
        p1 = ch.peek("b")
        p2 = ch.peek("b")
        assert p1 is not p2

    def test_multiple_messages_same_receiver(self):
        ch = DirectChannel()
        for i in range(5):
            ch.send(MessageProtocol(sender="a", receiver="b", content=i))
        received = ch.receive("b")
        assert len(received) == 5

    def test_thread_safety(self):
        ch = DirectChannel()
        errors = []

        def sender(agent_id, n):
            try:
                for i in range(n):
                    ch.send(MessageProtocol(sender="s", receiver=agent_id, content=i))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=sender, args=(f"agent_{j}", 50)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        total = sum(len(ch.receive(f"agent_{j}")) for j in range(4))
        assert total == 200


# ---------------------------------------------------------------------------
# SharedMemory
# ---------------------------------------------------------------------------


class TestSharedMemory:
    def test_write_and_read(self):
        sm = SharedMemory()
        sm.write("key1", "value1")
        assert sm.read("key1") == "value1"

    def test_read_default(self):
        sm = SharedMemory()
        assert sm.read("missing") is None
        assert sm.read("missing", default=42) == 42

    def test_overwrite(self):
        sm = SharedMemory()
        sm.write("k", 1)
        sm.write("k", 2)
        assert sm.read("k") == 2

    def test_read_all(self):
        sm = SharedMemory()
        sm.write("a", 1)
        sm.write("b", 2)
        all_data = sm.read_all()
        assert all_data == {"a": 1, "b": 2}

    def test_read_all_returns_copy(self):
        sm = SharedMemory()
        sm.write("a", 1)
        d = sm.read_all()
        d["a"] = 999
        assert sm.read("a") == 1

    def test_clear(self):
        sm = SharedMemory()
        sm.write("a", 1)
        sm.clear()
        assert sm.read("a") is None
        assert sm.keys == []

    def test_keys(self):
        sm = SharedMemory()
        sm.write("x", 1)
        sm.write("y", 2)
        assert set(sm.keys) == {"x", "y"}

    def test_numpy_values(self):
        sm = SharedMemory()
        arr = np.array([1.0, 2.0, 3.0])
        sm.write("positions", arr)
        np.testing.assert_allclose(sm.read("positions"), arr)

    def test_thread_safety(self):
        sm = SharedMemory()
        errors = []

        def writer(key, n):
            try:
                for i in range(n):
                    sm.write(key, i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(f"k{j}", 100)) for j in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        assert set(sm.keys) == {"k0", "k1", "k2", "k3"}

    def test_concurrent_read_write(self):
        sm = SharedMemory()
        sm.write("counter", 0)
        errors = []

        def reader(n):
            try:
                for _ in range(n):
                    sm.read("counter")
            except Exception as e:
                errors.append(e)

        def writer(n):
            try:
                for i in range(n):
                    sm.write("counter", i)
            except Exception as e:
                errors.append(e)

        t_read = threading.Thread(target=reader, args=(100,))
        t_write = threading.Thread(target=writer, args=(100,))
        t_read.start()
        t_write.start()
        t_read.join()
        t_write.join()
        assert not errors
