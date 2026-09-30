import json
import threading
import time

import pytest

rclpy = pytest.importorskip("rclpy")
from geometry_msgs.msg import TwistStamped

from eat_nav_command_delivery_bridge import DiagnosticNode, DiagnosticServer
from visualnav_transformer.eat_nav.command_delivery_journal import EventJournal


def test_subscriber_callback_runs_while_waiter_releases_lock(tmp_path):
    rclpy.init()
    journal = EventJournal(tmp_path / "events.jsonl")
    node = DiagnosticNode(1, journal)
    try:
        stamp = 2_000_000_050_000
        node.protocol.advance(stamp, reset=True)
        node.protocol.set_tf(stamp)
        node.protocol.set_path(stamp)
        node.tf_published_steady_ns = time.monotonic_ns() - 1
        node.path_published_steady_ns = time.monotonic_ns() - 1
        message = TwistStamped()
        message.header.stamp.sec, message.header.stamp.nanosec = divmod(stamp, 1_000_000_000)
        message.twist.linear.x = 0.2
        worker = threading.Thread(target=lambda: node.on_command(0, message))
        worker.start()
        assert node.cv_wait(lambda: node.protocol.ready(1), 1)
        worker.join(timeout=1)
        assert not worker.is_alive()
        assert node.sequence == [1]
        assert node.commands == [[0.2, 0.0, 0.0]]
        events = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
        assert [x["event"] for x in events] == [
            "subscriber_entry", "queue_insert", "subscriber_exit"
        ]
        assert events[1]["sequence"] == 1
    finally:
        node.destroy_node()
        journal.close()
        rclpy.shutdown()


def test_stopped_executor_is_reported_on_command_timeout(tmp_path):
    rclpy.init()
    journal = EventJournal(tmp_path / "events.jsonl")
    node = DiagnosticNode(1, journal)
    server = object.__new__(DiagnosticServer)
    server.node = node
    server.journal = journal
    server.followers = []
    server.spin = threading.Thread(target=lambda: None)
    try:
        started = time.monotonic()
        with pytest.raises(TimeoutError, match="fresh command timeout"):
            server.wait_command(42, 0.01)
        assert time.monotonic() - started < 3
        rows = [json.loads(x) for x in (tmp_path / "events.jsonl").read_text().splitlines()]
        error = next(row for row in rows if row["event"] == "waiter_error")
        assert error["executor_alive"] is False
        assert error["stamp_ns"] == 42
    finally:
        node.destroy_node()
        journal.close()
        rclpy.shutdown()
