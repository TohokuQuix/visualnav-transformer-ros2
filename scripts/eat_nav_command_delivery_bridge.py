"""Diagnostic-only bridge; event logging does not alter exact-stamp acceptance."""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from pathlib import Path

import rclpy

from eat_nav_exact_stamp_bridge import ExactNode, ExactServer
from visualnav_transformer.eat_nav.command_delivery_journal import EventJournal


def resource_counts():
    proc = Path("/proc/self")
    fd = list((proc / "fd").iterdir())
    sockets = 0
    for path in fd:
        try:
            sockets += "socket:" in os.readlink(path)
        except FileNotFoundError:
            pass  # File descriptors can close during a /proc snapshot.
    return {
        "rss_pages": int((proc / "statm").read_text().split()[1]),
        "fd_count": len(fd),
        "thread_count": len(list((proc / "task").iterdir())),
        "socket_count": sockets,
    }


class DiagnosticNode(ExactNode):
    def __init__(self, branches, journal):
        self.journal = journal
        super().__init__(branches)
        self.heartbeat_count = 0
        self.create_timer(0.5, self.heartbeat)

    def heartbeat(self):
        self.heartbeat_count += 1
        self.journal.emit("executor_heartbeat", count=self.heartbeat_count)

    def on_command(self, branch, msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.journal.emit("subscriber_entry", branch=branch, stamp_ns=stamp,
                          vx=float(msg.twist.linear.x), wz=float(msg.twist.angular.z))
        try:
            super().on_command(branch, msg)
            with self.cv:
                self.journal.emit("queue_insert", branch=branch, stamp_ns=stamp,
                                  sequence=self.sequence[branch],
                                  command_count=self.command_count[branch],
                                  rejected=self.rejected[branch], queue_length=len(self.events))
        finally:
            self.journal.emit("subscriber_exit", branch=branch, stamp_ns=stamp)

    def on_status(self, branch, msg):
        self.journal.emit("status_entry", branch=branch)
        super().on_status(branch, msg)

    def clock(self, ns, reset=False):
        self.journal.emit("clock_publish_entry", stamp_ns=ns, reset=reset)
        super().clock(ns, reset=reset)
        self.journal.emit("clock_publish_return", stamp_ns=ns)

    def poses(self, poses, ns):
        self.journal.emit("tf_publish_entry", stamp_ns=ns)
        super().poses(poses, ns)
        self.journal.emit("tf_publish_return", stamp_ns=ns)


class DiagnosticServer(ExactServer):
    def __init__(self, args):
        self.args = args
        self.log_dir = args.diagnostic_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.journal = EventJournal(self.log_dir / "bridge-events.jsonl")
        rclpy.init()
        self.node = DiagnosticNode(args.branches, self.journal)
        self.spin = threading.Thread(target=self._spin, daemon=True, name="ros-executor")
        self.spin.start()
        self.followers = []
        self.logs = []
        self.reset_number = 0
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name="resource-monitor")
        self.monitor.start()

    def _spin(self):
        try:
            rclpy.spin(self.node)
            self.journal.emit("executor_return")
        except BaseException as exc:
            self.journal.emit("executor_exception", error=repr(exc), traceback=traceback.format_exc())
            raise

    def _monitor(self):
        while True:
            try:
                self.journal.emit("resource", executor_alive=self.spin.is_alive(),
                                  followers_alive=sum(p.poll() is None for p in self.followers),
                                  **resource_counts())
            except BaseException as exc:
                self.journal.emit("resource_exception", error=repr(exc))
            time.sleep(1)

    def reset(self, request):
        self.journal.emit("reset_entry", reset_number=self.reset_number,
                          follower_count=len(self.followers))
        result = super().reset(request)
        self.journal.emit("reset_return", reset_number=self.reset_number,
                          follower_count=len(self.followers))
        return result

    def wait_command(self, ns, timeout):
        self.journal.emit("waiter_entry", stamp_ns=ns, timeout_sec=timeout,
                          executor_alive=self.spin.is_alive())
        try:
            response = super().wait_command(ns, timeout)
        except BaseException as exc:
            stacks = {str(tid): "".join(traceback.format_stack(frame))
                      for tid, frame in sys._current_frames().items()}
            self.journal.emit("waiter_error", stamp_ns=ns, error=repr(exc),
                              executor_alive=self.spin.is_alive(), stacks=stacks,
                              **resource_counts())
            raise
        self.journal.emit("waiter_wake", stamp_ns=ns,
                          sequence=response["sequence"], callback_count=response["callback_count"])
        return response

    def handle(self, request):
        self.journal.emit("tcp_request_entry", action=request.get("action"),
                          stamp_ns=request.get("stamp_ns"))
        try:
            result = super().handle(request)
            self.journal.emit("tcp_response", action=request.get("action"),
                              stamp_ns=request.get("stamp_ns"))
            return result
        except BaseException as exc:
            self.journal.emit("tcp_error", action=request.get("action"),
                              stamp_ns=request.get("stamp_ns"), error=repr(exc))
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47819)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--follower", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    DiagnosticServer(parser.parse_args()).serve()


if __name__ == "__main__":
    main()
