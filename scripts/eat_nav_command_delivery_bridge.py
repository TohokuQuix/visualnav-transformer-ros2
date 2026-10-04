"""Diagnostic-only bridge; event logging does not alter exact-stamp acceptance."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import threading
import time
import traceback
from pathlib import Path

import rclpy

from eat_nav_exact_stamp_bridge import ExactNode, ExactServer
from visualnav_transformer.eat_nav.command_delivery_journal import EventJournal
from eat_nav_e1_process_ownership import process_record


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
        self.last_heartbeat_ns = 0
        self.drop_stamp_ns = None
        self.heartbeat_guard = self.create_guard_condition(self.heartbeat)

    def heartbeat(self):
        self.heartbeat_count += 1
        self.last_heartbeat_ns = time.monotonic_ns()
        self.journal.emit("executor_heartbeat", count=self.heartbeat_count)

    def on_command(self, branch, msg):
        stamp = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.journal.emit("subscriber_entry", branch=branch, stamp_ns=stamp,
                          vx=float(msg.twist.linear.x), wz=float(msg.twist.angular.z))
        if stamp == self.drop_stamp_ns:
            self.journal.emit("fault_drop", branch=branch, stamp_ns=stamp)
            return
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

    def path(self, paths, ns):
        if any(self.command_count) or any(self.sequence):
            raise RuntimeError("pre-Path command or previous command state")
        subscribers = [publisher.get_subscription_count() for publisher in self.paths]
        tf_ready = [bool(self.tf_buffer.can_transform(
            "map", f"branch_{i}/chassis/body", rclpy.time.Time()))
            for i in range(self.branches)]
        clock_publishers = self.count_publishers("/clock")
        age_ns = time.monotonic_ns() - self.last_heartbeat_ns
        self.journal.emit("prepath_check", stamp_ns=ns, subscribers=subscribers,
                          tf_ready=tf_ready, clock_publishers=clock_publishers,
                          heartbeat_age_ns=age_ns, command_count=self.command_count,
                          sequence=self.sequence)
        if (subscribers != [1] * self.branches or not all(tf_ready)
                or clock_publishers != 1 or self.heartbeat_count == 0
                or age_ns > 2_000_000_000):
            raise RuntimeError("fresh bridge pre-Path readiness failed")
        super().path(paths, ns)


class DiagnosticServer(ExactServer):
    def __init__(self, args):
        self.args = args
        self.log_dir = args.diagnostic_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.journal = EventJournal(self.log_dir / "bridge-events.jsonl")
        rclpy.init()
        self.node = DiagnosticNode(args.branches, self.journal)
        self.node.drop_stamp_ns = args.fault_drop_stamp_ns
        self.spin = threading.Thread(target=self._spin, daemon=True, name="ros-executor")
        self.spin.start()
        self.followers = []
        self.logs = []
        self.reset_number = 0
        self.stop_monitor = threading.Event()
        self.monitor = threading.Thread(target=self._monitor, daemon=True, name="resource-monitor")
        self.monitor.start()

    def launch(self):
        super().launch()
        for branch, process in enumerate(self.followers):
            record=process_record(process.pid)
            if record is None:
                raise RuntimeError(f'follower_{branch} exited before process registration')
            self.journal.emit('follower_spawn',branch=branch,pid=process.pid,process=record)

    def _spin(self):
        try:
            rclpy.spin(self.node)
            self.journal.emit("executor_return")
        except BaseException as exc:
            self.journal.emit("executor_exception", error=repr(exc), traceback=traceback.format_exc())
            raise

    def _monitor(self):
        while not self.stop_monitor.wait(0.5):
            try:
                self.node.heartbeat_guard.trigger()
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
        for branch, (command, sequence) in enumerate(zip(response["commands"], response["sequence"])):
            self.journal.emit("command_adopted", branch=branch, stamp_ns=ns,
                              command=command, sequence=sequence)
        return response

    def handle(self, request):
        self.journal.emit("tcp_request_entry", action=request.get("action"),
                          stamp_ns=request.get("stamp_ns"))
        try:
            result = {"shutdown": True} if request.get("action") == "shutdown" else super().handle(request)
            self.journal.emit("tcp_response", action=request.get("action"),
                              stamp_ns=request.get("stamp_ns"))
            return result
        except BaseException as exc:
            self.journal.emit("tcp_error", action=request.get("action"),
                              stamp_ns=request.get("stamp_ns"), error=repr(exc))
            raise

    def shutdown_lifecycle(self):
        self.journal.emit("shutdown_entry", follower_count=len(self.followers))
        for process in self.followers:
            self.journal.emit("follower_sigterm", pid=process.pid)
            if process.poll() is None:
                process.terminate()
        for process in self.followers:
            try:
                process.wait(timeout=3)
            except Exception:
                self.journal.emit("follower_sigkill", pid=process.pid)
                process.kill()
                process.wait(timeout=3)
            self.journal.emit("follower_exit", pid=process.pid, returncode=process.returncode)
        for stream in self.logs:
            stream.flush()
            stream.close()
        self.stop_monitor.set()
        self.monitor.join(timeout=2)
        self.node.destroy_node()
        rclpy.shutdown()
        self.spin.join(timeout=5)
        self.journal.emit("shutdown_return", executor_alive=self.spin.is_alive())
        self.journal.close()

    def serve(self):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind((self.args.host, self.args.port))
                sock.listen(1)
                print(json.dumps({"stage": "listening", "port": self.args.port}), flush=True)
                while True:
                    conn, _ = sock.accept()
                    with conn, conn.makefile("rwb") as stream:
                        for line in stream:
                            request = json.loads(line)
                            try:
                                response = {"ok": True, **self.handle(request)}
                            except BaseException as exc:
                                response = {"ok": False, "error": repr(exc)}
                            stream.write((json.dumps(response) + "\n").encode())
                            stream.flush()
                            if request.get("action") == "shutdown":
                                return
        finally:
            self.shutdown_lifecycle()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=47819)
    parser.add_argument("--branches", type=int, default=8)
    parser.add_argument("--follower", type=Path, required=True)
    parser.add_argument("--params", type=Path, required=True)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--fault-drop-stamp-ns", type=int)
    DiagnosticServer(parser.parse_args()).serve()


if __name__ == "__main__":
    main()
