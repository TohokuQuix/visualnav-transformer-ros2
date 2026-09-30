"""Exact-stamp gates used only by the simulation-clock diagnostic."""
from __future__ import annotations
from dataclasses import dataclass
from .sim_clock_protocol import ClockProtocol


@dataclass
class ExactStampLatch:
    last_emitted_stamp_ns: int | None = None

    def may_emit(self, request_ns: int, returned_tf_ns: int) -> bool:
        return request_ns == returned_tf_ns and self.last_emitted_stamp_ns != request_ns

    def emitted(self, stamp_ns: int) -> None:
        if self.last_emitted_stamp_ns == stamp_ns:
            raise ValueError('duplicate logical stamp')
        self.last_emitted_stamp_ns = stamp_ns

    def reset(self) -> None:
        self.last_emitted_stamp_ns = None


class ExactClockProtocol(ClockProtocol):
    def reject_reason(self, branch: int, stamp_ns: int) -> str | None:
        if self.stamp_ns is None:
            return 'clock_missing'
        if stamp_ns < self.stamp_ns:
            return 'stale'
        if stamp_ns > self.stamp_ns:
            return 'future'
        if self.path_stamp_ns is None or self.path_stamp_ns > stamp_ns:
            return 'pre_path'
        if self.tf_stamp_ns != stamp_ns:
            return 'tf_not_published'
        if branch in self.commands:
            return 'duplicate'
        return None

    def accept(self, branch: int, stamp_ns: int, command: tuple[float, float, float]) -> bool:
        if self.reject_reason(branch, stamp_ns) is not None:
            return False
        self.commands[branch] = [command]
        return True
