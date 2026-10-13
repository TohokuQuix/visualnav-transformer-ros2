"""ROS-free copy of the policy bridge's turn-stall command preprocessing.

This runs between the Path follower and the RL policy. The bridge's launch
defaults are fixed here so the demo does not depend on a ROS bridge process.
"""

from __future__ import annotations

import math


class TurnStuckEscape:
    """Add alternating longitudinal motion when commanded turning is stuck."""

    def __init__(self):
        self._last_stamp_ns = None
        self.reset()

    def reset(self):
        self._low_response_since_ns = None
        self._direction = 1.0
        self._last_switch_ns = None

    def apply(self, follower_command, measured_yaw_rate, stamp_ns):
        command = tuple(float(value) for value in follower_command)
        if len(command) != 3 or not all(math.isfinite(value) for value in command):
            raise ValueError('follower command must be finite (linear x, linear y, angular z)')
        measured_yaw_rate = float(measured_yaw_rate)
        if not math.isfinite(measured_yaw_rate):
            raise ValueError('measured yaw rate must be finite')
        if not isinstance(stamp_ns, int) or stamp_ns < 0 or (
                self._last_stamp_ns is not None and stamp_ns <= self._last_stamp_ns):
            raise ValueError('turn escape stamp must increase')
        self._last_stamp_ns = stamp_ns

        commanded = abs(command[2])
        measured = abs(measured_yaw_rate)
        if commanded < 0.01 or measured > commanded * 0.5:
            self.reset()
            active = False
        elif self._low_response_since_ns is None:
            self._low_response_since_ns = stamp_ns
            active = False
        else:
            active = stamp_ns - self._low_response_since_ns >= 2_000_000_000

        offset = 0.0
        if active:
            if self._last_switch_ns is None:
                self._last_switch_ns = stamp_ns
            elif stamp_ns - self._last_switch_ns >= 1_000_000_000:
                self._direction *= -1.0
                self._last_switch_ns = stamp_ns
            offset = self._direction * min(0.2, max(0.0, commanded - measured))
        processed = (command[0] + offset, command[1], command[2])
        diagnostic = {
            'active': active,
            'direction': self._direction,
            'offset_mps': offset,
            'follower_command': command,
            'measured_yaw_rate_radps': measured_yaw_rate,
            'commanded_yaw_rate_radps': command[2],
            'low_response_since_ns': self._low_response_since_ns,
        }
        return processed, diagnostic
