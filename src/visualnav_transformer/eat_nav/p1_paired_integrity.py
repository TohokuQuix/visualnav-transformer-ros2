"""Frozen per-step abort checks for the P1 S/T diagnostic."""

from __future__ import annotations

import math


def _numbers(value):
    if isinstance(value, (list, tuple)):
        for item in value:
            yield from _numbers(item)
    elif isinstance(value, (int, float)):
        yield float(value)


def assess_step(row: dict, *, roll_pitch_cap_rad: float, speed_cap_mps: float,
                angular_cap_radps: float = 1.6) -> list[str]:
    reasons = []
    root = row['post_root_state']
    if len(root) != 13 or not all(math.isfinite(x) for x in root):
        return ['nonfinite_root']
    command = row['applied_command']
    if len(command) != 3 or not all(math.isfinite(x) for x in command):
        reasons.append('nonfinite_command')
    contact = row.get('contact_net_forces_w')
    if contact is not None and not all(math.isfinite(x) for x in _numbers(contact)):
        reasons.append('nonfinite_contact')
    _, _, _, w, x, y, z = root[:7]
    roll = math.atan2(2 * (w*x + y*z), 1 - 2 * (x*x + y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w*y - z*x))))
    if max(abs(roll), abs(pitch)) > roll_pitch_cap_rad:
        reasons.append('roll_pitch_cap')
    speed = math.sqrt(sum(v*v for v in root[7:10]))
    if speed > speed_cap_mps:
        reasons.append('root_speed_cap')
    if any(abs(command[i]) > cap + 1e-6 for i, cap in ((0, 0.3), (2, angular_cap_radps))):
        reasons.append('command_cap')
    return reasons
