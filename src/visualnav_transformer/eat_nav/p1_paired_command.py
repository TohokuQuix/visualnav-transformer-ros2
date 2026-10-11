"""Stamp-verified diagnostic command interventions for the P1 paired-noise pilot.

This module never changes the ROS follower.  Its input is the command accepted
by the bridge and the follower's exact-stamp ``command_pre_publish`` record.
"""

from __future__ import annotations

import math


def _angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _direction(record: dict) -> int:
    desired = math.atan2(record['carrot_y'] - record['tf_y'],
                         record['carrot_x'] - record['tf_x'])
    yaw = record['tf_yaw']
    error = record['heading_error']
    forward_error = abs(_angle(desired - yaw - error))
    reverse_error = abs(_angle(desired - _angle(yaw + math.pi) - error))
    if forward_error < 1e-5 and reverse_error > 1.0:
        return 1
    if reverse_error < 1e-5 and forward_error > 1.0:
        return -1
    raise ValueError('cannot infer follower travel direction from exact carrot/pose')


def intervene(command: list[float], record: dict, condition: str, *,
              stamp_ns: int, nominal_speed: float = 0.3,
              turn_threshold: float = 0.6) -> tuple[list[float], str]:
    """Change only the ordinary forward scale/turn stop, or fail closed.

    Condition A/B are exact pass-through.  S removes the soft scale inside the
    normal driving region.  T additionally permits forward motion when the
    normal turn-in-place branch is the *only* reason for zero forward speed.
    """
    if condition not in ('A', 'B', 'S', 'T'):
        raise ValueError('unknown condition')
    if len(command) != 3 or not all(math.isfinite(float(x)) for x in command):
        raise ValueError('invalid follower command')
    if record['phase'] != 'command_pre_publish':
        raise ValueError('follower diagnostic phase mismatch')
    if record['command_stamp_ns'] != stamp_ns or record['tf_stamp_ns'] != stamp_ns:
        raise ValueError('follower diagnostic stamp mismatch')
    if not record['tf_success'] or record['path_points'] == 0:
        raise ValueError('follower TF/Path invalid')
    if max(abs(command[0] - record['vx']), abs(command[2] - record['wz'])) > 1e-6:
        raise ValueError('follower diagnostic/accepted command mismatch')
    if abs(command[0]) > nominal_speed + 1e-6:
        raise ValueError('follower forward command over cap')
    original = [float(x) for x in command]
    if condition in ('A', 'B'):
        return original, 'normal'
    if record['goal_reached']:
        return original, 'goal_stop_preserved'
    if record['mode'] != 'default' or record['linear_recovery'] or record['angular_recovery']:
        return original, 'recovery_preserved'
    direction = _direction(record)
    turning = abs(record['heading_error']) >= turn_threshold
    if turning != bool(record['turn_in_place']):
        raise ValueError('turn-in-place diagnostic inconsistent with threshold')
    if turning:
        if abs(original[0]) > 1e-6:
            raise ValueError('ordinary turn stop has nonzero forward speed')
        if condition == 'S':
            return original, 'ordinary_turn_stop_preserved'
        return [direction * nominal_speed, original[1], original[2]], 'ordinary_turn_stop_removed'
    expected = direction * nominal_speed * max(0.0, 1.0 - abs(record['heading_error']) / math.pi)
    if abs(original[0] - expected) > 1e-6:
        raise ValueError('ordinary soft scale formula mismatch; independent override possible')
    return [direction * nominal_speed, original[1], original[2]], 'soft_scale_removed'
