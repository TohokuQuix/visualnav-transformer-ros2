"""Pure geometry and exact-stamp guards for the limited statistical pilot."""

from __future__ import annotations

import math


def local_pose_from_root(root_state, tile_origin_xy):
    w, x, y, z = root_state[3:7]
    yaw = math.atan2(2 * (w*z + x*y), 1 - 2 * (y*y + z*z))
    return [float(root_state[0]-tile_origin_xy[0]),
            float(root_state[1]-tile_origin_xy[1]), yaw]


def path_in_map(waypoints, pose):
    px, py, yaw = pose
    c, s = math.cos(yaw), math.sin(yaw)
    return [[float(px + c*x - s*y), float(py + s*x + c*y)]
            for x, y in waypoints]


def paths_for_slots(rows, path_lookup, poses):
    if len(rows) != 7 or len(poses) != 7 or {r['slot'] for r in rows} != set(range(7)):
        raise ValueError('seven-slot assignment required')
    result = [None] * 7
    for row in rows:
        identity = row['path_sha256']
        if identity not in path_lookup:
            raise ValueError('unknown frozen Path SHA')
        result[row['slot']] = path_in_map(path_lookup[identity]['waypoints'], poses[row['slot']])
    # The frozen bridge has eight production followers. Branch eight receives
    # an identical monitor Path and pose, and its command is never injected.
    return result + [result[0]]


def validate_bridge_response(response, stamp_ns, branches=8):
    if not response.get('fresh') or any(response.get(key) != stamp_ns for key in
                                       ('stamp_ns', 'clock_now_ns', 'tf_stamp_ns')):
        raise ValueError('follower command/clock/TF stamp mismatch')
    commands = response.get('commands')
    if len(commands) != branches or any(len(cmd) != 3 or
                                        not all(math.isfinite(v) for v in cmd) for cmd in commands):
        raise ValueError('follower command shape/nonfinite')
    return commands
