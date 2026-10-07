"""Pure numeric and PhysX tensor-buffer contracts for radial contact audits."""
from __future__ import annotations

import math
from typing import Sequence


def _normal(q: Sequence[float]) -> tuple[float, ...]:
    if len(q) != 4 or not all(math.isfinite(v) for v in q):
        raise ValueError('quaternion must be four finite wxyz components')
    n = math.sqrt(sum(v * v for v in q))
    if n < 1e-9:
        raise ValueError('zero quaternion')
    return tuple(v / n for v in q)


def _mul(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    w, x, y, z = a
    W, X, Y, Z = b
    return (w*W-x*X-y*Y-z*Z, w*X+x*W+y*Z-z*Y,
            w*Y-x*Z+y*W+z*X, w*Z+x*Y-y*X+z*W)


def _rotate(q: Sequence[float], axis: Sequence[float]) -> tuple[float, ...]:
    v = _mul(_mul(q, (0.0, *axis)), (q[0], -q[1], -q[2], -q[3]))
    return v[1:]


def _wrap(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def orientation_metrics(settled_wxyz: Sequence[float], requested_yaw_rad: float) -> dict:
    """H1 horizontal body-forward, H2 relative world-z twist; radians."""
    if not math.isfinite(requested_yaw_rad):
        raise ValueError('requested yaw nonfinite')
    q = _normal(settled_wxyz)
    qreq = (math.cos(requested_yaw_rad/2), 0.0, 0.0, math.sin(requested_yaw_rad/2))
    fwd = _rotate(q, (1.0, 0.0, 0.0))
    horizontal = math.hypot(fwd[0], fwd[1])
    h1 = None if horizontal < 1e-6 else _wrap(math.atan2(fwd[1], fwd[0]) - requested_yaw_rad)
    relative = _mul(q, (qreq[0], 0.0, 0.0, -qreq[3]))
    twist_norm = math.hypot(relative[0], relative[3])
    h2 = None if twist_norm < 1e-6 else _wrap(2 * math.atan2(relative[3], relative[0]))
    w, x, y, z = q
    yaw = math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
    roll = math.atan2(2*(w*x+y*z), 1-2*(x*x+y*y))
    pitch = math.asin(max(-1.0, min(1.0, 2*(w*y-z*x))))
    up = _rotate(q, (0.0, 0.0, 1.0))
    tilt = math.acos(max(-1.0, min(1.0, up[2])))
    dot = abs(sum(a*b for a, b in zip(q, qreq)))
    return {'h1_signed_rad': h1, 'h2_signed_rad': h2,
            'horizontal_forward_norm': horizontal,
            'euler_yaw_signed_rad': _wrap(yaw-requested_yaw_rad),
            'roll_rad': roll, 'pitch_rad': pitch, 'tilt_rad': tilt,
            'quaternion_geodesic_rad': 2*math.acos(max(-1.0, min(1.0, dot)))}


def body_category(path: str) -> str:
    name = path.rsplit('/', 1)[-1].lower()
    if 'sprocket' in name:
        return 'sprocket'
    if 'flipper' in name:
        return 'flipper'
    if 'track' in name:
        return 'main_track'
    if name in ('base', 'body', 'base_link', 'chassis') or 'chassis' in name:
        return 'chassis_body'
    if 'arm' in name or 'gripper' in name:
        return 'arm_other'
    return 'unknown_body_pair'


def decode_contact_buffer(*, sensor_paths: Sequence[str], filter_paths: Sequence[str],
                          forces: Sequence[float], points: Sequence[Sequence[float]],
                          normals: Sequence[Sequence[float]], separations: Sequence[float],
                          counts: Sequence[Sequence[int]], starts: Sequence[Sequence[int]],
                          capacity: int) -> dict:
    """Decode six-array RigidContactView result without treating unused slots as contacts."""
    if capacity <= 0 or any(len(a) < capacity for a in (forces, points, normals, separations)):
        raise ValueError('invalid contact buffer capacity')
    if len(counts) != len(sensor_paths) or len(starts) != len(sensor_paths):
        raise ValueError('sensor count mismatch')
    pairs = []
    used = 0
    for i, sensor in enumerate(sensor_paths):
        if len(counts[i]) != len(filter_paths) or len(starts[i]) != len(filter_paths):
            raise ValueError('filter count mismatch')
        for j, collider in enumerate(filter_paths):
            count, start = int(counts[i][j]), int(starts[i][j])
            if count < 0 or start < 0 or start + count > capacity:
                raise ValueError('contact indices out of bounds')
            used += count
            if not count:
                continue
            entries = []
            for k in range(start, start+count):
                f, s = float(forces[k]), float(separations[k])
                p, n = [float(v) for v in points[k]], [float(v) for v in normals[k]]
                if len(p) != 3 or len(n) != 3 or not all(math.isfinite(v) for v in (f, s, *p, *n)):
                    raise ValueError('contact data nonfinite or wrong shape')
                entries.append({'normal_force_n': f, 'point_w_m': p, 'normal_w': n,
                                'separation_m': s, 'penetration_m': max(0.0, -s)})
            pairs.append({'sensor_path': sensor, 'collider_path': collider,
                          'pair_identity': [sensor, collider],
                          'body_category': body_category(sensor),
                          'pair_kind': 'self_collision' if '/Robot/' in collider else
                                       'robot_terrain' if ('terrain' in collider.lower() or 'ground' in collider.lower())
                                       else 'unknown_body_pair',
                          'contact_count': count,
                          'min_separation_m': min(e['separation_m'] for e in entries),
                          'max_penetration_m': max(e['penetration_m'] for e in entries),
                          'contacts': entries})
    return {'pairs': pairs, 'contact_count': used,
            'buffer_capacity': capacity, 'buffer_overflow_or_full': used >= capacity}
