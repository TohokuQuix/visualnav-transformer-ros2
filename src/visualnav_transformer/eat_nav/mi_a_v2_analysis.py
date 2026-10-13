"""Descriptive one-StateGroup MI-A v2 pilot analysis; no population inference."""

from __future__ import annotations

import numpy as np

from .mi_a_v2 import CANDIDATES
from .motion_identifiability import relative_motion, wrap


def _labels(rows):
    if len(rows) != 4 or any(not row.get('available') for row in rows):
        raise ValueError('incomplete four-env motion label')
    xy = np.asarray([row['delta_body_xy_m'] for row in rows], dtype=np.float64)
    yaw = np.asarray([row['delta_body_yaw_rad'] for row in rows], dtype=np.float64)
    if xy.shape != (4, 2) or yaw.shape != (4,) or not np.isfinite(xy).all() or \
            not np.isfinite(yaw).all():
        raise ValueError('nonfinite motion label')
    return xy, yaw


def summarize_labels(entries: dict, direct_repeat: list[dict], *, seed: int) -> dict:
    if set(entries) != set(CANDIDATES):
        raise ValueError('six frozen candidates required')
    measured = {role: _labels(entries[role]) for role in CANDIDATES}
    repeated_xy, repeated_yaw = _labels(direct_repeat)
    forward_xy, forward_yaw = measured['forward_direct']
    repeat = [{'env': env,
               'delta_xy_m': (repeated_xy[env]-forward_xy[env]).tolist(),
               'delta_yaw_rad': wrap(float(repeated_yaw[env]-forward_yaw[env]))}
              for env in range(4)]
    differences = {}
    for role in CANDIDATES:
        if role.endswith('direct'):
            continue
        direction = 1 if role.startswith('forward') else -1
        angle = int(role.rsplit('_', 1)[1])
        direct = measured['forward_direct' if direction == 1 else 'reverse_direct']
        arc = measured[role]
        per_env = []
        for env in range(4):
            contrast = relative_motion(
                {'delta_body_xy_m': arc[0][env], 'delta_body_yaw_rad': arc[1][env]},
                {'delta_body_xy_m': direct[0][env], 'delta_body_yaw_rad': direct[1][env]},
                direction=direction, angle_deg=angle)
            per_env.append({'env': env, **contrast})
        differences[role] = {'per_env': per_env,
                             'mean_delta_xy_m': np.mean([r['delta_xy_m'] for r in per_env], axis=0).tolist(),
                             'mean_delta_yaw_rad': float(np.mean([r['delta_yaw_rad'] for r in per_env])),
                             'mean_signed_yaw_rad': float(np.mean([r['signed_yaw_rad'] for r in per_env])),
                             'mean_signed_travel_lateral_m': float(np.mean([
                                 r['signed_travel_lateral_m'] for r in per_env]))}
    raw = {role: [{'env': env, 'delta_body_xy_m': values[0][env].tolist(),
                   'delta_body_yaw_rad': float(values[1][env])} for env in range(4)]
           for role, values in measured.items()}
    return {'seed': seed, 'independent_terrain_seed_count': 1,
            'raw_motion': raw, 'candidate_differences': differences,
            'forward_direct_fresh_process_repeat_difference': repeat,
            'inference_limit': 'descriptive one-seed pilot; four envs are not independent terrain seeds'}
