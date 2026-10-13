"""Pure path/frame and cross-candidate pre-Path gates for vNext collection."""

from __future__ import annotations

import math
import numpy as np


PREPATH_NUMERIC_FIELDS = ('root', 'joint', 'actor_pre_noise', 'dem')
PREPATH_IDENTITY_FIELDS = ('material_sha256', 'terrain_sha256', 'seed')


def body_path_to_tile_local(path_body, pose_world, tile_origin_world):
    path = np.asarray(path_body, dtype=np.float64)
    pose = np.asarray(pose_world, dtype=np.float64)
    origin = np.asarray(tile_origin_world, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or pose.shape != (3,) or \
            origin.shape != (2,) or not np.isfinite(path).all() or \
            not np.isfinite(pose).all() or not np.isfinite(origin).all():
        raise ValueError('body Path/tile pose contract')
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.array([[c, -s], [s, c]])
    return path @ rotation.T + pose[:2] - origin


def compare_prepath(reference: dict, candidate: dict, *, atol: float = 1e-6) -> dict:
    """Strict identity plus predeclared floating tolerance; never inspect Outcome."""
    details = {}
    for key in PREPATH_IDENTITY_FIELDS:
        details[key] = {'pass': reference.get(key) == candidate.get(key),
                        'reference': reference.get(key), 'candidate': candidate.get(key)}
    for key in PREPATH_NUMERIC_FIELDS:
        left = np.asarray(reference[key], dtype=np.float64)
        right = np.asarray(candidate[key], dtype=np.float64)
        valid = left.shape == right.shape and np.isfinite(left).all() and np.isfinite(right).all()
        maximum = float(np.max(np.abs(left - right))) if valid and left.size else None
        details[key] = {'pass': bool(valid and maximum is not None and maximum <= atol),
                        'max_abs': maximum, 'shape': list(left.shape)}
    return {'pass': all(row['pass'] for row in details.values()), 'tolerance_abs': atol,
            'fields': details}
