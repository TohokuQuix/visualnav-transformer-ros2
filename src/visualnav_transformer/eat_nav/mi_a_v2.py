"""Frozen paired-state comparison for the independent MI-A v2 pilot."""

from __future__ import annotations

import numpy as np


SCHEMA = 'mi_a_protocol_v2'
IDENTITY_FIELDS = ('seed', 'terrain_sha256', 'material_sha256',
                   'episode_length_s', 'dem_stamps_ns', 'joint_names',
                   'dem_frame', 'dem_resolution_m', 'python_rng_sha256',
                   'numpy_rng_sha256', 'torch_cpu_rng_sha256',
                   'torch_cuda_rng_sha256')
NUMERIC_FIELDS = ('root', 'joint', 'actor_pre_noise', 'actor_post_noise',
                  'actor_input', 'dem', 'command', 'latency', 'contact_force')
ABS_TOLERANCE = 1e-6
CANDIDATES = ('forward_direct', 'forward_+30', 'forward_-30',
              'reverse_direct', 'reverse_+30', 'reverse_-30')


def compare_gate_p(reference: dict, current: dict) -> dict:
    """Require paired identity without requiring left/right self symmetry."""
    details = {}
    for key in IDENTITY_FIELDS:
        equal = key in reference['identity'] and key in current['identity'] and \
                reference['identity'][key] == current['identity'][key]
        details[key] = {'pass': bool(equal), 'kind': 'identity'}
    for key in NUMERIC_FIELDS:
        left, right = reference['arrays'].get(key), current['arrays'].get(key)
        if left is None or right is None:
            details[key] = {'pass': False, 'kind': 'numeric', 'reason': 'missing'}
            continue
        a, b = np.asarray(left), np.asarray(right)
        valid = a.shape == b.shape and a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if valid else None
        details[key] = {'pass': bool(valid and maximum <= ABS_TOLERANCE),
                        'kind': 'numeric', 'max_abs': maximum,
                        'shape': list(a.shape)}
    for key in set(reference['arrays']) | set(current['arrays']):
        if key in NUMERIC_FIELDS or key == 'root_tile_local' or key in details:
            continue
        a, b = reference['arrays'].get(key), current['arrays'].get(key)
        valid = a is not None and b is not None and np.asarray(a).shape == np.asarray(b).shape
        if valid:
            a, b = np.asarray(a), np.asarray(b)
            valid = a.size > 0 and np.isfinite(a).all() and np.isfinite(b).all()
        maximum = float(np.max(np.abs(a.astype(np.float64)-b.astype(np.float64)))) if valid else None
        details[key] = {'pass': bool(valid and maximum <= ABS_TOLERANCE),
                        'kind': 'numeric_extra', 'max_abs': maximum}
    return {'schema': SCHEMA, 'pass': all(item['pass'] for item in details.values()),
            'tolerance_abs': ABS_TOLERANCE, 'fields': details}


def pilot_plan(seed: int) -> list[dict]:
    if not isinstance(seed, int) or seed <= 0:
        raise ValueError('positive fixed terrain seed required')
    rows = [{'stage': 'prepath', 'candidate': None, 'seed': seed, 'index': index}
            for index in (0, 1)]
    rows.extend({'stage': 'candidate', 'candidate': candidate, 'seed': seed,
                 'index': index+2} for index, candidate in enumerate(CANDIDATES))
    rows.append({'stage': 'repeat', 'candidate': 'forward_direct',
                 'seed': seed, 'index': 8})
    return rows
