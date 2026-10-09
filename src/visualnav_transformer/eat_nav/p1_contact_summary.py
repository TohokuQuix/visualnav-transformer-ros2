"""Validate and summarize one physics-substep PhysX contact view sample."""

from __future__ import annotations

import numpy as np


def summarize_contact_arrays(arrays, capacity: int) -> dict:
    arrays = [np.asarray(x) for x in arrays]
    if len(arrays) != 6:
        raise ValueError('expected six contact arrays')
    forces, points, normals, separations, counts, starts = arrays
    if (forces.shape != (capacity, 1) or points.shape != (capacity, 3) or
            normals.shape != (capacity, 3) or separations.shape != (capacity, 1) or
            counts.shape != starts.shape or counts.ndim != 2):
        raise ValueError('contact array shape mismatch')
    if not all(np.isfinite(x).all() for x in arrays):
        raise ValueError('nonfinite contact array')
    if np.any(counts < 0) or np.any(starts < 0) or np.any(starts + counts > capacity):
        raise ValueError('contact count/start bounds failure')
    total = int(counts.sum())
    if total >= capacity:
        raise ValueError('contact buffer full or truncation suspected')
    active = []
    for count, start in zip(counts.flat, starts.flat):
        active.extend(range(int(start), int(start+count)))
    return {'contact_count': total,
            'force_max_n': float(np.max(np.abs(forces[active, 0]))) if active else 0.0,
            'min_separation_m': float(np.min(separations[active, 0])) if active else None,
            'bounds_valid': True, 'buffer_full': False}
