"""Descriptive, condition-paired MI-FI motion analysis."""

from __future__ import annotations

import numpy as np

from .mi_a_v2 import CANDIDATES
from .mi_fi_v1 import CONDITIONS
from .motion_identifiability import wrap


def _matrix(labels: list[dict]) -> np.ndarray:
    if len(labels) != 4 or any(not row.get('available') for row in labels):
        raise ValueError('four complete motion labels required')
    values = np.asarray([[*row['delta_body_xy_m'], row['delta_body_yaw_rad']]
                         for row in labels], dtype=np.float64)
    if values.shape != (4, 3) or not np.isfinite(values).all():
        raise ValueError('nonfinite motion label')
    return values


def _difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result = a - b
    result[:, 2] = [wrap(float(value)) for value in result[:, 2]]
    return result


def _serial(values: np.ndarray) -> dict:
    return {'per_env': [{'env': index, 'delta_xy_m': row[:2].tolist(),
                         'delta_yaw_rad': float(row[2])}
                        for index, row in enumerate(values)],
            'mean_delta_xy_m': values[:, :2].mean(axis=0).tolist(),
            'mean_delta_yaw_rad': float(values[:, 2].mean())}


def summarize_interactions(labels: dict[str, dict[str, list[dict]]]) -> dict:
    if set(labels) != set(CONDITIONS) or any(set(labels[c]) != set(CANDIDATES) for c in CONDITIONS):
        raise ValueError('two conditions × six frozen candidates required')
    measured = {condition: {candidate: _matrix(rows) for candidate, rows in group.items()}
                for condition, group in labels.items()}
    relative = {}
    for condition in CONDITIONS:
        relative[condition] = {}
        for candidate in CANDIDATES:
            if candidate.endswith('direct'):
                continue
            direct = 'forward_direct' if candidate.startswith('forward') else 'reverse_direct'
            relative[condition][candidate] = _difference(
                measured[condition][candidate], measured[condition][direct])
    interactions = {candidate: _serial(_difference(
        relative['rear_folded'][candidate], relative['front_folded'][candidate]))
        for candidate in relative['front_folded']}
    direct = {name: _serial(_difference(measured['rear_folded'][name],
                                        measured['front_folded'][name]))
              for name in ('forward_direct', 'reverse_direct')}
    return {'raw': {condition: {candidate: _serial(values)
                               for candidate, values in group.items()}
                    for condition, group in measured.items()},
            'arc_minus_direct': {condition: {candidate: _serial(values)
                                            for candidate, values in group.items()}
                                 for condition, group in relative.items()},
            'interactions': interactions, 'direct_rear_minus_front': direct,
            'independent_terrain_seed_count': 1,
            'interpretation': 'descriptive; four env tiles share one terrain seed'}
