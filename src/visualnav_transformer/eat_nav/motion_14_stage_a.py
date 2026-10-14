"""Frozen Stage A schedule and offline geometry checks for the 14-Path bank."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .bidirectional_bank import ANGLES, LENGTH_M, POINTS, audit_candidate, bank


SCHEMA = 'eatnav_motion_14_stage_a_v1'
PILOT_GROUP = {
    'state_group_id': 'random_rough-level2-seed2026101602',
    'terrain_type': 'random_rough', 'terrain_level': 2,
    'terrain_seed': 2026101602, 'env_tile_ids': [0, 1, 2, 3],
}
LEGACY_ROOT = Path('/home/kojima/lib/visualnav-transformer-ros2/'
                   '.eatnav-motion-vnext-evidence/mi-a-v31-formal-20261014-03')


def candidate_role(direction: int, angle: int) -> str:
    if direction not in (-1, 1) or angle not in ANGLES:
        raise ValueError('unfrozen candidate')
    prefix = 'forward' if direction == 1 else 'reverse'
    return f'{prefix}_direct' if angle == 0 else f'{prefix}_{angle:+d}'


def build_pilot_schedule() -> list[dict]:
    rows = []
    for posture in ('front_folded', 'rear_folded'):
        for repeat in (0, 1):
            rows.append({**PILOT_GROUP, 'index': len(rows), 'stage': 'prepath',
                         'candidate_id': None, 'repeat_index': repeat,
                         'flipper_initialization_mode': posture})
        for item in bank():
            rows.append({**PILOT_GROUP, 'index': len(rows), 'stage': 'candidate',
                         'candidate_id': candidate_role(item.direction, item.arc_angle_deg),
                         'repeat_index': 0, 'flipper_initialization_mode': posture,
                         'travel_direction': 'forward' if item.direction == 1 else 'reverse'})
    return rows


def validate_schedule(rows: list[dict]) -> None:
    if json.dumps(rows, sort_keys=True) != json.dumps(build_pilot_schedule(), sort_keys=True):
        raise ValueError('Stage A schedule differs from frozen design')


def path_sha(path: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(path, dtype='<f8').tobytes()).hexdigest()


def legacy_path_sha(legacy_root: Path = LEGACY_ROOT) -> dict[str, str]:
    """Read SHA references from sealed MI-A candidate manifests without editing them."""
    found = {}
    for manifest in sorted(legacy_root.glob('*/sim-output/input-manifest.json')):
        data = json.loads(manifest.read_text())
        if data.get('mode') != 'candidate':
            continue
        role = candidate_role(data['direction'], data['arc_angle_deg'])
        if role not in found:
            if not (manifest.parent/'host-seal.json').is_file():
                raise ValueError(f'unsealed legacy Path reference: {manifest}')
            points = np.asarray(data['path_body_xy_m'], dtype='<f8')
            digest = path_sha(points)
            if digest != data['path_body_sha256']:
                raise ValueError(f'legacy Path coordinate/SHA mismatch: {manifest}')
            found[role] = digest
    if len(found) != 6:
        raise ValueError(f'incomplete six-Path legacy reference: {sorted(found)}')
    return found


def geometry_audit(legacy_root: Path = LEGACY_ROOT) -> dict:
    references = legacy_path_sha(legacy_root)
    candidates = []
    for item in bank():
        geometry = audit_candidate(item)
        points = item.waypoints
        chords = np.diff(points, axis=0)
        intervals = np.linalg.norm(chords, axis=1)
        role = candidate_role(item.direction, item.arc_angle_deg)
        mirror = next(other for other in bank() if other.direction == item.direction
                      and other.arc_angle_deg == -item.arc_angle_deg)
        mirror_err = float(np.max(np.abs(points - mirror.waypoints * np.array([1., -1.]))))
        expected_chord = 2 * LENGTH_M / (POINTS - 1) * (
            math.sin(abs(math.radians(item.arc_angle_deg)) / (2 * (POINTS - 1))) /
            (abs(math.radians(item.arc_angle_deg)) / (POINTS - 1))
            if item.arc_angle_deg else .5)
        # The final chord length is an approximation to 0.75 m arc length;
        # the bank metadata retains the exact reference arc length.
        row = {'candidate_id': role, 'direction': item.direction,
               'angle_deg': item.arc_angle_deg, 'path_sha256': path_sha(points),
               'legacy_path_sha256': references.get(role),
               'geometry': geometry, 'mirror_max_abs_m': mirror_err,
               'interval_max_spread_m': float(np.ptp(intervals)),
               'expected_interval_m': expected_chord,
               'pass': bool(geometry['geometry_valid'] and points.shape == (8, 2)
                            and mirror_err < 1e-12 and np.ptp(intervals) < 1e-12
                            and abs(intervals[0] - expected_chord) < 1e-12
                            and (role not in references or path_sha(points) == references[role]))}
        candidates.append(row)
    return {'schema': SCHEMA, 'pass': all(row['pass'] for row in candidates),
            'legacy_reference_root': str(legacy_root), 'candidates': candidates}
