"""Outcome-blind contracts for the candidate-free all-terrain radial inventory."""
from __future__ import annotations

import hashlib
import math
from typing import Sequence

TERRAIN_ORDER = (
    'boxes', 'hf_pyramid_slope', 'pyramid_stairs', 'pyramid_stairs_inv',
    'random_rough', 'k_rail_field', 'k_rail_step_field',
)
FLAT_TYPES = frozenset(TERRAIN_ORDER[:4])
RADIUS_M = .30
ANGLE_COUNT = 16
VALIDITY_LIMITS = {
    'settled_error_xy_m': .10,
    'settled_yaw_error_rad': .10,
    'settled_speed_mps': .10,
    'settled_angular_speed_radps': .20,
    'dem_valid_fraction_min': .70,
}


def terrain_seed(batch_seed: int, column: int) -> int:
    if batch_seed < 0 or not 0 <= column < len(TERRAIN_ORDER):
        raise ValueError('invalid batch seed or column')
    digest = hashlib.sha256(f'eatnav-radial:{batch_seed}:{column}'.encode()).digest()
    return int.from_bytes(digest[:4], 'little')


def build_pose_schedule(batch_seeds: Sequence[int]) -> list[dict]:
    if len(set(batch_seeds)) != len(batch_seeds):
        raise ValueError('batch seeds must be distinct')
    rows = []
    for batch_seed in batch_seeds:
        for angle_index in range(ANGLE_COUNT):
            theta = angle_index * math.pi / 8
            for column, terrain in enumerate(TERRAIN_ORDER):
                rows.append({
                    'pose_id': f'b{batch_seed}-{terrain}-a{angle_index:02d}',
                    'batch_seed': int(batch_seed), 'terrain_seed': terrain_seed(batch_seed, column),
                    'terrain': terrain, 'terrain_class': 'source_flat_radial' if terrain in FLAT_TYPES else 'unguaranteed_center_radial',
                    'source_family': f'{batch_seed}:{terrain}',
                    'row': angle_index, 'column': column,
                    'env_index': angle_index * len(TERRAIN_ORDER) + column,
                    'angle_index': angle_index, 'theta_rad': theta,
                    'tile_local_xy_centered_m': [RADIUS_M * math.cos(theta), RADIUS_M * math.sin(theta)],
                    'tile_local_xy_corner_origin_m': [4 + RADIUS_M * math.cos(theta), 4 + RADIUS_M * math.sin(theta)],
                    'tile_local_yaw_rad': theta,
                })
    return rows


def pose_world(origin_xyz: Sequence[float], angle_index: int) -> dict:
    if len(origin_xyz) != 3 or not 0 <= angle_index < ANGLE_COUNT:
        raise ValueError('invalid origin or angle')
    theta = angle_index * math.pi / 8
    return {'world_x': float(origin_xyz[0] + RADIUS_M * math.cos(theta)),
            'world_y': float(origin_xyz[1] + RADIUS_M * math.sin(theta)),
            'world_z_reference': float(origin_xyz[2]), 'world_yaw': theta}


def validate_record(record: dict, schedule_row: dict, limits: dict = VALIDITY_LIMITS) -> dict:
    reasons = []
    for key in ('pose_id', 'batch_seed', 'terrain', 'row', 'column', 'source_family'):
        if record.get(key) != schedule_row[key]:
            reasons.append(f'identity:{key}')
    origin = record.get('env_origin')
    if not isinstance(origin, list) or len(origin) != 3 or not all(math.isfinite(v) for v in origin):
        reasons.append('env_origin')
    else:
        expected = pose_world(origin, schedule_row['angle_index'])
        actual = record.get('requested_pose', {})
        if any(not math.isclose(actual.get(key, math.nan), value, abs_tol=1e-6) for key, value in expected.items()):
            reasons.append('pose_transform')
    for key in ('settled_error_xy_m', 'settled_yaw_error_rad', 'settled_speed_mps', 'settled_angular_speed_radps'):
        value = record.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or value > limits[key]:
            reasons.append(key)
    coverage = record.get('dem_valid_fraction')
    if not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or coverage < limits['dem_valid_fraction_min']:
        reasons.append('dem_valid_fraction')
    if record.get('dem_history_shape') != [4, 2, 20, 20]:
        reasons.append('dem_history_shape')
    for key in ('dem_finite', 'timestamps_monotone', 'actor_finite', 'cleanup_ok'):
        if record.get(key) is not True:
            reasons.append(key)
    for key in ('penetration', 'unexpected_reset', 'termination'):
        if record.get(key) is not False:
            reasons.append(key)
    return {'valid': not reasons, 'reasons': reasons}
