"""Simulation-only, ROS-message-free boundary for Phase 3 GUI transport.

The ROS side and Isaac side both validate these plain JSON messages. No
production follower or robot command topic is exposed by this module.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .bidirectional_bank import bank
from .motion_14_stage_a import candidate_role


_BANK = bank()
IDS = tuple(candidate_role(c.direction, c.arc_angle_deg) for c in _BANK)
PATHS = np.asarray([c.waypoints for c in _BANK], dtype=np.float32)
WIRE_SHA = hashlib.sha256(np.ascontiguousarray(PATHS).tobytes()).hexdigest()


def validate_selection(value: dict, now_ns: int) -> int:
    """Require the exact frozen body-frame Path and a current ROS lease."""
    if value.get('type') != 'selection' or value.get('wire_path_sha256') != WIRE_SHA:
        raise ValueError('selection type/bank SHA mismatch')
    if not isinstance(now_ns, int) or not isinstance(value.get('stamp_ns'), int):
        raise ValueError('selection stamp')
    age = now_ns - value['stamp_ns']
    expiry = value.get('expires_ns')
    if not 0 <= age <= 300_000_000 or not isinstance(expiry, int) or \
            not now_ns < expiry <= value['stamp_ns'] + 500_000_000 or \
            not isinstance(value.get('generation'), int) or value['generation'] <= 0:
        raise ValueError('selection generation/stamp/lease')
    try:
        index = IDS.index(value['candidate_id'])
    except (ValueError, KeyError) as exc:
        raise ValueError('unknown selected Path') from exc
    actual = np.asarray(value.get('body_xy_m'), dtype=np.float32)
    if actual.shape != (16,) or not np.array_equal(actual, PATHS[index].reshape(-1)):
        raise ValueError('selected Path bytes differ from frozen bank')
    return index


def validate_source_frame(value: dict) -> None:
    """The published DEM and Actor state must describe one measured step."""
    if value.get('type') != 'source' or not isinstance(value.get('stamp_ns'), int) or \
            value['stamp_ns'] < 0 or not isinstance(value.get('episode_id'), int) or \
            value['episode_id'] < 0:
        raise ValueError('source identity')
    for key, shape in [('pose_map', (3,)), ('actor_pre_noise', (22,))]:
        array = np.asarray(value.get(key), dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{key} shape/nonfinite')
    if value.get('dem') is not None:
        dem = np.asarray(value['dem'], dtype=np.float32)
        if dem.shape != (2, 20, 20) or not np.isfinite(dem).all() or \
                not np.isin(dem[1], (0., 1.)).all() or \
                np.any(dem[0][dem[1] == 0] != 0) or \
                value.get('dem_stamp_ns') != value['stamp_ns'] or \
                not isinstance(value.get('dem_sensor_update_ns'), int) or \
                not 0 <= value['stamp_ns'] - value['dem_sensor_update_ns'] <= 300_000_000:
            raise ValueError('DEM frame/stamp/mask')


def validate_gated_command(value: dict, stamp_ns: int, generation: int) -> tuple[float, float]:
    if value.get('type') != 'gated_command' or value.get('stamp_ns') != stamp_ns or \
            value.get('generation') != generation or not value.get('gate_open'):
        raise ValueError('closed/stale command gate')
    command = value.get('command')
    if not isinstance(command, list) or len(command) != 2 or \
            not all(isinstance(x, (int, float)) and math.isfinite(x) for x in command):
        raise ValueError('command shape/nonfinite')
    return float(command[0]), float(command[1])


def validate_gated_stop(value: dict, stamp_ns: int, generation: int) -> tuple[float, float]:
    """Accept a supervisor stop only at the requested stamp and at zero speed."""
    if value.get('type') != 'gated_command' or value.get('stamp_ns') != stamp_ns or \
            value.get('generation') != generation or value.get('gate_open') is not False or \
            value.get('command') != [0., 0.]:
        raise ValueError('invalid supervisor stop command')
    return 0., 0.
