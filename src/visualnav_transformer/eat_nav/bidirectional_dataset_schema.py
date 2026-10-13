"""Provisional paired StateGroup schema for forward/reverse motion labels."""

from __future__ import annotations

import numpy as np


SCHEMA = 'eatnav_forward_reverse_motion_dataset_v1'


def validate_group(group: dict) -> None:
    """Validate shape/provenance without imposing a sign on measured motion."""
    if not isinstance(group.get('state_group_id'), str) or not group['state_group_id'] or \
            not isinstance(group.get('terrain_seed'), int):
        raise ValueError('StateGroup/terrain seed required')
    dem = np.asarray(group['dem'])
    state = np.asarray(group['actor_state'])
    paths = np.asarray(group['paths'])
    direction = np.asarray(group['direction'])
    motion = np.asarray(group['motion_xy'])
    yaw = np.asarray(group['delta_yaw'])
    valid = np.asarray(group['valid'])
    if dem.shape != (4, 2, 20, 20) or state.shape != (22,) or \
            paths.ndim != 3 or paths.shape[1:] != (8, 2):
        raise ValueError('DEM/Actor/Path shape')
    count = len(paths)
    if direction.shape != (count, 1) or not np.isin(direction, [-1, 1]).all():
        raise ValueError('direction shape/value')
    if motion.shape != (count, 2) or yaw.shape != (count,) or \
            valid.shape != (count,) or valid.dtype != np.bool_ or not valid.any():
        raise ValueError('motion/yaw/valid shape')
    if not np.isfinite(dem).all() or not np.isfinite(state).all() or \
            not np.isfinite(paths).all() or not np.isfinite(motion[valid]).all() or \
            not np.isfinite(yaw[valid]).all():
        raise ValueError('nonfinite group data')
