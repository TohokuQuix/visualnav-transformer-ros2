"""Robot-forward Direct and audited ±30/60/90 degree GUI arc candidates."""

from __future__ import annotations

import math

import numpy as np

from .arc_bank import ARC_LENGTH_M, STAGE1_ANGLES, arc_waypoints


DEMO_ARC_ANGLES = {
    'direct': 0,
    'left_30': 30,
    'right_30': -30,
    'left_60': 60,
    'right_60': -60,
    'left_90': 90,
    'right_90': -90,
}
DEMO_ARC_ROLES = tuple(DEMO_ARC_ANGLES)


def candidates_from_body_arcs(robot_pose_map):
    """Return seven fixed body-frame Paths and their map-frame copies."""
    pose = np.asarray(robot_pose_map, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('robot pose shape/nonfinite')
    if set(DEMO_ARC_ANGLES.values()) != set(STAGE1_ANGLES):
        raise ValueError('demo Arc Bank differs from audited stage one')
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    body = {role: arc_waypoints(angle) for role, angle in DEMO_ARC_ANGLES.items()}
    mapped = {role: path @ rotation.T + pose[:2] for role, path in body.items()}
    info = {'bank': 'eatnav_body_arc_bank_v1_stage1', 'arc_length_m': ARC_LENGTH_M,
            'angle_semantics': 'total_path_tangent_change_deg',
            'maximum_turn_deg': 90, 'roles': DEMO_ARC_ANGLES.copy()}
    return body, mapped, info
