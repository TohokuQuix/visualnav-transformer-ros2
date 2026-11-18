"""Versioned body-frame Forward/Reverse Path Bank; no follower rule changes."""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SCHEMA = 'eatnav_forward_reverse_path_bank_v1'
ANGLES = (0, -30, 30, -60, 60, -90, 90)
LENGTH_M = 0.75
POINTS = 8


@dataclass(frozen=True)
class PathCandidate:
    waypoints: np.ndarray
    direction: int
    arc_id: str
    arc_angle_deg: int
    arc_length_m: float
    geometry_version: str = SCHEMA


def candidate(direction: int, arc_angle_deg: int) -> PathCandidate:
    """Angle is tangent turn along travel, with positive turn to travel-left.

    Reverse travel frame is body frame rotated by pi. Thus reverse positive
    turn bends toward body-right, while the required body yaw change is
    positive. The C++ follower infers reverse from the initial tangent;
    explicit direction remains a separate bank/model/validation field.
    """
    if direction not in (-1, 1) or arc_angle_deg not in ANGLES:
        raise ValueError('unfrozen direction or arc angle')
    s = np.linspace(0., LENGTH_M, POINTS, dtype=np.float64)
    theta = math.radians(arc_angle_deg)
    if theta == 0:
        local = np.column_stack((s, np.zeros_like(s)))
    else:
        radius = LENGTH_M / theta
        phase = theta * s / LENGTH_M
        local = np.column_stack((radius * np.sin(phase), radius * (1. - np.cos(phase))))
    waypoints = np.ascontiguousarray(local * direction)
    waypoints.setflags(write=False)
    role = 'forward' if direction == 1 else 'reverse'
    return PathCandidate(waypoints, direction, f'{role}_{arc_angle_deg:+d}',
                         arc_angle_deg, LENGTH_M)


def bank() -> tuple[PathCandidate, ...]:
    return tuple(candidate(direction, angle) for direction in (1, -1)
                 for angle in ANGLES)


def audit_candidate(item: PathCandidate) -> dict:
    path = item.waypoints
    expected = candidate(item.direction, item.arc_angle_deg)
    if path.shape != (POINTS, 2) or not np.isfinite(path).all():
        raise ValueError('invalid Path shape/nonfinite')
    chords = np.diff(path, axis=0)
    chord_length = np.linalg.norm(chords, axis=1)
    first = math.atan2(chords[0, 1], chords[0, 0])
    desired_initial = 0. if item.direction == 1 else math.pi
    initial_error = math.atan2(math.sin(first - desired_initial),
                               math.cos(first - desired_initial))
    path_length = float(chord_length.sum())
    valid = bool(np.array_equal(path, expected.waypoints) and
                 np.linalg.norm(path[0]) < 1e-12 and
                 np.all(chord_length > 0) and abs(initial_error) <= math.radians(7))
    return {'schema': SCHEMA, 'arc_id': item.arc_id, 'direction': item.direction,
            'arc_angle_deg': item.arc_angle_deg, 'geometry_valid': valid,
            'path_length_m': path_length, 'arc_length_m': item.arc_length_m,
            'endpoint_xy_m': path[-1].tolist(),
            'first_chord_tangent_error_deg': math.degrees(initial_error),
            'curvature_signed_inv_m': math.radians(item.arc_angle_deg) / LENGTH_M,
            'body_yaw_change_deg': item.arc_angle_deg,
            'waypoint_intervals_m': chord_length.tolist()}
