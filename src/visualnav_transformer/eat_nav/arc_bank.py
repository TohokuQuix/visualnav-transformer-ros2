"""Frozen body-frame, forward-start constant-curvature Arc geometry."""

from __future__ import annotations

import math

import numpy as np


SCHEMA = 'eatnav_body_arc_bank_v1'
STAGE1_ANGLES = (0, -30, 30, -60, 60, -90, 90)
STAGE2_ANGLES = (-120, 120, -150, 150, -180, 180)
ARC_LENGTH_M = .75
POINTS = 8


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


def _self_intersects(path: np.ndarray) -> bool:
    """Detect crossings or overlap of nonadjacent Path segments."""
    eps = 1e-10
    for i in range(len(path) - 1):
        a, b = path[i:i + 2]
        r = b - a
        for j in range(i + 2, len(path) - 1):
            c, d = path[j:j + 2]
            s = d - c
            denom = _cross(r, s)
            ca = c - a
            if abs(denom) <= eps:
                if abs(_cross(ca, r)) > eps:
                    continue
                rr = float(np.dot(r, r))
                lo, hi = sorted((float(np.dot(c - a, r) / rr),
                                 float(np.dot(d - a, r) / rr)))
                if max(0., lo) <= min(1., hi) + eps:
                    return True
            else:
                t = _cross(ca, s) / denom
                u = _cross(ca, r) / denom
                if -eps <= t <= 1. + eps and -eps <= u <= 1. + eps:
                    return True
    return False


def arc_waypoints(angle_deg: int, *, arc_length_m: float = ARC_LENGTH_M,
                  points: int = POINTS) -> np.ndarray:
    """Sample a circular arc at equal arclength, including the origin."""
    if angle_deg not in STAGE1_ANGLES + STAGE2_ANGLES or points != POINTS or \
            not math.isfinite(arc_length_m) or arc_length_m <= 0:
        raise ValueError('Arc angle/length/point contract')
    s = np.linspace(0., arc_length_m, points, dtype=np.float64)
    angle = math.radians(angle_deg)
    if angle == 0:
        return np.stack((s, np.zeros_like(s)), axis=1)
    radius = arc_length_m / angle
    phi = s / arc_length_m * angle
    return np.stack((radius * np.sin(phi), radius * (1. - np.cos(phi))), axis=1)


def audit_arc(path_xy, angle_deg: int, *, arc_length_m: float = ARC_LENGTH_M) -> dict:
    """Compare sampled Path to its requested analytic arc; fail invalid shapes."""
    path = np.asarray(path_xy, dtype=np.float64)
    if path.shape != (POINTS, 2) or not np.isfinite(path).all() or \
            np.linalg.norm(path[0]) > 1e-9:
        raise ValueError('Arc waypoint shape/origin/nonfinite')
    segments = np.diff(path, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if (lengths <= 1e-9).any():
        raise ValueError('repeated or zero-length waypoint')
    expected = arc_waypoints(angle_deg, arc_length_m=arc_length_m)
    intersects = _self_intersects(path)
    geometry_valid = bool(np.allclose(path, expected, atol=1e-8, rtol=0) and not intersects)
    heading = np.unwrap(np.arctan2(segments[:, 1], segments[:, 0]))
    turn = np.diff(heading)
    curvature = np.abs(turn) / ((lengths[:-1] + lengths[1:]) / 2)
    radius = None if angle_deg == 0 else abs(arc_length_m / math.radians(angle_deg))
    return {'schema': SCHEMA, 'angle_deg': angle_deg, 'geometry_valid': geometry_valid,
            'origin_xy_m': path[0].tolist(), 'endpoint_xy_m': path[-1].tolist(),
            'endpoint_bearing_deg': math.degrees(math.atan2(path[-1, 1], path[-1, 0])),
            'tangent_change_deg': angle_deg, 'first_chord_heading_deg': math.degrees(heading[0]),
            'last_chord_heading_deg': math.degrees(heading[-1]),
            'path_length_m': float(lengths.sum()), 'waypoint_intervals_m': lengths.tolist(),
            'curvature_max_inv_m': float(curvature.max()), 'turn_radius_m': radius,
            'self_intersection': intersects}
