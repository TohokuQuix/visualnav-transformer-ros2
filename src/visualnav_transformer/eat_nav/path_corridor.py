"""Anchor-local DEM and candidate Path corridor geometry."""
from __future__ import annotations

import math
import numpy as np


def local_to_world(points, origin, yaw):
    points = np.asarray(points, dtype=float)
    c, s = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    return points @ rotation.T + np.asarray(origin, dtype=float)


def bilinear_world(height, mask, origin, resolution, xy):
    height = np.asarray(height)
    mask = np.asarray(mask)
    col, row = (np.asarray(xy) - np.asarray(origin)) / resolution
    if not (np.isfinite(row) and np.isfinite(col)):
        return 0.0, False
    # Floating-point positions infinitesimally outside a grid remain outside.
    if row < 0 or col < 0 or row > height.shape[0] - 1 or col > height.shape[1] - 1:
        return 0.0, False
    lo_row, lo_col = int(math.floor(row)), int(math.floor(col))
    hi_row, hi_col = min(lo_row + 1, height.shape[0] - 1), min(lo_col + 1, height.shape[1] - 1)
    fr, fc = row - lo_row, col - lo_col
    contributions = ((lo_row, lo_col, (1-fr)*(1-fc)),
                     (lo_row, hi_col, (1-fr)*fc),
                     (hi_row, lo_col, fr*(1-fc)),
                     (hi_row, hi_col, fr*fc))
    value = 0.0
    for r, c, weight in contributions:
        if weight <= 0:
            continue
        if not bool(mask[r, c]) or not np.isfinite(height[r, c]):
            return 0.0, False
        value += weight * float(height[r, c])
    return value, True


def resample_path(path, stations):
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) != 8 or not np.isfinite(path).all():
        raise ValueError('candidate Path must contain eight finite XY points')
    polyline = np.concatenate([np.zeros((1, 2)), path], axis=0)
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    if np.any(lengths <= 0):
        raise ValueError('candidate Path has zero-length segment')
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    distances = np.linspace(0, cumulative[-1], stations)
    xy = np.stack([np.interp(distances, cumulative, polyline[:, axis]) for axis in (0, 1)], axis=1)
    segments = np.minimum(np.searchsorted(cumulative, distances, side='right') - 1, len(lengths) - 1)
    tangents = np.diff(polyline, axis=0)[segments] / lengths[segments, None]
    return xy, tangents, distances


def corridor_points(stations, tangents, lateral_offsets):
    normal = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)
    return stations[:, None, :] + normal[:, None, :] * np.asarray(lateral_offsets)[None, :, None]


def sample_source_grid(dem, world_xy, reference_height):
    grid = np.zeros((2,) + world_xy.shape[:-1], dtype=np.float32)
    for index in np.ndindex(world_xy.shape[:-1]):
        value, valid = bilinear_world(dem['source_height'], dem['source_mask'],
                                      dem['origin_xy'], dem['resolution_m'], world_xy[index])
        if valid:
            grid[(0,) + index] = value - reference_height
            grid[(1,) + index] = 1
    return grid


def anchor_reference_height(anchor_frame, inner=0.3, outer=0.8):
    dem = anchor_frame['dem']
    height = np.asarray(dem['source_height'], dtype=float)
    mask = np.asarray(dem['source_mask'], dtype=bool)
    y, x = np.indices(height.shape)
    xy = np.stack([x, y], axis=-1) * float(dem['resolution_m']) + np.asarray(dem['origin_xy'])
    radius = np.linalg.norm(xy - np.asarray(anchor_frame['pose'][:2]), axis=-1)
    selected = mask & np.isfinite(height) & (radius >= inner) & (radius <= outer)
    if not np.any(selected):
        raise ValueError('no valid anchor reference annulus')
    return float(np.median(height[selected]))
