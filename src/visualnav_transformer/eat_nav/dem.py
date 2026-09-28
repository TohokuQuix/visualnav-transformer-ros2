"""Robot-centric DEM conversion shared by simulation, datasets, and ROS.

SourceGrid uses an explicit logical raster convention: rows increase with world
+y and columns increase with world +x. origin_xy is the world position of the
logical [row=0, col=0] cell center. start_index_xy identifies that logical
cell's position in the circular storage array. A ROS GridMap adapter must
establish its own message-to-SourceGrid mapping from a measured message.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, floor, isfinite, sin
from typing import Tuple

import numpy as np

from .config import EATConfig


@dataclass(frozen=True)
class SourceGrid:
    height: np.ndarray
    valid: np.ndarray
    origin_xy: Tuple[float, float]
    resolution_m: float
    start_index_xy: Tuple[int, int]
    frame_id: str
    stamp_ns: int

    def __post_init__(self) -> None:
        if self.height.ndim != 2 or self.valid.shape != self.height.shape:
            raise ValueError("height and valid must be matching 2-D arrays")
        if not isfinite(self.resolution_m) or self.resolution_m <= 0:
            raise ValueError("resolution_m must be positive")
        if not all(isfinite(value) for value in self.origin_xy):
            raise ValueError("origin_xy must be finite")
        if len(self.start_index_xy) != 2:
            raise ValueError("start_index_xy must have two values")
        if not self.frame_id:
            raise ValueError("frame_id must be nonempty")


def _height_at(grid: SourceGrid, x_index: float, y_index: float) -> tuple[float, bool]:
    rows, cols = grid.height.shape
    for index, limit in ((x_index, cols), (y_index, rows)):
        if index < -1e-8 or index > limit - 1 + 1e-8:
            return 0.0, False
    x_index = min(max(x_index, 0.0), cols - 1.0)
    y_index = min(max(y_index, 0.0), rows - 1.0)
    if abs(x_index - round(x_index)) < 1e-8:
        x_index = float(round(x_index))
    if abs(y_index - round(y_index)) < 1e-8:
        y_index = float(round(y_index))
    x0, y0 = floor(x_index), floor(y_index)
    x1, y1 = min(x0 + 1, cols - 1), min(y0 + 1, rows - 1)
    wx, wy = x_index - x0, y_index - y0
    result = 0.0
    for lx, ly, weight in (
        (x0, y0, (1 - wx) * (1 - wy)),
        (x1, y0, wx * (1 - wy)),
        (x0, y1, (1 - wx) * wy),
        (x1, y1, wx * wy),
    ):
        if weight <= 0:
            continue
        bx = (lx + grid.start_index_xy[0]) % cols
        by = (ly + grid.start_index_xy[1]) % rows
        value = grid.height[by, bx]
        if not grid.valid[by, bx] or not np.isfinite(value):
            return 0.0, False
        result += weight * float(value)
    return result, True


def robot_dem(
    grid: SourceGrid, pose_xy_yaw: tuple[float, float, float], config: EATConfig
) -> np.ndarray:
    """Return [height minus local scalar reference, valid mask] in base-yaw raster."""
    if grid.height.shape != (config.dem_height_cells, config.dem_width_cells):
        raise ValueError("source DEM shape differs from EAT config")
    if not np.isclose(grid.resolution_m, config.dem_resolution_m, rtol=0, atol=1e-9):
        raise ValueError("source DEM resolution differs from EAT config")
    pose_x, pose_y, yaw = pose_xy_yaw
    if not all(isfinite(value) for value in pose_xy_yaw):
        raise ValueError("robot pose must be finite")
    rows, cols = config.dem_height_cells, config.dem_width_cells
    heights = np.zeros((rows, cols), dtype=np.float32)
    valid = np.zeros((rows, cols), dtype=bool)
    c, s = cos(yaw), sin(yaw)
    for row in range(rows):
        base_y = (row - (rows - 1) / 2) * config.dem_resolution_m
        for col in range(cols):
            base_x = (col - (cols - 1) / 2) * config.dem_resolution_m
            world_x = pose_x + c * base_x - s * base_y
            world_y = pose_y + s * base_x + c * base_y
            gx = (world_x - grid.origin_xy[0]) / grid.resolution_m
            gy = (world_y - grid.origin_xy[1]) / grid.resolution_m
            heights[row, col], valid[row, col] = _height_at(grid, gx, gy)
    if valid.mean() < config.dem_min_valid_fraction:
        raise ValueError("DEM has too few valid cells")
    ys = (np.arange(rows) - (rows - 1) / 2) * config.dem_resolution_m
    xs = (np.arange(cols) - (cols - 1) / 2) * config.dem_resolution_m
    radius = np.hypot(ys[:, None], xs[None, :])
    reference_cells = valid & (radius >= 0.3) & (radius <= 0.8)
    if not reference_cells.any():
        raise ValueError("DEM has no valid reference-height cells")
    reference_height = float(np.median(heights[reference_cells]))
    heights[valid] -= reference_height
    return np.stack((heights, valid.astype(np.float32)))
