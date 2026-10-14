"""Explicit GridMap storage mapping; real layout remains a measured gate."""

from dataclasses import dataclass

import numpy as np

from visualnav_transformer.eat_nav.dem import SourceGrid
from .wire import stamp_ns


@dataclass(frozen=True)
class GridMapLayout:
    layer: str
    memory_order: str  # C or F
    raw_axes: str  # yx or xy, in outer/inner storage order


def source_grid_from_gridmap(message, layout: GridMapLayout) -> SourceGrid:
    """Convert only a declared and measured GridMap layout, never guess it."""
    if layout.memory_order not in ('C', 'F') or layout.raw_axes not in ('yx', 'xy') or \
            message.header.frame_id != 'map' or layout.layer not in message.layers:
        raise ValueError('GridMap layout/frame/layer not verified')
    resolution = float(message.info.resolution)
    if not np.isclose(resolution, .1, rtol=0, atol=1e-9) or \
            not np.isclose(message.info.length_x, 2., rtol=0, atol=1e-6) or \
            not np.isclose(message.info.length_y, 2., rtol=0, atol=1e-6):
        raise ValueError('GridMap geometry differs from frozen DEM')
    layer = message.data[message.layers.index(layout.layer)]
    if len(layer.data) != 400 or len(layer.layout.dim) != 2 or \
            any(int(dim.size) != 20 for dim in layer.layout.dim) or \
            int(layer.layout.data_offset) != 0:
        raise ValueError('GridMap dimensions/offset')
    raw = np.asarray(layer.data, dtype=np.float32).reshape(20, 20,
                                                         order=layout.memory_order)
    if layout.raw_axes == 'yx':
        storage = raw
        sx, sy = int(message.inner_start_index), int(message.outer_start_index)
    else:
        storage = raw.T
        sx, sy = int(message.outer_start_index), int(message.inner_start_index)
    if not 0 <= sx < 20 or not 0 <= sy < 20:
        raise ValueError('GridMap circular start index')
    center = (float(message.info.pose.position.x), float(message.info.pose.position.y))
    if not np.isfinite(center).all():
        raise ValueError('GridMap center')
    valid = np.isfinite(storage)
    height = np.where(valid, storage, 0).astype(np.float32)
    origin = (center[0]-9.5*resolution, center[1]-9.5*resolution)
    return SourceGrid(height, valid, origin, resolution, (sx, sy), 'map',
                      stamp_ns(message.header.stamp))
