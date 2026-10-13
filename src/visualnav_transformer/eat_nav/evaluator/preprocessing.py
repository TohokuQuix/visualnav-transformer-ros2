"""Shared, fail-closed pre-Path DEM, state and Path preprocessing."""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn.functional as F


SCHEMA = 'eatnav_baseline_input_v1'
DEM_FRAMES = 4
DEM_SHAPE = (20, 20)
STATE_DIM = 22
PATH_POINTS = 8


def body_local_path(path_tile_xy, robot_xy, yaw_rad):
    path = np.asarray(path_tile_xy, dtype=np.float64)
    origin = np.asarray(robot_xy, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2 or origin.shape != (2,) or \
            not np.isfinite(path).all() or not np.isfinite(origin).all() or not math.isfinite(yaw_rad):
        raise ValueError('Path frame/shape/nonfinite')
    delta = path - origin
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    return np.stack((c*delta[:, 0] + s*delta[:, 1],
                     -s*delta[:, 0] + c*delta[:, 1]), axis=-1).astype(np.float32)


def resample_path(path_xy, count=PATH_POINTS):
    path = np.asarray(path_xy, dtype=np.float64)
    if count < 2 or path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2 or \
            not np.isfinite(path).all():
        raise ValueError('invalid Path')
    arc = np.concatenate(([0.], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))))
    if arc[-1] <= 1e-9:
        raise ValueError('zero-length Path')
    keep = np.r_[np.diff(arc) > 1e-9, True]
    arc, path = arc[keep], path[keep]
    target = np.linspace(0., arc[-1], count)
    return np.stack([np.interp(target, arc, path[:, i]) for i in range(2)], axis=-1).astype(np.float32)


def _dem_array(dem):
    array = np.asarray(dem, dtype=np.float32)
    if array.ndim == 4:
        array = array[None]
    if array.ndim != 5 or array.shape[1:3] != (DEM_FRAMES, 2):
        raise ValueError('DEM must be [B,4,2,H,W]')
    height, mask = array[:, :, 0], array[:, :, 1]
    if not np.isfinite(mask).all() or not np.logical_or(mask == 0, mask == 1).all() or \
            not np.isfinite(height[mask == 1]).all() or np.any(mask.mean(axis=(-2, -1)) < .25):
        raise ValueError('DEM mask/height invalid')
    return array


def _resize_dem(array):
    batch, frames, _, height, width = array.shape
    # A larger local raster is center-cropped before any resize; geometry is
    # only valid when the caller's resolution/frame matches the input contract.
    if height > DEM_SHAPE[0] and width > DEM_SHAPE[1]:
        row=(height-DEM_SHAPE[0])//2;col=(width-DEM_SHAPE[1])//2
        array=array[...,row:row+DEM_SHAPE[0],col:col+DEM_SHAPE[1]]
        height,width=DEM_SHAPE
    mask = torch.from_numpy(array[:, :, 1].copy()).reshape(batch*frames, 1, height, width)
    values = torch.from_numpy(np.where(array[:, :, 1] == 1, array[:, :, 0], 0).copy())
    values = values.reshape(batch*frames, 1, height, width)
    if (height, width) != DEM_SHAPE:
        valid_weight = F.interpolate(mask, size=DEM_SHAPE, mode='bilinear', align_corners=False)
        values = F.interpolate(values*mask, size=DEM_SHAPE, mode='bilinear', align_corners=False)
        values = values / valid_weight.clamp_min(1e-6)
        mask = (valid_weight >= .5).float()
    result = torch.stack((values.reshape(batch, frames, *DEM_SHAPE),
                          mask.reshape(batch, frames, *DEM_SHAPE)), dim=2)
    return result.numpy()


def _state_paths(state, paths, batch):
    state = np.asarray(state, dtype=np.float32)
    paths = np.asarray(paths, dtype=np.float32)
    if state.ndim == 1:
        state = state[None]
    if paths.ndim == 3:
        paths = paths[None]
    if state.shape != (batch, STATE_DIM) or paths.ndim != 4 or paths.shape[0] != batch or \
            paths.shape[2:] != (PATH_POINTS, 2) or paths.shape[1] < 1 or \
            not np.isfinite(state).all() or not np.isfinite(paths).all():
        raise ValueError('state/Path shape or nonfinite')
    return state, paths


def fit_normalization(dem, state, paths):
    array = _resize_dem(_dem_array(dem))
    state, paths = _state_paths(state, paths, len(array))
    mask = array[:, :, 1] == 1
    heights = array[:, :, 0][mask]
    if not len(heights):
        raise ValueError('no valid DEM heights')
    state_mean, state_std = state.mean(0), state.std(0)
    path_mean, path_std = paths.mean((0, 1, 2)), paths.std((0, 1, 2))
    return {'schema': SCHEMA, 'source': 'train_split_only',
            'dem_height_mean': float(heights.mean()),
            'dem_height_std': max(float(heights.std()), 1e-3),
            'state_mean': state_mean.tolist(), 'state_std': np.maximum(state_std, 1e-3).tolist(),
            'path_mean': path_mean.tolist(), 'path_std': np.maximum(path_std, 1e-3).tolist(),
            'dem_frames': DEM_FRAMES, 'dem_shape': list(DEM_SHAPE),
            'state_dim': STATE_DIM, 'path_points': PATH_POINTS,
            'dem_frame': 'robot_base_yaw', 'path_frame': 'robot_base_yaw',
            'dem_resolution_m': .1, 'height_unit': 'm', 'path_unit': 'm'}


def prepare_inputs(dem, state, paths, stats):
    if stats.get('schema') != SCHEMA or stats.get('source') != 'train_split_only' or \
            stats.get('dem_frames') != DEM_FRAMES or stats.get('dem_shape') != list(DEM_SHAPE) or \
            stats.get('state_dim') != STATE_DIM or stats.get('path_points') != PATH_POINTS or \
            stats.get('dem_frame') != 'robot_base_yaw' or stats.get('path_frame') != 'robot_base_yaw':
        raise ValueError('normalization/input schema mismatch')
    array = _resize_dem(_dem_array(dem))
    state, paths = _state_paths(state, paths, len(array))
    mask = array[:, :, 1]
    heights = np.where(mask == 1, (array[:, :, 0] - stats['dem_height_mean']) /
                       stats['dem_height_std'], 0).astype(np.float32)
    dem_tensor = np.stack((heights, mask), axis=2).reshape(len(array), 8, *DEM_SHAPE)
    state_tensor = (state - np.asarray(stats['state_mean'], np.float32)) / np.asarray(stats['state_std'], np.float32)
    path_tensor = (paths - np.asarray(stats['path_mean'], np.float32)) / np.asarray(stats['path_std'], np.float32)
    if not all(np.isfinite(x).all() for x in (dem_tensor, state_tensor, path_tensor)):
        raise ValueError('normalized nonfinite input')
    return {'dem': dem_tensor, 'state': state_tensor.astype(np.float32),
            'paths': path_tensor.astype(np.float32)}
