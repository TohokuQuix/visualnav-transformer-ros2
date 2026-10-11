"""Persist pre-Path raycaster evidence before applying the frozen DEM gate."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


ARRAY_FIELDS = ('ray_hits_w', 'ray_starts_w', 'source_height', 'source_valid',
                'transformed', 'env_origin_w', 'robot_pose_w', 'root_state_w',
                'joint_pos', 'joint_vel')


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def _durable_json(path: Path, value) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _durable_npz(path: Path, arrays: dict) -> None:
    temp = path.with_suffix('.tmp')
    with temp.open('xb') as stream:
        np.savez(stream, **arrays)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)
    directory = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def compare_direct_frames(first: dict, second: dict) -> tuple[dict, dict]:
    for key in ('identity', 'timestamp_ns', 'shape', 'frame', 'resolution_m'):
        if first[key] != second[key]:
            raise ValueError('direct DEM metadata mismatch:' + key)
    a, b = np.asarray(first['transformed']), np.asarray(second['transformed'])
    if a.shape != b.shape or a.shape[0] != 2 or not (
            np.isfinite(a).all() and np.isfinite(b).all()):
        raise ValueError('direct DEM transformed shape/nonfinite')
    heights = a[0] - b[0]
    mask_a, mask_b = a[1] > .5, b[1] > .5
    mask_diff = np.logical_xor(mask_a, mask_b)
    union = int(np.logical_or(mask_a, mask_b).sum())
    iou = float(np.logical_and(mask_a, mask_b).sum() / union) if union else 1.0
    starts_a, starts_b = np.asarray(first['ray_starts_w']), np.asarray(second['ray_starts_w'])
    if starts_a.shape != starts_b.shape or starts_a.ndim != 2 or starts_a.shape[1] != 3:
        raise ValueError('direct ray origin shape mismatch')
    origin_a, origin_b = np.asarray(first['env_origin_w']), np.asarray(second['env_origin_w'])
    world_delta = starts_a - starts_b
    tile_local_delta = (starts_a - origin_a) - (starts_b - origin_b)
    hits_a, hits_b = np.asarray(first['ray_hits_w']), np.asarray(second['ray_hits_w'])
    if hits_a.shape != starts_a.shape or hits_b.shape != starts_b.shape:
        raise ValueError('direct ray hit shape mismatch')
    hit_finite = np.isfinite(hits_a).all(axis=1) & np.isfinite(hits_b).all(axis=1)
    hit_world_delta = np.full(hits_a.shape, np.nan, dtype=np.float64)
    hit_local_delta = np.full(hits_a.shape, np.nan, dtype=np.float64)
    hit_world_delta[hit_finite] = hits_a[hit_finite] - hits_b[hit_finite]
    hit_local_delta[hit_finite] = ((hits_a[hit_finite] - origin_a) -
                                    (hits_b[hit_finite] - origin_b))
    root_a, root_b = np.asarray(first['root_state_w']), np.asarray(second['root_state_w'])
    root_local_a, root_local_b = root_a.copy(), root_b.copy()
    root_local_a[:3] -= origin_a
    root_local_b[:3] -= origin_b
    def finite_max(value):
        finite = np.isfinite(value)
        return float(np.max(np.abs(value[finite]))) if finite.any() else None
    metrics = {'identity': first['identity'], 'timestamp_ns': first['timestamp_ns'],
        'height_rms_m': float(np.sqrt(np.mean(heights ** 2))),
        'height_max_abs_m': float(np.max(np.abs(heights))),
        'height_difference_cells': int(np.count_nonzero(heights)),
        'height_over_frozen_max_cells': int(np.count_nonzero(np.abs(heights) > 1e-4)),
        'mask_iou': iou, 'mask_difference_cells': int(mask_diff.sum()),
        'ray_origin_world_max_abs_m': float(np.max(np.abs(world_delta))),
        'ray_origin_tile_local_max_abs_m': float(np.max(np.abs(tile_local_delta))),
        'ray_hit_finite_pair_count': int(hit_finite.sum()),
        'ray_hit_valid_difference_count': int(np.count_nonzero(
            np.isfinite(hits_a[:, 2]) ^ np.isfinite(hits_b[:, 2]))),
        'ray_hit_world_max_abs_m': finite_max(hit_world_delta),
        'ray_hit_tile_local_max_abs_m': finite_max(hit_local_delta),
        'root_state_tile_local_max_abs': finite_max(root_local_a-root_local_b),
        'joint_pos_max_abs': finite_max(np.asarray(first['joint_pos'])-np.asarray(second['joint_pos'])),
        'joint_vel_max_abs': finite_max(np.asarray(first['joint_vel'])-np.asarray(second['joint_vel'])),
        'sensor_update_timestamp_s': [float(first['sensor_update_timestamp_s']),
                                      float(second['sensor_update_timestamp_s'])]}
    maps = {'height_signed': heights.astype(np.float32),
        'mask_difference': mask_diff,
        'ray_origin_world_delta': world_delta.astype(np.float32),
        'ray_origin_tile_local_delta': tile_local_delta.astype(np.float32),
        'ray_hit_world_delta': hit_world_delta.astype(np.float32),
        'ray_hit_tile_local_delta': hit_local_delta.astype(np.float32),
        'ray_hit_finite_pair': hit_finite}
    return metrics, maps


def persist_dem_evidence(output: Path, frames: list[list[dict]]) -> dict:
    output = Path(output)
    if len(frames) != 4 or any(len(env) != 4 for env in frames):
        raise ValueError('four-env four-frame DEM history required')
    output.mkdir(parents=True, exist_ok=True)
    ledger = {}
    metadata = []
    for env, history in enumerate(frames):
        for frame, item in enumerate(history):
            name = f'prepath-dem-env-{env}-frame-{frame}.npz'
            _durable_npz(output / name, {key: np.asarray(item[key]) for key in ARRAY_FIELDS})
            ledger[name] = _sha(output / name)
            metadata.append({'env': env, 'frame_index': frame,
                'array_file': name, 'array_sha256': ledger[name],
                **{key: item[key] for key in ('identity', 'timestamp_ns', 'shape',
                    'frame', 'resolution_m', 'sensor_timestamp_s',
                    'sensor_update_timestamp_s', 'sensor_is_outdated',
                    'sensor_update_period_s')},
                'env_origin_w': np.asarray(item['env_origin_w']).tolist(),
                'robot_pose_w': np.asarray(item['robot_pose_w']).tolist()})
    _durable_json(output / 'prepath-dem-frame-metadata.json', metadata)
    ledger['prepath-dem-frame-metadata.json'] = _sha(output / 'prepath-dem-frame-metadata.json')
    comparisons = []
    for frame in range(4):
        metrics, maps = compare_direct_frames(frames[0][frame], frames[1][frame])
        name = f'prepath-dem-diff-frame-{frame}.npz'
        _durable_npz(output / name, maps)
        ledger[name] = _sha(output / name)
        comparisons.append({**metrics, 'difference_file': name,
                            'difference_sha256': ledger[name]})
    result = {'schema': 'p1_g3_prepath_dem_diagnostic_v1', 'frames': comparisons,
              'frozen_limits': {'height_rms_m': 1e-5, 'height_max_abs_m': 1e-4,
                                'mask_iou_min': 1.0}}
    _durable_json(output / 'prepath-dem-comparison.json', result)
    ledger['prepath-dem-comparison.json'] = _sha(output / 'prepath-dem-comparison.json')
    _durable_json(output / 'prepath-dem-sha-ledger.json', ledger)
    return result


def persist_then_gate(output: Path, frames: list[list[dict]], frozen_gate):
    """The caller's unmodified gate runs only after raw data and metrics are durable."""
    persist_dem_evidence(output, frames)
    return frozen_gate(frames)
