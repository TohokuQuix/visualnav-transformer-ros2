"""Non-cumulative, fail-closed logging for the frozen radial five-pose smoke."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .contact_raw_capture import capture_contact_api
from .radial_contact_audit import decode_contact_buffer

RAW_FIXED_STEPS = frozenset((50, 100, 150, 180, 200))
ACTIVE_FORCE_FLOOR_N = 1e-6
FROZEN_CAPACITY = 4096


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return path


def append_step(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, sort_keys=True, allow_nan=False) + '\n').encode()
    with path.open('ab') as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def read_steps(path):
    data = Path(path).read_bytes()
    complete = data.rfind(b'\n') + 1
    rows = [json.loads(line) for line in data[:complete].splitlines()]
    tail = None if complete == len(data) else data[complete:].decode(errors='replace')
    return rows, tail


def _distribution(values):
    if not values:
        return {'count': 0, 'min': None, 'mean': None, 'p50': None, 'p95': None, 'max': None}
    a = np.asarray(values, dtype=np.float64)
    return {'count': len(values), 'min': float(a.min()), 'mean': float(a.mean()),
            'p50': float(np.percentile(a, 50)), 'p95': float(np.percentile(a, 95)),
            'max': float(a.max())}


class LeanContactRecorder:
    def __init__(self, root, paths_by_pose, *, expected_shapes=None):
        self.root = Path(root)
        self.paths_by_pose = paths_by_pose
        self.expected_shapes = expected_shapes
        self.first_contact_saved = set()
        self.last_samples = {}

    def save_last_failure(self):
        for pose_id, (step, arrays, sensors, filters, capacity) in self.last_samples.items():
            raw = self.root/'raw'/f'step-{step:04d}'/f'{pose_id}.npz'
            if not raw.exists():
                capture_contact_api(self.root, step, pose_id, arrays, sensors, filters, capacity)

    def record(self, pose_id, step, arrays, capacity):
        if capacity != FROZEN_CAPACITY:
            raise ValueError('contact capacity must remain 4096')
        arrays = [np.asarray(a) for a in arrays]
        sensors, filters = self.paths_by_pose[pose_id]
        if len(arrays) != 6:
            raise ValueError('expected six PhysX arrays')
        self.last_samples[pose_id] = (step, arrays, sensors, filters, capacity)
        shapes = [list(a.shape) for a in arrays]
        expected = self.expected_shapes or [[4096, 1], [4096, 3], [4096, 3], [4096, 1], [len(sensors), len(filters)], [len(sensors), len(filters)]]
        shape_ok = shapes == expected
        count, start = arrays[4], arrays[5]
        try:
            bounds_ok = shape_ok and bool(np.all(count >= 0) and np.all(start >= 0) and
                                          np.all(start.astype(np.int64) + count.astype(np.int64) <= capacity))
            occupancy = int(count.sum()) if shape_ok else -1
            full = occupancy >= capacity
            decoded = None
            if bounds_ok and not full:
                decoded = decode_contact_buffer(
                    sensor_paths=sensors, filter_paths=filters,
                    forces=arrays[0].reshape(-1).tolist(), points=arrays[1].tolist(),
                    normals=arrays[2].tolist(), separations=arrays[3].reshape(-1).tolist(),
                    counts=count.tolist(), starts=start.tolist(), capacity=capacity)
            event = step in RAW_FIXED_STEPS or (decoded is not None and occupancy > 0 and pose_id not in self.first_contact_saved)
            if not bounds_ok or full or event or decoded is None:
                capture_contact_api(self.root, step, pose_id, arrays, sensors, filters, capacity)
            if not bounds_ok:
                raise ValueError('contact bounds or shape mismatch')
            if full:
                raise RuntimeError('contact buffer full or truncation suspected')
            if decoded is None:
                raise ValueError('contact decode failed')
            if occupancy > 0:
                self.first_contact_saved.add(pose_id)
            entries = [entry for pair in decoded['pairs'] for entry in pair['contacts']]
            forces = [abs(entry['normal_force_n']) for entry in entries]
            separations = [entry['separation_m'] for entry in entries]
            categories = {}
            for pair in decoded['pairs']:
                name = pair['body_category']
                slot = categories.setdefault(name, {'contact_count': 0, 'force_sum_n': 0.,
                                                     'force_max_n': 0., 'min_separation_m': None})
                slot['contact_count'] += pair['contact_count']
                slot['force_sum_n'] += sum(abs(e['normal_force_n']) for e in pair['contacts'])
                slot['force_max_n'] = max(slot['force_max_n'], *(abs(e['normal_force_n']) for e in pair['contacts']))
                x = pair['min_separation_m']
                slot['min_separation_m'] = x if slot['min_separation_m'] is None else min(slot['min_separation_m'], x)
            return {'contact_count': occupancy, 'buffer_occupancy': occupancy,
                    'buffer_capacity': capacity, 'capacity_ratio': occupancy/capacity,
                    'bounds_valid': True, 'buffer_full_or_truncation_suspected': False,
                    'backend_overflow_flag': 'not_exposed',
                    'min_separation_m': min(separations) if separations else None,
                    'max_decoded_penetration_m': max((max(0., -s) for s in separations), default=0.),
                    'force_sum_n': float(sum(forces)), 'force_max_n': max(forces, default=0.),
                    'zero_force_fraction': float(sum(f <= ACTIVE_FORCE_FLOOR_N for f in forces)/len(forces)) if forces else None,
                    'all_force_distribution_n': _distribution(forces),
                    'active_force_distribution_n': _distribution([f for f in forces if f > ACTIVE_FORCE_FLOOR_N]),
                    'active_force_floor_n': ACTIVE_FORCE_FLOOR_N,
                    'body_categories': categories, 'finite': True, 'shape_valid': True,
                    'status': 'valid'}
        except (ValueError, RuntimeError, TypeError) as exc:
            raw = self.root/'raw'/f'step-{step:04d}'/f'{pose_id}.npz'
            if not raw.exists():
                capture_contact_api(self.root, step, pose_id, arrays, sensors, filters, capacity)
            atomic_json(self.root/'contact-invalid.json', {'pose_id': pose_id, 'physics_step': step,
                                                         'reason': repr(exc), 'raw_path': str(raw)})
            raise


def save_dem_frame(root, pose_id, index, frame):
    values = np.asarray(frame['transformed'], dtype=float)
    source = np.asarray(frame['source_height'], dtype=float)
    mask = np.asarray(frame['source_mask'], dtype=int)
    if not np.isfinite(values).all() or not np.isfinite(source).all() or not np.isin(mask, (0, 1)).all():
        raise ValueError('nonfinite DEM or invalid mask')
    if values.shape != (2, 20, 20) or source.shape != (20, 20) or mask.shape != (20, 20):
        raise ValueError('DEM shape mismatch')
    if not frame.get('frame_id') or not frame.get('axis') or frame.get('logical_stamp_ns') is None or frame.get('resolution_m', 0) <= 0:
        raise ValueError('DEM identity, time, axis, or resolution missing')
    item = dict(frame)
    item.update({'frame_index': index, 'shape': list(values.shape), 'finite': True,
                 'raw_sha256': hashlib.sha256(values.astype(np.float32).tobytes()).hexdigest()})
    return atomic_json(Path(root)/'dem'/pose_id/f'frame-{index:02d}.json', item)


def save_actor(root, pose_id, phase, values, field_order, config_identity):
    values = [float(v) for v in values]
    if len(values) != 22 or len(field_order) != 22 or not all(math.isfinite(v) for v in values):
        raise ValueError('Actor observation shape/order/finite mismatch')
    return atomic_json(Path(root)/'actor'/pose_id/f'{phase}.json', {
        'pose_id': pose_id, 'phase': phase, 'shape': [22], 'values': values,
        'field_order': list(field_order), 'config_identity': config_identity, 'finite': True})


def write_complete_before_close(root, value, close):
    path = atomic_json(Path(root)/'sim-finished.json', value)
    if json.loads(path.read_text()) != value:
        raise RuntimeError('completion marker read-back mismatch')
    close()
    return path


def process_matches(owned, pid, start_tick):
    return int(owned['pid']) == int(pid) and int(owned['proc_start_ticks']) == int(start_tick)
