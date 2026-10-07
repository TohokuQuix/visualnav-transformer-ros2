"""Frozen, Outcome-blind path-corridor extraction and support audit."""
from __future__ import annotations

from collections import defaultdict, Counter
import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from eat_nav_corridor_audit import ROOT, read, sha, source_frames_path
from visualnav_transformer.eat_nav.path_corridor import (
    anchor_reference_height, corridor_points, local_to_world, resample_path,
    sample_source_grid,
)


def _masked_mean(values, mask, axis):
    return np.sum(values * mask, axis=axis) / np.maximum(1, np.sum(mask, axis=axis))


def profile_from_corridor(corridor: np.ndarray, path_length: float) -> np.ndarray:
    """Four-frame station features; last feature is invalid ratio."""
    rows = []
    spacing = path_length / (corridor.shape[-2] - 1)
    for frame in corridor:
        height, valid = frame
        mid = height.shape[-1] // 2
        mean = _masked_mean(height, valid, axis=1)
        safe_min = np.min(np.where(valid > 0, height, np.inf), axis=1)
        safe_max = np.max(np.where(valid > 0, height, -np.inf), axis=1)
        safe_min[~np.isfinite(safe_min)] = 0
        safe_max[~np.isfinite(safe_max)] = 0
        left = _masked_mean(height[:, :mid], valid[:, :mid], axis=1)
        right = _masked_mean(height[:, mid + 1:], valid[:, mid + 1:], axis=1)
        step = np.concatenate([[0], np.diff(mean)])
        features = np.stack([height[:, mid] * valid[:, mid], mean, safe_min, safe_max,
                             left, right, step, step / max(spacing, 1e-9),
                             left - right], axis=1)
        rows.append(features)
    invalid_ratio = 1 - np.mean(corridor[:, 1], axis=(0, 2))
    return np.concatenate(rows + [invalid_ratio[:, None]], axis=1).astype(np.float32)


def candidate_metrics(corridor: np.ndarray, path_length: float, threshold: float) -> dict:
    height, valid = corridor[:, 0], corridor[:, 1] > .5
    anchor_height = height[-1]
    anchor_valid = valid[-1]
    station_height = _masked_mean(anchor_height, anchor_valid, axis=1)
    station_valid = anchor_valid.any(axis=1)
    steps = np.diff(station_height)
    spacing = path_length / 15
    lateral = np.diff(anchor_height, axis=1)
    lateral_valid = anchor_valid[:, 1:] & anchor_valid[:, :-1]
    left = _masked_mean(anchor_height[:, :3], anchor_valid[:, :3], axis=1)
    right = _masked_mean(anchor_height[:, 4:], anchor_valid[:, 4:], axis=1)
    nonflat = np.flatnonzero(station_valid & (np.abs(station_height - station_height[0]) > threshold))
    first = float(nonflat[0] * spacing) if len(nonflat) else None
    return {'height_std_m': float(np.std(anchor_height[anchor_valid])) if anchor_valid.any() else None,
            'max_longitudinal_step_m': float(np.max(np.abs(steps))) if steps.size else 0.0,
            'max_longitudinal_slope': float(np.max(np.abs(steps)) / max(spacing, 1e-9)) if steps.size else 0.0,
            'max_lateral_slope': float(np.max(np.abs(lateral[lateral_valid])) / (0.8 / 6)) if lateral_valid.any() else None,
            'left_right_height_difference_m': float(np.max(np.abs(left - right))),
            'invalid_ratio': float(1 - np.mean(valid)),
            'valid_ratio': float(np.mean(valid)),
            'first_nonflat_station_distance_m': first,
            'path_length_m': float(path_length)}


def pair_metrics(a: np.ndarray, b: np.ndarray, ma: dict, mb: dict) -> dict:
    mask = (a[-1, 1] > .5) & (b[-1, 1] > .5)
    rms = float(np.sqrt(np.mean((a[-1, 0][mask] - b[-1, 0][mask])**2))) if mask.any() else None
    return {'corridor_rms_m': rms,
            'max_step_difference_m': abs(ma['max_longitudinal_step_m'] - mb['max_longitudinal_step_m']),
            'roughness_difference_m': abs((ma['height_std_m'] or 0) - (mb['height_std_m'] or 0))}


def classify_group(corridors: dict[str, np.ndarray], path_length: float | dict,
                   thresholds: dict) -> dict:
    lengths = path_length if isinstance(path_length, dict) else {key: path_length for key in corridors}
    metrics = {key: candidate_metrics(value, lengths[key],
               thresholds['nonflat_station_height_delta_min_m']) for key, value in corridors.items()}
    pairs = {}
    keys = sorted(corridors)
    for i, left in enumerate(keys):
        for right in keys[i + 1:]:
            pairs[left + '|' + right] = pair_metrics(corridors[left], corridors[right], metrics[left], metrics[right])
    if any(value['valid_ratio'] < thresholds['min_candidate_valid_ratio'] for value in metrics.values()):
        group_class = 'insufficient_or_unknown'
    elif any(value['first_nonflat_station_distance_m'] is not None and
             thresholds['transition_first_nonflat_range_m'][0] <= value['first_nonflat_station_distance_m'] <=
             thresholds['transition_first_nonflat_range_m'][1] for value in metrics.values()):
        group_class = 'transition_corridor'
    elif any(value['corridor_rms_m'] is not None and
             (value['corridor_rms_m'] > thresholds['differential_pair_rms_min_m'] or
              value['max_step_difference_m'] > thresholds['differential_step_difference_min_m'])
             for value in pairs.values()):
        group_class = 'differential_corridor'
    elif all(value['height_std_m'] is not None and value['height_std_m'] <= thresholds['flat_height_std_max_m'] and
             value['max_longitudinal_step_m'] <= thresholds['flat_longitudinal_step_max_m'] for value in metrics.values()):
        group_class = 'flat_corridor'
    else:
        group_class = 'common_rough_corridor'
    # Connected components over similarity graph; records ambiguous chaining explicitly.
    parent = {key: key for key in keys}
    def find(key):
        while parent[key] != key:
            key = parent[key]
        return key
    for pair, value in pairs.items():
        if value['corridor_rms_m'] is not None and value['corridor_rms_m'] <= thresholds['cluster_pair_rms_max_m']:
            left, right = pair.split('|')
            parent[find(right)] = find(left)
    return {'class': group_class, 'candidate_metrics': metrics, 'pair_metrics': pairs,
            'cluster_count': len({find(key) for key in keys})}


def _grid_points():
    axis = (np.arange(20) - 9.5) * .1
    yy, xx = np.meshgrid(axis, axis, indexing='ij')
    return np.stack([xx, yy], axis=-1)


def prepare(root: Path, config_path: Path, output: Path) -> dict:
    config = yaml.safe_load(config_path.read_text())
    source_audit = read(output / 'source-provenance.json')
    if source_audit['status'] != 'pass' or source_audit['record_count'] != 252:
        raise ValueError('C0 source audit missing or failed')
    records = read(root / 'record-identity.json')
    with np.load(root / 'combined-features.npz') as data:
        paths = data['candidate_path'].copy()
        historical_dem = data['dem_history'].copy()
        goals = data['relative_goal'].copy()
    if paths.shape != (252, 8, 2) or historical_dem.shape != (252, 4, 2, 20, 20):
        raise ValueError('frozen feature shape mismatch')
    full = np.empty_like(historical_dem)
    corridor = np.empty((252, 4, 2, 16, 7), dtype=np.float32)
    profile = np.empty((252, 16, 37), dtype=np.float32)
    by_group = defaultdict(dict)
    lengths = defaultdict(dict)
    source_sha = {}
    frame_cache = {}
    full_local = _grid_points()
    lateral = np.linspace(-.4, .4, 7)
    for index, record in enumerate(records):
        group = record['group']
        if group not in frame_cache:
            source_path = source_frames_path(root, group)
            source_sha[group] = {'path': str(source_path), 'sha256': sha(source_path)}
            frame_cache[group] = read(source_path)
        frames = frame_cache[group]
        anchor = frames[-1]
        pose = anchor['pose']
        world_full = local_to_world(full_local, pose[:2], anchor['yaw'])
        reference = anchor_reference_height(anchor)
        stations, tangents, distance = resample_path(paths[index], 16)
        corridor_local = corridor_points(stations, tangents, lateral)
        world_corridor = local_to_world(corridor_local, pose[:2], anchor['yaw'])
        if not np.isfinite(goals[index]).all():
            raise ValueError(f'nonfinite goal: {record["process_id"]}')
        for frame_index, frame in enumerate(frames):
            full[index, frame_index] = sample_source_grid(frame['dem'], world_full, reference)
            corridor[index, frame_index] = sample_source_grid(frame['dem'], world_corridor, reference)
            observed = historical_dem[index, frame_index]
            saved = np.asarray(frame['dem']['transformed'], dtype=np.float32)
            both = (observed[1] > .5) & (saved[1] > .5)
            union = (observed[1] > .5) | (saved[1] > .5)
            rms = float(np.sqrt(np.mean((observed[0][both] - saved[0][both])**2))) if both.any() else float('inf')
            iou = float(both.sum() / union.sum()) if union.any() else 0
            if rms > .03 or iou < .95:
                raise ValueError(f'feature/source DEM mismatch {record["process_id"]} frame {frame_index}: {rms}, {iou}')
        profile[index] = profile_from_corridor(corridor[index], distance[-1])
        if record['candidate'] not in by_group[group]:
            by_group[group][record['candidate']] = corridor[index]
            lengths[group][record['candidate']] = float(distance[-1])
    support = {group: classify_group(by_group[group], lengths[group], config['support'])
               for group in sorted(by_group)}
    output.mkdir(parents=True, exist_ok=True)
    temp_npz = output / 'corridor-features.npz.tmp'
    with temp_npz.open('wb') as stream:
        np.savez_compressed(stream, full_dem=full, corridor=corridor, profile=profile)
    temp_npz.replace(output / 'corridor-features.npz')
    audit = {'status': 'pass', 'config_sha256': sha(config_path),
             'input_features_sha256': sha(root / 'combined-features.npz'),
             'source_frames': source_sha, 'group_support': support,
             'class_counts': dict(Counter(row['class'] for row in support.values())),
             'features_sha256': sha(output / 'corridor-features.npz')}
    temp = output / 'corridor-support-audit.json.tmp'
    temp.write_text(json.dumps(audit, indent=2, sort_keys=True) + '\n')
    temp.replace(output / 'corridor-support-audit.json')
    return audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = prepare(args.root, args.config, args.output)
    print(json.dumps({'status': result['status'], 'class_counts': result['class_counts'],
                      'config_sha256': result['config_sha256']}))


if __name__ == '__main__':
    main()
