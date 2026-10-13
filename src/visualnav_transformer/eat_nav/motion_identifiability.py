"""Seed-clustered paired motion estimands for Phase B-MI."""

from __future__ import annotations

import math

import numpy as np


SCHEMA = 'eatnav_motion_identifiability_v1'
CANDIDATES = ('forward_direct', 'forward_-30', 'forward_+30',
              'reverse_direct', 'reverse_-30', 'reverse_+30')


def mirror_audit(root_tile_local: np.ndarray, dem: np.ndarray) -> dict:
    """Check an actual observed y-reflection, never infer it from terrain name."""
    root = np.asarray(root_tile_local, dtype=np.float64)
    grid = np.asarray(dem, dtype=np.float64)
    if root.shape[0] != 4 or root.shape[1] < 7 or grid.shape != (4, 4, 2, 20, 20):
        raise ValueError('mirror audit shape')
    if not np.isfinite(root).all() or not np.isfinite(grid).all():
        raise ValueError('mirror audit nonfinite')
    q = root[:, 3:7]
    yaw = np.arctan2(2 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]),
                     1 - 2 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    rows = []
    for env in range(4):
        masks = grid[env, :, 1] > .5
        reflected = masks[:, ::-1, :]
        union = np.logical_or(masks, reflected).sum(axis=(1, 2))
        intersection = np.logical_and(masks, reflected).sum(axis=(1, 2))
        iou = np.divide(intersection, union, out=np.ones(4), where=union > 0)
        common = np.logical_and(masks, reflected)
        heights = grid[env, :, 0]
        difference = heights - heights[:, ::-1, :]
        rms = [float(np.sqrt(np.mean(difference[frame][common[frame]] ** 2)))
               if common[frame].any() else None for frame in range(4)]
        passed = bool(abs(root[env, 1]) <= .02 and abs(yaw[env]) <= .02 and
                      min(iou) >= .99 and all(value is not None and value <= .005
                                               for value in rms))
        rows.append({'env': env, 'tile_local_y_m': float(root[env, 1]),
                     'yaw_rad': float(yaw[env]), 'mask_iou': iou.tolist(),
                     'height_rms_m': rms, 'pass': passed})
    return {'pass': all(row['pass'] for row in rows), 'rows': rows,
            'thresholds': {'tile_local_y_m': .02, 'yaw_rad': .02,
                           'mask_iou_min': .99, 'height_rms_m_max': .005}}


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def relative_motion(candidate: dict, direct: dict, *, direction: int, angle_deg: int) -> dict:
    if direction not in (-1, 1) or angle_deg not in (-30, 30):
        raise ValueError('unfrozen contrast')
    xy = np.asarray(candidate['delta_body_xy_m'], dtype=np.float64) - np.asarray(
        direct['delta_body_xy_m'], dtype=np.float64)
    yaw = wrap(float(candidate['delta_body_yaw_rad']) - float(direct['delta_body_yaw_rad']))
    if xy.shape != (2,) or not np.isfinite(xy).all() or not math.isfinite(yaw):
        raise ValueError('nonfinite paired motion')
    sign = 1 if angle_deg > 0 else -1
    return {'delta_xy_m': xy.tolist(), 'delta_yaw_rad': yaw,
            'signed_yaw_rad': sign * yaw,
            'signed_travel_lateral_m': direction * sign * float(xy[1])}


def _candidate_role(candidate: str) -> tuple[str, int, int]:
    if candidate not in CANDIDATES or candidate.endswith('direct'):
        raise ValueError('candidate must be ±30 arc')
    role, angle = candidate.rsplit('_', 1)
    return role + '_direct', 1 if role == 'forward' else -1, int(angle)


def seed_effects(rows: list[dict], candidate: str) -> dict:
    """Pair within seed×repeat×env, then average env/repeat before inference."""
    direct_name, direction, angle = _candidate_role(candidate)
    lookup = {}
    for row in rows:
        key = (int(row['seed']), int(row['repeat']), int(row['env']), row['candidate'])
        if key in lookup:
            raise ValueError('duplicate seed/repeat/env/candidate')
        lookup[key] = row
    seeds = sorted({key[0] for key in lookup})
    if len(seeds) != 6:
        raise ValueError('six seed clusters required')
    values = []
    repeat_differences = {'signed_yaw_rad': [], 'signed_lateral_m': [], 'delta_xy_m': []}
    for seed in seeds:
        per_repeat = []
        conditions = set()
        for repeat in (0, 1):
            per_env = []
            for env in range(4):
                direct_key = (seed, repeat, env, direct_name)
                arc_key = (seed, repeat, env, candidate)
                if direct_key not in lookup or arc_key not in lookup:
                    raise ValueError('missing paired candidate/direct')
                direct = lookup[direct_key]
                arc = lookup[arc_key]
                if direct['condition'] != arc['condition']:
                    raise ValueError('paired condition mismatch')
                conditions.add(arc['condition'])
                per_env.append(relative_motion(
                    {'delta_body_xy_m': arc['xy'], 'delta_body_yaw_rad': arc['yaw']},
                    {'delta_body_xy_m': direct['xy'], 'delta_body_yaw_rad': direct['yaw']},
                    direction=direction, angle_deg=angle))
            per_repeat.append({'signed_yaw_rad': float(np.mean([x['signed_yaw_rad'] for x in per_env])),
                               'signed_lateral_m': float(np.mean([x['signed_travel_lateral_m'] for x in per_env])),
                               'delta_xy_m': np.mean([x['delta_xy_m'] for x in per_env], axis=0)})
        if len(conditions) != 1:
            raise ValueError('seed condition changed across repeat')
        values.append({'seed': seed, 'condition': conditions.pop(),
                       'signed_yaw_rad': float(np.mean([x['signed_yaw_rad'] for x in per_repeat])),
                       'signed_lateral_m': float(np.mean([x['signed_lateral_m'] for x in per_repeat])),
                       'delta_xy_m': np.mean([x['delta_xy_m'] for x in per_repeat], axis=0).tolist()})
        for key in ('signed_yaw_rad', 'signed_lateral_m'):
            repeat_differences[key].append(abs(per_repeat[1][key] - per_repeat[0][key]))
        repeat_differences['delta_xy_m'].append(float(np.linalg.norm(
            per_repeat[1]['delta_xy_m'] - per_repeat[0]['delta_xy_m'])))
    return {'candidate': candidate, 'seed_values': values,
            'repeat_p90_yaw_rad': float(np.quantile(repeat_differences['signed_yaw_rad'], .9)),
            'repeat_p90_lateral_m': float(np.quantile(repeat_differences['signed_lateral_m'], .9)),
            'repeat_p90_xy_m': float(np.quantile(repeat_differences['delta_xy_m'], .9)),
            'repeat_differences': repeat_differences}


def bootstrap_interval(values, *, seed: int = 130013, samples: int = 10_000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all() or samples != 10_000:
        raise ValueError('six finite seed effects and frozen bootstrap count required')
    draws = np.random.default_rng(seed).integers(0, 6, (samples, 6))
    means = values[draws].mean(axis=1)
    return float(np.quantile(means, .0125)), float(np.quantile(means, .9875))


def candidate_decision(estimate: dict) -> dict:
    values = estimate['seed_values']
    conditions = [row['condition'] for row in values]
    if len(values) != 6 or conditions.count('mirror') != 2 or conditions.count('rough') != 4:
        return {'status': 'inconclusive', 'reason': 'independent/mirror support missing'}
    metrics = []
    for name, threshold, repeat_key in (
            ('signed_yaw_rad', .05, 'repeat_p90_yaw_rad'),
            ('signed_lateral_m', .03, 'repeat_p90_lateral_m')):
        seed_values = np.asarray([float(row[name]) for row in values])
        lower, upper = bootstrap_interval(seed_values)
        aligned = int((seed_values > 0).sum())
        mirror_aligned = any(row['condition'] == 'mirror' and row[name] > 0 for row in values)
        mean = float(seed_values.mean())
        passed = bool(lower > threshold and aligned >= 5 and mirror_aligned and
                      mean > 2 * estimate[repeat_key])
        metrics.append({'name': name, 'threshold': threshold, 'mean': mean,
                        'adjusted_lower': lower, 'adjusted_upper': upper,
                        'aligned_seeds': aligned,
                        'mirror_aligned': mirror_aligned,
                        'repeat_p90': estimate[repeat_key], 'pass': passed})
    status = ('pass' if any(item['pass'] for item in metrics) else
              'fail' if all(item['adjusted_upper'] < item['threshold'] for item in metrics)
              else 'inconclusive')
    return {'status': status,
            'metrics': metrics}
