"""Frozen, outcome-blind ingredients for counterfactual feasible-set probes."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np


def select_stratified_states(rows, quotas, seed: int, episode_cap: int = 1):
    """Select deterministically without reading future outcomes.

    A short stratum stays short; other strata are never used as replacements.
    The caller supplies train-derived strata and independently fixed split rows.
    """
    if episode_cap < 1:
        raise ValueError('episode_cap must be positive')
    grouped = defaultdict(list)
    for row in rows:
        if row['split'] == 'heldout':
            raise ValueError('held-out rows cannot be selected')
        if row['stratum'] in quotas:
            grouped[(row['terrain_type'], row['stratum'])].append(row)
    terrains = sorted({terrain for terrain, _ in grouped})
    used = defaultdict(int)
    selected = []

    def rank(row):
        identity = [seed, row['split'], row['terrain_type'], row['stratum'],
                    row['episode_key'], row['view_index']]
        return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    for terrain in terrains:
        for stratum, quota in quotas.items():
            if quota < 0:
                raise ValueError('quota must be nonnegative')
            for row in sorted(grouped[(terrain, stratum)], key=rank):
                if sum(r['terrain_type'] == terrain and r['stratum'] == stratum for r in selected) >= quota:
                    break
                episode = tuple(row['episode_key'])
                if used[episode] >= episode_cap:
                    continue
                used[episode] += 1
                selected.append(row)
    return selected


def maneuver_bank(nominal, goal, slight_scale: float, strong_scale: float):
    """Endpoint-aligned nominal and symmetric lateral arcs in robot base frame."""
    nominal = np.asarray(nominal, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if nominal.ndim != 2 or nominal.shape[1] != 2 or goal.shape != (2,):
        raise ValueError('expected (N,2) trajectory and (2,) relative goal')
    if len(nominal) < 2 or not np.isfinite(nominal).all() or not np.isfinite(goal).all():
        raise ValueError('invalid trajectory or goal')
    if not 0 < slight_scale < strong_scale or not np.isfinite(strong_scale):
        raise ValueError('lateral scales must be finite, positive and ordered')
    t = np.arange(1, len(nominal) + 1, dtype=np.float64) / len(nominal)
    aligned = nominal + t[:, None] * (goal - nominal[-1])
    unit = goal / np.linalg.norm(goal) if np.linalg.norm(goal) > 1e-8 else np.array([1.0, 0.0])
    left = np.array([-unit[1], unit[0]])
    bump = np.sin(np.pi * t)[:, None] * left
    paths = {'nominal': aligned}
    for label, scale in [('slight', slight_scale), ('strong', strong_scale)]:
        paths[f'{label}_left'] = aligned + scale * bump
        paths[f'{label}_right'] = aligned - scale * bump
    for path in paths.values():
        path[-1] = goal
    return paths


def dem_difficulty(dem, goal):
    """Outcome-blind input score used solely for within-terrain strata.

    The height channel is robot-relative; the mask channel marks known cells.
    """
    dem = np.asarray(dem, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    if dem.ndim != 4 or dem.shape[1] != 2 or goal.shape != (2,):
        raise ValueError('expected DEM history (T,2,H,W) and a 2D goal')
    height, known = dem[-1]
    mask = known > .5
    if not mask.any() or not np.isfinite(height[mask]).all():
        return float('inf')
    values = height[mask]
    roughness = float(np.std(values))
    gradients = []
    adjacent = mask[1:] & mask[:-1]
    if adjacent.any():
        gradients.append(np.abs(np.diff(height, axis=0))[adjacent])
    adjacent = mask[:, 1:] & mask[:, :-1]
    if adjacent.any():
        gradients.append(np.abs(np.diff(height, axis=1))[adjacent])
    slope = float(np.mean(np.concatenate(gradients))) if gradients else 0.0
    unknown = 1.0 - float(mask.mean())
    bearing = abs(float(np.arctan2(goal[1], goal[0]))) / np.pi
    return roughness + .5 * slope + .2 * unknown + .05 * bearing
