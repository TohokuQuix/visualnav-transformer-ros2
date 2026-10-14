"""Seed-clustered descriptive metrics for fourteen paired Path outcomes."""

from __future__ import annotations

import numpy as np

from .motion_14_predictor import CANDIDATE_IDS, contrast_indices


def _rmse(error2: np.ndarray) -> float:
    return float(np.sqrt(np.mean(error2)))


def evaluate_predictions(pred: np.ndarray, target: np.ndarray, rows: list[dict],
                         *, bootstrap_replicates: int = 1000,
                         bootstrap_seed: int = 20261014) -> dict:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape or pred.shape != (len(rows), len(CANDIDATE_IDS), 2) or \
            not np.isfinite(pred).all() or not np.isfinite(target).all():
        raise ValueError('fourteen-Path prediction/target shape or finite contract')
    seed_ids = np.asarray([int(row['terrain_seed']) for row in rows])
    seeds = np.unique(seed_ids)
    if len(seeds) < 1 or bootstrap_replicates < 1:
        raise ValueError('independent seed/bootstrap contract')
    absolute2 = np.sum((pred-target)**2, axis=-1)
    contrast2 = np.stack([np.sum(((pred[:, arc]-pred[:, direct]) -
                                  (target[:, arc]-target[:, direct]))**2, axis=-1)
                          for arc, direct in contrast_indices()], axis=-1)
    by_seed = {str(int(seed)): {'absolute_xy_rmse_m': _rmse(absolute2[seed_ids == seed]),
                               'contrast_rmse_m': _rmse(contrast2[seed_ids == seed])}
               for seed in seeds}
    random = np.random.default_rng(bootstrap_seed)
    bootstrap = []
    for _ in range(bootstrap_replicates):
        draw = random.choice(seeds, size=len(seeds), replace=True)
        indices = np.concatenate([np.where(seed_ids == seed)[0] for seed in draw])
        bootstrap.append(_rmse(contrast2[indices]))
    stratified = {}
    for field in ('posture', 'terrain_type', 'terrain_level'):
        stratified[field] = {}
        for value in sorted({row[field] for row in rows}):
            indices = np.asarray([i for i, row in enumerate(rows) if row[field] == value])
            stratified[field][str(value)] = {
                'group_count': len(indices),
                'independent_seed_count': len(set(seed_ids[indices].tolist())),
                'absolute_xy_rmse_m': _rmse(absolute2[indices]),
                'contrast_rmse_m': _rmse(contrast2[indices])}
    by_candidate = {role: {'absolute_xy_rmse_m': _rmse(absolute2[:, index]),
                           'mean_actual_xy_m': target[:, index].mean(0).tolist(),
                           'mean_predicted_xy_m': pred[:, index].mean(0).tolist()}
                    for index, role in enumerate(CANDIDATE_IDS)}
    angle_contrast = {}
    for angle in (30, 60, 90):
        columns = [j for j, (arc, _) in enumerate(contrast_indices())
                   if abs(int(CANDIDATE_IDS[arc].rsplit('_', 1)[1])) == angle]
        angle_contrast[str(angle)] = _rmse(contrast2[:, columns])
    direction_contrast = {'forward': _rmse(contrast2[:, :6]),
                          'reverse': _rmse(contrast2[:, 6:])}
    posture_lookup = {(row['terrain_seed'], row['env_tile_id'], row['posture']): i
                      for i, row in enumerate(rows)}
    interaction = []
    for (seed, tile, posture), front in posture_lookup.items():
        if posture != 'front_folded':
            continue
        rear = posture_lookup.get((seed, tile, 'rear_folded'))
        if rear is None:
            continue
        for arc, direct in contrast_indices():
            predicted = (pred[rear, arc]-pred[rear, direct]) - \
                        (pred[front, arc]-pred[front, direct])
            measured = (target[rear, arc]-target[rear, direct]) - \
                       (target[front, arc]-target[front, direct])
            interaction.append(np.sum((predicted-measured)**2))
    return {'group_count': len(rows), 'independent_seed_count': len(seeds),
            'absolute_xy_rmse_m': _rmse(absolute2),
            'contrast_rmse_m': _rmse(contrast2),
            'contrast_bootstrap_seed_cluster_95_interval_m':
                np.quantile(bootstrap, [.025, .975]).tolist(),
            'seed': by_seed, 'strata': stratified,
            'candidate': by_candidate,
            'angle_contrast_rmse_m': angle_contrast,
            'direction_contrast_rmse_m': direction_contrast,
            'front_rear_path_interaction_rmse_m': _rmse(np.asarray(interaction))
                if interaction else None}
