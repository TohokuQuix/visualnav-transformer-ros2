"""Read-only subgroup analysis of frozen MI-B test predictions."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    from .eat_nav_mi_b_pilot import atomic_json, load
except ImportError:
    from eat_nav_mi_b_pilot import atomic_json, load
from visualnav_transformer.eat_nav.mi_b_pilot import CANDIDATES, contrast_indices, sha


def subgroup_metrics(pred: np.ndarray, target: np.ndarray, groups: np.ndarray,
                     candidate_indices: tuple[int, ...]) -> dict:
    result = {}
    for group in sorted(set(groups.tolist())):
        chosen = groups == group
        error = (pred[chosen][:, candidate_indices] - target[chosen][:, candidate_indices])**2
        result[str(group)] = {'groups': int(chosen.sum()),
                              'xy_rmse_m': float(np.sqrt(error.sum(-1).mean()))}
    return result


def contrast_by_group(pred: np.ndarray, target: np.ndarray, groups: np.ndarray) -> dict:
    errors = np.stack([np.sum(((pred[:, arc] - pred[:, direct]) -
                                (target[:, arc] - target[:, direct]))**2, -1)
                       for arc, direct in contrast_indices()], axis=1)
    return {str(group): {'groups': int((groups == group).sum()),
                         'contrast_rmse_m': float(np.sqrt(errors[groups == group].mean()))}
            for group in sorted(set(groups.tolist()))}


def contrast_vectors(pred: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for (arc, direct) in contrast_indices():
        measured = target[:, arc] - target[:, direct]
        predicted = pred[:, arc] - pred[:, direct]
        dot = np.sum(measured * predicted, axis=-1)
        supported = np.linalg.norm(measured, axis=-1) > 1e-6
        result[CANDIDATES[arc]] = {
            'measured_mean_xy_m': measured.mean(0).tolist(),
            'predicted_mean_xy_m': predicted.mean(0).tolist(),
            'contrast_rmse_m': float(np.sqrt(np.sum((predicted - measured)**2, -1).mean())),
            'same_direction_fraction': float((dot[supported] > 0).mean()) if supported.any() else None,
            'supported_groups': int(supported.sum())}
    return result


def posture_interaction(pred: np.ndarray, target: np.ndarray, metadata: list[dict]) -> dict:
    pairs = {}
    for i, row in enumerate(metadata):
        key = (row['terrain_seed'], row['env_tile_id'])
        pairs.setdefault(key, {})[row['posture']] = i
    if any(set(pair) != {'front_folded', 'rear_folded'} for pair in pairs.values()):
        raise ValueError('posture pairing incomplete')
    errors = []
    measured = []
    predicted = []
    for pair in pairs.values():
        f, r = pair['front_folded'], pair['rear_folded']
        for arc, direct in contrast_indices():
            actual = (target[r, arc] - target[r, direct]) - (target[f, arc] - target[f, direct])
            estimate = (pred[r, arc] - pred[r, direct]) - (pred[f, arc] - pred[f, direct])
            errors.append(float(np.sum((estimate - actual)**2)))
            measured.append(actual)
            predicted.append(estimate)
    return {'paired_conditions': len(pairs), 'interaction_rmse_m': float(np.sqrt(np.mean(errors))),
            'measured_mean_xy_m': np.mean(measured, axis=0).tolist(),
            'predicted_mean_xy_m': np.mean(predicted, axis=0).tolist()}


def analyze(root: Path) -> dict:
    arrays, metadata, split, config = load(root)
    ledger = json.loads((root / 'results-ledger.json').read_text())
    if any(sha(root / name) != digest for name, digest in ledger.items()):
        raise ValueError('test result ledger mismatch')
    ids = split['test']
    with np.load(root / 'test-predictions.npz', allow_pickle=False) as z:
        prediction = {name: z[name].copy() for name in z.files}
    if not np.array_equal(prediction['test_group_indices'], ids) or \
            not np.array_equal(prediction['actual'], arrays['target'][ids]):
        raise ValueError('test prediction identity mismatch')
    target = prediction['actual']
    posture = np.asarray([metadata[i]['posture'] for i in ids])
    terrain = np.asarray([metadata[i]['terrain_type'] for i in ids])
    seed_ids = np.asarray([metadata[i]['terrain_seed'] for i in ids])
    result = {'schema': 'eatnav_mi_b_pilot_subgroups_v2', 'test_seed_ids': sorted(set(seed_ids.tolist())),
              'models': {}, 'target_candidate_mean_xy_m': {
                  name: target[:, k].mean(0).tolist() for k, name in enumerate(CANDIDATES)}}
    for model_name in sorted(set(prediction) - {'actual', 'test_group_indices'}):
        pred = prediction[model_name]
        if pred.shape != target.shape or not np.isfinite(pred).all():
            raise ValueError('prediction shape/finite mismatch')
        result['models'][model_name] = {
            'posture_absolute': subgroup_metrics(pred, target, posture, tuple(range(6))),
            'posture_contrast': contrast_by_group(pred, target, posture),
            'terrain_absolute': subgroup_metrics(pred, target, terrain, tuple(range(6))),
            'terrain_contrast': contrast_by_group(pred, target, terrain),
            'direction_absolute': subgroup_metrics(pred, target,
                np.asarray(['forward'] * len(pred)), (0, 1, 2)) |
                subgroup_metrics(pred, target, np.asarray(['reverse'] * len(pred)), (3, 4, 5)),
            'seed_contrast': contrast_by_group(pred, target, seed_ids),
            'arc_minus_direct_vectors': contrast_vectors(pred, target),
            'posture_interaction': posture_interaction(
                pred, target, [metadata[int(i)] for i in ids]),
            'candidate': {name: {
                'predicted_mean_xy_m': pred[:, k].mean(0).tolist(),
                'measured_mean_xy_m': target[:, k].mean(0).tolist(),
                'xy_rmse_m': float(np.sqrt(np.sum((pred[:, k] - target[:, k])**2, -1).mean()))}
                for k, name in enumerate(CANDIDATES)},
            'worst_group': None,
        }
        per_group = np.sum((pred - target)**2, axis=-1).mean(axis=-1)
        worst = int(np.argmax(per_group))
        result['models'][model_name]['worst_group'] = {
            'metadata': metadata[int(ids[worst])], 'absolute_xy_rmse_m': float(np.sqrt(per_group[worst]))}
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    path = args.root / 'pilot-subgroups-v2.json'
    if path.exists():
        raise FileExistsError('subgroup analysis exists')
    value = analyze(args.root)
    atomic_json(path, value)
    print({'models': len(value['models']), 'test_seeds': value['test_seed_ids']})
