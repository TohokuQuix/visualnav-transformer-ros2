"""Frozen, resumable 40-fold × 3-seed × A1–A4 offline comparison."""
from __future__ import annotations

from collections import defaultdict
import argparse
import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

from visualnav_transformer.eat_nav.evaluator_offline_train import fit_scaler
from visualnav_transformer.eat_nav.path_corridor_models import PathCorridorEvaluator, parameter_count


HEAD_KEYS = ('reward', 'direct_relative_reward', 'goal', 'endpoint',
             'path_progress_2s', 'cross_track_p95_2s')
COMMON_KEYS = ('relative_goal', 'candidate_path', 'path_geometry', 'path_dem_profile',
               'path_coverage', 'body_angular_velocity', 'projected_gravity',
               'flipper_fold_mode', 'flipper_position', 'availability_mask',
               'stale_mask', 'arm_applicability_mask')
MASK_KEYS = {'availability_mask', 'stale_mask', 'arm_applicability_mask'}


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for piece in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(piece)
    return digest.hexdigest()


def _atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def fit_train_scalers(common, targets, groups, train_indices, binary_mask):
    train_indices = np.asarray(train_indices)
    xm, xs = fit_scaler(common[train_indices], [groups[i] for i in train_indices])
    ym, ys = fit_scaler(targets[train_indices], [groups[i] for i in train_indices])
    xm[np.asarray(binary_mask)] = 0
    xs[np.asarray(binary_mask)] = 1
    return {'common_mean': xm, 'common_scale': xs, 'target_mean': ym, 'target_scale': ys}


def group_pair_loss(pred, rows, indices, pairs, weights):
    """Exact historical Bradley–Terry group-macro loss, strict pairs only."""
    indices = list(indices)
    groups = sorted({rows[i]['group'] for i in indices})
    losses = []
    for group in groups:
        by_candidate = defaultdict(list)
        for position, index in enumerate(indices):
            if rows[index]['group'] == group:
                by_candidate[rows[index]['candidate']].append(position)
        terms = {0: [], 2: []}
        for pair in pairs.get(group, []):
            left, right = pair['candidate_i'], pair['candidate_j']
            if left not in by_candidate or right not in by_candidate:
                continue
            for key, head in (('reward_label', 0), ('goal_label', 2)):
                label = pair[key]
                if label not in ('i_better', 'j_better'):
                    continue
                a = pred[by_candidate[left], head].mean()
                b = pred[by_candidate[right], head].mean()
                sign = 1 if label == 'i_better' else -1
                terms[head].append(F.softplus(-sign * (a - b)))
        weighted = [float(weight) * torch.stack(terms[head]).mean()
                    for head, weight in zip((0, 2), weights) if terms[head]]
        if weighted:
            losses.append(torch.stack(weighted).sum())
    return torch.stack(losses).mean() if losses else pred.sum() * 0


def common_array(features, count):
    blocks, binary = [], []
    for key in COMMON_KEYS:
        block = np.asarray(features[key], dtype=np.float32).reshape(count, -1)
        blocks.append(block)
        binary.extend([key in MASK_KEYS] * block.shape[1])
    return np.concatenate(blocks, axis=1), np.asarray(binary, dtype=bool)


def _height_scaler(full_dem, train_indices, groups):
    group_values = []
    for group in sorted({groups[i] for i in train_indices}):
        subset = full_dem[[i for i in train_indices if groups[i] == group]]
        valid = subset[:, :, 1] > .5
        height = subset[:, :, 0][valid]
        if not len(height):
            raise ValueError(f'no valid DEM heights in train group {group}')
        group_values.append((float(np.mean(height)), float(np.mean(height**2))))
    mean = float(np.mean([value[0] for value in group_values]))
    scale = max(float(np.sqrt(max(0, np.mean([value[1] for value in group_values]) - mean**2))), 1e-6)
    return mean, scale


def _normal_inputs(common, full_dem, corridor, profile, indices, groups, target, binary):
    scaler = fit_train_scalers(common, target, groups, indices, binary)
    pm, ps = fit_scaler(profile[indices].transpose(0, 2, 1).reshape(-1, profile.shape[-1]),
                        [groups[i] for i in indices for _ in range(profile.shape[1])])
    hm, hs = _height_scaler(full_dem, indices, groups)
    full = full_dem.copy()
    strip = corridor.copy()
    for values in (full, strip):
        height, valid = values[:, :, 0], values[:, :, 1]
        height[:] = np.where(valid > .5, (height - hm) / hs, 0)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite normalized DEM')
    return {'common': ((common - scaler['common_mean']) / scaler['common_scale']).astype(np.float32),
            'full_dem': full.reshape(len(full), 8, *full.shape[-2:]).astype(np.float32),
            'corridor': strip.reshape(len(strip), 8, *strip.shape[-2:]).astype(np.float32),
            'profile': ((profile - pm) / ps).transpose(0, 2, 1).astype(np.float32)}, scaler, (hm, hs), (pm, ps)


def _pair_plan(rows, indices, pairs):
    """Vectorized group-macro ranking terms for repeated training steps."""
    local = {int(index): position for position, index in enumerate(indices)}
    groups = sorted({rows[i]['group'] for i in indices})
    candidate_positions = defaultdict(list)
    for index in indices:
        candidate_positions[(rows[index]['group'], rows[index]['candidate'])].append(local[int(index)])
    entries = []
    for group_index, group in enumerate(groups):
        for pair in pairs.get(group, []):
            left = candidate_positions.get((group, pair['candidate_i']))
            right = candidate_positions.get((group, pair['candidate_j']))
            if not left or not right:
                continue
            for head, key, weight in ((0, 'reward_label', .25), (2, 'goal_label', .25)):
                label = pair[key]
                if label in ('i_better', 'j_better'):
                    entries.append((group_index, head, left, right, 1 if label == 'i_better' else -1, weight))
    return groups, entries


def _fast_pair_loss(pred, plan):
    groups, entries = plan
    by_group_head = defaultdict(list)
    for group, head, left, right, sign, weight in entries:
        by_group_head[(group, head)].append(F.softplus(-sign * (pred[left, head].mean() - pred[right, head].mean())))
    per_group = []
    for group in range(len(groups)):
        terms = [weight * torch.stack(by_group_head[(group, head)]).mean()
                 for head, weight in ((0, .25), (2, .25)) if by_group_head[(group, head)]]
        if terms:
            per_group.append(torch.stack(terms).sum())
    return torch.stack(per_group).mean() if per_group else pred.sum() * 0


def _fit_one(variant, seed, fold, rows, targets, common, binary, full, corridor,
             profile, pairs, config, device):
    indices = np.asarray([i for i, row in enumerate(rows) if row['group'] in set(fold['train'])])
    test = np.asarray([i for i, row in enumerate(rows) if row['group'] in set(fold['test'])])
    groups = [row['group'] for row in rows]
    normalized, scaler, height_scaler, profile_scaler = _normal_inputs(
        common, full, corridor, profile, indices, groups, targets, binary)
    tensor = {key: torch.from_numpy(value).to(device) for key, value in normalized.items()}
    target = torch.from_numpy(((targets - scaler['target_mean']) / scaler['target_scale']).astype(np.float32)).to(device)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    model = PathCorridorEvaluator(variant, common.shape[1], profile.shape[-1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['training']['learning_rate'],
                                 weight_decay=config['training']['weight_decay'])
    train_tensor = torch.as_tensor(indices, device=device)
    plan = _pair_plan(rows, indices, pairs)
    group_position = {group: [position for position, index in enumerate(indices) if groups[index] == group]
                      for group in plan[0]}
    weight = torch.tensor(config['training']['regression_weights'], device=device)
    best = float('inf')
    best_state, best_epoch, wait = None, 0, 0
    started = time.monotonic()
    for epoch in range(config['training']['max_epochs']):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(**{key: value[train_tensor] for key, value in tensor.items()})
        per_group = []
        for position in group_position.values():
            regression = F.smooth_l1_loss(prediction[position], target[train_tensor[position]],
                                          beta=config['training']['regression_beta'], reduction='none')
            per_group.append((regression * weight).mean())
        loss = torch.stack(per_group).mean() + _fast_pair_loss(prediction, plan)
        if not torch.isfinite(loss):
            raise ValueError(f'nonfinite loss: {fold["fold_id"]} {variant} {seed} epoch {epoch}')
        loss.backward()
        optimizer.step()
        current = float(loss.detach())
        if current < best - config['training']['min_delta']:
            best, best_epoch, wait = current, epoch + 1, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            wait += 1
            if wait >= config['training']['patience']:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predicted = model(**{key: value for key, value in tensor.items()}).cpu().numpy()
    predicted = predicted * scaler['target_scale'] + scaler['target_mean']
    if not np.isfinite(predicted).all():
        raise ValueError('nonfinite predictions')
    meta = {'fold_id': fold['fold_id'], 'mode': fold['mode'], 'variant': variant, 'seed': seed,
            'best_train_loss': best, 'best_epoch': best_epoch, 'epochs_run': epoch + 1,
            'elapsed_sec': time.monotonic() - started, 'parameter_count': parameter_count(model),
            'train_group_count': len(fold['train']), 'test_group_count': len(fold['test']),
            'train_indices': indices.tolist(), 'test_indices': test.tolist(),
            'input_scaler_sha256': hashlib.sha256(scaler['common_mean'].tobytes() + scaler['common_scale'].tobytes()).hexdigest(),
            'target_scaler_sha256': hashlib.sha256(scaler['target_mean'].tobytes() + scaler['target_scale'].tobytes()).hexdigest(),
            'height_scaler': height_scaler,
            'profile_scaler_sha256': hashlib.sha256(profile_scaler[0].tobytes() + profile_scaler[1].tobytes()).hexdigest()}
    return predicted.astype(np.float32), meta


def folds_for_shard(folds, shard_index: int, shard_count: int):
    if shard_count < 1 or not 0 <= shard_index < shard_count:
        raise ValueError('invalid shard index/count')
    return [fold for index, fold in enumerate(folds) if index % shard_count == shard_index]


def run(root: Path, output: Path, config_path: Path, limit_fits: int | None = None,
        shard_index: int = 0, shard_count: int = 1):
    torch.set_num_threads(1)
    config = yaml.safe_load(config_path.read_text())
    audit = json.loads((output / 'corridor-support-audit.json').read_text())
    if audit['status'] != 'pass' or audit['config_sha256'] != _sha(config_path):
        raise ValueError('C1 config or support audit mismatch')
    if audit['features_sha256'] != _sha(output / 'corridor-features.npz'):
        raise ValueError('C1 tensor SHA mismatch')
    with np.load(root / 'combined-features.npz') as source, np.load(output / 'corridor-features.npz') as prepared:
        common, binary = common_array(source, 252)
        full = prepared['full_dem'].copy()
        corridor = prepared['corridor'].copy()
        profile = prepared['profile'].copy()
    rows = json.loads((root / 'record-identity.json').read_text())
    labels = {row['process_id']: row for row in map(json.loads, gzip.open(root / 'combined-labels.jsonl.gz', 'rt'))}
    targets = np.asarray([[float(labels[row['process_id']][key]) for key in HEAD_KEYS] for row in rows], dtype=np.float32)
    if not np.isfinite(targets).all() or not all(labels[row['process_id']]['valid'] for row in rows):
        raise ValueError('nonfinite or invalid frozen teacher labels')
    pair_data = json.loads((root / 'expanded-pairwise-labels.json').read_text())
    pairs = defaultdict(list)
    for pair in pair_data['candidate_pairs']:
        pairs[pair['state_group_id']].append(pair)
    folds = json.loads((root / 'split-folds.json').read_text())['folds']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output.mkdir(parents=True, exist_ok=True)
    fits = output / 'fits'
    fits.mkdir(exist_ok=True)
    architecture = {variant: parameter_count(PathCorridorEvaluator(variant, common.shape[1], profile.shape[-1]))
                    for variant in config['models']['variants']}
    if shard_index == 0:
        _atomic_json(output / 'architecture-freeze.json', {'config_sha256': _sha(config_path),
                       'parameter_counts': architecture, 'common_dim': common.shape[1],
                       'profile_dim': profile.shape[-1], 'device': str(device),
                       'torch_version': torch.__version__})
    count = 0
    status_path = output / f'training-status-shard-{shard_index}-of-{shard_count}.json'
    for fold in folds_for_shard(folds, shard_index, shard_count):
        for seed in config['training']['seeds']:
            for variant in config['models']['variants']:
                stem = f'{fold["mode"]}_{fold["fold_id"]}_{seed}_{variant}'.replace('/', '_')
                marker = fits / (stem + '.json')
                output_npz = fits / (stem + '.npz')
                if marker.exists() and output_npz.exists():
                    saved = json.loads(marker.read_text())
                    if saved['prediction_sha256'] != _sha(output_npz) or saved['config_sha256'] != _sha(config_path):
                        raise ValueError(f'resume artifact mismatch: {stem}')
                    continue
                predicted, meta = _fit_one(variant, seed, fold, rows, targets, common, binary, full,
                                           corridor, profile, pairs, config, device)
                temporary = output_npz.with_suffix('.npz.tmp')
                with temporary.open('wb') as stream:
                    np.savez_compressed(stream, prediction=predicted)
                os.replace(temporary, output_npz)
                meta.update({'prediction_sha256': _sha(output_npz), 'config_sha256': _sha(config_path)})
                _atomic_json(marker, meta)
                count += 1
                _atomic_json(status_path, {'status': 'running', 'completed_new': count,
                            'completed_total_observed': len(list(fits.glob('*.json'))), 'expected_total': 480,
                            'last_fit': stem, 'last_elapsed_sec': meta['elapsed_sec'], 'device': str(device)})
                print(json.dumps({'fit': stem, 'elapsed_sec': meta['elapsed_sec'],
                                  'epochs_run': meta['epochs_run'], 'completed_new': count}), flush=True)
                if limit_fits is not None and count >= limit_fits:
                    return
    _atomic_json(status_path, {'status': 'complete', 'completed_total_observed': len(list(fits.glob('*.json'))),
                               'expected_total': 480, 'device': str(device)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('/tmp/eatnav-evaluator-independent-expansion-20261006-v7'))
    parser.add_argument('--output', type=Path, default=Path('/tmp/eatnav-path-corridor-architecture-20261007'))
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--limit-fits', type=int)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shard-count', type=int, default=1)
    args = parser.parse_args()
    run(args.root, args.output, args.config, args.limit_fits, args.shard_index, args.shard_count)


if __name__ == '__main__':
    main()
