"""Offline MI-B pilot. Never launches Isaac or writes to source evidence."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from visualnav_transformer.eat_nav.mi_b_pilot import (
    PilotPredictor, build_grouped_dataset, contrast_indices, fit_normalization,
    normalized_pair_loss, paired_metrics, sha, split_indices, transform_arrays,
    validate_grouped_arrays,
)

REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO / 'docs/eat-nav/motion-predictor-vnext/mi_b_pilot_contract.json'


def atomic_json(path: Path, value: dict) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    with temp.open('x') as file:
        json.dump(value, file, indent=2, sort_keys=True)
        file.write('\n')
        file.flush()
        os.fsync(file.fileno())
    os.replace(temp, path)


def build(root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError('MI-B dataset requires fresh output root')
    config = json.loads(CONTRACT.read_text())
    evidence = Path(config['source_evidence_root'])
    arrays, metadata, audit = build_grouped_dataset(
        evidence, config['source_manifest_sha256'], config['split_by_terrain_seed'])
    root.mkdir(parents=True)
    try:
        np.savez_compressed(root / 'grouped.npz', **arrays)
        atomic_json(root / 'metadata.json', {'rows': metadata})
        atomic_json(root / 'audit.json', audit)
        atomic_json(root / 'contract.json', config)
        atomic_json(root / 'dataset-ledger.json', {
            name: sha(root / name) for name in
            ('grouped.npz', 'metadata.json', 'audit.json', 'contract.json')})
    except Exception:
        atomic_json(root / 'build-failure.json', {'stage': 'artifact-write'})
        raise
    print(json.dumps({key: audit[key] for key in ('verified_process_count',
          'candidate_trial_count', 'group_count', 'split_group_count', 'split_seed_count')}))


def load(root: Path):
    ledger = json.loads((root / 'dataset-ledger.json').read_text())
    for name, digest in ledger.items():
        if sha(root / name) != digest:
            raise ValueError('MI-B dataset SHA mismatch: ' + name)
    with np.load(root / 'grouped.npz', allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in z.files}
    validate_grouped_arrays(arrays)
    metadata = json.loads((root / 'metadata.json').read_text())['rows']
    config = json.loads((root / 'contract.json').read_text())
    if len(metadata) != len(arrays['dem']):
        raise ValueError('metadata/group count mismatch')
    split = split_indices(metadata, config['split_by_terrain_seed'])
    return arrays, metadata, split, config


def tensors(arrays: dict, indices: np.ndarray) -> tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(arrays[key][indices]) for key in
                 ('dem', 'state', 'paths', 'direction', 'mask', 'target'))


def forward(model: PilotPredictor, values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return model(*values[:5])


def save_checkpoint(path: Path, model: PilotPredictor, details: dict) -> None:
    temp = path.with_suffix('.tmp')
    torch.save({'model_state': model.state_dict(), **details}, temp)
    with temp.open('rb') as file:
        os.fsync(file.fileno())
    os.replace(temp, path)


def overfit(root: Path) -> None:
    arrays, metadata, split, config = load(root)
    path = root / 'overfit-result.json'
    if path.exists():
        raise FileExistsError('overfit result already exists')
    torch.set_num_threads(4)
    seed = config['overfit']['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    norm = fit_normalization(arrays, split['train'])
    scaled = transform_arrays(arrays, norm)
    ids = split['train'][:config['overfit']['train_group_count']]
    values = tensors(scaled, ids)
    model = PilotPredictor('full')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['learning_rate'],
                                  weight_decay=0)
    for step in range(config['overfit']['max_steps']):
        model.train(); optimizer.zero_grad()
        loss = normalized_pair_loss(forward(model, values), values[-1], values[-2])
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        scaled_pred = forward(model, values).numpy()
    pred = scaled_pred * np.asarray(norm['target_std']) + np.asarray(norm['target_mean'])
    actual = arrays['target'][ids]
    error = float(np.max(np.abs(pred - actual)))
    contrast_error = float(max(np.max(np.abs((pred[:, arc] - pred[:, direct]) -
                                              (actual[:, arc] - actual[:, direct])))
                               for arc, direct in contrast_indices()))
    result = {'groups': [metadata[i] for i in ids], 'steps': config['overfit']['max_steps'],
              'final_loss': float(loss.item()), 'max_absolute_xy_error_m': error,
              'max_arc_minus_direct_xy_error_m': contrast_error,
              'pass': error < config['overfit']['target_max_absolute_xy_error_m'] and
                      contrast_error < config['overfit']['target_max_arc_minus_direct_xy_error_m']}
    atomic_json(path, result)
    print(json.dumps(result))


def train(root: Path) -> None:
    arrays, _, split, config = load(root)
    overfit_result = json.loads((root / 'overfit-result.json').read_text())
    if not overfit_result['pass']:
        raise ValueError('small overfit gate failed; diagnose before training')
    torch.set_num_threads(4)
    norm = fit_normalization(arrays, split['train'])
    if (root / 'normalization.json').exists():
        raise FileExistsError('normalization already exists')
    atomic_json(root / 'normalization.json', norm)
    scaled = transform_arrays(arrays, norm)
    train_ids, val_ids = split['train'], split['validation']
    val_data = tensors(scaled, val_ids)
    training_dir = root / 'training'
    training_dir.mkdir()
    source = {'model_sha256': sha(REPO / 'src/visualnav_transformer/eat_nav/mi_b_pilot.py'),
              'runner_sha256': sha(Path(__file__)), 'contract_sha256': sha(root / 'contract.json'),
              'dataset_ledger_sha256': sha(root / 'dataset-ledger.json')}
    atomic_json(training_dir / 'source.json', source)
    for variant in config['training']['variants']:
        for seed in config['training']['training_seeds']:
            run_dir = training_dir / f'{variant}-seed{seed}'
            run_dir.mkdir()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = PilotPredictor(variant)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=config['training']['learning_rate'],
                                          weight_decay=config['training']['weight_decay'])
            best, wait, history = float('inf'), 0, []
            generator = np.random.default_rng(seed)
            for epoch in range(config['training']['max_epochs']):
                model.train()
                order = generator.permutation(train_ids)
                losses = []
                for chunk in np.array_split(order, max(1, int(np.ceil(len(order) /
                                              config['training']['batch_groups'])))):
                    values = tensors(scaled, chunk)
                    optimizer.zero_grad()
                    loss = normalized_pair_loss(forward(model, values), values[-1], values[-2])
                    loss.backward(); optimizer.step()
                    losses.append(float(loss.item()))
                model.eval()
                with torch.no_grad():
                    val_loss = float(normalized_pair_loss(forward(model, val_data),
                                                          val_data[-1], val_data[-2]).item())
                history.append({'epoch': epoch + 1, 'train_loss': float(np.mean(losses)),
                                'validation_loss': val_loss})
                if val_loss < best - config['training']['min_delta']:
                    best, wait = val_loss, 0
                    save_checkpoint(run_dir / 'best.pt', model, {
                        'variant': variant, 'training_seed': seed, 'best_epoch': epoch + 1,
                        'best_validation_loss': best, 'source': source, 'normalization': norm})
                else:
                    wait += 1
                if wait >= config['training']['patience']:
                    break
            atomic_json(run_dir / 'history.json', {'epochs': history,
                        'best_validation_loss': best, 'stopped_epoch': len(history)})
            print(json.dumps({'variant': variant, 'training_seed': seed,
                              'best_validation_loss': best, 'epochs': len(history)}), flush=True)
    atomic_json(training_dir / 'complete.json', {'runs': len(config['training']['variants']) *
                len(config['training']['training_seeds']), 'source': source})


def evaluate(root: Path) -> None:
    arrays, metadata, split, config = load(root)
    if not (root / 'training/complete.json').is_file():
        raise ValueError('all models must finish before test evaluation')
    if (root / 'pilot-results.json').exists():
        raise FileExistsError('test result already exists')
    torch.set_num_threads(4)
    norm = json.loads((root / 'normalization.json').read_text())
    scaled = transform_arrays(arrays, norm)
    ids = split['test']
    seed_ids = np.asarray([metadata[i]['terrain_seed'] for i in ids])
    target = arrays['target'][ids]
    mask = arrays['mask'][ids]
    train_ids = split['train']
    baseline = np.zeros_like(target)
    for candidate in range(6):
        direction_indices = list(range(0, 3)) if candidate < 3 else list(range(3, 6))
        baseline[:, candidate] = arrays['target'][train_ids][:, direction_indices].mean(axis=(0, 1))
    baseline_metrics = paired_metrics(baseline, target, mask, seed_ids)
    results = {'schema': 'eatnav_mi_b_pilot_result_v1', 'baseline': baseline_metrics,
               'variants': {}, 'test_seed_ids': sorted(set(seed_ids.tolist())),
               'test_group_count': len(ids), 'source': json.loads((root / 'training/source.json').read_text())}
    predictions = {'actual': target, 'baseline': baseline, 'test_group_indices': ids}
    for variant in config['training']['variants']:
        results['variants'][variant] = {}
        for seed in config['training']['training_seeds']:
            run = root / 'training' / f'{variant}-seed{seed}'
            saved = torch.load(run / 'best.pt', map_location='cpu', weights_only=False)
            if saved['source'] != results['source'] or saved['normalization'] != norm:
                raise ValueError('checkpoint source/normalization mismatch')
            model = PilotPredictor(variant)
            model.load_state_dict(saved['model_state']); model.eval()
            with torch.no_grad():
                pred_scaled = forward(model, tensors(scaled, ids)).numpy()
            pred = pred_scaled * np.asarray(norm['target_std']) + np.asarray(norm['target_mean'])
            pred = pred.astype(np.float32)
            predictions[f'{variant}-seed{seed}'] = pred
            metrics = paired_metrics(pred, target, mask, seed_ids)
            metrics['best_epoch'] = saved['best_epoch']
            metrics['best_validation_loss'] = saved['best_validation_loss']
            results['variants'][variant][str(seed)] = metrics
    full_errors = [results['variants']['full'][str(seed)]['contrast_rmse_m']
                   for seed in config['training']['training_seeds']]
    no_path_errors = [results['variants']['no_path'][str(seed)]['contrast_rmse_m']
                      for seed in config['training']['training_seeds']]
    results['pilot_decision'] = ('pass' if max(full_errors) < min(
        baseline_metrics['contrast_rmse_m'], *no_path_errors) else 'fail')
    results['pilot_decision_note'] = 'descriptive pilot criterion, not a frozen formal MI-B gate'
    np.savez_compressed(root / 'test-predictions.npz', **predictions)
    atomic_json(root / 'pilot-results.json', results)
    atomic_json(root / 'results-ledger.json', {name: sha(root / name) for name in
                ('normalization.json', 'test-predictions.npz', 'pilot-results.json')})
    print(json.dumps({'pilot_decision': results['pilot_decision'],
                      'baseline_contrast_rmse_m': baseline_metrics['contrast_rmse_m'],
                      'full_contrast_rmse_m': full_errors}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'verify', 'overfit', 'train', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'build':
        build(args.root)
    elif args.command == 'verify':
        arrays, metadata, split, _ = load(args.root)
        print({'groups': len(metadata), 'candidate_trials': len(metadata) * 6,
               'split_groups': {name: len(ids) for name, ids in split.items()},
               'dem_valid_height_range': [float(arrays['dem'][:, :, 0][arrays['dem'][:, :, 1].astype(bool)].min()),
                                          float(arrays['dem'][:, :, 0][arrays['dem'][:, :, 1].astype(bool)].max())]})
    elif args.command == 'overfit':
        overfit(args.root)
    elif args.command == 'train':
        train(args.root)
    else:
        evaluate(args.root)


if __name__ == '__main__':
    main()
