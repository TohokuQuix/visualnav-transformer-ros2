"""Offline, sealed-evidence-only training of the fourteen-Path motion predictor."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from visualnav_transformer.eat_nav.mi_b_pilot import (
    PilotPredictor, fit_normalization, paired_metrics, sha, split_indices,
    transform_arrays,
)
from visualnav_transformer.eat_nav.motion_14_dataset import build_grouped_dataset
from visualnav_transformer.eat_nav.motion_14_metrics import evaluate_predictions
from visualnav_transformer.eat_nav.motion_14_predictor import (
    CANDIDATE_COUNT, CANDIDATE_IDS, Motion14Predictor, contrast_indices,
    normalized_pair_loss, validate_grouped_arrays,
)
from eat_nav_motion_14_collection_run import MANIFEST as COLLECTION_MANIFEST, load_frozen


REPO = Path(__file__).resolve().parents[1]
CONTRACT = REPO/'docs/eat-nav/motion-predictor-vnext/motion_14_training_contract.json'
LEGACY = Path('/home/kojima/lib/visualnav-transformer-ros2/'
              '.eatnav-motion-vnext-evidence/mi-b-pilot-20261014-01/training/full-seed456/best.pt')
LEGACY_COMMON_SIX = (0, 2, 1, 7, 9, 8)


def atomic_json(path: Path, value: dict) -> None:
    tmp = path.with_suffix(path.suffix+'.tmp')
    with tmp.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def build(root: Path, collection_root: Path) -> None:
    if root.exists() or root.is_symlink():
        raise FileExistsError('fresh training evidence root required')
    contract = json.loads(CONTRACT.read_text())
    frozen = load_frozen()
    if contract['candidate_count'] != CANDIDATE_COUNT or \
            contract['candidate_order'] != list(CANDIDATE_IDS) or \
            frozen['schema'] != contract['source_collection_schema']:
        raise ValueError('training/collection candidate contract mismatch')
    arrays, metadata, audit = build_grouped_dataset(collection_root)
    if len(metadata) != 288 or audit['candidate_trial_count'] != 4032:
        raise ValueError('full 36-seed dataset incomplete')
    root.mkdir(parents=True)
    np.savez_compressed(root/'grouped.npz', **arrays)
    atomic_json(root/'metadata.json', {'rows': metadata})
    atomic_json(root/'audit.json', audit)
    atomic_json(root/'contract.json', contract)
    atomic_json(root/'provenance.json', {'collection_manifest_sha256': sha(COLLECTION_MANIFEST),
                'collection_root': str(collection_root), 'contract_sha256': sha(CONTRACT),
                'model_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
                'dataset_source_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/motion_14_dataset.py'),
                'training_source_sha256': sha(Path(__file__)),
                'legacy_checkpoint_sha256': sha(LEGACY)})
    atomic_json(root/'dataset-ledger.json', {name: sha(root/name) for name in
                ('grouped.npz', 'metadata.json', 'audit.json', 'contract.json', 'provenance.json')})


def load(root: Path):
    ledger = json.loads((root/'dataset-ledger.json').read_text())
    for name, digest in ledger.items():
        if sha(root/name) != digest:
            raise ValueError(f'training dataset SHA mismatch: {name}')
    with np.load(root/'grouped.npz', allow_pickle=False) as z:
        arrays = {name: z[name].copy() for name in z.files}
    validate_grouped_arrays(arrays)
    metadata = json.loads((root/'metadata.json').read_text())['rows']
    contract = json.loads((root/'contract.json').read_text())
    provenance = json.loads((root/'provenance.json').read_text())
    for key, actual in (
        ('collection_manifest_sha256', sha(COLLECTION_MANIFEST)),
        ('contract_sha256', sha(CONTRACT)),
        ('model_source_sha256', sha(REPO/'src/visualnav_transformer/eat_nav/motion_14_predictor.py')),
        ('dataset_source_sha256', sha(REPO/'src/visualnav_transformer/eat_nav/motion_14_dataset.py')),
        ('training_source_sha256', sha(Path(__file__))),
        ('legacy_checkpoint_sha256', sha(LEGACY)),
    ):
        if provenance.get(key) != actual:
            raise ValueError(f'training provenance mismatch: {key}')
    frozen = load_frozen()
    if len(metadata) != len(arrays['dem']) or contract['candidate_order'] != list(CANDIDATE_IDS):
        raise ValueError('training metadata/candidate mismatch')
    split = split_indices(metadata, frozen['split_seeds'])
    if {name: len(ids) for name, ids in split.items()} != \
            {'train': 96, 'validation': 96, 'test': 96}:
        raise ValueError('train/validation/test grouped support mismatch')
    return arrays, metadata, split, contract


def tensors(arrays: dict, indices: np.ndarray):
    return tuple(torch.from_numpy(arrays[key][indices]) for key in
                 ('dem', 'state', 'paths', 'direction', 'mask', 'target'))


def save_checkpoint(path: Path, model: Motion14Predictor, details: dict) -> None:
    tmp = path.with_suffix('.tmp')
    torch.save({'model_state': model.state_dict(), 'candidate_count': CANDIDATE_COUNT,
                'candidate_order': CANDIDATE_IDS, **details}, tmp)
    with tmp.open('rb') as stream:
        os.fsync(stream.fileno())
    os.replace(tmp, path)


def overfit(root: Path) -> None:
    arrays, metadata, split, contract = load(root)
    if (root/'overfit-result.json').exists():
        raise FileExistsError('small overfit result already exists')
    torch.set_num_threads(4)
    seed = contract['training']['training_seeds'][0]
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    normalization = fit_normalization(arrays, split['train'])
    scaled = transform_arrays(arrays, normalization)
    indices = split['train'][:contract['small_overfit']['train_group_count']]
    values = tensors(scaled, indices)
    model = Motion14Predictor('full')
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=contract['training']['learning_rate'], weight_decay=0)
    for step in range(contract['small_overfit']['max_steps']):
        model.train(); optimizer.zero_grad()
        loss = normalized_pair_loss(model(*values[:5]), values[-1], values[-2])
        loss.backward(); optimizer.step()
    model.eval()
    with torch.no_grad():
        scaled_pred = model(*values[:5]).numpy()
    pred = scaled_pred * np.asarray(normalization['target_std']) + \
           np.asarray(normalization['target_mean'])
    target = arrays['target'][indices]
    error = float(np.max(np.abs(pred-target)))
    contrast = float(max(np.max(np.abs((pred[:, arc]-pred[:, direct]) -
                                       (target[:, arc]-target[:, direct])))
                         for arc, direct in contrast_indices()))
    result = {'train_groups': [metadata[i] for i in indices],
              'max_absolute_xy_error_m': error,
              'max_arc_minus_direct_xy_error_m': contrast,
              'final_loss': float(loss.item()),
              'pass': error < contract['small_overfit']['max_absolute_xy_error_m'] and
                      contrast < contract['small_overfit']['max_arc_minus_direct_xy_error_m']}
    atomic_json(root/'overfit-result.json', result)
    print(json.dumps(result), flush=True)


def train(root: Path) -> None:
    arrays, _, split, contract = load(root)
    if not json.loads((root/'overfit-result.json').read_text())['pass']:
        raise ValueError('small overfit failed; diagnose before main training')
    if (root/'normalization.json').exists():
        raise FileExistsError('normalization already exists')
    torch.set_num_threads(4)
    norm = fit_normalization(arrays, split['train'])
    atomic_json(root/'normalization.json', norm)
    scaled = transform_arrays(arrays, norm)
    train_ids, validation_ids = split['train'], split['validation']
    validation = tensors(scaled, validation_ids)
    output = root/'training'
    output.mkdir()
    source = {'model_sha256': sha(REPO/'src/visualnav_transformer/eat_nav/motion_14_predictor.py'),
              'runner_sha256': sha(Path(__file__)), 'contract_sha256': sha(root/'contract.json'),
              'dataset_ledger_sha256': sha(root/'dataset-ledger.json')}
    atomic_json(output/'source.json', source)
    settings = contract['training']
    for variant in settings['variants']:
        for seed in settings['training_seeds']:
            run_root = output/f'{variant}-seed{seed}'
            run_root.mkdir()
            random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
            model = Motion14Predictor(variant)
            optimizer = torch.optim.AdamW(model.parameters(),
                                          lr=settings['learning_rate'],
                                          weight_decay=settings['weight_decay'])
            best, wait, history = float('inf'), 0, []
            generator = np.random.default_rng(seed)
            for epoch in range(settings['max_epochs']):
                model.train()
                order = generator.permutation(train_ids)
                losses = []
                for chunk in np.array_split(order, max(1, int(np.ceil(len(order) /
                                              settings['batch_groups'])))):
                    values = tensors(scaled, chunk)
                    optimizer.zero_grad()
                    loss = normalized_pair_loss(model(*values[:5]), values[-1], values[-2])
                    loss.backward(); optimizer.step()
                    losses.append(float(loss.item()))
                model.eval()
                with torch.no_grad():
                    value = float(normalized_pair_loss(model(*validation[:5]),
                                                       validation[-1], validation[-2]).item())
                history.append({'epoch': epoch+1, 'train_loss': float(np.mean(losses)),
                                'validation_loss': value})
                if value < best-settings['min_delta']:
                    best, wait = value, 0
                    save_checkpoint(run_root/'best.pt', model, {
                        'variant': variant, 'training_seed': seed,
                        'best_epoch': epoch+1, 'best_validation_loss': best,
                        'source': source, 'normalization': norm})
                else:
                    wait += 1
                if wait >= settings['patience']:
                    break
            atomic_json(run_root/'history.json', {'epochs': history,
                        'best_validation_loss': best, 'stopped_epoch': len(history)})
            print(json.dumps({'variant': variant, 'seed': seed,
                              'best_validation_loss': best, 'epochs': len(history)}), flush=True)
    atomic_json(output/'complete.json', {'runs': len(settings['variants']) *
                len(settings['training_seeds']), 'source': source})


def _predict_legacy(arrays: dict, indices: np.ndarray) -> np.ndarray:
    payload = torch.load(LEGACY, map_location='cpu', weights_only=True)
    if payload['variant'] != 'full' or payload['training_seed'] != 456:
        raise ValueError('legacy checkpoint variant/seed mismatch')
    old = PilotPredictor('full')
    old.load_state_dict(payload['model_state'], strict=True)
    old.eval()
    common = np.asarray(LEGACY_COMMON_SIX)
    subset = {name: value[indices].copy() for name, value in arrays.items()}
    for name in ('paths', 'direction', 'mask', 'target'):
        subset[name] = subset[name][:, common]
    normalized = transform_arrays(subset, payload['normalization'])
    with torch.no_grad():
        scaled = old(*tuple(torch.from_numpy(normalized[key]) for key in
                             ('dem', 'state', 'paths', 'direction', 'mask'))).numpy()
    norm = payload['normalization']
    return scaled * np.asarray(norm['target_std']) + np.asarray(norm['target_mean'])


def evaluate(root: Path) -> None:
    arrays, metadata, split, contract = load(root)
    if not (root/'training/complete.json').is_file() or (root/'results.json').exists():
        raise ValueError('training incomplete or test already evaluated')
    torch.set_num_threads(4)
    norm = json.loads((root/'normalization.json').read_text())
    scaled = transform_arrays(arrays, norm)
    ids = split['test']
    rows = [metadata[i] for i in ids]
    target = arrays['target'][ids]
    seed_ids = np.asarray([row['terrain_seed'] for row in rows])
    train_target = arrays['target'][split['train']]
    baseline = np.zeros_like(target)
    baseline[:, :7] = train_target[:, :7].mean(axis=(0, 1))
    baseline[:, 7:] = train_target[:, 7:].mean(axis=(0, 1))
    bootstrap = contract['bootstrap']
    score = lambda prediction: evaluate_predictions(
        prediction, target, rows,
        bootstrap_replicates=bootstrap['replicates'], bootstrap_seed=bootstrap['random_seed'])
    results = {'schema': 'eatnav_motion_14_training_result_v1',
               'baseline': score(baseline), 'variants': {},
               'test_seed_ids': sorted(set(seed_ids.tolist())),
               'source': json.loads((root/'training/source.json').read_text())}
    prediction_artifact = {'actual': target, 'baseline': baseline,
                           'test_group_indices': ids}
    values = tensors(scaled, ids)
    for variant in contract['training']['variants']:
        results['variants'][variant] = {}
        for seed in contract['training']['training_seeds']:
            path = root/'training'/f'{variant}-seed{seed}'/'best.pt'
            saved = torch.load(path, map_location='cpu', weights_only=True)
            if saved['source'] != results['source'] or saved['normalization'] != norm or \
                    saved['candidate_count'] != 14 or \
                    tuple(saved['candidate_order']) != CANDIDATE_IDS:
                raise ValueError('checkpoint provenance/candidate mismatch')
            model = Motion14Predictor(variant)
            model.load_state_dict(saved['model_state'], strict=True)
            model.eval()
            with torch.no_grad():
                scaled_pred = model(*values[:5]).numpy()
            pred = (scaled_pred*np.asarray(norm['target_std']) +
                    np.asarray(norm['target_mean'])).astype(np.float32)
            prediction_artifact[f'{variant}-seed{seed}'] = pred
            metrics = score(pred)
            metrics['best_epoch'] = saved['best_epoch']
            metrics['best_validation_loss'] = saved['best_validation_loss']
            metrics['checkpoint_sha256'] = sha(path)
            results['variants'][variant][str(seed)] = metrics
    selected = min(contract['training']['training_seeds'],
                   key=lambda seed: (results['variants']['full'][str(seed)]['best_validation_loss'], seed))
    new = results['variants']['full'][str(selected)]
    old_prediction = _predict_legacy(arrays, ids)
    common = np.asarray(LEGACY_COMMON_SIX)
    old_metrics = paired_metrics(old_prediction, target[:, common],
                                 np.ones((len(target), 6), bool), seed_ids)
    new_metrics = paired_metrics(prediction_artifact[f'full-seed{selected}'][:, common],
                                 target[:, common], np.ones((len(target), 6), bool), seed_ids)
    results['legacy_common_six'] = {'old_checkpoint_sha256': sha(LEGACY),
                                    'old': old_metrics, 'new': new_metrics,
                                    'indices_in_new_bank': list(LEGACY_COMMON_SIX)}
    gate = contract['demo_adoption_gate']
    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator > 1e-12 else None

    ratios = {'all14_contrast_to_direction_baseline':
                  ratio(new['contrast_rmse_m'], results['baseline']['contrast_rmse_m']),
              'absolute_to_direction_baseline':
                  ratio(new['absolute_xy_rmse_m'], results['baseline']['absolute_xy_rmse_m']),
              'abs90_contrast_to_direction_baseline':
                  ratio(new['angle_contrast_rmse_m']['90'],
                        results['baseline']['angle_contrast_rmse_m']['90']),
              'common6_contrast_to_old_checkpoint':
                  ratio(new_metrics['contrast_rmse_m'], old_metrics['contrast_rmse_m'])}
    criteria = {
        'all14_contrast': ratios['all14_contrast_to_direction_baseline'] is not None and
            ratios['all14_contrast_to_direction_baseline'] <=
            gate['selected_full_all14_contrast_rmse_ratio_to_train_direction_baseline_max'],
        'absolute': ratios['absolute_to_direction_baseline'] is not None and
            ratios['absolute_to_direction_baseline'] <=
            gate['selected_full_absolute_rmse_ratio_to_train_direction_baseline_max'],
        'abs90': ratios['abs90_contrast_to_direction_baseline'] is not None and
            ratios['abs90_contrast_to_direction_baseline'] <=
            gate['selected_full_abs90_contrast_rmse_ratio_to_baseline_max'],
        'common6_legacy': ratios['common6_contrast_to_old_checkpoint'] is not None and
            ratios['common6_contrast_to_old_checkpoint'] <=
            gate['selected_full_common6_contrast_rmse_ratio_to_old_mib_full_seed456_max'],
        'validation_plateau': new['best_epoch'] <=
            contract['training']['max_epochs'] -
            gate['selected_full_best_epoch_must_precede_max_epoch_by_at_least'],
        'twelve_test_seeds': len(results['test_seed_ids']) == 12,
        'finite_predictions': all(np.isfinite(value).all() for name, value in
                                  prediction_artifact.items() if name.startswith('full-seed')),
    }
    results['selected_full_seed'] = selected
    decision = ('inconclusive' if not criteria['validation_plateau'] or
                any(value is None for value in ratios.values()) else
                ('pass' if all(criteria.values()) else 'fail'))
    results['demo_adoption'] = {'decision': decision,
                                'criteria': criteria, 'ratios': ratios,
                                'contract_sha256': sha(CONTRACT)}
    np.savez_compressed(root/'test-predictions.npz', **prediction_artifact,
                        legacy_full_seed456_common6=old_prediction)
    atomic_json(root/'results.json', results)
    atomic_json(root/'results-ledger.json', {name: sha(root/name) for name in
                ('normalization.json', 'test-predictions.npz', 'results.json')})
    print(json.dumps({'selected_seed': selected, 'adoption': results['demo_adoption']}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('build', 'verify', 'overfit', 'train', 'evaluate'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--collection-root', type=Path)
    options = parser.parse_args()
    if options.command == 'build':
        if options.collection_root is None:
            raise ValueError('build requires --collection-root')
        build(options.root, options.collection_root)
    elif options.command == 'verify':
        arrays, metadata, split, _ = load(options.root)
        print(json.dumps({'groups': len(metadata), 'candidates': arrays['paths'].shape[1],
                          'split_groups': {name: len(ids) for name, ids in split.items()}}))
    elif options.command == 'overfit':
        overfit(options.root)
    elif options.command == 'train':
        train(options.root)
    else:
        evaluate(options.root)


if __name__ == '__main__':
    main()
