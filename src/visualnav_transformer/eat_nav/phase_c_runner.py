"""Frozen, resumable Phase C-0 reference training and timing probe."""
from __future__ import annotations

import argparse
import copy
import gzip
import io
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import resource
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from .checkpoint import save_checkpoint
from .config import assert_checkpoint_compatible, load_eat_config
from .dataset import EATDataset
from .model import EATNav
from .phase_c_reference import CompleteOnlyView, fit_complete_train_statistics
from visualnav_transformer.train.train_eat_nav import EarlyStopper, train_batch


def capture_rng(sampler: torch.Generator) -> dict:
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'cpu': torch.get_rng_state(),
            'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'sampler': sampler.get_state()}


def restore_rng(state: dict, sampler: torch.Generator) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['cpu'])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state['cuda'])
    sampler.set_state(state['sampler'])


def gate_c0_status(validation_losses: list[float], last_epoch: int, max_epochs: int) -> str:
    if last_epoch < max_epochs:
        return 'pending'
    # A run exhausted at the cap is not evidence of a clear plateau.
    return 'inconclusive'


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    os.replace(temp, path)


def atomic_torch(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


def best_checkpoint_payload(full_payload: dict) -> dict:
    # Keep the deployable best checkpoint readable with torch.load(weights_only=True).
    # Full Python/NumPy RNG and sampler state live in atomic latest.pt.
    best = {key: value for key, value in full_payload.items()
            if key not in ('rng_state', 'sampler_state', 'history')}
    if 'software_provenance' in best:
        best['software_provenance'] = {key: str(value) for key, value in best['software_provenance'].items()}
    return best


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1 << 20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def save_identities(path: Path, train_view, val_view) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as binary:
        with gzip.GzipFile(fileobj=binary, filename='', mode='wb', mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding='utf-8') as stream:
                for split, view in [('train', train_view), ('val', val_view)]:
                    for index, (key, step) in zip(view.indices, view.identities):
                        window = view.dataset.windows[index]
                        stream.write(json.dumps({'split': split, 'episode_key': list(key),
                                                 'step': step, 'terrain_seed': int(window['terrain_seed']),
                                                 'terrain_type': window['terrain_type']}, sort_keys=True) + '\n')
    os.replace(temp, path)
    return sha256_file(path)


def validate_frozen_identities(path: Path, train_view, val_view) -> None:
    expected = {'train': set(train_view.identities), 'val': set(val_view.identities)}
    actual = {'train': set(), 'val': set()}
    with gzip.open(path, 'rt') as stream:
        for line in stream:
            row = json.loads(line)
            split = row['split']
            if split in actual:
                actual[split].add((tuple(row['episode_key']), int(row['step'])))
    for split in expected:
        if expected[split] != actual[split]:
            raise ValueError(f'{split} identities mismatch frozen complete-only audit')
    if expected['train'] & expected['val']:
        raise ValueError('train/validation identity leakage')


def software_provenance() -> dict:
    import diffusers
    return {'python': platform.python_version(), 'torch': str(torch.__version__), 'cuda': torch.version.cuda,
            'numpy': np.__version__, 'diffusers': diffusers.__version__,
            'git_commit': os.environ.get('EAT_GIT_COMMIT', 'unknown')}


def validation_loss(model, loader, device) -> float:
    model.eval()
    weighted = 0.0
    count = 0
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == 'cuda' else []):
        torch.manual_seed(41237)
        if device.type == 'cuda':
            torch.cuda.manual_seed_all(41237)
        with torch.no_grad():
            for batch in loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                context = model.encode(batch['dem'], batch['goal'], batch['quality'])
                loss = model.noise_loss(batch['action'], batch['action_mask'], context)
                n = len(batch['action'])
                weighted += float(loss) * n
                count += n
    return weighted / count


def run(args) -> dict:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = load_eat_config(args.config)
    if config.quality_enabled or config.max_epochs != 100 or config.early_stopping_patience != 10 or config.early_stopping_min_delta != 0:
        raise ValueError('reference config must be Quality-disabled, max100, patience10, min_delta0')
    if args.batch_size != 64 or args.learning_rate != 1e-4 or args.seed != 123:
        raise ValueError('frozen reference batch/LR/seed mismatch')
    if args.mode not in ('timing', 'train'):
        raise ValueError('mode must be timing or train')
    if args.mode == 'timing' and args.epochs not in (1, 2):
        raise ValueError('timing probe must run 1 or 2 epochs')
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    atomic_json(output/'status.json', {'stage': 'loading_dataset', 'mode': args.mode})
    train_data = EATDataset(Path(args.manifest), 'train', config, split_seed=123)
    val_data = EATDataset(Path(args.manifest), 'val', config, split_seed=123)
    train_view = CompleteOnlyView(train_data, config.len_traj_pred)
    val_view = CompleteOnlyView(val_data, config.len_traj_pred)
    if (len(train_view), len(val_view)) != (49598, 6171):
        raise ValueError(f'complete-only count mismatch: {len(train_view)}, {len(val_view)}')
    validate_frozen_identities(Path(args.frozen_ids), train_view, val_view)
    id_sha = save_identities(output/'complete-window-ids.jsonl.gz', train_view, val_view)
    stats, ecdf = fit_complete_train_statistics(train_view)
    if stats.count != len(train_view) * config.len_traj_pred or len(ecdf.values) != len(train_view):
        raise ValueError('complete-only normalization count mismatch')
    split_hash = hashlib.sha256(json.dumps(train_data.split_map, sort_keys=True).encode()).hexdigest()
    source = {'manifest_sha256': sha256_file(Path(args.manifest)),
              'config_sha256': sha256_file(Path(args.config)),
              'frozen_ids_sha256': sha256_file(Path(args.frozen_ids)),
              'complete_ids_sha256': id_sha,
              'split_hash': split_hash}
    atomic_json(output/'data-audit.json', {'source': source, 'train_windows': len(train_view),
                                           'validation_windows': len(val_view),
                                           'train_action_count': stats.count,
                                           'train_action_mean': stats.mean.tolist(),
                                           'train_action_std': stats.std.tolist(),
                                           'partial_zero_included': 0})
    sampler = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_view, batch_size=args.batch_size, shuffle=True,
                              generator=sampler, num_workers=0)
    val_loader = DataLoader(val_view, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = EATNav(config).to(device)
    model.set_action_statistics(stats.mean, stats.std)
    ema = copy.deepcopy(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    stopper = EarlyStopper('min', 10, 0.)
    history = []
    global_step = 0
    start_epoch = 1
    latest = output/'latest.pt'
    if args.resume:
        payload = torch.load(latest, map_location='cpu', weights_only=False)
        assert_checkpoint_compatible(config, payload['metadata'])
        if payload['source'] != source:
            raise ValueError('resume source mismatch')
        if payload['action_statistics']['count'] != stats.count or not np.allclose(payload['action_statistics']['mean'], stats.mean):
            raise ValueError('resume normalization mismatch')
        model.load_state_dict(payload['model_state_dict'])
        ema.load_state_dict(payload['ema_state_dict'])
        optimizer.load_state_dict(payload['optimizer_state_dict'])
        history = payload['history']
        global_step = int(payload['global_step'])
        stopper.best = payload['best_metric']
        stopper.best_epoch = payload['best_epoch']
        stopper.bad_epochs = payload['bad_epochs']
        start_epoch = int(payload['epoch']) + 1
        restore_rng(payload['rng_state'], sampler)
    provenance = software_provenance()
    limit = args.epochs if args.mode == 'timing' else config.max_epochs
    for epoch in range(start_epoch, limit + 1):
        atomic_json(output/'status.json', {'stage': 'epoch_running', 'mode': args.mode,
                                           'epoch': epoch, 'global_step': global_step})
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        model.train()
        weighted_loss = 0.0
        train_count = 0
        for batch in train_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            loss = train_batch(model, optimizer, batch, ema)
            n = len(batch['action'])
            weighted_loss += loss * n
            train_count += n
            global_step += 1
        epoch_train_sec = time.monotonic() - started
        val_metric = validation_loss(ema, val_loader, device)
        improved = stopper.update(val_metric, epoch)
        metrics = {'epoch': epoch, 'global_step': global_step, 'train_loss': weighted_loss/train_count,
                   'validation_loss': val_metric, 'best': improved,
                   'train_seconds': epoch_train_sec, 'epoch_seconds': time.monotonic()-started,
                   'gpu_peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0,
                   'gpu_peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else 0,
                   'cpu_peak_rss_kb': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
        history.append(metrics)
        base = {'metadata': config.checkpoint_metadata(), 'split_hash': split_hash,
                'model_state_dict': model.state_dict(), 'ema_state_dict': ema.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'action_statistics': {'mean': stats.mean.tolist(), 'std': stats.std.tolist(), 'count': stats.count},
                'quality_ecdf': list(ecdf.values), 'epoch': epoch, 'global_step': global_step,
                'best_metric': stopper.best, 'best_epoch': stopper.best_epoch,
                'bad_epochs': stopper.bad_epochs, 'history': history,
                'rng_state': capture_rng(sampler), 'sampler_state': sampler.get_state(),
                'source': source, 'software_provenance': provenance,
                'seed': args.seed, 'batch_size': args.batch_size, 'learning_rate': args.learning_rate}
        if improved:
            atomic_torch(output/'best.pt', best_checkpoint_payload(base))
        atomic_torch(latest, base)
        metrics['latest_checkpoint_bytes'] = latest.stat().st_size
        metrics['best_checkpoint_bytes'] = (output/'best.pt').stat().st_size
        atomic_json(output/'history.json', history)
        atomic_json(output/'status.json', {'stage': 'epoch_complete', 'mode': args.mode,
                                           'epoch': epoch, 'global_step': global_step,
                                           'best_epoch': stopper.best_epoch, 'best_validation_loss': stopper.best,
                                           'metrics': metrics})
        print('PHASE_C_EPOCH ' + json.dumps(metrics, sort_keys=True), flush=True)
        if args.mode == 'train' and stopper.should_stop:
            break
    if args.mode == 'timing':
        result = {'stage': 'timing_probe_complete', 'epochs': len(history), 'metrics': history,
                  'checkpoint_bytes': latest.stat().st_size, 'source': source}
    else:
        plateau = stopper.should_stop
        result = {'stage': 'reference_training_complete', 'gate_c0': 'pass' if plateau else gate_c0_status(
                  [entry['validation_loss'] for entry in history], history[-1]['epoch'], config.max_epochs),
                  'best_epoch': stopper.best_epoch, 'best_validation_loss': stopper.best,
                  'epochs': len(history), 'source': source}
    atomic_json(output/'result.json', result)
    atomic_json(output/'status.json', result)
    atomic_json(output/(result['stage'] + '.marker.json'), {'source': source, 'latest_sha256': sha256_file(latest),
                                                             'result_sha256': sha256_file(output/'result.json')})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['timing', 'train'], required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--frozen-ids', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--learning-rate', type=float, default=1e-4)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--device', default=None)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
