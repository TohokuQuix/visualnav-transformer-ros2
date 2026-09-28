"""Phase C goal-endpoint shortcut audit; Phase B G/GT weights remain frozen."""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
import yaml

from visualnav_transformer.eat_nav.phase_c_shortcut import endpoint_fixed_shuffle

PHASE_B_SOURCE = Path('/tmp/eatnav-train/eat_nav_forward_evaluator.py')
specification = importlib.util.spec_from_file_location('frozen_phase_b', PHASE_B_SOURCE)
base = importlib.util.module_from_spec(specification)
specification.loader.exec_module(base)
HEADS = base.HEADS


class ShortcutEvaluator(base.Evaluator):
    def __init__(self, variant):
        super().__init__(trajectory=True, v0=False)
        if variant not in ('GTE', 'GTS'):
            raise ValueError(variant)
        self.variant = variant

    def forward(self, dem, goal, action, value):
        steps = action.reshape(-1, 8, 2)
        if self.variant == 'GTE':
            transformed = torch.zeros_like(steps)
            transformed[:, -1] = steps.sum(dim=1)
        else:
            transformed = steps - steps.mean(dim=1, keepdim=True)
        return super().forward(dem, goal, transformed.reshape(-1, 16), value)


def train_one(variant, seed, train, val, spec, root):
    torch.manual_seed(seed); np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ShortcutEvaluator(variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'])
    tr, va = base.tensors(train, device), base.tensors(val, device)
    n, batch = len(tr[0]), spec['batch_size']
    best = math.inf; best_epoch = -1; wait = 0; curve = []
    folder = root/'checkpoints'/f'{variant}-{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(spec['max_epochs']):
        model.train()
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed*100000+epoch)).to(device)
        for ids in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*(a[ids] for a in tr[:4]))
            loss = ((prediction-tr[4][ids])**2).mean()
            loss.backward(); optimizer.step()
        prediction = base.predict(model, va, batch)
        vl = float(np.mean((prediction-va[4].cpu().numpy())**2))
        curve.append(vl)
        if vl < best-spec['min_delta']:
            best = vl; best_epoch = epoch; wait = 0
            temp = folder/'best.pt.tmp'
            torch.save({'model': model.state_dict(), 'variant': variant, 'seed': seed,
                        'epoch': epoch, 'validation_loss': vl,
                        'source_split_sha256': spec['source_split_sha256']}, temp)
            temp.replace(folder/'best.pt')
        else:
            wait += 1
            if wait >= spec['patience']:
                break
    result = {'variant': variant, 'seed': seed, 'best_epoch': best_epoch, 'last_epoch': epoch,
              'best_val_loss': best, 'curve': curve,
              'parameter_count': sum(p.numel() for p in model.parameters()),
              'checkpoint_sha256': base.sha(folder/'best.pt')}
    base.write(folder/'training.json', result)
    return result


def load_model(variant, seed, device, root, spec, frozen_runs):
    if variant in ('G', 'GT'):
        model = base.Evaluator(*base.VARIANTS[variant]).to(device)
        checkpoint = base.ROOT/'checkpoints'/f'{variant}-{seed}'/'best.pt'
        matched = next(r for r in frozen_runs if r['variant'] == variant and r['seed'] == seed)
        if base.sha(checkpoint) != matched['checkpoint_sha256']:
            raise ValueError('frozen Phase B checkpoint SHA mismatch')
    else:
        model = ShortcutEvaluator(variant).to(device)
        checkpoint = root/'checkpoints'/f'{variant}-{seed}'/'best.pt'
    payload = torch.load(checkpoint, map_location=device, weights_only=True)
    if payload['source_split_sha256'] != spec['source_split_sha256']:
        raise ValueError('checkpoint split SHA mismatch')
    model.load_state_dict(payload['model'])
    model.eval()
    return model, base.sha(checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=Path('/tmp/eatnav-phase-c/shortcut'))
    args = parser.parse_args()
    root = args.output; root.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(base.SPEC.read_text())
    torch.set_num_threads(4)
    started = time.monotonic()
    base.write(root/'status.json', {'stage': 'audit'})
    train = base.load_split('train', spec)
    val = base.load_split('val', spec)
    frozen_runs = json.loads((base.ROOT/'training-summary.json').read_text())['runs']
    if {(r['variant'], r['seed']) for r in frozen_runs if r['variant'] in ('G','GT')} != {(v,s) for v in ('G','GT') for s in spec['seeds']}:
        raise ValueError('frozen G/GT runs missing')
    runs = []
    for seed in spec['seeds']:
        for variant in ('GTE', 'GTS'):
            runs.append(train_one(variant, seed, train, val, spec, root))
            base.write(root/'status.json', {'stage': 'training', 'completed': len(runs), 'total': 6})
    base.write(root/'training-summary.json', {'runs': runs, 'source_split_sha256': spec['source_split_sha256']})
    # Held-out is read only after all diagnostic checkpoints are selected by validation.
    test = base.load_split('test', spec)
    audit = json.loads((base.SOURCE/'output'/'relative-quality-extraction.json').read_text())
    scalers = audit['train_only_scalers']
    y = np.column_stack([test['z_'+head] for head in HEADS])
    episodes = test['episode']
    arrays = base.tensors(test, 'cuda' if torch.cuda.is_available() else 'cpu')
    prediction = {}; shuffled = {}; hashes = {}
    shuffled_action = endpoint_fixed_shuffle(test['action'], test['terrain'], seed=613)
    shuffled_arrays = (arrays[0], arrays[1], torch.as_tensor(shuffled_action.reshape(-1,16), dtype=torch.float32, device=arrays[2].device), arrays[3], arrays[4])
    for seed in spec['seeds']:
        for variant in ('G','GTE','GTS','GT'):
            model, digest = load_model(variant, seed, arrays[0].device, root, spec, frozen_runs)
            name = f'{variant}-{seed}'
            hashes[name] = digest
            prediction[name] = base.predict(model, arrays, spec['batch_size'])
            if variant == 'GT':
                shuffled[name] = base.predict(model, shuffled_arrays, spec['batch_size'])
    result = {'source_split_sha256': spec['source_split_sha256'], 'checkpoint_sha256': hashes,
              'frozen_phase_b_retrained': False, 'heldout_model_selection': False,
              'counts': spec['counts'], 'metrics': {}, 'paired': {}, 'endpoint_fixed_shuffle': {},
              'gts_endpoint_sum_max_abs': float(np.max(np.abs((test['action'] - test['action'].mean(axis=1, keepdims=True)).sum(axis=1))))}
    for name, p in prediction.items():
        m = base.metrics(y, p, scalers)
        m['aggregate_episode_ci95'] = base.episode_ci(((y-p)**2).mean(1), episodes)
        result['metrics'][name] = m
    for seed in spec['seeds']:
        for better, worse in [('GT','GTE'), ('GTS','G'), ('GTE','G')]:
            a, b = prediction[f'{better}-{seed}'], prediction[f'{worse}-{seed}']
            gain = (b-y)**2 - (a-y)**2
            result['paired'][f'{better}-{seed}_vs_{worse}-{seed}'] = {
                'aggregate_gain': float(gain.mean()),
                'aggregate_episode_ci95': base.episode_ci(gain.mean(axis=1), episodes),
                'head_gain': {h: float(gain[:, j].mean()) for j,h in enumerate(HEADS)},
                'head_episode_ci95': {h: base.episode_ci(gain[:, j], episodes) for j,h in enumerate(HEADS)}}
        original = prediction[f'GT-{seed}']; changed = shuffled[f'GT-{seed}']
        degradation = ((changed-y)**2 - (original-y)**2).mean(axis=1)
        result['endpoint_fixed_shuffle'][str(seed)] = {
            'aggregate_degradation': float(degradation.mean()),
            'aggregate_episode_ci95': base.episode_ci(degradation, episodes),
            'prediction_rms_change': float(np.sqrt(np.mean((changed-original)**2))),
            'endpoint_max_abs_error': float(np.max(np.abs(shuffled_action.sum(axis=1)-test['action'].sum(axis=1))))}
    result['elapsed_seconds'] = time.monotonic()-started
    base.write(root/'result.json', result)
    base.write(root/'status.json', {'stage': 'complete', 'elapsed_seconds': result['elapsed_seconds']})
    base.write(root/'complete.marker.json', {'result_sha256': base.sha(root/'result.json'), 'checkpoint_sha256': hashes})
    print(json.dumps({'elapsed_seconds': result['elapsed_seconds'], 'paired': result['paired'],
                      'endpoint_fixed_shuffle': result['endpoint_fixed_shuffle']}), flush=True)


if __name__ == '__main__':
    main()
