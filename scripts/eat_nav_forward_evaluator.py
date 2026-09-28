"""Frozen-log Phase B forward evaluator. Run inside the Isaac container."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
import yaml

ROOT = Path('/tmp/eatnav-quality-identifiability-gate/forward-evaluator')
SOURCE = Path('/tmp/eatnav-quality-identifiability-gate/relative-quality')
BASE = Path('/tmp/eatnav-train')
SPEC = BASE / 'forward-evaluator-config.yaml'
HEADS = ('delta_v', 'advantage_h', 'drawdown', 'recovery')
VARIANTS = {'G': (False, False), 'GT': (True, False), 'GV0': (False, True), 'GTV0': (True, True)}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def mark(name, value):
    write(ROOT / 'stages' / f'{name}.json', value)
    write(ROOT / 'status.json', {'stage': name, 'time_unix': time.time(), **value})
    print('FORWARD_STAGE', name, value.get('status'), flush=True)


def load_split(name, spec):
    path = SOURCE / 'splits' / f'{name}.npz'
    if sha(path) != spec['source_split_sha256'][name]:
        raise ValueError(f'{name} source SHA mismatch')
    with np.load(path) as source:
        data = {k: source[k] for k in source.files}
    if len(data['index']) != spec['counts'][name]:
        raise ValueError(f'{name} window count mismatch')
    assert data['dem'].shape[1:] == (4, 2, 20, 20)
    assert data['goal'].shape[1:] == (2,)
    assert data['action'].shape[1:] == (8, 2)
    assert data['episode'].shape[1:] == (3,)
    for key in ('dem', 'goal', 'action', 'v0', *HEADS):
        if not np.isfinite(data[key]).all():
            raise ValueError(f'{name}:{key} nonfinite')
    return data


def trajectory_features(action):
    action = np.asarray(action)
    xy = np.cumsum(action, axis=1)
    length = np.linalg.norm(action, axis=2).sum(axis=1)
    angle = np.unwrap(np.arctan2(action[:, :, 1], action[:, :, 0]), axis=1)
    turn = np.abs(np.diff(angle, axis=1)).sum(axis=1)
    curvature = turn / np.maximum(length, 1e-4)
    step_max = np.linalg.norm(action, axis=2).max(axis=1)
    return np.column_stack((xy[:, -1], xy[:, -1, 0], xy[:, -1, 1], length, turn, curvature, step_max)).astype(np.float32)


def simple_features(data, trajectory=False, v0=False):
    valid = data['dem'][:, :, 1] > .5
    heights = data['dem'][:, :, 0]
    mean = (heights * valid).sum(axis=(1, 2, 3)) / np.maximum(valid.sum(axis=(1, 2, 3)), 1)
    rough = np.sqrt((((heights - mean[:, None, None, None]) * valid) ** 2).sum(axis=(1, 2, 3)) / np.maximum(valid.sum(axis=(1, 2, 3)), 1))
    fields = [data['goal'], np.linalg.norm(data['goal'], axis=1)[:, None], rough[:, None], valid.mean(axis=(1, 2, 3))[:, None]]
    if trajectory:
        fields.append(trajectory_features(data['action']))
    if v0:
        fields.append(data['v0'][:, None])
    return np.column_stack(fields).astype(np.float32)


class Evaluator(nn.Module):
    def __init__(self, trajectory, v0):
        super().__init__()
        self.trajectory = trajectory
        self.v0 = v0
        self.dem = nn.Sequential(nn.Conv2d(8, 16, 3, stride=2, padding=1), nn.ReLU(), nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(), nn.Flatten(), nn.Linear(800, 32), nn.ReLU())
        self.goal = nn.Sequential(nn.Linear(2, 16), nn.ReLU(), nn.Linear(16, 16), nn.ReLU())
        self.traj = nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 32), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(1, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(88, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        self.heads = nn.ModuleList([nn.Linear(64, 1) for _ in HEADS])

    def forward(self, dem, goal, action, value):
        # Every branch and parameter exists in every variant. Disabled inputs are fixed zero.
        if not self.trajectory:
            action = torch.zeros_like(action)
        if not self.v0:
            value = torch.zeros_like(value)
        z = torch.cat((self.dem(dem), self.goal(goal), self.traj(action), self.value(value)), dim=1)
        z = self.fuse(z)
        return torch.cat([h(z) for h in self.heads], dim=1)


def tensors(data, device):
    def t(x):
        return torch.as_tensor(np.asarray(x, dtype=np.float32), device=device)
    return (t(data['dem'].reshape(-1, 8, 20, 20)), t(data['goal']), t(data['action'].reshape(-1, 16)), t(data['z_v0'][:, None]), t(np.column_stack([data['z_' + h] for h in HEADS])))


@torch.no_grad()
def predict(model, arrays, batch):
    model.eval()
    n = len(arrays[0])
    return np.concatenate([model(*(a[i:i + batch] for a in arrays[:4])).cpu().numpy() for i in range(0, n, batch)])


def train_one(name, seed, train, val, spec, smoke=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Evaluator(*VARIANTS[name]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec['learning_rate'])
    tr, va = tensors(train, device), tensors(val, device)
    batch = spec['batch_size']
    epochs = 2 if smoke else spec['max_epochs']
    n = len(tr[0]); best = math.inf; best_epoch = -1; wait = 0; curve = []
    folder = ROOT / ('smoke' if smoke else 'checkpoints') / f'{name}-{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    for epoch in range(epochs):
        model.train()
        # Common order across variants for each seed.
        order = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 100000 + epoch), device='cpu').to(device)
        for ids in order.split(batch):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(*(a[ids] for a in tr[:4]))
            loss = ((prediction - tr[4][ids]) ** 2).mean()
            loss.backward(); optimizer.step()
        p = predict(model, va, batch)
        vl = float(np.mean((p - va[4].cpu().numpy()) ** 2))
        curve.append(vl)
        if vl < best - spec['min_delta']:
            best = vl; best_epoch = epoch; wait = 0
            torch.save({'model': model.state_dict(), 'variant': name, 'seed': seed, 'epoch': epoch, 'validation_loss': vl, 'source_split_sha256': spec['source_split_sha256']}, folder / 'best.pt')
        else:
            wait += 1
            if wait >= spec['patience']:
                break
    outcome = {'variant': name, 'seed': seed, 'best_epoch': best_epoch, 'last_epoch': epoch, 'best_val_loss': best,
               'curve': curve, 'parameter_count': sum(p.numel() for p in model.parameters()),
               'underfit': epoch == epochs - 1 and best_epoch >= epochs - 1 - spec['patience'],
               'checkpoint_sha256': sha(folder / 'best.pt')}
    write(folder / 'training.json', outcome)
    return outcome


def fit_linear(train, val, spec, smoke=False):
    # Select ridge strength using validation only, then freeze for held-out.
    result = {}
    y = np.column_stack([train['z_' + h] for h in HEADS])
    vy = np.column_stack([val['z_' + h] for h in HEADS])
    for name, use_traj, use_v0 in [('goal_dem_linear', False, False), ('trajectory_stats_linear', True, False), ('v0_linear', False, True), ('trajectory_v0_linear', True, True)]:
        x = simple_features(train, use_traj, use_v0)
        vx = simple_features(val, use_traj, use_v0)
        if name == 'trajectory_stats_linear':
            x, vx = trajectory_features(train['action']), trajectory_features(val['action'])
        if name == 'v0_linear':
            x, vx = train['v0'][:, None].astype(np.float32), val['v0'][:, None].astype(np.float32)
        mean = x.mean(0); std = np.maximum(x.std(0), 1e-6)
        x = np.column_stack((np.ones(len(x)), (x - mean) / std))
        vx = np.column_stack((np.ones(len(vx)), (vx - mean) / std))
        best = None
        for ridge in (1., 100., 10000.):
            matrix = x.T @ x + ridge * np.eye(x.shape[1])
            matrix[0, 0] -= ridge
            coef = np.linalg.solve(matrix, x.T @ y)
            mse = float(np.mean((vx @ coef - vy) ** 2))
            if best is None or mse < best['validation_loss']:
                best = {'ridge': ridge, 'validation_loss': mse, 'mean': mean.tolist(), 'std': std.tolist(), 'coef': coef.tolist(), 'trajectory': use_traj, 'v0': use_v0}
        result[name] = best
    write(ROOT / ('smoke' if smoke else 'baselines') / 'linear.json', result)
    return result


def linear_predict(data, name, params):
    if name == 'trajectory_stats_linear':
        x = trajectory_features(data['action'])
    elif name == 'v0_linear':
        x = data['v0'][:, None].astype(np.float32)
    else:
        x = simple_features(data, params['trajectory'], params['v0'])
    x = np.column_stack((np.ones(len(x)), (x - np.asarray(params['mean'])) / np.asarray(params['std'])))
    return x @ np.asarray(params['coef'])


def metrics(y, pred, scalers):
    result = {}
    for j, h in enumerate(HEADS):
        truth = y[:, j]; p = pred[:, j]; error = p - truth
        r2 = 1 - float(np.mean(error ** 2)) / max(float(np.var(truth)), 1e-12)
        variable_prediction = float(np.ptp(p)) > 1e-7
        variable_truth = float(np.ptp(truth)) > 1e-7
        slope, intercept = np.polyfit(p, truth, 1) if variable_prediction else (None, None)
        rank = spearmanr(truth, p).statistic if variable_prediction and variable_truth else None
        rank = float(rank) if rank is not None and np.isfinite(rank) else None
        result[h] = {'mae_raw': float(np.mean(np.abs(error)) * scalers[h]['std']), 'rmse_raw': float(np.sqrt(np.mean(error ** 2)) * scalers[h]['std']),
                     'standardized_mse': float(np.mean(error ** 2)), 'r2': r2, 'spearman': rank,
                     'calibration_slope': None if slope is None else float(slope), 'calibration_intercept_z': None if intercept is None else float(intercept)}
    result['aggregate_mse'] = float(np.mean((y - pred) ** 2))
    result['macro_r2'] = float(np.mean([result[h]['r2'] for h in HEADS]))
    valid_ranks = [result[h]['spearman'] for h in HEADS if result[h]['spearman'] is not None]
    result['macro_spearman'] = float(np.mean(valid_ranks)) if valid_ranks else None
    return result


def episode_ci(values, episodes, seed=613, repeats=500):
    groups, inverse = np.unique(episodes, axis=0, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(groups)); counts = np.bincount(inverse, minlength=len(groups))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(groups), (repeats, len(groups)))
    boot = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(x) for x in np.quantile(boot, [.025, .975])]


def pair_episode_ci(correct, left, right, seed=613, repeats=500):
    episodes, inverse = np.unique(np.vstack((left, right)), axis=0, return_inverse=True)
    n = len(left); li, ri = inverse[:n], inverse[n:]
    rng = np.random.default_rng(seed)
    counts = rng.multinomial(len(episodes), np.full(len(episodes), 1/len(episodes)), size=repeats)
    weight = counts[:, li] * counts[:, ri]
    denom = weight.sum(1)
    valid = denom > 0
    if valid.sum() < repeats // 2:
        return None
    scores = (weight[valid] @ np.asarray(correct)) / denom[valid]
    return [float(x) for x in np.quantile(scores, [.025, .975])]


def audit(spec):
    expected = json.loads((BASE / 'relative-quality-followup-actor-provenance.json').read_text())
    if expected['actor_observation_status_counts'] != {'missing': 360000}:
        raise ValueError('actor observation provenance changed')
    if sha(BASE / 'relative-quality-config.yaml') != spec['source_relative_config_sha256']:
        raise ValueError('Phase A config SHA mismatch')
    if sha(BASE / 'relative-quality-followup-config.yaml') != spec['source_followup_config_sha256']:
        raise ValueError('Phase A followup SHA mismatch')
    split = {n: load_split(n, spec) for n in ('train', 'val')}
    for n, d in split.items():
        assert len({tuple(x) for x in d['episode']}) > 50
    source = yaml.safe_load((BASE / 'relative-quality-followup-config.yaml').read_text())
    if spec['source_split_sha256'] != source['source_aligned_split_sha256']:
        raise ValueError('frozen split changed')
    provenance = {'status': 'missing', 'actor_observation_dim': 22, 'record_count': 360000,
                  'anchor_alignment': 'unavailable: exact pre-action obs[policy] was not serialized',
                  'fields': ['base_ang_vel', 'projected_gravity', 'velocity_commands', 'flipper_fold_mode', 'joint_pos_rel', 'track_mean_velocity', 'joint_effort_with_track_sum'],
                  'normalization': 'empirical normalization disabled; configured scale/clip and policy observation noise enabled',
                  'history': 'no actor observation history configured', 'privileged': False,
                  'deployability': 'some joints and command available on robot, but exact noisy vector absent from log',
                  'available_variants': ['G', 'GT', 'GV0', 'GTV0'], 'unavailable_variants': ['GB', 'GBT', 'GL', 'GLT', 'GBTV0', 'GLTV0']}
    write(ROOT / 'actor-input-provenance.json', provenance)
    write(ROOT / 'actor-latent-provenance.json', {'status': 'unavailable', 'reason': 'no saved latent or exact actor observation',
          'actor_checkpoint_sha256': expected['actor_checkpoint_sha256'], 'exact_recomputation': False, 'action_consistency_test': 'unavailable'})
    write(ROOT / 'critic-v0-deployability.json', {'deployable': False, 'status': 'privileged upper bound only',
          'critic_observation_dim': 267, 'privileged_fields': ['base_lin_vel', 'control_latency', 'roller_contact_binary', 'height_scan_critic'],
          'note': 'No on-robot equivalence established for critic observation. V0 is never used for deployable gate.'})
    write(ROOT / 'trajectory-provenance.json', {'shape': [8, 2], 'interval_sec': .25, 'horizon_sec': 2.,
          'representation': 'increments of 8 cumulative robot-local XY waypoints', 'axes': 'forward x, left y',
          'goal_frame': 'same robot-local frame', 'source': 'frozen complete aligned window',
          'future_fields_allowed': ['candidate XY trajectory'], 'episode_boundary_crossings': 0})
    mark('audit', {'status': 'pass_with_actor_input_unavailable', 'counts': {n: len(d['index']) for n, d in split.items()}})
    return split


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--smoke', action='store_true'); parser.add_argument('--evaluate-only', action='store_true'); args = parser.parse_args()
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = yaml.safe_load(SPEC.read_text()); torch.set_num_threads(4)
    started = time.monotonic()
    try:
        if args.evaluate_only:
            runs = json.loads((ROOT / 'training-summary.json').read_text())['runs']
            if len(runs) != 12 or {(r['variant'], r['seed']) for r in runs} != {(v, seed) for v in VARIANTS for seed in spec['seeds']}:
                raise ValueError('frozen training run set incomplete')
            evaluate(spec, runs)
            mark('complete', {'status': 'pass', 'evaluation_only': True})
            return
        data = audit(spec)
        if args.smoke:
            data = {k: {n: v[:256] for n, v in d.items()} for k, d in data.items()}
        fit_linear(data['train'], data['val'], spec, args.smoke)
        mark('baselines', {'status': 'pass', 'smoke': args.smoke})
        runs = []
        for seed in ([11] if args.smoke else spec['seeds']):
            for name in VARIANTS:
                run = train_one(name, seed, data['train'], data['val'], spec, args.smoke)
                runs.append(run)
                mark('training', {'status': 'running', 'completed_runs': len(runs), 'total_runs': 4 if args.smoke else 12})
        write(ROOT / ('smoke/training-summary.json' if args.smoke else 'training-summary.json'), {'runs': runs, 'elapsed_sec': time.monotonic() - started})
        if args.smoke:
            mark('smoke', {'status': 'pass', 'elapsed_sec': time.monotonic() - started,
                           'peak_gpu_bytes': int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None})
            return
        evaluate(spec, runs)
        mark('complete', {'status': 'pass', 'elapsed_sec': time.monotonic() - started})
    except Exception as exc:
        mark('failed', {'status': 'failed', 'error': repr(exc)})
        raise


def evaluate(spec, runs):
    # The held-out split is read only after all variants, seeds and validation checkpoints are fixed.
    test = load_split('test', spec)
    audit_result = json.loads((SOURCE / 'output' / 'relative-quality-extraction.json').read_text())
    scalers = audit_result['train_only_scalers']
    y = np.column_stack([test['z_' + h] for h in HEADS]); episodes = test['episode']
    baselines = json.loads((ROOT / 'baselines' / 'linear.json').read_text())
    prediction = {'train_mean': np.zeros_like(y)}
    train = load_split('train', spec)
    means = {}
    for terrain in set(train['terrain']):
        ids = train['terrain'] == terrain
        means[str(terrain)] = np.column_stack([train['z_' + h] for h in HEADS])[ids].mean(0)
    prediction['terrain_train_mean'] = np.asarray([means.get(str(t), np.zeros(4)) for t in test['terrain']])
    for name, params in baselines.items():
        prediction[name] = linear_predict(test, name, params)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    arrays = tensors(test, device)
    for run in runs:
        name = run['variant']; seed = run['seed']; model = Evaluator(*VARIANTS[name]).to(device)
        ckpt = ROOT / 'checkpoints' / f'{name}-{seed}' / 'best.pt'
        if sha(ckpt) != run['checkpoint_sha256']:
            raise ValueError('checkpoint SHA mismatch')
        saved = torch.load(ckpt, map_location=device, weights_only=True)
        if saved['source_split_sha256'] != spec['source_split_sha256'] or saved['variant'] != name:
            raise ValueError('checkpoint compatibility mismatch')
        model.load_state_dict(saved['model']); prediction[f'{name}-{seed}'] = predict(model, arrays, spec['batch_size'])
    endpoint = np.cumsum(test['action'], axis=1)[:, -1]
    goal_difference = np.linalg.norm(endpoint-test['goal'],axis=1)
    result = {'counts': spec['counts'], 'metrics': {}, 'paired': {}, 'ordering': {}, 'train_only_scalers': scalers,
              'goal_from_logged_future_endpoint': {'exact_fraction': float(np.mean(goal_difference < 1e-5)), 'clip_radius_m': .8, 'interpretation': 'This frozen dataset derives relative goal from the same logged future path, clipping distant endpoints. Matched ranking is an on-policy proxy.'},
              'missing_variants': ['GB', 'GBT', 'GL', 'GLT', 'GBTV0', 'GLTV0'], 'heldout_model_selection': False}
    for name, p in prediction.items():
        m = metrics(y, p, scalers)
        m['episode_loss_ci95'] = episode_ci(((p-y)**2).mean(1), episodes)
        for j, h in enumerate(HEADS):
            m[h]['standardized_mse_ci95'] = episode_ci((p[:,j]-y[:,j])**2, episodes)
        m['terrain'] = {}
        for terrain in sorted(set(test['terrain'])):
            ids = test['terrain'] == terrain
            m['terrain'][str(terrain)] = {'count': int(ids.sum()), 'metrics': metrics(y[ids], p[ids], scalers) if ids.sum() >= 30 else None}
        result['metrics'][name] = m
    for seed in spec['seeds']:
        for better, worse in [('GT','G'), ('GTV0','GV0'), ('GV0','G'), ('GTV0','GT')]:
            key = f'{better}-{seed}_vs_{worse}-{seed}'
            pa, pb = prediction[f'{better}-{seed}'], prediction[f'{worse}-{seed}']
            improvement = (pb-y)**2 - (pa-y)**2
            result['paired'][key] = {'aggregate_gain': float(improvement.mean()), 'aggregate_ci95': episode_ci(improvement.mean(1), episodes),
                 'head_gain': {h: float(improvement[:, j].mean()) for j,h in enumerate(HEADS)},
                 'head_ci95': {h: episode_ci(improvement[:, j], episodes) for j,h in enumerate(HEADS)}}
    for seed in spec['seeds']:
        better = prediction[f'GT-{seed}']
        for baseline in ('train_mean','goal_dem_linear','trajectory_stats_linear'):
            loser = prediction[baseline]
            gain = ((loser-y)**2 - (better-y)**2).mean(1)
            result['paired'][f'GT-{seed}_vs_{baseline}'] = {'aggregate_gain': float(gain.mean()), 'aggregate_ci95': episode_ci(gain, episodes)}
    train_terrains = set(train['terrain']); val_terrains = set(load_split('val', spec)['terrain'])
    common = sorted(train_terrains & val_terrains & set(test['terrain']))
    result['common_support_terrains'] = common
    result['common_support'] = {}
    for name,p in prediction.items():
        ids = np.isin(test['terrain'], common)
        result['common_support'][name] = {'count': int(ids.sum()), 'metrics': metrics(y[ids],p[ids],scalers)}
    result['seed_variance'] = {variant: {key: float(np.std([result['metrics'][f'{variant}-{seed}'][key] for seed in spec['seeds']],ddof=1)) for key in ('aggregate_mse','macro_r2')} for variant in VARIANTS}
    ordering(result, prediction, test, spec)
    result['gate'] = decision(result, runs, spec)
    write(ROOT / 'heldout-results.json', result)
    write(ROOT / 'decision.json', result['gate'])
    write(ROOT / 'artifact-hashes.json', {p.name: sha(p) for p in ROOT.glob('*.json') if p.name != 'artifact-hashes.json'})
    mark('evaluation', {'status': 'pass', 'gate': result['gate']['predictive_signal']})


def ordering(result, prediction, test, spec):
    frozen = json.loads((BASE / 'relative-quality-matched-pairs.json').read_text())
    positions = {int(v): i for i,v in enumerate(test['index'])}
    for head in ('delta_v', 'advantage_h', 'drawdown'):
        records = frozen['metrics'][head]['cross']['pairs']
        pairs = [(positions[r['left_index']], positions[r['right_index']]) for r in records if r['left_index'] in positions and r['right_index'] in positions]
        j = HEADS.index(head)
        truth = test['z_'+head]
        valid = [(a,b) for a,b in pairs if abs(truth[a]-truth[b]) > 1e-6]
        for name,p in prediction.items():
            correct = np.asarray([np.sign(p[a,j]-p[b,j]) == np.sign(truth[a]-truth[b]) for a,b in valid], dtype=float)
            left_episodes = np.asarray([test['episode'][a] for a,b in valid])
            right_episodes = np.asarray([test['episode'][b] for a,b in valid])
            ci = pair_episode_ci(correct, left_episodes, right_episodes) if len(correct) else None
            result['ordering'].setdefault(name,{})[head] = {'pairs': len(valid), 'accuracy': float(correct.mean()) if len(correct) else None, 'episode_cluster_ci95': ci,
                  'note': 'cross-episode near-neighborhood proxy, not same-state counterfactual ranking'}


def decision(result, runs, spec):
    checks = {}
    for seed in spec['seeds']:
        m = result['metrics'][f'GT-{seed}']; benefit = result['paired'][f'GT-{seed}_vs_G-{seed}']
        checks[str(seed)] = {'two_primary_r2': sum(m[h]['r2'] > .05 for h in HEADS[:3]) >= 2,
            'delta_spearman': m['delta_v']['spearman'] is not None and m['delta_v']['spearman'] > .2,
            'trajectory_ci': benefit['aggregate_ci95'][0] > 0,
            'linear_proxy': m['aggregate_mse'] < result['metrics']['trajectory_stats_linear']['aggregate_mse'],
            'simple_baseline': all(result['paired'][f'GT-{seed}_vs_{baseline}']['aggregate_ci95'][0] > 0 for baseline in ('train_mean','goal_dem_linear','trajectory_stats_linear'))}
    common = result['common_support_terrains']
    terrain_consistent = all(result['metrics'][f'G-{seed}']['terrain'][terrain]['metrics']['aggregate_mse'] > result['metrics'][f'GT-{seed}']['terrain'][terrain]['metrics']['aggregate_mse'] for seed in spec['seeds'] for terrain in common)
    ordering_supported = all(result['ordering'][f'GT-{seed}']['delta_v']['episode_cluster_ci95'][0] > .5 for seed in spec['seeds'])
    episode_count = len({tuple(row) for row in load_split('test', spec)['episode']})
    underfit = any(r['underfit'] for r in runs if r['variant'] in ('G', 'GT'))
    passes = all(all(v.values()) for v in checks.values()) and terrain_consistent and episode_count >= 30 and not underfit
    return {'predictive_signal': 'present' if passes else ('inconclusive_underfit' if underfit else 'not_established'),
            'checks_by_seed': checks, 'underfit': underfit, 'body_benefit': 'unavailable_actor_observation',
            'latent_benefit': 'unavailable_actor_observation', 'oracle_rollout_recommended': bool(passes and ordering_supported),
            'common_support_direction_consistent': terrain_consistent, 'ordering_delta_ci_above_chance': ordering_supported, 'heldout_episodes': episode_count,
            'note': 'No simulator rollout or policy integration was performed.'}


if __name__ == '__main__':
    main()
