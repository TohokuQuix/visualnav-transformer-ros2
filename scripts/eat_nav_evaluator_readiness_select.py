"""Freeze seven new complete-only train states before sampling or rollout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.evaluator_readiness import select_states, TERRAINS


EXPECTED = {
    'manifest': '55d9e2d89631c2afa19dabfb02b2753616bc2b71907843f5570bd3a8e41e1475',
    'reference_config': 'caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992',
    'reference_checkpoint': '191d6a03266bca2ce8d43afac44b04828ae0d2571632be4f47a5f0dc3564b4e0',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'frozen artifact changed: {path}')
        return
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(content)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for name in ('manifest', 'reference-config', 'reference-checkpoint', 'prior-selection', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    args = p.parse_args()
    source = {'manifest': args.manifest, 'reference_config': args.reference_config,
              'reference_checkpoint': args.reference_checkpoint}
    hashes = {k: sha(v) for k, v in source.items()}
    if hashes != EXPECTED:
        raise ValueError(f'split/checkpoint/config SHA mismatch: {hashes}')
    cfg = load_eat_config(args.reference_config)
    if cfg.context_size != 3 or cfg.len_traj_pred != 8:
        raise ValueError('reference NoMaD shape changed')
    view = CompleteOnlyView(EATDataset(args.manifest, 'train', cfg, split_seed=123), 8)
    if len(view) != 49598:
        raise ValueError('complete-only train count changed')
    prior = json.loads(args.prior_selection.read_text())['selected']
    if len(prior) != 7:
        raise ValueError('prior seven-state selection changed')
    prior_episodes = {tuple(r['episode_key']) for r in prior}
    rows = []
    for vi, di in enumerate(view.indices):
        w = view.dataset.windows[di]
        dem = np.asarray(w['dem'], dtype=np.float32)
        goal = np.asarray(w['goal'], dtype=np.float32)
        rows.append({'split': 'train', 'complete': bool(np.asarray(w['action_mask']).all()),
                     'terrain_type': w['terrain_type'], 'episode_key': list(w['episode_key']),
                     'step': int(w['step']), 'local_step': int(w['local_step']),
                     'view_index': vi, 'goal': goal.astype(float).tolist(),
                     'dem_sha256': hashlib.sha256(dem.tobytes()).hexdigest(),
                     'dem_finite': bool(np.isfinite(dem).all()),
                     'goal_finite': bool(np.isfinite(goal).all()),
                     'boundary_clean': bool(w['episode_id'] == w['episode_key'][2] and
                                            np.asarray(w['action_mask']).all()),
                     'terrain_seed': int(w['terrain_seed'])})
    result = select_states(rows, TERRAINS, prior_episodes=prior_episodes)
    result.update({'source_sha256': hashes, 'prior_selection_sha256': sha(args.prior_selection),
                   'complete_only_train_windows': len(view), 'dem_history_frames': 4,
                   'teacher_trajectory_points': 8, 'validation_used': False,
                   'heldout_used': False, 'candidate_outcomes_used': False})
    atomic_once(args.output / 'selection.json', result)
    atomic_once(args.output / 'status.json', {'stage': 'selection_frozen',
                'selection_sha256': sha(args.output / 'selection.json'),
                'selected_count': len(result['selected']), 'rollout_started': False})
    print('READINESS_SELECTION_FROZEN',
          [(r['terrain_type'], r['view_index'], r['episode_local_prefix_steps'])
           for r in result['selected']], flush=True)


if __name__ == '__main__':
    main()
