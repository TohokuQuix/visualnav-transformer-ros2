"""Freeze 112-run inputs from seven new train states, without candidate outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import yaml

from visualnav_transformer.eat_nav.checkpoint import load_checkpoint
from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.candidate_coverage_sampling import sample_paths
from visualnav_transformer.eat_nav.feasible_set import maneuver_bank
from visualnav_transformer.eat_nav.candidate_coverage_offline import candidate_constraints, path_metrics
from eat_nav_evaluator_readiness_select import EXPECTED, atomic_once, sha

NAMES = ('nominal', 'slight_left', 'slight_right', 'strong_left', 'strong_right')
SEEDS = tuple(range(1001, 1009))


def save_npz_once(path, **arrays):
    path = Path(path)
    if path.exists():
        with np.load(path) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise FileExistsError(f'frozen NPZ differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.npz.tmp')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for name in ('manifest', 'reference-config', 'reference-checkpoint', 'selection',
                 'maneuver-thresholds', 'constraint-limits', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    selection = json.loads(a.selection.read_text())
    hashes = {'manifest': sha(a.manifest), 'reference_config': sha(a.reference_config),
              'reference_checkpoint': sha(a.reference_checkpoint)}
    if hashes != EXPECTED or selection['source_sha256'] != hashes or len(selection['selected']) != 7:
        raise ValueError('source or selection mismatch')
    cfg = load_eat_config(a.reference_config)
    view = CompleteOnlyView(EATDataset(a.manifest, 'train', cfg, split_seed=123), 8)
    if len(view) != 49598:
        raise ValueError('complete-only train count changed')
    manifest = json.loads(a.manifest.read_text())
    max_step = max(s['step'] for s in selection['selected']) + 120
    actions, anchors, commands = {}, {}, {i: {} for i in range(7)}
    for chunk in manifest['chunks']:
        with gzip.open(a.manifest.parent / chunk['file'], 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                step = int(row['step'])
                if step > max_step:
                    continue
                env = int(row['env_id'])
                actions.setdefault(step, {})[env] = row['action']
                for j, state in enumerate(selection['selected']):
                    if env != state['episode_key'][1] or step > state['step']:
                        continue
                    if [row['session_id'], env, row['episode_id']] != state['episode_key']:
                        raise ValueError('episode boundary crossed in prefix')
                    commands[j][step] = row['command']
                    if step == state['step']:
                        if row['pre_reset']:
                            raise ValueError('anchor is pre-reset')
                        anchors[j] = row
        if len(anchors) == 7 and all(len(actions.get(step, {})) == 100 for step in range(max_step + 1)):
            break
    if len(anchors) != 7 or not all(len(actions.get(step, {})) == 100 for step in range(max_step + 1)):
        raise ValueError('incomplete all-env replay actions')
    all_actions = np.asarray([[actions[step][env] for env in range(100)]
                              for step in range(max_step + 1)], dtype=np.float32)
    if all_actions.shape != (max_step + 1, 100, 6):
        raise ValueError('recorded Actor action shape changed')
    model, _, _ = load_checkpoint(a.reference_checkpoint, cfg)
    model = model.to('cuda').eval()
    scales = json.loads(a.maneuver_thresholds.read_text())['lateral_scale_m']
    limits = json.loads(a.constraint_limits.read_text())
    if limits['source_split'] != 'complete_only_train' or limits['sample_count'] != 49598:
        raise ValueError('constraint limits changed')
    identities = []
    for j, state in enumerate(selection['selected']):
        w = view.dataset.windows[view.indices[state['view_index']]]
        dem = np.asarray(w['dem'], dtype=np.float32)
        goal = np.asarray(w['goal'], dtype=np.float32)
        if hashlib.sha256(dem.tobytes()).hexdigest() != state['dem_sha256']:
            raise ValueError('frozen DEM changed')
        if list(w['episode_key']) != state['episode_key'] or int(w['step']) != state['step']:
            raise ValueError('frozen window identity changed')
        if not np.isfinite(dem).all() or not np.isfinite(goal).all() or not w['action_mask'].all():
            raise ValueError('incomplete or nonfinite selected window')
        folder = a.output / f'state-{j:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        step = int(state['step'])
        command = np.asarray([commands[j][t] for t in range(step + 1)], dtype=np.float32)
        save_npz_once(folder / 'prefix-actions.npz', actions=all_actions[:step + 1],
                      selected_env_commands=command)
        save_npz_once(folder / 'suffix-actions-120.npz', actions=all_actions[step + 1:step + 121])
        atomic_once(folder / 'logged-anchor.json', anchors[j])
        state_cfg = {'schema_version': 2, 'control_steps': 120,
                     'anchor_limits': {'position_m': .02, 'orientation_rad': .03,
                                       'dem_rms_m': .03, 'mask_iou_min': .95},
                     'source_state': {'split': 'train', 'state_index': j,
                                      'env_index': state['episode_key'][1],
                                      'episode_id': 0, 'anchor_step': step}}
        cfg_path = folder / 'config-120.yaml'
        content = yaml.safe_dump(state_cfg, sort_keys=True)
        if cfg_path.exists() and cfg_path.read_text() != content:
            raise FileExistsError('state config changed')
        if not cfg_path.exists():
            cfg_path.write_text(content)
        with torch.inference_mode():
            sampled = sample_paths(model, torch.as_tensor(dem[None], device='cuda'),
                                   torch.as_tensor(goal[None], device='cuda'), SEEDS)
        if sampled.shape != (8, 8, 2) or not np.isfinite(sampled).all():
            raise ValueError('NoMaD path shape or finiteness changed')
        bank = maneuver_bank(sampled[0], goal, scales['slight'], scales['strong'])
        paths = {f'nomad-seed-{seed}': sampled[k] for k, seed in enumerate(SEEDS)}
        paths.update({f'bank-{name}': bank[name] for name in NAMES})
        candidate_table = []
        for name, path in paths.items():
            points = [[0., 0.]] + np.asarray(path, dtype=float).tolist()
            atomic_once(folder / f'{name}-base-path.json', points)
            candidate_table.append({'candidate_id': name, 'seed': int(name.split('-')[-1]) if name.startswith('nomad-') else None,
                                    'generator': 'nomad' if name.startswith('nomad-') else 'fixed_bank',
                                    'base_path_sha256': sha(folder / f'{name}-base-path.json'),
                                    'planned_metrics': path_metrics(path),
                                    'constraints': candidate_constraints(path, dem, resolution_m=cfg.dem_resolution_m, limits=limits)})
        atomic_once(folder / 'candidate-table.json', candidate_table)
        identities.append({'state_id': state['state_id'], 'terrain_type': state['terrain_type'],
                           'episode_key': state['episode_key'], 'view_index': state['view_index'],
                           'anchor_step': step, 'prefix_steps': step + 1,
                           'goal_distance_m': state['goal_distance_m'],
                           'prefix_sha256': sha(folder / 'prefix-actions.npz'),
                           'suffix_sha256': sha(folder / 'suffix-actions-120.npz'),
                           'logged_anchor_sha256': sha(folder / 'logged-anchor.json'),
                           'config_sha256': sha(cfg_path),
                           'candidate_table_sha256': sha(folder / 'candidate-table.json')})
    atomic_once(a.output / 'selected-states.json', {'train': selection['selected'], 'heldout_used': False})
    atomic_once(a.output / 'input-freeze.json', {'selection_sha256': sha(a.selection),
                'manifest_sha256': sha(a.manifest), 'reference_checkpoint_sha256': sha(a.reference_checkpoint),
                'maneuver_thresholds_sha256': sha(a.maneuver_thresholds),
                'constraint_limits_sha256': sha(a.constraint_limits), 'seeds': list(SEEDS),
                'bank_order': list(NAMES), 'states': identities,
                'candidate_outcomes_used': False})
    print('READINESS_INPUTS_FROZEN', sha(a.output / 'input-freeze.json'), flush=True)


if __name__ == '__main__':
    main()
