"""Freeze the old four goal slots and sample 32 NoMaD paths for v2 states."""
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
from eat_nav_goal_ood_prepare import classify
from eat_nav_goal_ood_freeze import geometry_screen
from eat_nav_goal_subgoal_v2_select import REPO, DOC, OUT, CONFIG, sha, atomic

SOURCE_DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood'
SOURCE_RAW = Path('/tmp/eatnav-candidate-coverage/goal-ood')
CHECKPOINT = Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')
GOALS = OUT / 'goals-v2.json'
PATHS = OUT / 'paths-v2.jsonl.gz'


def read(path):
    return json.loads(Path(path).read_text())


def frozen():
    cfg = yaml.safe_load(CONFIG.read_text())
    selected = read(OUT / 'selected-states-v2.json')
    if read(OUT / 'status.json')['stage'] != 'V2_STATES_FROZEN' or len(selected['states']) != 4:
        raise ValueError('v2 states not frozen')
    if selected['selection_config_sha256'] != sha(CONFIG) or sha(CHECKPOINT) != cfg['reference_checkpoint_sha256']:
        raise ValueError('selection config or checkpoint drift')
    if sha(SOURCE_RAW / 'train-derived.npz') != read(SOURCE_RAW / 'g1.done.json')['train_derived_sha256']:
        raise ValueError('train-only goal distribution drift')
    if sha(SOURCE_DOC / 'goal-support-thresholds.json') != read(SOURCE_RAW / 'g1.done.json')['support_sha256']:
        raise ValueError('train-only support thresholds drift')
    prior = yaml.safe_load((SOURCE_DOC / 'goal-ood-config.yaml').read_text())
    slots = {slot['id']: slot for slot in prior['goal_slots']}
    if list(cfg['goal_ids']) != ['logged', 'id_forward_long', 'near_left_lateral', 'far_diag_right']:
        raise ValueError('goal slot order drift')
    if cfg['diffusion_seeds'] != [1001, 1002]:
        raise ValueError('diffusion seed drift')
    return cfg, selected['states'], prior, slots


def build_goals():
    cfg, states, prior, slots = frozen()
    if GOALS.exists():
        raise FileExistsError(GOALS)
    train_goals = np.load(SOURCE_RAW / 'train-derived.npz')['goals']
    support = read(SOURCE_DOC / 'goal-support-thresholds.json')
    arrays = np.load(OUT / 'selected-states-v2.npz')
    rows = []
    for index, state in enumerate(states):
        dem = arrays['dem'][index]
        if hashlib.sha256(dem.astype(np.float32).tobytes()).hexdigest() != state['dem_sha256']:
            raise ValueError('selected DEM drift')
        for goal_index, goal_id in enumerate(cfg['goal_ids']):
            if goal_id == 'logged':
                goal = np.asarray(state['logged_goal'], dtype=np.float32)
                declared = 'logged'
            else:
                slot = slots[goal_id]
                a = np.deg2rad(slot['angle_deg'])
                goal = np.asarray([slot['radius_m'] * np.cos(a), slot['radius_m'] * np.sin(a)], dtype=np.float32)
                declared = slot['class']
            support_bin, distances = classify([goal], train_goals, support)
            if goal_id != 'logged' and support_bin[0] != declared:
                raise ValueError(f'frozen support class drift for {goal_id}')
            screen = geometry_screen(goal, dem, prior)
            radius = float(np.linalg.norm(goal))
            angle = float(np.arctan2(goal[1], goal[0]))
            rows.append({'state_id': state['state_id'], 'state_index': index,
                         'terrain': state['terrain_name'], 'episode_key': state['episode_key'],
                         'goal_order': goal_index, 'goal_id': goal_id, 'declared_class': declared,
                         'actual_train_support_class': support_bin[0], 'goal_xy': goal.astype(float).tolist(),
                         'goal_radius_m': radius, 'goal_angle_rad': angle,
                         'train_knn_distance_m': float(distances[0]), 'geometry_screen': screen,
                         'dem_sha256': state['dem_sha256'],
                         'input_sha256': hashlib.sha256(dem.astype(np.float32).tobytes() + goal.tobytes()).hexdigest()})
    if len(rows) != 16:
        raise ValueError('goal count')
    atomic(GOALS, {'goals': rows, 'selected_states_sha256': sha(OUT / 'selected-states-v2.json'),
                   'train_support_sha256': sha(SOURCE_DOC / 'goal-support-thresholds.json'),
                   'goal_ood_config_sha256': sha(SOURCE_DOC / 'goal-ood-config.yaml'),
                   'heldout_used': False, 'validation_used': False})
    atomic(OUT / 'v3-goals.done.json', {'goals_sha256': sha(GOALS), 'goal_count': 16})
    print('V3_GOALS_FROZEN', len(rows), 'screen_valid', sum(r['geometry_screen']['geometry_screen_valid'] for r in rows))


def build_paths():
    cfg, states, _, _ = frozen()
    goal_data = read(GOALS)
    if goal_data['selected_states_sha256'] != sha(OUT / 'selected-states-v2.json') or PATHS.exists():
        raise ValueError('goal freeze drift or v2 paths already exist')
    model_cfg = load_eat_config(REPO / 'docs/quality-identifiability-gate/phase-c/reference-eat.yaml')
    model, stats, _ = load_checkpoint(CHECKPOINT, model_cfg)
    if not np.allclose(stats.mean, [.028116857632994652, -4.290757715352811e-05], atol=1e-8):
        raise ValueError('train-only action normalization drift')
    model = model.to('cuda').eval()
    arrays = np.load(OUT / 'selected-states-v2.npz')
    rows = []
    with torch.inference_mode():
        for item in goal_data['goals']:
            dem = np.asarray(arrays['dem'][item['state_index']], dtype=np.float32)
            goal = np.asarray(item['goal_xy'], dtype=np.float32)
            if hashlib.sha256(dem.tobytes() + goal.tobytes()).hexdigest() != item['input_sha256']:
                raise ValueError('NoMaD input SHA drift')
            d = torch.as_tensor(dem[None], device='cuda')
            g = torch.as_tensor(goal[None], device='cuda')
            for seed in cfg['diffusion_seeds']:
                increments = model.sample(d, g, None, seed=seed)[0].detach().cpu().numpy()
                if increments.shape != (8, 2) or not np.isfinite(increments).all():
                    raise ValueError('nonfinite or malformed sampled Path')
                waypoints = np.cumsum(increments, axis=0)
                endpoint_error = float(np.linalg.norm(waypoints[-1] - goal))
                path_sha = hashlib.sha256(increments.astype(np.float32).tobytes()).hexdigest()
                rows.append({**item, 'diffusion_seed': seed, 'generation_order': len(rows),
                             'path_increments': increments.astype(float).tolist(),
                             'cumulative_waypoints': waypoints.astype(float).tolist(),
                             'endpoint_error_m': endpoint_error, 'path_sha256': path_sha,
                             'model_checkpoint_sha256': sha(CHECKPOINT)})
    if len(rows) != 32:
        raise ValueError('Path count')
    temp = PATHS.with_name(PATHS.name + '.tmp')
    with temp.open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())
    temp.replace(PATHS)
    atomic(OUT / 'v3-paths.done.json', {'paths_sha256': sha(PATHS), 'path_count': 32,
                                        'goals_sha256': sha(GOALS), 'checkpoint_sha256': sha(CHECKPOINT)})
    print('V3_PATHS_FROZEN', len(rows), sha(PATHS))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('goals', 'paths'), required=True)
    args = parser.parse_args()
    (build_goals if args.stage == 'goals' else build_paths)()
