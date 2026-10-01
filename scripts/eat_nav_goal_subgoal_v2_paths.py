"""Freeze the old four goal slots and sample 32 NoMaD paths for v2 states."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import yaml

from eat_nav_goal_ood_prepare import classify
from eat_nav_goal_ood_freeze import geometry_screen
from eat_nav_goal_subgoal_v2_select import REPO, DOC, OUT, CONFIG, sha, atomic

SOURCE_DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood'
SOURCE_RAW = Path('/tmp/eatnav-candidate-coverage/goal-ood')
CHECKPOINT = Path('/tmp/eatnav-quality-identifiability-gate/phase-c-remediation/reference/best.pt')
GOALS = OUT / 'goals-v2.json'
PATHS = OUT / 'paths-v2.jsonl.gz'
REMOTE_INFER = '/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2/v2-infer.py'


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
    cfg, _, _, _ = frozen()
    goal_data = read(GOALS)
    if goal_data['selected_states_sha256'] != sha(OUT / 'selected-states-v2.json') or PATHS.exists():
        raise ValueError('goal freeze drift or v2 paths already exist')
    if cfg['diffusion_seeds'] != [1001, 1002] or sha(GOALS) != read(OUT / 'v3-goals.done.json')['goals_sha256']:
        raise ValueError('frozen seeds or goals SHA drift')
    remote = str(OUT)
    subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', remote], check=True)
    for source, name in [(GOALS, 'goals-v2.json'), (OUT / 'selected-states-v2.npz', 'selected-states-v2.npz'),
                         (REPO / 'scripts/eat_nav_goal_subgoal_v2_infer.py', 'v2-infer.py')]:
        subprocess.run(['docker', 'cp', str(source), f'isaac-lab-onix:{remote}/{name}'], check=True)
        digest = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', f'{remote}/{name}'], text=True).split()[0]
        if digest != sha(source):
            raise ValueError('Docker input copy SHA mismatch: ' + name)
    command = ('cd /tmp/eatnav-train && PYTHONPATH=/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src '
               f'/isaac-sim/python.sh {REMOTE_INFER}')
    with (OUT / 'v3-inference.log').open('w') as log:
        subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'bash', '-lc', command],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    for name in ('paths-v2.jsonl.gz', 'v3-paths.done.json'):
        target = OUT / name
        if target.exists():
            raise FileExistsError(target)
        temp = target.with_name(target.name + '.copytmp')
        subprocess.run(['docker', 'cp', f'isaac-lab-onix:{remote}/{name}', str(temp)], check=True)
        temp.replace(target)
    marker = read(OUT / 'v3-paths.done.json')
    if marker['paths_sha256'] != sha(PATHS) or marker['goals_sha256'] != sha(GOALS) or marker['path_count'] != 32:
        raise ValueError('Docker sampled path SHA/count drift')
    print('V3_PATHS_FROZEN', marker['path_count'], marker['paths_sha256'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('goals', 'paths'), required=True)
    args = parser.parse_args()
    (build_goals if args.stage == 'goals' else build_paths)()
