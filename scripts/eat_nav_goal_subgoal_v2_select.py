"""Outcome-blind, reproducible cost-aware state selection for the v2 pilot."""
from __future__ import annotations

from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

from visualnav_transformer.eat_nav.config import load_eat_config
from visualnav_transformer.eat_nav.dataset import EATDataset
from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView
from visualnav_transformer.eat_nav.quality_followup import is_complete_window

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator'
OUT = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator-v2')
SOURCE = Path('/tmp/eatnav-candidate-coverage/goal-ood')
REF = REPO / 'docs/quality-identifiability-gate/phase-c'
MANIFEST = SOURCE / 'input-copy/manifest.json'
CONFIG = DOC / 'v2-selection-config.yaml'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != content:
            raise FileExistsError(f'frozen result differs: {path}')
        return
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(content)
    temp.replace(path)


def frozen_config():
    cfg = yaml.safe_load(CONFIG.read_text())
    if cfg['schema_version'] != 1 or cfg['source_split'] != 'train' or cfg['selection_salt'] != 'goal-subgoal-cost-aware-v2-20261001':
        raise ValueError('v2 selection config drift')
    if cfg['terrain_order'] != ['k_rail_step_field', 'pyramid_stairs', 'pyramid_stairs_inv', 'boxes']:
        raise ValueError('terrain identity/order drift')
    if sha(MANIFEST) != cfg['manifest_sha256'] or sha(REF / 'complete-window-ids.jsonl.gz') != cfg['complete_window_ids_sha256']:
        raise ValueError('manifest or complete window identity SHA drift')
    if sha(REF / 'reference-eat.yaml') != 'caa88735e765fc0358930dfa41579ecaac9eb96dc4052dc01a81390377279992':
        raise ValueError('reference config SHA drift')
    return cfg


def identity(window):
    key = tuple(int(x) for x in window['episode_key'])
    return {'terrain_seed': int(window['terrain_seed']), 'env_id': key[1],
            'episode_id': key[2], 'global_step': int(window['step']),
            'local_step': int(window['local_step']), 'terrain_name': str(window['terrain_type'])}


def rank(window, config):
    fields = identity(window)
    values = [fields[name] for name in config['selection_rank_fields']]
    encoded = config['selection_salt'] + '\n' + json.dumps(values, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def exclusion_reasons(window, config):
    fields = identity(window)
    reasons = []
    if fields['terrain_name'] not in config['terrain_order']:
        reasons.append('terrain_not_in_frozen_set')
    if fields['episode_id'] != config['episode_number']:
        reasons.append('episode_not_zero')
    if fields['global_step'] != fields['local_step']:
        reasons.append('global_local_step_differ')
    if fields['global_step'] + 1 > config['max_global_prefix_steps']:
        reasons.append('prefix_exceeds_80')
    if not is_complete_window(window, 8) or np.asarray(window['dem']).shape[0] != 4:
        reasons.append('incomplete_dem_or_teacher')
    if float(np.linalg.norm(window['goal'])) < config['min_goal_distance_m']:
        reasons.append('goal_too_close')
    return reasons


def select(windows, config):
    by_terrain = defaultdict(list)
    candidate_rows = []
    for window in windows:
        fields = identity(window)
        if fields['terrain_name'] not in config['terrain_order'] or fields['episode_id'] != 0:
            continue
        reasons = exclusion_reasons(window, config)
        row = {**fields, 'episode_key': list(map(int, window['episode_key'])),
               'prefix_steps': fields['global_step'] + 1,
               'goal_distance_m': float(np.linalg.norm(window['goal'])),
               'window_identity': [list(map(int, window['episode_key'])), fields['global_step']],
               'selection_rank': rank(window, config), 'exclusion_reasons': reasons}
        candidate_rows.append(row)
        if not reasons:
            by_terrain[fields['terrain_name']].append((row, window))
    selected = []
    used_episodes = set()
    for terrain in config['terrain_order']:
        for row, window in sorted(by_terrain[terrain], key=lambda pair: (pair[0]['selection_rank'], pair[0]['window_identity'])):
            key = tuple(row['episode_key'])
            if key not in used_episodes:
                selected.append((row, window))
                used_episodes.add(key)
                break
    return selected, candidate_rows


def main():
    config = frozen_config()
    if not (OUT / 'v0.done.json').exists():
        raise ValueError('V0 audit marker missing')
    if (OUT / 'selected-states-v2.json').exists():
        raise FileExistsError('v2 state selection already frozen')
    with gzip.open(REF / 'complete-window-ids.jsonl.gz', 'rt') as stream:
        frozen = {(tuple(row['episode_key']), int(row['step'])) for row in map(json.loads, stream)
                  if row['split'] == 'train'}
    model_cfg = load_eat_config(REF / 'reference-eat.yaml')
    dataset = EATDataset(MANIFEST, 'train', model_cfg, split_seed=123)
    view = CompleteOnlyView(dataset, 8)
    split_hash = hashlib.sha256(json.dumps(dataset.split_map, sort_keys=True).encode()).hexdigest()
    if split_hash != config['split_hash'] or len(view) != 49598:
        raise ValueError('split or complete train count drift')
    windows = [dataset.windows[i] for i in view.indices]
    actual = {(tuple(w['episode_key']), int(w['step'])) for w in windows}
    if actual != frozen or len(actual) != len(windows):
        raise ValueError('complete-only train identity mismatch')
    selected, rows = select(windows, config)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'v2-selection-candidates.jsonl.gz'
    with path.with_suffix('.tmp').open('wb') as raw:
        with gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as stream:
            for row in rows:
                stream.write((json.dumps(row, sort_keys=True, allow_nan=False) + '\n').encode())
    path.with_suffix('.tmp').replace(path)
    counts = Counter(r['terrain_name'] for r in rows if not r['exclusion_reasons'])
    excluded = {terrain: dict(Counter(reason for row in rows if row['terrain_name'] == terrain
                                      for reason in row['exclusion_reasons'])) for terrain in config['terrain_order']}
    audit = {'status': 'pass' if len(selected) == 4 else 'v2_selection_unavailable',
             'config_sha256': sha(CONFIG), 'complete_train_windows': len(windows),
             'candidate_count_by_terrain': {terrain: counts[terrain] for terrain in config['terrain_order']},
             'exclusion_counts_by_terrain': excluded,
             'candidate_list_sha256': sha(path), 'selected_count': len(selected),
             'terrain_order': config['terrain_order'],
             'prefix_quantiles_by_terrain': {}, 'goal_distance_quantiles_by_terrain': {}}
    for terrain in config['terrain_order']:
        terrain_rows = [r for r in rows if r['terrain_name'] == terrain]
        for name, field in [('prefix_quantiles_by_terrain', 'prefix_steps'),
                            ('goal_distance_quantiles_by_terrain', 'goal_distance_m')]:
            values = [r[field] for r in terrain_rows]
            audit[name][terrain] = np.quantile(values, [0, .25, .5, .75, 1]).tolist() if values else None
    atomic(OUT / 'v2-selection-audit.json', audit)
    if len(selected) != 4:
        atomic(OUT / 'status.json', {'stage': 'v2_selection_unavailable',
                                     'selection_audit_sha256': sha(OUT / 'v2-selection-audit.json')})
        print(json.dumps(audit, indent=2))
        return 2
    output = []
    for index, (row, window) in enumerate(selected):
        dem = np.asarray(window['dem'], dtype=np.float32)
        action = np.asarray(window['action'], dtype=np.float32)
        goal = np.asarray(window['goal'], dtype=np.float32)
        output.append({**row, 'state_id': f'v2-train-{index:03d}', 'state_index': index,
                       'split': 'train', 'dem_sha256': hashlib.sha256(dem.tobytes()).hexdigest(),
                       'action_sha256': hashlib.sha256(action.tobytes()).hexdigest(),
                       'logged_goal': goal.tolist(), 'manifest_sha256': sha(MANIFEST)})
    atomic(OUT / 'selected-states-v2.json', {'states': output, 'selection_config_sha256': sha(CONFIG),
                                              'candidate_list_sha256': sha(path), 'heldout_used': False,
                                              'outcomes_used': False})
    np.savez_compressed(OUT / 'selected-states-v2.npz', dem=np.stack([w['dem'] for _, w in selected]),
                        logged_goals=np.stack([w['goal'] for _, w in selected]),
                        actions=np.stack([w['action'] for _, w in selected]))
    atomic(OUT / 'status.json', {'stage': 'V2_STATES_FROZEN', 'selected_states_sha256': sha(OUT / 'selected-states-v2.json'),
                                 'candidate_runs': 0})
    print('V2_STATES_FROZEN', [(r['state_id'], r['terrain_name'], r['prefix_steps']) for r in output])
    return 0


if __name__ == '__main__':
    sys.exit(main())
