"""Freeze 48-process schedule and episode-zero replay inputs; no simulator startup."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from eat_nav_goal_subgoal_v2_select import OUT, CONFIG, MANIFEST, sha, atomic


def read(path):
    return json.loads(Path(path).read_text())


def npz_once(path, **arrays):
    path = Path(path)
    if path.exists():
        with np.load(path) as prior:
            if set(prior.files) != set(arrays) or any(not np.array_equal(prior[key], value) for key, value in arrays.items()):
                raise FileExistsError(f'frozen NPZ differs: {path}')
        return
    temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    temp.replace(path)


def main():
    if (OUT / 'v4.done.json').exists():
        raise FileExistsError('V4 schedule already frozen')
    cfg = yaml.safe_load(CONFIG.read_text())
    frozen = read(OUT / 'selected-states-v2.json')
    states = frozen['states']
    goals = read(OUT / 'goals-v2.json')['goals']
    paths_marker = read(OUT / 'v3-paths.done.json')
    if sha(OUT / 'paths-v2.jsonl.gz') != paths_marker['paths_sha256'] or paths_marker.get('resampling_performed') is not False:
        raise ValueError('NoMaD Path correction missing or SHA drift')
    with gzip.open(OUT / 'paths-v2.jsonl.gz', 'rt') as stream:
        paths = [json.loads(line) for line in stream]
    if len(states) != 4 or len(goals) != 16 or len(paths) != 32:
        raise ValueError('v2 fixed counts differ')
    if [s['terrain_name'] for s in states] != cfg['terrain_order'] or any(s['episode_id'] != 0 for s in states):
        raise ValueError('terrain or episode order differs')
    max_step = max(s['global_step'] for s in states) + 120
    actions = {}
    commands = [dict() for _ in states]
    anchors = {}
    manifest = read(MANIFEST)
    if sha(MANIFEST) != cfg['manifest_sha256']:
        raise ValueError('manifest SHA drift')
    for chunk in manifest['chunks']:
        file = MANIFEST.parent / chunk['file']
        if sha(file) != chunk['sha256']:
            raise ValueError('manifest chunk SHA drift')
        with gzip.open(file, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                step = int(row['step'])
                if step > max_step:
                    continue
                env = int(row['env_id'])
                actions.setdefault(step, {})[env] = row['action']
                for index, state in enumerate(states):
                    if env != state['env_id'] or step > state['global_step']:
                        continue
                    commands[index][step] = row['command']
                    if step == state['global_step']:
                        if ([row['session_id'], env, row['episode_id']] != state['episode_key']
                                or row['pre_reset']):
                            raise ValueError('selected anchor episode/pre-reset identity mismatch')
                        anchors[index] = row
        if len(anchors) == 4 and all(len(actions.get(step, {})) == 100 for step in range(max_step + 1)):
            break
    if len(anchors) != 4 or not all(len(actions.get(step, {})) == 100 for step in range(max_step + 1)):
        raise ValueError('incomplete 100-env prefix/suffix recorded actions')
    all_actions = np.asarray([[actions[step][env] for env in range(100)]
                              for step in range(max_step + 1)], dtype=np.float32)
    if all_actions.shape != (max_step + 1, 100, 6) or not np.isfinite(all_actions).all():
        raise ValueError('recorded Actor action array malformed')
    input_states = []
    input_ledger = []
    schedule = []
    for index, state in enumerate(states):
        folder = OUT / 'inputs' / f'state-{index:03d}'
        folder.mkdir(parents=True, exist_ok=True)
        step = state['global_step']
        command = np.asarray([commands[index][t] for t in range(step + 1)], dtype=np.float32)
        if command.shape != (step + 1, 3) or not np.isfinite(command).all():
            raise ValueError('recorded command prefix malformed')
        npz_once(folder / 'prefix-actions.npz', actions=all_actions[:step + 1], selected_env_commands=command)
        npz_once(folder / 'suffix-actions-120.npz', actions=all_actions[step + 1:step + 121])
        atomic(folder / 'logged-anchor.json', anchors[index])
        input_state = {**state, 'step': step, 'local_step': state['local_step'],
                       'terrain_type': state['terrain_name'], 'state_index': index}
        input_states.append(input_state)
        state_config = {'schema_version': 3, 'control_steps': 120,
                        'anchor_limits': {'position_m': .02, 'orientation_rad': .03,
                                          'dem_rms_m': .03, 'mask_iou_min': .95},
                        'source_state': {'split': 'train', 'state_index': index,
                                         'env_index': state['env_id'], 'episode_id': 0,
                                         'anchor_step': step}}
        config_path = folder / 'config-120.yaml'
        content = yaml.safe_dump(state_config, sort_keys=True)
        if config_path.exists() and config_path.read_text() != content:
            raise FileExistsError('state config drift')
        if not config_path.exists():
            config_path.write_text(content)
        candidates = []
        for row in paths[index * 8:(index + 1) * 8]:
            if row['state_id'] != state['state_id'] or not row['geometry_screen']['geometry_screen_valid']:
                raise ValueError('Path state or geometry screen mismatch')
            candidate_id = f"{row['goal_id']}-seed-{row['diffusion_seed']}"
            waypoints = np.asarray(row['cumulative_waypoints'], dtype=np.float32)
            if waypoints.shape != (8, 2) or hashlib.sha256(waypoints.tobytes()).hexdigest() != row['path_sha256']:
                raise ValueError('Path byte identity differs from frozen NoMaD sample')
            base = [[0., 0.]] + waypoints.astype(float).tolist()
            path_file = folder / f'{candidate_id}-base-path.json'
            atomic(path_file, base)
            candidates.append({'candidate_id': candidate_id, 'goal_id': row['goal_id'],
                               'goal_class': row['declared_class'], 'goal_xy': row['goal_xy'],
                               'seed': row['diffusion_seed'], 'base_path_sha256': sha(path_file),
                               'nomad_path_sha256': row['path_sha256'], 'generation_order': row['generation_order'],
                               'geometry_screen_valid': True})
        if len(candidates) != 8:
            raise ValueError('candidate count per state')
        nominal = candidates[0]
        anchor_entry = {**nominal, 'state_id': state['state_id'], 'state_index': index,
                        'terrain': state['terrain_name'], 'episode_key': state['episode_key'],
                        'anchor_step': step, 'prefix_steps': step + 1, 'logical_t0_ns': 2_000_000_000_000 + index * 100_000_000_000,
                        'kind': 'anchor', 'run_id': f"{state['state_id']}-anchor", 'order': len(schedule)}
        schedule.append(anchor_entry)
        for repeat in range(3):
            schedule.append({**anchor_entry, 'kind': 'duplicate', 'repeat': repeat,
                             'run_id': f"{state['state_id']}-nominal-{repeat:02d}", 'order': len(schedule)})
        for candidate in candidates:
            schedule.append({**anchor_entry, **candidate, 'kind': 'candidate',
                             'run_id': f"{state['state_id']}-{candidate['candidate_id']}", 'order': len(schedule)})
        for entry in schedule[-11:]:
            path_file = folder / f"{entry['candidate_id']}-base-path.json"
            metadata = {'run_id': entry['run_id'], 'state_id': state['state_id'],
                        'goal_id': entry['goal_id'], 'goal_class': entry['goal_class'],
                        'goal_xy': entry['goal_xy'], 'mission_logged_goal': state['logged_goal'],
                        'base_path': read(path_file), 'base_path_sha256': sha(path_file),
                        'diffusion_seed': entry['seed']}
            atomic(folder / f"{entry['run_id']}-goal-metadata.json", metadata)
        input_ledger.append({'state_id': state['state_id'], 'episode_key': state['episode_key'],
                             'prefix_steps': step + 1, 'prefix_sha256': sha(folder / 'prefix-actions.npz'),
                             'suffix_sha256': sha(folder / 'suffix-actions-120.npz'),
                             'anchor_sha256': sha(folder / 'logged-anchor.json'),
                             'config_sha256': sha(config_path)})
    if len(schedule) != 48 or [r['kind'] for r in schedule] != (['anchor'] + ['duplicate'] * 3 + ['candidate'] * 8) * 4:
        raise ValueError('48-process schedule shape')
    atomic(OUT / 'inputs/selected-states.json', {'train': input_states, 'heldout_used': False})
    atomic(OUT / 'run-schedule-v2.json', {'runs': schedule, 'anchor_count': 4, 'live_count': 44,
                                         'nominal_duplicate_count': 12, 'candidate_count': 32,
                                         'control_hz': 20, 'control_steps': 120,
                                         'selected_states_sha256': sha(OUT / 'selected-states-v2.json'),
                                         'paths_sha256': sha(OUT / 'paths-v2.jsonl.gz'),
                                         'input_ledger': input_ledger})
    atomic(OUT / 'v4.done.json', {'schedule_sha256': sha(OUT / 'run-schedule-v2.json'),
                                  'selected_input_sha256': sha(OUT / 'inputs/selected-states.json'),
                                  'run_count': 48, 'candidate_outcome_count': 0})
    print('V4_SCHEDULE_FROZEN', len(schedule), [s['prefix_steps'] for s in states])


if __name__ == '__main__':
    main()
