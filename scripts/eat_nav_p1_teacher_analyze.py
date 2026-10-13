"""Read-only verification and fixed-window labels for a sealed teacher batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_teacher_labels import label_window
from visualnav_transformer.eat_nav.p1_teacher_chunks import verify_chunks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(batch):
    batch = Path(batch)
    seal = verify_seal(batch)
    output = batch / 'output'
    chunks = output / 'teacher-chunks'
    chunk_status = verify_chunks(chunks)
    index = json.loads((output / 'teacher-index-mapping.json').read_text())
    publication = json.loads((output / 'path-activation.json').read_text())['trial_paths']
    terrain = json.loads((output / 'terrain.json').read_text())
    paths = [output / f'trace-slot-{env}.jsonl' for env in range(4)]
    traces = [[json.loads(line) for line in path.read_text().splitlines()] for path in paths]
    rows = [[] for _ in range(4)]
    reward_max_abs_error = 0.0
    reward_term_max_abs_error = 0.0
    substep_counts = []
    for path in sorted((chunks / 'chunks').glob('*.npz')):
        with np.load(path, allow_pickle=False) as arrays:
            for j in range(len(arrays['env_id'])):
                env = int(arrays['env_id'][j]); step = int(arrays['control_step'][j])
                if not 0 <= env < 4 or step != len(rows[env]):
                    raise ValueError('teacher_row_order')
                trace = traces[env][step]
                if int(arrays['logical_stamp_ns'][j]) != trace['logical_sim_time_ns']:
                    raise ValueError('teacher_trace_stamp')
                reward = float(arrays['reward_total'][j])
                reward_max_abs_error = max(reward_max_abs_error,
                                           abs(reward - float(trace['reward'])))
                reward_term_max_abs_error = max(reward_term_max_abs_error,
                    abs(float(arrays['reward_term_weighted'][j].sum()) - reward))
                substep_counts.append(int(arrays['substep_contact_count'][j].max()))
                rows[env].append({
                    'step': step, 'stamp_ns': step * 50_000_000,
                    'root_pre': arrays['root_pre'][j].tolist(),
                    'root_post': arrays['root_post'][j].tolist(),
                    'command': arrays['follower_command'][j].tolist(),
                    'reward': reward,
                    'goal_reached_pre': bool(arrays['goal_reached_pre'][j]),
                    'torque_Nm': arrays['applied_torque_pre'][j].tolist(),
                    'body_force_world_N': arrays['body_force_pre_world'][j].tolist(),
                    'terminated': bool(trace['terminal'] and trace['terminal'].get('terminated')),
                    'censored': bool(trace['terminal'] and trace['terminal'].get('truncated')),
                })
    if [len(group) for group in rows] != [len(trace) for trace in traces] or \
            sum(map(len, rows)) != chunk_status['rows']:
        raise ValueError('teacher_trace_row_count')
    if reward_max_abs_error > 1e-5 or reward_term_max_abs_error > 1e-5:
        raise ValueError('teacher_reward_mismatch')
    allowables = {'schema': 'eatnav_pain_allowables_v1',
                  'joint_torque_ref_Nm': {}, 'body_force_ref_N': {}}
    labels = []
    for env in range(4):
        label = label_window(rows[env],
            common_goal_xy=publication[env]['common_goal_map_xy'],
            env_origin_xy=terrain['world_tile_origins'][env][:2],
            allowables=allowables,
            joint_names=[item['name'] for item in index['joints']],
            body_names=[item['name'] for item in index['bodies']])
        labels.append({'env_id': env, 'path_role': publication[env]['path_role'],
                       'path_sha256': publication[env]['path_sha256'], **label})
    return {'schema': 'eatnav_p1_teacher_offline_analysis_v1',
            'batch': str(batch), 'host_seal_sha256': sha(batch / 'host-seal.json'),
            'host_seal': seal, 'chunk_verification': chunk_status,
            'reward_trace_max_abs_error': reward_max_abs_error,
            'reward_term_max_abs_error': reward_term_max_abs_error,
            'max_contact_substeps_with_nonzero_force': max(substep_counts),
            'labels': labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists() or args.out.is_relative_to(args.batch):
        raise ValueError('analysis_output_must_be_new_and_outside_seal')
    result = analyze(args.batch)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(result, stream, sort_keys=True, indent=2)
        stream.write('\n')
    print(json.dumps({'out': str(args.out), 'rows': result['chunk_verification']['rows'],
                      'labels': len(result['labels'])}, sort_keys=True))


if __name__ == '__main__':
    main()
