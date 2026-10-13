#!/usr/bin/env python3
"""Compare A/B/C read-only sealed residual diagnostics on one logical clock."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean

from visualnav_transformer.eat_nav.demo_residual import compare_step_fields, summarize_cancel


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(root, mode):
    root = Path(root)
    output = root/'sim-output'
    seal = json.loads((output/'host-seal.json').read_text())
    ledger = json.loads((output/'sha-ledger.json').read_text())
    if sha(output/'sha-ledger.json') != seal['ledger_sha256'] or any(
            sha(output/name) != digest for name, digest in ledger.items()):
        raise ValueError(f'{mode}: inner seal mismatch')
    outer = json.loads((root/'trial-host-ledger.json').read_text())
    outer_seal = json.loads((root/'trial-host-seal.json').read_text())
    if sha(root/'trial-host-ledger.json') != outer_seal['ledger_sha256'] or any(
            sha(root/name) != digest for name, digest in outer.items()):
        raise ValueError(f'{mode}: outer seal mismatch')
    result = json.loads((root/'trial-controller-result.json').read_text())
    if result['mode'] != mode or result['status'] != 'protocol_valid' or \
            sha(output/'trace.jsonl') != result['trace_sha256']:
        raise ValueError(f'{mode}: protocol/trace provenance')
    records = [json.loads(line) for line in (output/'trace.jsonl').open()]
    steps = [row for row in records if row['event'] == 'step']
    if len(steps) != 280 or any(row['step'] != i or row['stamp_ns'] != i*50_000_000
                                for i, row in enumerate(steps)):
        raise ValueError(f'{mode}: logical trace incomplete')
    return {'root': str(root), 'records': records, 'steps': steps,
            'input': json.loads((output/'input-ledger.json').read_text()),
            'tile': json.loads((output/'terrain-provenance.json').read_text())['selected_tiles'][0]['tile'],
            'inner_seal_sha256': sha(output/'host-seal.json'),
            'outer_seal_sha256': sha(root/'trial-host-seal.json')}


def metrics(rows):
    names = rows[0]['physical_post']['joint_names']
    sprockets = [i for i, name in enumerate(names) if 'sprocket' in name]
    flippers = [i for i, name in enumerate(names) if 'flipper_joint' in name]
    contact = [row['physical_post']['contact_net_forces_w'] for row in rows]
    return {
        'path_length_xy_m': sum(math.dist(row['pose_pre'][:2], row['pose_post'][:2]) for row in rows),
        'net_xy_m': math.dist(rows[0]['pose_pre'][:2], rows[-1]['pose_post'][:2]),
        'yaw_change_rad': rows[-1]['pose_post'][2] - rows[0]['pose_pre'][2],
        'root_xy_speed_mean_mps': mean(math.hypot(*row['physical_post']['root_lin_vel_w'][:2]) for row in rows),
        'root_xy_speed_final_mps': math.hypot(*rows[-1]['physical_post']['root_lin_vel_w'][:2]),
        'nonzero_follower_command_steps': sum(any(abs(v)>1e-6 for v in row['command']) for row in rows),
        'nonzero_actor_command_steps': sum(any(abs(v)>1e-6 for v in row['actor_input_command']) for row in rows),
        'actor_action_mean_abs': [mean(abs(row['actor_action'][j]) for row in rows) for j in range(6)],
        'sprocket_joint_velocity_mean_abs_radps': mean(abs(row['physical_post']['joint_vel'][j])
                                                       for row in rows for j in sprockets),
        'sprocket_applied_torque_mean_abs_Nm': mean(abs(row['physical_post']['applied_torque'][j])
                                                   for row in rows for j in sprockets),
        'flipper_joint_velocity_mean_abs_radps': mean(abs(row['physical_post']['joint_vel'][j])
                                                      for row in rows for j in flippers),
        'sum_body_contact_net_force_norm_mean_N': mean(sum(math.sqrt(sum(v*v for v in body)) for body in step)
                                                   for step in contact),
        'done_steps': sum(bool(row['done']) for row in rows),
    }


def main():
    parser = argparse.ArgumentParser()
    for mode in 'ABC':
        parser.add_argument('--root-'+mode.lower(), type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {mode: load(getattr(args, 'root_'+mode.lower()), mode) for mode in 'ABC'}
    identity = ('checkpoint_sha256', 'rl_checkpoint_sha256', 'bank_sha256',
                'eat_config_sha256', 'seed', 'max_steps', 'replan_steps',
                'num_envs', 'terrain_type', 'terrain_level', 'source_sha256')
    for field in identity:
        if len({json.dumps(runs[mode]['input'][field], sort_keys=True) for mode in 'ABC'}) != 1:
            raise ValueError(f'input mismatch: {field}')
    if len({json.dumps(runs[mode]['tile'], sort_keys=True) for mode in 'ABC'}) != 1:
        raise ValueError('terrain tile mismatch')
    b_cancel = next(row for row in runs['B']['records'] if row['event'] == 'cancel')['step']
    fields = ('pose_pre', 'pose_post', 'command', 'actor_input_command', 'actor_action',
              'physical_pre', 'physical_post', 'reward', 'done', 'dem_sensor_update_ns')
    exact = compare_step_fields(runs['A']['steps'], runs['B']['steps'], fields)
    result = {'schema': 'eatnav_cancel_residual_abc_comparison_v1',
              'analysis_source_sha256': sha(__file__),
              'independent_process_count': 3, 'repeats_per_condition': 1,
              'cancel_step_B': b_cancel, 'paired_AB_exact_fields': exact,
              'input_sha256': {field: runs['A']['input'][field] for field in identity},
              'terrain_tile': runs['A']['tile'],
              'runs': {mode: {'root': data['root'],
                              'inner_seal_sha256': data['inner_seal_sha256'],
                              'outer_seal_sha256': data['outer_seal_sha256'],
                              'event_steps': [(r['event'], r['step']) for r in data['records'] if r['event'] != 'step'],
                              'all_280_steps': metrics(data['steps']),
                              'steps_70_to_190': metrics(data['steps'][70:191]),
                              'steps_191_to_279': metrics(data['steps'][191:]),
                              'steps_240_to_279': metrics(data['steps'][240:])}
                       for mode, data in runs.items()},
              'B_cancel_summary': summarize_cancel(runs['B']['records'])}
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    report = args.output/'comparison.json'
    with report.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    with (args.output/'sha-ledger.json').open('x') as stream:
        json.dump({'comparison.json': sha(report)}, stream, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    with (args.output/'host-seal.json').open('x') as stream:
        json.dump({'ledger_sha256': sha(args.output/'sha-ledger.json'), 'file_count': 1}, stream, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    print(json.dumps({'output': str(args.output), 'AB_exact': all(v['equal'] for v in exact.values()),
                      'B_cancel_step': b_cancel, 'C_zero_command_distance_m':
                      result['runs']['C']['all_280_steps']['net_xy_m']}, sort_keys=True))


if __name__ == '__main__':
    main()
