"""Read-only descriptive analysis of a completed 14-Path Stage A pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eat_nav_motion_14_stage_a_run import (
    load_frozen, row_name, verify_attempt,
)


def summarize(root: Path) -> dict:
    manifest = load_frozen()
    status = json.loads((root/'status.json').read_text())
    if status.get('state') != 'completed' or status.get('completed') != 32:
        raise ValueError('Stage A protocol incomplete')
    grouped = {}
    total = 0
    for row in manifest['assignment']:
        attempt = root/row_name(row)
        verify_attempt(attempt, row)
        if row['stage'] != 'candidate':
            continue
        output = attempt/'sim-output'
        labels = json.loads((output/'motion-labels.json').read_text())['labels']
        samples = []
        for env in range(4):
            trace = [json.loads(line) for line in
                     (output/f'trace-env-{env}.jsonl').read_text().splitlines()]
            if len(trace) != 40:
                raise ValueError('incomplete candidate trace')
            command = np.asarray([entry['follower_command'] for entry in trace], dtype=float)
            velocity = np.asarray([entry['body_velocity_xy_mps'] for entry in trace], dtype=float)
            contact = np.asarray([entry['contact_net_forces_w'] for entry in trace], dtype=float)
            torque = np.asarray([entry['applied_torque'] for entry in trace], dtype=float)
            diagnostics = [entry['follower_diagnostic'] for entry in trace]
            samples.append({'env_tile': env, 'delta_body_xy_m': labels[env]['delta_body_xy_m'],
                            'delta_body_yaw_rad': labels[env]['delta_body_yaw_rad'],
                            'mean_command_linear_mps': float(command[:, 0].mean()),
                            'mean_command_angular_rps': float(command[:, 2].mean()),
                            'mean_body_xy_speed_mps': float(np.linalg.norm(velocity, axis=1).mean()),
                            'peak_contact_force_n': float(np.linalg.norm(contact, axis=-1).max()),
                            'peak_applied_torque_nm': float(np.abs(torque).max()),
                            'goal_reached_any': any(d.get('goal_reached', False) for d in diagnostics),
                            'linear_recovery_any': any(d.get('linear_recovery', False) for d in diagnostics),
                            'angular_recovery_any': any(d.get('angular_recovery', False) for d in diagnostics),
                            'termination_count': sum(t['terminal'] is not None for t in trace)})
        key = row['flipper_initialization_mode']
        grouped.setdefault(key, {})[row['candidate_id']] = {
            'samples': samples,
            'mean_delta_body_xy_m': np.asarray([s['delta_body_xy_m'] for s in samples]).mean(0).tolist(),
            'mean_delta_body_yaw_rad': float(np.mean([s['delta_body_yaw_rad'] for s in samples])),
            'mean_command_linear_mps': float(np.mean([s['mean_command_linear_mps'] for s in samples])),
            'mean_command_angular_rps': float(np.mean([s['mean_command_angular_rps'] for s in samples])),
        }
        total += len(samples)
    return {'schema': 'eatnav_motion_14_stage_a_analysis_v1', 'decision': 'pass',
            'independent_terrain_seed_count': 1, 'process_count': 32,
            'candidate_trial_count': total, 'by_posture_and_path': grouped,
            'interpretation_limit': 'Single-terrain-seed feasibility, no terrain generalization'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    options = parser.parse_args()
    options.output.write_text(json.dumps(summarize(options.root), sort_keys=True,
                                         indent=2, allow_nan=False)+'\n')
