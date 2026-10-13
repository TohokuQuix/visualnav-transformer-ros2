"""Read-only descriptive analysis of the sealed Stage B Arc feasibility sweep."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from eat_nav_motion_arc_stage_b import name, verify_existing


def yaw(root):
    w, x, y, z = root[3:7]
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def analyze(parent: Path, angles: list[int]) -> dict:
    rows = []
    for angle in angles:
        root = parent/name(angle)
        seal = verify_existing(root, angle)
        output = root/'sim-output'
        manifest = json.loads((output/'input-manifest.json').read_text())
        gate = json.loads((output/'prepath-pair-gate.json').read_text())
        labels = json.loads((output/'motion-labels.json').read_text())['labels']
        traces = [[json.loads(line) for line in (output/f'trace-env-{env}.jsonl').read_text().splitlines()]
                  for env in range(4)]
        motion = np.asarray([label['delta_body_xy_m'] for label in labels], dtype=float)
        if not gate['pass'] or motion.shape != (4, 2) or not np.isfinite(motion).all():
            raise ValueError('Stage B label/prepath gate incomplete')
        initial = [trace[0]['follower_command'] for trace in traces]
        if any(len(trace) != 40 for trace in traces):
            raise ValueError('Stage B trace length')
        rows.append({'angle_deg': angle, 'stage': 1 if abs(angle) <= 90 else 2,
                     'geometry': manifest['arc_geometry'],
                     'motion_body_xy_m': motion.tolist(),
                     'motion_mean_xy_m': motion.mean(axis=0).tolist(),
                     'motion_mean_magnitude_m': float(np.linalg.norm(motion, axis=1).mean()),
                     'initial_command': initial,
                     'goal_reached_by_step_39': [any(row['follower_diagnostic']['goal_reached']
                         for row in trace) for trace in traces],
                     'turn_in_place_steps': [sum(row['follower_diagnostic']['turn_in_place']
                         for row in trace) for trace in traces],
                     'commanded_linear_integral_m': [sum(row['follower_command'][0] * .05
                         for row in trace) for trace in traces],
                     'commanded_yaw_integral_rad': [sum(row['follower_command'][2] * .05
                         for row in trace) for trace in traces],
                     'observed_yaw_delta_rad': [wrap(yaw(trace[-1]['post_root_state']) -
                         yaw(trace[0]['root_state'])) for trace in traces],
                     'linear_recovery_steps': [sum(row['follower_diagnostic']['linear_recovery']
                         for row in trace) for trace in traces],
                     'max_contact_force_n': [float(np.max(np.linalg.norm(
                         np.asarray([row['contact_net_forces_w'] for row in trace]), axis=-1)))
                         for trace in traces],
                     'prepath_max_abs': {field: result['max_abs'] for field, result in
                         gate.get('fields', {}).items() if 'max_abs' in result},
                     **seal})
    direct = np.asarray(rows[0]['motion_body_xy_m'])
    for row in rows:
        motion = np.asarray(row['motion_body_xy_m'])
        row['delta_from_direct_xy_m'] = (motion - direct).tolist()
        row['mean_delta_from_direct_m'] = float(np.linalg.norm(motion - direct, axis=1).mean())
    return {'schema': 'eatnav_motion_arc_stage_b_analysis_v1',
            'independent_terrain_seed_count': 1,
            'candidate_count': len(rows), 'trial_count': 4 * len(rows),
            'runtime_feasibility_scope': 'one_seed_random_rough_level_2_only',
            'rows': rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads((args.parent/'stage-b-control/frozen.json').read_text())
    result = analyze(args.parent, frozen['ordered_angles_deg'])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
