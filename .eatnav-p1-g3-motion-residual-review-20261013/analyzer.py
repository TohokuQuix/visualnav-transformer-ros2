"""Read-only motion residual review of the three sealed ROS-free G3 periods."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from eat_nav_p1_g3_rosfree_crossover_complete_analyze import analyze as verify_crossover


DEFAULT_ROOTS = [
    Path(f"/home/kojima/lib/onix_isaac_lab/.eatnav-p1-g3-rosfree-crossover-v1-attempt-{p + 1}")
    for p in range(3)
]
DT = 0.05


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def yaw(state):
    w, x, y, z = map(float, state[3:7])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def wrap(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


def project_s(point, path):
    best = (float('inf'), 0.0)
    run = 0.0
    for a, b in zip(path, path[1:]):
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        fraction = max(0.0, min(1.0, ((point[0] - a[0]) * dx +
                                       (point[1] - a[1]) * dy) / (length * length)))
        near = (a[0] + fraction * dx, a[1] + fraction * dy)
        candidate = (math.dist(point, near), run + fraction * length)
        if candidate[0] < best[0]:
            best = candidate
        run += length
    return best[1]


def row_metrics(row, origin, path, goal):
    pre, post = row['root_state'], row['post_root_state']
    pre_yaw, post_yaw = yaw(pre), yaw(post)
    dx, dy = float(post[0] - pre[0]), float(post[1] - pre[1])
    forward_delta = dx * math.cos(pre_yaw) + dy * math.sin(pre_yaw)
    v_cmd, omega_cmd = float(row['follower_command'][0]), float(row['follower_command'][2])
    v_actual, omega_actual = forward_delta / DT, wrap(post_yaw - pre_yaw) / DT
    loc_pre = (pre[0] - origin[0], pre[1] - origin[1])
    loc_post = (post[0] - origin[0], post[1] - origin[1])
    heading = float(row['follower_diagnostic']['heading_error'])
    cap = 0.3 * max(0.0, 1.0 - abs(heading) / math.pi) if abs(heading) < 0.6 and not row['follower_diagnostic']['goal_reached'] else 0.0
    force = statistics.mean(math.dist(vec, (0, 0, 0)) for vec in row['contact_net_forces_w'])
    torque = statistics.mean(abs(value) for value in row['applied_torque'])
    result = {
        'step': row['control_step'], 'stamp_ns': row['logical_sim_time_ns'],
        'v_command_mps': v_cmd, 'v_actual_interval_mps': v_actual,
        'v_residual_signed_mps': v_cmd - v_actual,
        'yaw_command_radps': omega_cmd, 'yaw_actual_interval_radps': omega_actual,
        'yaw_residual_signed_radps': omega_cmd - omega_actual,
        'heading_abs_rad': abs(heading), 'nominal_forward_cap_mps': cap,
        'zero_forward_cap': cap == 0.0,
        'commanded_forward_delta_m': v_cmd * DT, 'actual_forward_delta_m': forward_delta,
        'commanded_yaw_delta_rad': omega_cmd * DT,
        'actual_yaw_delta_rad': wrap(post_yaw - pre_yaw),
        'path_progress_delta_m': project_s(loc_post, path) - project_s(loc_pre, path),
        'common_goal_progress_delta_m': math.dist(loc_pre, goal) - math.dist(loc_post, goal),
        'contact_pre_step_mean_body_force_N': force,
        'torque_pre_step_mean_abs_Nm': torque,
        'reward_post_step': float(row['reward']),
        'critic_value_pre_step': float(row['critic_value']),
        'goal_reached_pre_step': bool(row['follower_diagnostic']['goal_reached']),
    }
    if not all(math.isfinite(x) for x in result.values() if isinstance(x, float)):
        raise ValueError('nonfinite motion metric')
    return result


def window_metrics(rows):
    if not rows:
        raise ValueError('empty window')
    result = {'steps': len(rows), 'duration_s': len(rows) * DT}
    for stem in ('v', 'yaw'):
        vals = [r[f'{stem}_residual_signed_mps' if stem == 'v' else 'yaw_residual_signed_radps'] for r in rows]
        result[f'{stem}_residual_signed_mean'] = statistics.mean(vals)
        result[f'{stem}_residual_abs_mean'] = statistics.mean(map(abs, vals))
        result[f'{stem}_residual_rmse'] = math.sqrt(statistics.mean(v * v for v in vals))
    for key in ('heading_abs_rad', 'nominal_forward_cap_mps', 'v_command_mps',
                'v_actual_interval_mps', 'yaw_command_radps', 'yaw_actual_interval_radps',
                'contact_pre_step_mean_body_force_N', 'torque_pre_step_mean_abs_Nm'):
        result[key + '_mean'] = statistics.mean(row[key] for row in rows)
    for key in ('commanded_forward_delta_m', 'actual_forward_delta_m',
                'commanded_yaw_delta_rad', 'actual_yaw_delta_rad', 'path_progress_delta_m',
                'common_goal_progress_delta_m', 'reward_post_step'):
        result[key + '_sum'] = sum(row[key] for row in rows)
    result['zero_forward_cap_steps'] = sum(row['zero_forward_cap'] for row in rows)
    return result


def analyze(roots=DEFAULT_ROOTS):
    verified = verify_crossover(roots)  # validates the three original seals and frozen provenance
    trials = []
    for period, root in enumerate(roots):
        output = root / f'batch-first-differential-0-period-{period}' / 'output'
        publication = json.loads((output / 'path-activation.json').read_text())['trial_paths']
        summary = json.loads((output / 'pilot-summary.json').read_text())['arrivals']
        terrain = json.loads((output / 'terrain.json').read_text())
        for env in range(4):
            trace_path = output / f'trace-slot-{env}.jsonl'
            raw = [json.loads(line) for line in trace_path.read_text().splitlines()]
            path = publication[env]['map_xy']
            goal = publication[env]['common_goal_map_xy']
            origin = terrain['world_tile_origins'][env]
            rows = [row_metrics(row, origin, path, goal) for row in raw]
            arrival_step = summary[env]['arrival_step']
            if arrival_step is None or arrival_step <= 0 or len(rows) < 40:
                raise ValueError('arrival/window contract differs')
            trials.append({
                'period': period, 'env_id': env, 'path_role': publication[env]['path_role'],
                'path_sha256': publication[env]['path_sha256'],
                'trace_sha256': sha(trace_path), 'arrival_step': arrival_step,
                'fixed_2s': window_metrics(rows[:40]),
                'pre_arrival': window_metrics(rows[:arrival_step]),
                'step_series': rows,
            })
    by_env = {(t['env_id'], t['path_role']): t for t in trials}
    paired = []
    keys = ('v_residual_abs_mean', 'yaw_residual_abs_mean', 'commanded_forward_delta_m_sum',
            'actual_forward_delta_m_sum', 'commanded_yaw_delta_rad_sum',
            'actual_yaw_delta_rad_sum', 'path_progress_delta_m_sum',
            'common_goal_progress_delta_m_sum', 'contact_pre_step_mean_body_force_N_mean',
            'torque_pre_step_mean_abs_Nm_mean', 'reward_post_step_sum')
    for env in range(4):
        direct = by_env[env, 'direct']
        for role in ('left_20', 'right_20'):
            candidate = by_env[env, role]
            paired.append({'env_id': env, 'role': role,
                           'fixed_2s_candidate_minus_direct': {
                               key: candidate['fixed_2s'][key] - direct['fixed_2s'][key] for key in keys},
                           'pre_arrival_candidate_minus_direct': {
                               key: candidate['pre_arrival'][key] - direct['pre_arrival'][key] for key in keys}})
    return {'schema': 'eatnav_p1_g3_motion_residual_review_v1',
            'source_sha256': sha(Path(__file__)),
            'verified_original_seals': [x['host_seal_sha256'] for x in verified['periods']],
            'assignment_sha256': verified['assignment_sha256'],
            'method': 'pre-step command versus same 50 ms pre-to-post root displacement projected on pre-step body x; yaw is wrapped yaw difference / 0.05 s; force/torque are pre-step sensor values; reward is post-step',
            'independent_state_group_seed_conditions': 1,
            'trials': trials, 'paired': paired}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = analyze()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    print(args.output)


if __name__ == '__main__':
    main()
