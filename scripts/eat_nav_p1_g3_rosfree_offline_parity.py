"""Read-only exact-stamp ROS G2/G3 trace replay through the ROS-free C++ core."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core


CONFIG = Path(__file__).resolve().parents[1] / 'config/eat_nav/p1_g3_rosfree_phase_a_parity.json'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replay_output(output: Path, library: Path, limits: dict) -> dict:
    publication_path = output / 'path-publication.json'
    publication = json.loads(publication_path.read_text())
    if publication['count'] != 1 or len(publication['trial_paths']) != 4:
        raise ValueError('expected one Path publication and four slots')
    result = {'output': str(output), 'path_publication_sha256': sha(publication_path),
              'slots': [], 'max_command_abs': 0.0, 'max_carrot_abs': 0.0,
              'max_heading_abs': 0.0, 'mismatches': [], 'recovery_ros_steps': 0,
              'max_motion_abs': 0.0,
              'recovery_core_steps': 0, 'goal_ros_steps': 0, 'goal_core_steps': 0}
    for slot, trial in enumerate(publication['trial_paths']):
        trace_path = output / f'trace-slot-{slot}.jsonl'
        follower = PureFollower(trial['map_xy'], library=library)
        slot_result = {'slot': slot, 'trace_sha256': sha(trace_path), 'rows': 0,
                       'first_stamp_ns': None, 'last_stamp_ns': None}
        previous_stamp = None
        for raw in trace_path.open():
            row = json.loads(raw)
            stamp = int(row['logical_sim_time_ns'])
            if previous_stamp is not None and stamp - previous_stamp != limits['logical_step_ns']:
                raise ValueError(f'noncontinuous logical time: {trace_path}:{row["control_step"]}')
            previous_stamp = stamp
            ros = row['follower_diagnostic']
            if not ros['tf_success'] or int(ros['tf_stamp_ns']) != stamp or int(ros['command_stamp_ns']) != stamp:
                raise ValueError(f'invalid exact-stamp trace: {trace_path}:{row["control_step"]}')
            command, diagnostic = follower.step_diagnostic(
                float(ros['tf_x']), float(ros['tf_y']), float(ros['tf_yaw']), stamp)
            error = max(abs(a - b) for a, b in zip(command, row['follower_command']))
            result['max_command_abs'] = max(result['max_command_abs'], error)
            tracking = not bool(ros['goal_reached'])
            point_error = max(abs(diagnostic[k] - float(ros[k])) for k in ('carrot_x', 'carrot_y')) if tracking else 0.0
            result['max_carrot_abs'] = max(result['max_carrot_abs'], point_error)
            heading_error = abs(diagnostic['heading_error'] - float(ros['heading_error'])) if tracking else 0.0
            result['max_heading_abs'] = max(result['max_heading_abs'], heading_error)
            motion_keys = {'pre_commanded_distance': 'commanded_distance',
                           'pre_actual_distance': 'actual_distance',
                           'pre_motion_ratio': 'motion_ratio'}
            motion_error = max(abs(diagnostic[k] - float(ros[v])) for k, v in motion_keys.items())
            result['max_motion_abs'] = max(result['max_motion_abs'], motion_error)
            motion_exact = (diagnostic['pre_motion_samples'] == int(ros['motion_samples'])
                            and diagnostic['pre_motion_ready'] == bool(ros['motion_ready'])
                            and int(diagnostic['pre_motion_oldest_ns']) == int(ros['motion_oldest_ns'])
                            and int(diagnostic['pre_motion_newest_ns']) == int(ros['motion_newest_ns']))
            fields = {'goal_reached': ('goal_reached', 'goal_reached'),
                      'linear_recovery': ('linear_recovery', 'linear_recovery'),
                      'angular_recovery': ('angular_recovery', 'angular_recovery')}
            boolean_diff = [label for label, (a, b) in fields.items()
                            if bool(diagnostic[a]) != bool(ros[b])]
            if (error > limits['command_max_abs'] or point_error > limits['geometry_max_abs']
                    or heading_error > limits['heading_max_abs']
                    or motion_error > limits['motion_scalar_max_abs']
                    or not motion_exact or boolean_diff):
                result['mismatches'].append({'slot': slot, 'step': row['control_step'],
                    'command_max_abs': error, 'carrot_max_abs': point_error,
                    'heading_abs': heading_error, 'motion_abs': motion_error,
                    'motion_exact': motion_exact, 'booleans': boolean_diff})
            result['recovery_ros_steps'] += int(bool(ros['linear_recovery']) or bool(ros['angular_recovery']))
            result['recovery_core_steps'] += int(diagnostic['linear_recovery'] or diagnostic['angular_recovery'])
            result['goal_ros_steps'] += int(bool(ros['goal_reached']))
            result['goal_core_steps'] += int(diagnostic['goal_reached'])
            slot_result['rows'] += 1
            slot_result['first_stamp_ns'] = slot_result['first_stamp_ns'] or stamp
            slot_result['last_stamp_ns'] = stamp
        follower.close()
        result['slots'].append(slot_result)
    result['rows'] = sum(s['rows'] for s in result['slots'])
    result['pass'] = not result['mismatches']
    result['mismatch_count'] = len(result['mismatches'])
    result['mismatches'] = result['mismatches'][:20]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    limits = json.loads(CONFIG.read_text())
    library = compile_core(args.report.parent / 'core-build')
    results = [replay_output(p, library, limits) for p in args.output_dir]
    report = {'protocol': limits['protocol'], 'limits_sha256': sha(CONFIG),
              'cpp_sha256': sha(Path(__file__).resolve().parents[1] /
                  'src/visualnav_transformer/eat_nav/p1_follower_core.cpp'),
              'inputs': results, 'pass': all(r['pass'] for r in results),
              'total_rows': sum(r['rows'] for r in results),
              'measured_recovery_covered': any(r['recovery_ros_steps'] for r in results)}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k: report[k] for k in ('pass', 'total_rows', 'measured_recovery_covered')}))


if __name__ == '__main__':
    main()
