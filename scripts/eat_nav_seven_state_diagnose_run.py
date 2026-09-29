"""Freeze the prior state 42 follower startup failure without rerunning physics."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

from scripts.eat_nav_seven_state_diagnose import (
    classify_command_delivery, map_path_goal_geometry, summarize_follower_log,
)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def atomic(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    p = argparse.ArgumentParser()
    for key in ('anchor', 'map-path', 'failure', 'bridge-log', 'follower-source', 'follower-binary', 'follower-params'):
        p.add_argument('--' + key, type=Path, required=True)
    p.add_argument('--follower-log', type=Path, action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if len(a.follower_log) != 8:
        raise ValueError('exactly eight failed-run follower logs required')
    out = a.output
    out.mkdir(parents=True, exist_ok=True)
    anchor = json.loads(a.anchor.read_text())
    state = anchor['state']
    if state['episode_key'] != [0, 56, 0] or state['step'] != 60:
        raise ValueError('state 42 anchor identity mismatch')
    geometry = map_path_goal_geometry(json.loads(a.map_path.read_text()), state['goal'], 0.2)
    failure = json.loads(a.failure.read_text())
    event = failure['bridge_timeout']
    bridge_events = [json.loads(line) for line in a.bridge_log.read_text().splitlines() if line.startswith('{')]
    if sum(row.get('event') == 'step_timeout' for row in bridge_events) != 1:
        raise ValueError('expected exactly one frozen bridge timeout')
    if event != next(row for row in bridge_events if row.get('event') == 'step_timeout'):
        raise ValueError('failure and bridge timeout evidence mismatch')
    delivered = classify_command_delivery(
        goal_distance_m=geometry['published_goal_distance_m'], tolerance_m=.2,
        before=event['sequence_before'], after=event['sequence_after'],
        commands=[None] * 8, goal_status=[False] * 8,
    )
    logs = {}
    for path in a.follower_log:
        summary = summarize_follower_log(path.read_text(errors='replace'))
        logs[path.name] = {'sha256': sha(path), **summary}
        target = out / 'ros-logs' / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha(target) != sha(path):
            raise ValueError('existing frozen ROS log differs')
        if not target.exists():
            shutil.copy2(path, target)
    if any(row['path_callback_count_lower_bound'] < 1 for row in logs.values()):
        raise ValueError('Path callback not supported by all logs')
    if any(row['control_debug_count'] < 1 for row in logs.values()):
        raise ValueError('later control computation not supported by all logs')
    shutil.copy2(a.bridge_log, out / 'prior-bridge.log')
    shutil.copy2(a.failure, out / 'prior-state42-interruption.json')
    provenance = {name: sha(path) for name, path in {
        'anchor': a.anchor, 'map_path': a.map_path, 'failure': a.failure,
        'bridge_log': a.bridge_log, 'follower_source': a.follower_source,
        'follower_binary': a.follower_binary, 'follower_params': a.follower_params,
        'bridge_source': Path('scripts/eat_nav_phase_c_ros_bridge.py'),
        'diagnostic_source': Path(__file__),
        'diagnostic_core_source': Path('scripts/eat_nav_seven_state_diagnose.py'),
    }.items()}
    atomic(out / 'source-provenance.json', {'sha256': provenance,
        'follower_log_sha256': {name: value['sha256'] for name, value in logs.items()}})
    report = {'state_index': 42, 'episode_key': state['episode_key'], 'anchor_step': state['step'],
        'robot_pose_world': anchor['anchor']['root_state'][:7],
        'robot_pose_map_xy': geometry['robot_pose_map_xy'],
        'relative_goal': state['goal'], 'goal_geometry': geometry,
        'follower_reference_goal_pose_map_xy': geometry['path_endpoint_map_xy'],
        'follower_goal_tolerance_m': .2, 'command_delivery': delivered,
        'prior_bridge_timeout': event, 'follower_logs': logs,
        'clock': {'source': 'ROS system time by default; follower uses create_wall_timer',
                  'use_sim_time_in_params': 'use_sim_time' in a.follower_params.read_text(),
                  'clock_topic_observed': None, 'two_clock_ticks_observed': None},
        'pipeline': {'process_alive': event['followers_alive'],
                     'path_subscriber_count': event['subscriber_counts'],
                     'path_callback_evidence': 'Activated default path in all eight follower ROS logs',
                     'tf_initial_failure_evidence': 'Failed to lookup robot pose in all eight follower logs',
                     'later_control_evidence': 'control debug line in all eight follower logs; exact timer and command publish counts unavailable',
                     'bridge_sequence_before': event['sequence_before'],
                     'bridge_sequence_after': event['sequence_after']},
        'diagnosis': 'goal_tolerance_excluded; no TwistStamped received by bridge for two seconds; startup TF/discovery timing is a candidate but the exact missing boundary at timeout remains unresolved',
        'classification': 'diagnosis_pending',
        'pilot_allowed': False, 'source_provenance': 'source-provenance.json'}
    atomic(out / 'state42-diagnosis.json', report)
    print('STATE42_OFFLINE_DIAGNOSIS', delivered['classification'], geometry['published_goal_distance_m'], flush=True)


if __name__ == '__main__':
    main()
