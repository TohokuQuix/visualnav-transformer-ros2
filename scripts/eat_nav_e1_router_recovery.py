"""External-router E1 recovery gate; never owns the router lifecycle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    tmp.replace(path)


def verify_readiness(samples):
    if len(samples) < 3:
        raise ValueError('three stable readiness samples required')
    clocks = [int(s['clock_now_ns']) for s in samples]
    if any(b <= a for a, b in zip(clocks, clocks[1:])):
        raise ValueError('logical clock did not advance')
    for index, snap in enumerate(samples):
        if snap['path_subscribers'] != [1] * 8:
            raise ValueError(f'Path subscriber count not stable at sample {index}')
        if snap['follower_alive'] != [True] * 8 or any(x < 2 for x in snap['status_counts']):
            raise ValueError(f'follower lifecycle/status not ready at sample {index}')
        if snap['tf_ready'] != [True] * 8:
            raise ValueError(f'TF not ready at sample {index}')
        if snap['clock_publishers'] != 1 or snap['use_sim_time'] is not True:
            raise ValueError(f'clock publisher/config mismatch at sample {index}')
        if any(snap[key] != [0] * 8 for key in ('command_sequence', 'callback_count')):
            raise ValueError(f'pre-Path command adopted at sample {index}')
        if snap['protocol_errors']:
            raise ValueError(f'protocol error at sample {index}')
    return {'status': 'pass', 'samples': len(samples), 'total_path_subscribers': 8,
            'first_clock_ns': clocks[0], 'last_clock_ns': clocks[-1],
            'candidate_path_published': False, 'accepted_pre_path_commands': 0}


def recovery_clock_start(root, attempt_zero, *, pending, readiness_status):
    root, attempt_zero = Path(root), Path(attempt_zero)
    if not (attempt_zero/'failure-0.marker.json').is_file():
        raise ValueError('attempt-0 failure marker missing')
    if readiness_status != 'pass' or pending != 18:
        raise ValueError('readiness or frozen pending count changed')
    marker = root/'recovery-continuation-started.marker.json'
    schedule_marker = root/'continuation-started.marker.json'
    if marker.exists() or schedule_marker.exists():
        raise FileExistsError('recovery continuation already started')
    start_ns = time.time_ns()
    value = {'start_wall_ns': start_ns, 'pending_valid_processes': 18,
             'attempt_0_failure_marker_sha256': sha(attempt_zero/'failure-0.marker.json'),
             'attempt_0_excluded_from_outcomes': True}
    atomic(marker, value)
    atomic(schedule_marker, {'start_wall_ns': start_ns, 'pending_processes': 18,
                             'recovery_marker_sha256': sha(marker)})
    return value


def diagnostic_smoke(root, follower, params):
    """Exercise the real production follower/bridge ROS graph with empty Paths only."""
    from eat_nav_command_delivery_bridge import DiagnosticServer
    from eat_nav_sim_clock_bridge import make_stamp
    from rosgraph_msgs.msg import Clock
    root = Path(root)
    args = argparse.Namespace(branches=8, follower=Path(follower), params=Path(params),
                              diagnostic_dir=root/'readiness-follower-logs', fault_drop_stamp_ns=None)
    server = DiagnosticServer(args)
    samples = []
    t0 = 2_300_000_000_000
    pose = [0.0, 0.0, 0.0]
    try:
        node = server.node
        node.clock(t0, reset=True)
        node.empty_paths(t0)
        node.poses([pose] * 8, t0)
        server.launch()
        deadline = time.monotonic() + 15
        while True:
            snap = server.snapshot()
            if (snap['path_subscribers'] == [1] * 8
                    and snap['follower_alive'] == [True] * 8
                    and all(x >= 2 for x in snap['status_counts'])
                    and snap['tf_ready'] == [True] * 8):
                break
            if time.monotonic() > deadline:
                raise TimeoutError('empty-Path bridge/follower readiness ' + repr(snap))
            node.clock_pub.publish(Clock(clock=make_stamp(t0)))
            node.poses([pose] * 8, t0)
            time.sleep(0.25)
        for index in range(3):
            ns = t0 + index * 50_000_000
            if index:
                node.clock(ns)
                node.poses([pose] * 8, ns)
            node.empty_paths(ns)
            time.sleep(0.3)
            samples.append(server.snapshot())
        result = verify_readiness(samples)
        result['empty_path_clear_published_at_each_stamp'] = True
        result['bridge_connected_by_follower_status'] = True
        atomic(root/'router-readiness.json', {'gate': result, 'samples': samples})
        return result
    finally:
        server.shutdown_lifecycle()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--follower', type=Path, required=True)
    p.add_argument('--params', type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(diagnostic_smoke(a.root, a.follower, a.params), sort_keys=True), flush=True)
