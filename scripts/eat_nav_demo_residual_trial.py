#!/usr/bin/env /usr/bin/python3
"""One bounded residual-motion diagnostic; never manages the external router."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


REPO = Path(__file__).resolve().parents[1]
ONIX = Path('/home/kojima/lib/onix_isaac_lab')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def durable(path, payload):
    with path.open('x') as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())


def verify(root):
    output = root/'sim-output'
    seal = json.loads((output/'host-seal.json').read_text())
    ledger = json.loads((output/'sha-ledger.json').read_text())
    if sha(output/'sha-ledger.json') != seal['ledger_sha256'] or any(
            sha(output/name) != digest for name, digest in ledger.items()):
        raise ValueError('simulator SHA/seal mismatch')
    if not (output/'complete.json').exists() or (output/'failure.json').exists():
        raise ValueError('simulator incomplete/failed')
    rows = [json.loads(line) for line in (output/'trace.jsonl').open()]
    steps = [row for row in rows if row['event'] == 'step']
    if len(steps) != 280 or any(row.get('physical_post') is None for row in steps):
        raise ValueError('diagnostic trace incomplete')
    if any(row['done'] for row in steps):
        raise ValueError('unexpected episode termination/reset')
    return rows


def latest_step(trace):
    if not trace.exists():
        return None
    for line in reversed(trace.read_text().splitlines()):
        row = json.loads(line)
        if row.get('event') == 'step':
            return row['step']
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=('A', 'B', 'C'), required=True)
    args = parser.parse_args()
    before = set(ONIX.glob('.eatnav-ros2-gui-demo-cancel-residual-v3-*'))
    cmd = ['/usr/bin/python3', str(REPO/'scripts/eat_nav_demo_launch.py'),
           '--residual-diagnostics', '--headless', '--num-envs', '1',
           '--terrain-type', 'boxes', '--terrain-level', '0', '--seed', '2026101101',
           '--max-steps', '280', '--replan-steps', '10']
    run = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True)
    root = None
    try:
        deadline = time.monotonic() + 120
        while root is None:
            fresh = set(ONIX.glob('.eatnav-ros2-gui-demo-cancel-residual-v3-*')) - before
            if len(fresh) > 1:
                raise ValueError('more than one new attempt root')
            if fresh:
                root = fresh.pop()
                break
            if run.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('launcher did not create a fresh attempt root')
            time.sleep(.25)
        trial = {'mode': args.mode, 'goal_distance_m': .75 if args.mode != 'C' else None,
                 'cancel_trigger_step': 186 if args.mode == 'B' else None,
                 'max_steps': 280, 'seed': 2026101101, 'terrain_type': 'boxes',
                 'terrain_level': 0, 'source_sha256': sha(__file__), 'launcher_command': cmd}
        durable(root/'trial-controller-input.json', trial)
        if args.mode != 'C':
            sender = subprocess.run(['/usr/bin/python3', str(REPO/'scripts/eat_nav_demo_goal_once.py'),
                                     '--distance', '.75', '--timeout', '180'], cwd=REPO,
                                    env={**os.environ, 'RMW_IMPLEMENTATION': 'rmw_zenoh_cpp'},
                                    capture_output=True, text=True, timeout=190)
            (root/'trial-goal-sender.log').write_text(sender.stdout + sender.stderr)
            if sender.returncode:
                raise RuntimeError('goal sender failed')
        if args.mode == 'B':
            trace = root/'sim-output/trace.jsonl'
            deadline = time.monotonic() + 240
            while (latest := latest_step(trace)) is None or latest < 186:
                if run.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('simulator ended before Cancel trigger')
                time.sleep(.05)
            cancel = subprocess.run(['ros2', 'topic', 'pub', '--once', '/eat_nav/cancel',
                                     'std_msgs/msg/Empty', '{}'], cwd=REPO,
                                    env={**os.environ, 'RMW_IMPLEMENTATION': 'rmw_zenoh_cpp'},
                                    capture_output=True, text=True, timeout=30)
            (root/'trial-cancel-sender.log').write_text(cancel.stdout + cancel.stderr)
            if cancel.returncode:
                raise RuntimeError('Cancel publisher failed')
        output, _ = run.communicate(timeout=360)
        (root/'trial-launcher.log').write_text(output)
        if run.returncode:
            raise RuntimeError(f'launcher failed: {run.returncode}')
        rows = verify(root)
        goals = sum(row['event'] == 'goal' for row in rows)
        cancels = sum(row['event'] == 'cancel' for row in rows)
        if (goals, cancels) != {'A': (1, 0), 'B': (1, 1), 'C': (0, 0)}[args.mode]:
            raise ValueError(f'goal/Cancel mismatch: {goals}/{cancels}')
        durable(root/'trial-controller-result.json', {
            'status': 'protocol_valid', 'mode': args.mode, 'trace_sha256': sha(root/'sim-output/trace.jsonl'),
            'host_seal_sha256': sha(root/'sim-output/host-seal.json'), 'goal_count': goals,
            'cancel_count': cancels})
        outer = {p.name: sha(p) for p in root.iterdir() if p.is_file()}
        durable(root/'trial-host-ledger.json', outer)
        durable(root/'trial-host-seal.json', {'ledger_sha256': sha(root/'trial-host-ledger.json'),
                                            'file_count': len(outer)})
        print(root)
    finally:
        if run.poll() is None:
            run.terminate()
            run.wait(timeout=30)


if __name__ == '__main__':
    main()
