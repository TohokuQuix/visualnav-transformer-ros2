#!/usr/bin/env /usr/bin/python3
"""Sequential 1-env terrain demo smoke; leaves each simulator seal untouched."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ONIX = Path('/home/kojima/lib/onix_isaac_lab')
CONDITIONS = (
    ('random_rough', 2), ('pyramid_stairs', 1),
    ('k_rail_step_field', 2), ('k_rail_field', 2),
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root, terrain_type, level):
    output = root/'sim-output'
    ledger = json.loads((output/'sha-ledger.json').read_text())
    seal = json.loads((output/'host-seal.json').read_text())
    if sha(output/'sha-ledger.json') != seal['ledger_sha256'] or \
            any(sha(output/name) != digest for name, digest in ledger.items()):
        raise ValueError('SHA/seal verification failed')
    if (output/'failure.json').exists() or not (output/'complete.json').exists():
        raise ValueError('simulator failure/incomplete')
    startup = json.loads((output/'startup-ready.json').read_text())
    terrain = json.loads((output/'terrain-provenance.json').read_text())
    if startup['actor_shape'] != [1, 22] or startup['critic_shape'] != [1, 267] or \
            terrain['terrain_type'] != terrain_type or terrain['terrain_level'] != level or \
            terrain['selected_tiles'][0]['row'] != level:
        raise ValueError('one-env actor/terrain contract mismatch')
    events = [json.loads(line) for line in (output/'trace.jsonl').open()]
    steps = [row for row in events if row['event'] == 'step']
    selections = [row for row in events if row['event'] == 'selection']
    goals = [row for row in events if row['event'] == 'goal']
    if not steps or not selections or not goals:
        raise ValueError('Goal/Evaluator/Actor/physics pipeline incomplete')
    goal = goals[0]['goal_map']
    distance = lambda pose: math.dist(pose[:2], goal)
    first = next(row for row in steps if row['step'] >= goals[0]['step'])
    travel = sum(math.dist(row['pose_pre'][:2], row['pose_post'][:2]) for row in steps)
    return {'root': str(root), 'terrain_type': terrain_type, 'level': level,
            'protocol_valid': True, 'steps': len(steps), 'replans': len(selections),
            'selection_counts': dict(Counter(row['selected'] for row in selections)),
            'start_pose_map': steps[0]['pose_pre'], 'end_pose_map': steps[-1]['pose_post'],
            'goal_map': goal, 'goal_initial_m': distance(first['pose_pre']),
            'goal_min_m': min(distance(row['pose_post']) for row in steps),
            'goal_final_m': distance(steps[-1]['pose_post']),
            'travel_m': travel,
            'nonzero_command_steps': sum(any(abs(v) > 1e-6 for v in row['command']) for row in steps),
            'done_steps': sum(row['done'] for row in steps),
            'terrain_tile': terrain['selected_tiles'][0]['tile'],
            'sha_ledger_sha256': sha(output/'sha-ledger.json'),
            'host_seal_sha256': sha(output/'host-seal.json')}


def run_one(terrain_type, level, *, gui):
    launch = [sys.executable, str(REPO/'scripts/eat_nav_demo_launch.py'),
              '--num-envs', '1', '--terrain-type', terrain_type,
              '--terrain-level', str(level), '--max-steps', '160', '--replan-steps', '10']
    if not gui:
        launch.append('--headless')
    before = set(ONIX.glob('.eatnav-ros2-gui-demo-terrain-v2-*'))
    simulator = subprocess.Popen(launch, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    goal = subprocess.Popen(['/usr/bin/python3', str(REPO/'scripts/eat_nav_demo_goal_once.py'),
                             '--distance', '0.75', '--timeout', '180'], cwd=REPO,
                            env={**os.environ, 'RMW_IMPLEMENTATION': 'rmw_zenoh_cpp'},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        deadline = time.monotonic() + 210
        while goal.poll() is None:
            if simulator.poll() is not None:
                raise RuntimeError(f'{terrain_type}: simulator exited before ROS goal')
            if time.monotonic() > deadline:
                raise TimeoutError(f'{terrain_type}: goal publication timeout')
            time.sleep(.5)
        goal_output, _ = goal.communicate()
        if goal.returncode != 0:
            raise RuntimeError(f'{terrain_type}: goal sender failed: {goal_output}')
        print(terrain_type, goal_output.strip(), flush=True)
        roots = set(ONIX.glob('.eatnav-ros2-gui-demo-terrain-v2-*')) - before
        if len(roots) != 1:
            raise ValueError('expected one fresh demo attempt root')
        root = roots.pop()
        output, _ = simulator.communicate(timeout=300)
        if simulator.returncode != 0:
            raise RuntimeError(f'{terrain_type}: launcher exit {simulator.returncode}: {output[-1500:]}')
        return verify(root, terrain_type, level)
    finally:
        if goal.poll() is None:
            goal.terminate()
            goal.wait(timeout=10)
        if simulator.poll() is None:
            simulator.terminate()
            simulator.wait(timeout=30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--summary', type=Path, required=True)
    args = parser.parse_args()
    if args.summary.exists():
        raise FileExistsError(args.summary)
    records = []
    try:
        for index, (terrain_type, level) in enumerate(CONDITIONS):
            print('starting', terrain_type, level, flush=True)
            record = run_one(terrain_type, level, gui=index == 0)
            records.append(record)
            print('verified', terrain_type, record['steps'], record['travel_m'], flush=True)
    finally:
        with args.summary.open('x') as file:
            json.dump({'runs': records, 'expected_runs': len(CONDITIONS)}, file, indent=2, sort_keys=True)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())


if __name__ == '__main__':
    main()
