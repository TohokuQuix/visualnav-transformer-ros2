"""One fresh readiness-002 logging parity run against the frozen prior trace."""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import pathlib
import shutil
import subprocess
import sys
import time

import eat_nav_fresh_bridge_run as fresh

REPO = pathlib.Path('/tmp/eatnav-vnt-goal-ood-simulator')
OUT = pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001')
OLD = pathlib.Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
PARITY = OUT / 'parity'
REMOTE = str(OUT / 'sim-parity.py')
NAME = 'latest-source-readiness-002'
FIELDS = (
    'follower_command', 'command', 'pre_noise', 'post_noise', 'actor_input',
    'actor_latent', 'actor_action', 'critic_observation', 'critic_value',
    'reward', 'root_state', 'post_root_state', 'joint_position',
    'joint_velocity', 'contact_force', 'raycast_hits', 'dem', 'terminal',
    'logical_sim_time_ns', 'tf_stamp_ns', 'follower_command_stamp_ns',
    'processed_action', 'previous_action', 'path_stamp_ns', 'clock_stamp_ns',
    'dem_timestamp_ns', 'raw_joint_effort', 'raw_joint_effort_grouped',
    'raw_effort_scaled_clipped_expected', 'actor_effort_pre_noise',
)


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    tmp.replace(path)


def load_trace(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def finite(value):
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return not isinstance(value, float) or math.isfinite(value)


def run():
    if not (OUT / 'timing-source-provenance.json').exists():
        raise ValueError('T0 provenance missing')
    if (PARITY / 'runs' / NAME).exists() or (PARITY / 'parity-result.json').exists():
        raise FileExistsError('T1 parity has already been attempted')
    PARITY.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(OLD / 'parity/r0-freeze.json', PARITY / 'r0-freeze.json')
    subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', str(OUT)], check=True)
    subprocess.run(['docker', 'cp', str(REPO / 'scripts/eat_nav_goal_subgoal_sim.py'),
                    f'isaac-lab-onix:{REMOTE}'], check=True)
    remote_sha = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', REMOTE], text=True).split()[0]
    local_sha = sha(REPO / 'scripts/eat_nav_goal_subgoal_sim.py')
    if remote_sha != local_sha:
        raise ValueError('T1 simulator source copy SHA mismatch')
    fresh.REPO = REPO
    fresh.ROOT = PARITY
    fresh.REMOTE_CODE = REMOTE
    fresh.REMOTE_ROOT = str(PARITY / 'runs')
    entry = json.loads((fresh.DOC / 'schedule.json').read_text())['runs'][2]
    if entry['run_id'] != 'readiness-002' or entry['control_steps'] != 120:
        raise ValueError('readiness parity entry drift')
    atomic(OUT / 'status.json', {'stage': 'T1_RUNNING', 'run_id': entry['run_id'], 'source_sha256': local_sha,
                                 'started_unix': time.time()})
    result = fresh.run_one(entry, NAME, timeout_sec=700)
    atomic(PARITY / 'run-result.json', result)
    print('T1_RUN_COMPLETE', result['elapsed_sec'], flush=True)


def analyze():
    new = PARITY / 'runs' / NAME / 'live-0.trace.json.gz'
    old = OLD / 'parity/runs/logging-parity-readiness-002-router/live-0.trace.json.gz'
    prior = json.loads((REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator/parity-smoke.json').read_text())
    if sha(old) != prior['logging_trace_sha256']:
        raise ValueError('prior logging trace SHA changed')
    current, reference = load_trace(new), load_trace(old)
    if len(current) != len(reference) or len(current) != 120:
        raise ValueError('T1 trace length differs from 120')
    first = None
    max_effort_abs = 0.0
    for step, (left, right) in enumerate(zip(current, reference)):
        for field in FIELDS:
            if field not in left or field not in right:
                first = {'step': step, 'signal': field, 'reason': 'missing field'}
                break
            if not finite(left[field]):
                first = {'step': step, 'signal': field, 'reason': 'nonfinite'}
                break
            if left[field] != right[field]:
                first = {'step': step, 'signal': field, 'reason': 'not exactly equal'}
                break
        if first:
            break
        stamp = left['logical_sim_time_ns']
        if any(left[field] != stamp for field in ('tf_stamp_ns', 'follower_command_stamp_ns', 'clock_stamp_ns', 'dem_timestamp_ns')):
            first = {'step': step, 'signal': 'stamp', 'reason': 'same-step stamp mismatch'}
            break
        if left['path_stamp_ns'] != current[0]['path_stamp_ns']:
            first = {'step': step, 'signal': 'path_stamp_ns', 'reason': 'Path stamp changed'}
            break
        raw = left['raw_joint_effort']
        grouped = left['raw_joint_effort_grouped']
        expected = left['raw_effort_scaled_clipped_expected']
        observed = left['actor_effort_pre_noise']
        if len(raw) != 52 or len(grouped) != 6 or len(expected) != 6 or len(observed) != 6:
            first = {'step': step, 'signal': 'raw_joint_effort', 'reason': 'joint/group shape'}
            break
        if any(abs(min(100.0, max(-100.0, grouped[i])) * .01 - expected[i]) > 1e-6 for i in range(6)):
            first = {'step': step, 'signal': 'raw_effort_scaled_clipped_expected', 'reason': 'clip/scale mismatch'}
            break
        max_effort_abs = max(max_effort_abs, *(abs(expected[i] - observed[i]) for i in range(6)))
        if max_effort_abs >= 1e-5:
            first = {'step': step, 'signal': 'actor_effort_pre_noise', 'reason': 'exceeds prior 1e-5 tolerance'}
            break
        if left['raw_effort_source'] != 'robot.data.applied_torque' or left['raw_effort_capture_phase'] != 'pre_env_step_after_previous_physics_before_actor_action':
            first = {'step': step, 'signal': 'raw_effort_source', 'reason': 'capture source/phase mismatch'}
            break
    result = {'status': 'pass' if first is None else 'fail', 'first_difference': first,
              'compared_steps': len(current), 'compared_fields': list(FIELDS),
              'source_sha256': sha(REPO / 'scripts/eat_nav_goal_subgoal_sim.py'),
              'new_trace_sha256': sha(new), 'old_trace_sha256': sha(old),
              'max_raw_to_pre_noise_effort_abs': max_effort_abs,
              'production_follower_sha256': sha(fresh.PROD),
              'rl_checkpoint_sha256': json.loads((OUT / 'timing-source-provenance.json').read_text())['rl_checkpoint_sha256']}
    atomic(PARITY / 'parity-result.json', result)
    atomic(OUT / 'status.json', {'stage': 'T1_PASS' if first is None else 'T1_FAIL',
                                 'parity_result_sha256': sha(PARITY / 'parity-result.json')})
    print(json.dumps(result, sort_keys=True), flush=True)
    if first is not None:
        raise ValueError(f'T1 parity failed: {first}')


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('run', 'analyze'):
        raise SystemExit('usage: python eat_nav_goal_subgoal_timing_parity.py run|analyze')
    globals()[sys.argv[1]]()
