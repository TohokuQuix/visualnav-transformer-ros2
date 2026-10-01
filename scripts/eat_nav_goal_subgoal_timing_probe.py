"""Run the single approved train-003 full-prefix timing anchor."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import time

import eat_nav_goal_subgoal_run as replay

REPO = Path('/tmp/eatnav-vnt-goal-ood-simulator')
OLD = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
OUT = Path('/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001')
RUN_NAME = 'train-003-timing-anchor'
SOURCE = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
REMOTE_SOURCE = str(OUT / 'sim-timing.py')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, data):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def main():
    if read(OUT / 'parity/parity-result.json')['status'] != 'pass':
        raise RuntimeError('T1 parity must pass before timing probe')
    if (OUT / 'runs' / RUN_NAME).exists():
        raise FileExistsError('approved timing probe has already been attempted')
    freeze = read(REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator/run-identities.json')
    state = read(OLD / 'inputs/selected-states.json')['train'][0]
    entry = {**freeze['runs'][0], 'kind': 'anchor', 'run_id': RUN_NAME, 'order': -1}
    if state['state_id'] != 'train-003' or state['step'] != 1620 or entry['anchor_step'] != 1620:
        raise ValueError('train-003 fixed state changed')
    if subprocess.run(['tmux', 'has-session', '-t', 'eatnav-goal-subgoal-router'], capture_output=True).returncode != 0:
        raise RuntimeError('router session not running')
    subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', str(OUT)], check=True)
    subprocess.run(['docker', 'cp', str(SOURCE), f'isaac-lab-onix:{REMOTE_SOURCE}'], check=True)
    remote_sha = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', REMOTE_SOURCE], text=True).split()[0]
    if remote_sha != sha(SOURCE):
        raise ValueError('timing simulator source SHA copy mismatch')
    replay.REPO = REPO
    replay.ROOT = OUT
    replay.INPUT = OLD / 'inputs'
    replay.REMOTE_CODE = REMOTE_SOURCE
    replay.REMOTE_ROOT = str(OUT / 'runs')
    replay.REMOTE_INPUT_ROOT = OLD
    atomic(OUT / 'timing-probe-source.json', {'source_sha256': remote_sha,
        'reference_source_before_telemetry_sha256': read(OLD / 'source-revision.json')['revised_sim_sha256'],
        'runner_sha256': sha(REPO / 'scripts/eat_nav_goal_subgoal_run.py'),
        'monitor_sha256': sha(REPO / 'scripts/eat_nav_goal_subgoal_timing_monitor.py'),
        'entry': entry, 'started_unix': time.time()})
    atomic(OUT / 'timing-monitor-config.json', {'heartbeat_interval_sec': 10,
        'heartbeat_stale_sec': 90, 'step_stall_sec': 120, 'gpu_cap_mib': 6656,
        'cgroup_memory_cap_bytes': 8 * 1024**3, 'run_timeout_sec': 3600,
        'frozen_before_probe': True})
    atomic(OUT / 'status.json', {'stage': 'T3_RUNNING', 'run_id': RUN_NAME, 'completed_runs': 0,
                                 'source_sha256': remote_sha, 'started_unix': time.time()})
    try:
        result = replay.run_one(entry, RUN_NAME, timeout_sec=3600, timing_telemetry=True)
        folder = OUT / 'runs' / RUN_NAME
        summary = read(folder / 'anchor-0.json')
        durations = read(folder / 'timing-step-durations.json')['step_duration_ns']
        events = [json.loads(line) for line in (folder / 'timing-events.jsonl').read_text().splitlines()]
        names = [event['event'] for event in events]
        required = ('isaac_start', 'scene_construction_complete', 'initial_reset_complete',
                    'prefix_replay_start', 'target_episode_start', 'target_anchor_reached',
                    'anchor_verification_complete', 'cleanup_start')
        if any(name not in names for name in required) or len(durations) != 1621:
            raise ValueError('timing events or 1621 step durations missing')
        errors = summary['anchor_error_vs_logged']
        if summary['state']['state_id'] != 'train-003' or summary['state']['episode_key'] != state['episode_key']:
            raise ValueError('anchor episode identity mismatch')
        if not errors['dem_history_sha_match'] or errors['prefix_command_max_abs'] > 1e-5:
            raise ValueError('anchor DEM history or command mismatch')
        if len(summary['raw_joint_effort_field_order']) != 52:
            raise ValueError('anchor raw effort joint order missing')
        if (folder / 'timing-stall.failure.json').exists():
            raise RuntimeError('external monitor recorded a stall')
        ordered = sorted(durations)
        stats = {'count': len(durations), 'median_sec': statistics.median(durations) / 1e9,
                 'p95_sec': ordered[int(.95 * (len(ordered) - 1))] / 1e9,
                 'max_sec': max(durations) / 1e9}
        remote_map = str(OUT / 'inputs/state-000/logged-seed-1001-map-path.json')
        host_map = OUT / 'inputs/state-000/logged-seed-1001-map-path.json'
        host_map.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(['docker', 'cp', f'isaac-lab-onix:{remote_map}', str(host_map)], check=True)
        outcome = {'status': 'pass', 'run_id': RUN_NAME, 'run_complete_sha256': sha(folder / 'run-complete.json'),
                   'anchor_sha256': sha(folder / 'anchor-0.json'), 'timing_events_sha256': sha(folder / 'timing-events.jsonl'),
                   'timing_durations_sha256': sha(folder / 'timing-step-durations.json'),
                   'timing_heartbeat_sha256': sha(folder / 'timing-heartbeat.json'),
                   'map_path_sha256': sha(host_map), 'step_timing': stats, 'anchor_error': errors,
                   'wall_elapsed_sec': result['elapsed_sec'], 'gpu_peak_mib': result['gpu_peak_mib'],
                   'cpu_anon_peak_bytes': result['cpu_anon_peak_bytes'],
                   'event_count': len(events), 'reset_count': names.count('episode_reset'),
                   'residual_processes': read(folder / 'final-status.json')['residual_processes']}
        atomic(OUT / 'timing-probe-result.json', outcome)
        atomic(OUT / 'status.json', {'stage': 'T3_PASS', 'run_id': RUN_NAME,
                                     'result_sha256': sha(OUT / 'timing-probe-result.json')})
        print('T3_PASS', json.dumps(outcome, sort_keys=True), flush=True)
    except BaseException as exc:
        atomic(OUT / 'status.json', {'stage': 'T3_FAIL', 'run_id': RUN_NAME, 'error': repr(exc),
                                     'failed_unix': time.time()})
        raise


if __name__ == '__main__':
    main()
