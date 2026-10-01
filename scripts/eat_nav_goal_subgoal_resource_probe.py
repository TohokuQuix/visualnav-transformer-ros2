"""One approved resource-aware train-003 anchor, then the frozen 48-run forecast."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback

import eat_nav_goal_subgoal_run as replay
from eat_nav_goal_subgoal_resource_report import (atomic, forecast, sha, suffix_timing,
                                                   summarize_probe, write_report)

REPO = Path('/tmp/eatnav-vnt-goal-ood-simulator')
OLD = Path('/tmp/eatnav-candidate-coverage/goal-ood-simulator')
PREV = Path('/tmp/eatnav-candidate-coverage/goal-ood-timing-20261001')
OUT = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')
DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/goal-ood-simulator'
RUN_NAME = 'train-003-resource-anchor'
SOURCE = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
REMOTE_SOURCE = str(OUT / 'sim-resource.py')


def read(path):
    return json.loads(Path(path).read_text())


def copy_remote_if_available(name: str):
    destination = OUT / 'runs' / RUN_NAME / name
    if destination.exists():
        return
    remote = str(OUT / 'runs' / RUN_NAME / name)
    exists = subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'test', '-f', remote],
                            capture_output=True).returncode == 0
    if exists:
        subprocess.run(['docker', 'cp', f'isaac-lab-onix:{remote}', str(destination)], check=True)


def assert_frozen():
    provenance = read(OUT / 'resource-probe-provenance.json')
    thresholds = OUT / 'resource-thresholds.json'
    expected_config_sha = (OUT / 'resource-thresholds.sha256').read_text().split()[0]
    if sha(thresholds) != expected_config_sha or provenance['status'] != 'pass':
        raise ValueError('R0/R2 provenance or threshold SHA differs')
    parity = read(OUT / 'parity/parity-result.json')
    if parity['status'] != 'pass' or parity['source_sha256'] != sha(SOURCE):
        raise ValueError('R4 current-source parity missing or source differs')
    if sha(SOURCE) != provenance['current_simulator_source_sha256']:
        raise ValueError('simulator source changed since R0')
    if sha(OLD / 'inputs/selected-states.json') != read(DOC / 'pilot-input-freeze.json')['selected_state_sha256']:
        raise ValueError('fixed state SHA changed')
    if sha(DOC / 'run-identities.json') != provenance['fixed_run_order_sha256']:
        raise ValueError('run order SHA changed')
    if (OUT / 'runs' / RUN_NAME).exists():
        raise FileExistsError('resource-aware timing probe already attempted')
    if subprocess.run(['tmux', 'has-session', '-t', 'eatnav-goal-resource-router'],
                      capture_output=True).returncode:
        raise RuntimeError('dedicated router absent')
    if sys.executable != '/usr/bin/python3':
        raise RuntimeError('probe host runner must use fixed NumPy Python environment')
    return read(thresholds), provenance, parity


def collect_anchor(result, config):
    folder = OUT / 'runs' / RUN_NAME
    summary = read(folder / 'anchor-0.json')
    state = read(OLD / 'inputs/selected-states.json')['train'][0]
    logged = read(OLD / 'inputs/state-000/logged-anchor.json')
    if summary['state']['state_id'] != 'train-003' or summary['state']['episode_key'] != state['episode_key']:
        raise ValueError('anchor state/episode identity mismatch')
    if summary['state']['step'] != 1620 or state['local_step'] != 20:
        raise ValueError('global/local prefix changed')
    errors = summary['anchor_error_vs_logged']
    limits = {'position_m': .02, 'orientation_rad': .03, 'dem_rms_m': .03, 'mask_iou_min': .95}
    if (errors['position_m'] > limits['position_m'] or errors['orientation_rad'] > limits['orientation_rad']
            or errors['dem_rms_m'] > limits['dem_rms_m'] or errors['mask_iou'] < limits['mask_iou_min']
            or not errors['dem_history_sha_match'] or errors['prefix_command_max_abs'] > 1e-5):
        raise ValueError('anchor pose/DEM/command differs from frozen record')
    if len(summary['raw_joint_effort_field_order']) != 52:
        raise ValueError('ordered 52-joint effort schema unavailable')
    anchor = summary['anchor']
    required = ('root_state', 'joint_position', 'joint_velocity', 'command_manager',
                'dem_history', 'policy_pre_noise', 'policy_post_noise_pre_delay',
                'policy_actor_input', 'actor_latent', 'actor_action',
                'critic_observation', 'critic_value', 'action_manager_previous_action')
    if any(anchor.get(key) is None for key in required):
        raise ValueError('anchor diagnostic fields missing')
    for key in required:
        if key == 'dem_history':
            continue
        def finite(value):
            if isinstance(value, list):
                return all(finite(item) for item in value)
            return not isinstance(value, (int, float)) or math.isfinite(value)
        if not finite(anchor[key]):
            raise ValueError(f'nonfinite anchor field {key}')
    events = [json.loads(line) for line in (folder / 'timing-events.jsonl').read_text().splitlines()]
    durations = read(folder / 'timing-step-durations.json')['step_duration_ns']
    if (folder / 'resource-stop.failure.json').exists():
        raise RuntimeError('resource monitor failure marker present after anchor')
    heartbeats = [json.loads(line) for line in (folder / 'resource-timeseries.jsonl').read_text().splitlines()]
    if not heartbeats:
        raise ValueError('no external heartbeat timeline')
    if any(b['stop_reason'] for b in heartbeats):
        raise ValueError('resource monitor recorded stop reason')
    if any(b['memory_events_delta']['oom'] or b['memory_events_delta']['oom_kill'] for b in heartbeats):
        raise ValueError('OOM event in heartbeat timeline')
    gaps = [(b['monotonic_ns'] - a['monotonic_ns']) / 1e9 for a, b in zip(heartbeats, heartbeats[1:])]
    if gaps and max(gaps) > config['heartbeat_stale_sec']:
        raise ValueError('heartbeat gap exceeded 90 seconds')
    if read(folder / 'final-status.json')['residual_processes']:
        raise ValueError('residual simulator/bridge/follower process')
    return {
        'status': 'pass', 'state_id': state['state_id'], 'episode_key': state['episode_key'],
        'global_prefix_steps': len(durations), 'episode_local_prefix_steps': state['local_step'],
        'anchor_error': errors, 'logged_actor_action_max_abs': max(abs(x-y) for x,y in zip(anchor['actor_action'],logged['action'])),
        'logged_critic_value_abs': abs(anchor['critic_value'] - logged['critic_value']),
        'logged_reference_fields': sorted(logged.keys()), 'detailed_anchor_fields': sorted(anchor.keys()),
        'unavailable_historical_reference': ['joint position/velocity', 'Actor observation/latent',
                                             'Critic observation', 'previous action', 'raw joint effort'],
        'heartbeat_count': len(heartbeats), 'max_heartbeat_gap_sec': max(gaps) if gaps else 0,
        'gpu_peak_mib': max(b['gpu']['used_mib'] for b in heartbeats),
        'anon_shmem_peak_bytes': max(b['resource']['anonymous_plus_shmem_bytes'] for b in heartbeats),
        'host_mem_available_min_bytes': min(b['host']['mem_available_bytes'] for b in heartbeats),
        'cgroup_file_peak_bytes': max(b['resource']['memory_stat']['file'] for b in heartbeats),
        'simulator_rss_peak_bytes': max((b['process']['simulator'] or {}).get('rss_bytes', 0) for b in heartbeats),
        'simulator_pss_peak_bytes': max((b['process']['simulator'] or {}).get('pss_bytes') or 0 for b in heartbeats),
        'run_complete_sha256': sha(folder / 'run-complete.json'),
        'anchor_sha256': sha(folder / 'anchor-0.json'),
        'timing_events_sha256': sha(folder / 'timing-events.jsonl'),
        'timing_durations_sha256': sha(folder / 'timing-step-durations.json'),
        'heartbeat_timeline_sha256': sha(folder / 'resource-timeseries.jsonl'),
        'wall_elapsed_sec': result['elapsed_sec'], 'cpu_anon_peak_bytes_runner': result['cpu_anon_peak_bytes'],
        'events': events, 'durations_ns': durations,
    }


def finish_forecast(anchor, config):
    folder = OUT / 'runs' / RUN_NAME
    host_start = read(folder / 'docker-exec-start.json')['monotonic_ns']
    host_return = read(folder / 'docker-exec-return.json')['monotonic_ns']
    probe = summarize_probe(anchor['events'], anchor['durations_ns'], host_start, host_return,
                            anchor['wall_elapsed_sec'])
    parity_folder = OUT / 'parity/runs/latest-source-readiness-002'
    parity_summary = read(parity_folder / 'live-0.json')
    parity_final = read(parity_folder / 'final-status.json')
    suffix = suffix_timing((parity_folder / 'simulator.log').read_text(),
                           parity_final['elapsed_sec'], parity_summary['elapsed_sec'])
    storage_each = max(sum(path.stat().st_size for path in parity_folder.rglob('*') if path.is_file()),
                       sum(path.stat().st_size for path in folder.rglob('*') if path.is_file()))
    prediction = forecast(probe, suffix, storage_bytes_per_run=storage_each,
                          observed_gpu_peak_mib=anchor['gpu_peak_mib'],
                          observed_anon_peak_bytes=anchor['anon_shmem_peak_bytes'])
    prediction['suffix_timing'] = suffix
    prediction['source_sha256'] = {'parity_trace': sha(parity_folder / 'live-0.trace.json.gz'),
                                   'anchor_timing': anchor['timing_durations_sha256'],
                                   'resource_config': sha(OUT / 'resource-thresholds.json')}
    artifact = write_report(OUT, probe, prediction)
    margin = prediction['conservative_plus_20_percent_sec']
    if margin > 4 * 3600:
        decision = 'runtime_gate_fail'
    elif (prediction['storage_prediction_bytes'] > config['storage_cap_bytes']
          or anchor['gpu_peak_mib'] > config['gpu_cap_mib']
          or anchor['anon_shmem_peak_bytes'] > config['anon_shmem_cap_bytes']):
        decision = 'runtime_gate_inconclusive'
    else:
        decision = 'runtime_gate_pass'
    atomic(OUT / 'resource-gate-result.json', {'decision': decision,
        'conservative_plus_20_percent_sec': margin, 'point_sec': prediction['point']['total_sec'],
        'conservative_sec': prediction['conservative_p95_step']['total_sec'],
        'worst_observed_sec': prediction['worst_observed']['total_sec'],
        'forecast_artifact_sha256': artifact['sha256'], 'anchor_result_sha256': sha(OUT / 'resource-anchor-result.json'),
        'candidate_runs': 0, 'duplicate_runs': 0, 'anchor_runs': 1, 'pilot_started': False})
    return decision


def main():
    config, provenance, parity = assert_frozen()
    state = read(OLD / 'inputs/selected-states.json')['train'][0]
    entry = {**read(DOC / 'run-identities.json')['runs'][0], 'kind': 'anchor',
             'run_id': RUN_NAME, 'order': -1}
    if state['state_id'] != 'train-003' or state['step'] != 1620 or state['local_step'] != 20:
        raise ValueError('fixed target state identity changed')
    subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', str(OUT)], check=True)
    subprocess.run(['docker', 'cp', str(SOURCE), f'isaac-lab-onix:{REMOTE_SOURCE}'], check=True)
    remote_sha = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', REMOTE_SOURCE], text=True).split()[0]
    if remote_sha != sha(SOURCE):
        raise ValueError('remote simulator source SHA differs')
    replay.REPO = REPO
    replay.ROOT = OUT
    replay.INPUT = OLD / 'inputs'
    replay.REMOTE_CODE = REMOTE_SOURCE
    replay.REMOTE_ROOT = str(OUT / 'runs')
    replay.REMOTE_INPUT_ROOT = OLD
    atomic(OUT / 'resource-probe-source.json', {'simulator_sha256': remote_sha,
        'runner_sha256': sha(REPO / 'scripts/eat_nav_goal_subgoal_run.py'),
        'monitor_sha256': sha(REPO / 'scripts/eat_nav_goal_subgoal_resource_monitor.py'),
        'report_sha256': sha(REPO / 'scripts/eat_nav_goal_subgoal_resource_report.py'),
        'config_sha256': sha(OUT / 'resource-thresholds.json'), 'parity_sha256': sha(OUT / 'parity/parity-result.json'),
        'entry': entry, 'runner_start_monotonic_ns': time.monotonic_ns(), 'runner_start_wall_ns': time.time_ns()})
    atomic(OUT / 'status.json', {'stage': 'R5_RUNNING', 'state_id': state['state_id'],
                                 'candidate_runs': 0, 'duplicate_runs': 0, 'source_sha256': remote_sha})
    try:
        result = replay.run_one(entry, RUN_NAME, timeout_sec=config['run_timeout_sec'],
            timing_telemetry=True,
            timing_monitor_script=REPO / 'scripts/eat_nav_goal_subgoal_resource_monitor.py',
            timing_monitor_config=OUT / 'resource-thresholds.json')
        anchor = collect_anchor(result, config)
        events = anchor.pop('events')
        durations = anchor.pop('durations_ns')
        atomic(OUT / 'resource-anchor-result.json', anchor)
        anchor['events'], anchor['durations_ns'] = events, durations
        decision = finish_forecast(anchor, config)
        atomic(OUT / 'status.json', {'stage': decision, 'anchor_runs': 1, 'candidate_runs': 0,
                                     'duplicate_runs': 0, 'report_generated_monotonic_ns': time.monotonic_ns(),
                                     'gate_result_sha256': sha(OUT / 'resource-gate-result.json')})
        print('RESOURCE_TIMING_DONE', decision, flush=True)
    except BaseException as exc:
        for name in ('timing-events.jsonl', 'timing-progress.json', 'timing-step-durations.json'):
            copy_remote_if_available(name)
        atomic(OUT / 'status.json', {'stage': 'runtime_gate_inconclusive', 'reason': repr(exc),
                                     'traceback': traceback.format_exc(), 'anchor_runs': 0,
                                     'candidate_runs': 0, 'duplicate_runs': 0})
        raise


if __name__ == '__main__':
    main()
