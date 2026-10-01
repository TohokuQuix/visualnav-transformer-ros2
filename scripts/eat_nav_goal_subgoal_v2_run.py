"""Fresh-process v2 anchors and conditionally approved live outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import traceback

import numpy as np

from visualnav_transformer.eat_nav.candidate_coverage_pilot import map_paths_from_anchor
import eat_nav_goal_subgoal_run as replay
from eat_nav_goal_subgoal_v2_select import REPO, DOC, OUT, CONFIG, sha, atomic

TIMING = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')
SOURCE = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
REMOTE_SOURCE = str(OUT / 'sim-v2.py')


def read(path):
    return json.loads(Path(path).read_text())


def status(data):
    target = OUT / 'status.json'
    temp = target.with_name('status.json.tmp')
    temp.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(target)


def bytes_in_root():
    return sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())


class Heartbeat:
    def __init__(self, stage, total):
        self.start = time.monotonic()
        self.state = {'stage': stage, 'completed_processes': 0, 'total_processes': total,
                      'candidate_outcomes': 0, 'stop_reason': None}
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def update(self, **fields):
        with self.lock:
            self.state.update(fields)

    def loop(self):
        while not self.stop.is_set():
            with self.lock:
                data = dict(self.state)
            run_id = data.get('run_id')
            path = OUT / 'timing-heartbeat.json'
            if run_id and path.exists():
                try:
                    beat = read(path)
                    if beat.get('run_id') == run_id:
                        process = beat['process'].get('simulator') or {}
                        mem = beat['resource']['memory_stat']
                        data.update({'global_step': beat.get('global_step'),
                                     'local_step': beat.get('episode_local_step'),
                                     'suffix_control_step': beat.get('suffix_control_step'),
                                     'simulator_logical_time_sec': beat['resource'].get('simulator_logical_time_sec'),
                                     'last_progress_monotonic_ns': beat.get('last_progress_monotonic_ns'),
                                     'simulator_rss_bytes': process.get('rss_bytes'),
                                     'simulator_pss_bytes': process.get('pss_bytes'),
                                     'cgroup_anon_bytes': mem['anon'], 'cgroup_shmem_bytes': mem['shmem'],
                                     'cgroup_file_bytes': mem['file'],
                                     'host_mem_available_bytes': beat['host']['mem_available_bytes'],
                                     'gpu_used_mib': beat['gpu']['used_mib'],
                                     'resource_stop_reason': beat.get('stop_reason')})
                except (OSError, ValueError, KeyError):
                    pass
            data['elapsed_sec'] = time.monotonic() - self.start
            data['artifact_bytes'] = bytes_in_root()
            data['latest_marker'] = str(OUT / 'runs' / run_id / 'run-complete.json') if run_id else None
            status(data)
            self.stop.wait(10)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=15)


def frozen(entry):
    schedule = read(OUT / 'run-schedule-v2.json')['runs']
    if schedule[entry['order']] != entry:
        raise ValueError('v2 run order/identity drift')
    state = OUT / 'inputs' / f"state-{entry['state_index']:03d}"
    if sha(state / f"{entry['candidate_id']}-base-path.json") != entry['base_path_sha256']:
        raise ValueError('v2 candidate Path SHA drift')
    if entry['kind'] != 'anchor':
        metadata = read(state / f"{entry['run_id']}-goal-metadata.json")
        if metadata['base_path_sha256'] != entry['base_path_sha256'] or metadata['run_id'] != entry['run_id']:
            raise ValueError('v2 goal metadata drift')
    return state


def setup():
    if read(OUT / 'v5.done.json')['decision'] != 'pass' or read(OUT / 'v6.done.json')['status'] != 'pass':
        raise ValueError('V5/V6 gate incomplete')
    if read(OUT / 'v6.done.json')['source_sha256'] != sha(SOURCE):
        raise ValueError('simulator source changed after V6 parity')
    if read(OUT / 'v4.done.json')['schedule_sha256'] != sha(OUT / 'run-schedule-v2.json'):
        raise ValueError('v2 schedule SHA drift')
    if subprocess.run(['tmux', 'has-session', '-t', 'eatnav-v2-router'], capture_output=True).returncode:
        raise RuntimeError('dedicated v2 router stopped')
    replay.REPO = REPO
    replay.DOC = DOC
    replay.ROOT = OUT
    replay.INPUT = OUT / 'inputs'
    replay.REMOTE_CODE = REMOTE_SOURCE
    replay.REMOTE_ROOT = str(OUT / 'runs')
    replay.REMOTE_INPUT_ROOT = OUT
    replay.frozen = frozen
    remote_inputs = str(OUT / 'inputs')
    remote_source = REMOTE_SOURCE
    subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'mkdir', '-p', str(OUT)], check=True)
    if subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'test', '-e', remote_inputs]).returncode:
        subprocess.run(['docker', 'cp', str(OUT / 'inputs'), f'isaac-lab-onix:{remote_inputs}'], check=True)
    if subprocess.run(['docker', 'exec', 'isaac-lab-onix', 'test', '-e', remote_source]).returncode:
        subprocess.run(['docker', 'cp', str(SOURCE), f'isaac-lab-onix:{remote_source}'], check=True)
    paths = [OUT / 'inputs/selected-states.json', SOURCE]
    for path in paths:
        remote = remote_source if path == SOURCE else str(path)
        digest = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sha256sum', remote], text=True).split()[0]
        if digest != sha(path):
            raise ValueError('Docker source/input SHA differs')
    atomic(OUT / 'v7-setup.json', {'schedule_sha256': sha(OUT / 'run-schedule-v2.json'),
                                   'simulator_source_sha256': sha(SOURCE),
                                   'selected_input_sha256': sha(OUT / 'inputs/selected-states.json'),
                                   'parity_result_sha256': read(OUT / 'v6.done.json')['parity_result_sha256']})


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, (int, float)) or np.isfinite(value)


def verify_anchor(entry):
    folder = OUT / 'runs' / entry['run_id']
    summary = read(folder / 'anchor-0.json')
    logged = read(OUT / 'inputs' / f"state-{entry['state_index']:03d}" / 'logged-anchor.json')
    state = summary['state']
    err = summary['anchor_error_vs_logged']
    if (state['state_id'] != entry['state_id'] or state['episode_key'] != entry['episode_key']
            or state['step'] != entry['anchor_step'] or state['local_step'] != entry['anchor_step']):
        raise ValueError('anchor identity/global/local step differs')
    if (err['position_m'] > .02 or err['orientation_rad'] > .03 or err['dem_rms_m'] > .03
            or err['mask_iou'] < .95 or not err['dem_history_sha_match']
            or err['prefix_command_max_abs'] > 1e-5):
        raise ValueError('anchor pose/DEM/mask/command gate failed')
    anchor = summary['anchor']
    if anchor['action_manager_action'] != logged['action'] or not finite(anchor):
        raise ValueError('logged action differs from ActionManager or nonfinite anchor')
    if len(summary['raw_joint_effort_field_order']) != 52:
        raise ValueError('raw effort joint order unavailable')
    result = {'status': 'pass', 'state_id': entry['state_id'], 'episode_key': entry['episode_key'],
              'terrain': entry['terrain'], 'global_prefix_steps': entry['prefix_steps'],
              'anchor_error': err, 'action_manager_matches_logged_action': True,
              'critic_value_abs_vs_logged': abs(anchor['critic_value'] - logged['critic_value']),
              'actor_action_max_abs_vs_logged': max(abs(x - y) for x, y in zip(anchor['actor_action'], logged['action'])),
              'detailed_fields_saved': sorted(anchor),
              'historical_reference_NA': ['joint position/velocity', 'Actor observation/latent',
                                          'Critic observation', 'previous action', 'raw joint effort'],
              'run_complete_sha256': sha(folder / 'run-complete.json'),
              'anchor_summary_sha256': sha(folder / 'anchor-0.json'),
              'simulator_elapsed_sec': read(folder / 'final-status.json')['elapsed_sec']}
    atomic(OUT / f"anchor-{entry['state_id']}.json", result)
    return summary


def map_paths(entry, summary):
    index = entry['state_index']
    folder = OUT / 'inputs' / f'state-{index:03d}'
    candidates = [r for r in read(OUT / 'run-schedule-v2.json')['runs'] if r['state_index'] == index and r['kind'] == 'candidate']
    base = {r['candidate_id']: read(folder / f"{r['candidate_id']}-base-path.json") for r in candidates}
    mapped = map_paths_from_anchor(summary, {'bank-nominal': base['logged-seed-1001'], **base})
    nominal = folder / 'logged-seed-1001-map-path.json'
    subprocess.run(['docker', 'cp', f'isaac-lab-onix:{folder}/logged-seed-1001-map-path.json', str(nominal)], check=True)
    if np.max(abs(np.asarray(read(nominal)) - np.asarray(mapped['bank-nominal']))) > 1e-6:
        raise ValueError('simulator nominal map Path differs from host conversion')
    hashes = {}
    for name, points in mapped.items():
        if name == 'bank-nominal':
            continue
        path = folder / f'{name}-map-path.json'
        if path.exists():
            if np.max(abs(np.asarray(read(path)) - np.asarray(points))) > 1e-6:
                raise ValueError('frozen map Path differs: ' + name)
        else:
            atomic(path, points)
        subprocess.run(['docker', 'cp', str(path), f'isaac-lab-onix:{path}'], check=True)
        hashes[name] = sha(path)
    atomic(OUT / f"map-path-freeze-{entry['state_id']}.json", {'state_id': entry['state_id'],
        'anchor_summary_sha256': sha(OUT / 'runs' / entry['run_id'] / 'anchor-0.json'),
        'map_path_sha256': hashes})


def anchors():
    setup()
    if (OUT / 'anchor-results-v2.json').exists():
        raise FileExistsError('v2 anchors already finalized')
    schedule = read(OUT / 'run-schedule-v2.json')['runs']
    entries = [row for row in schedule if row['kind'] == 'anchor']
    results = []
    with Heartbeat('V7_ANCHORS_RUNNING', 4) as beat:
        for index, entry in enumerate(entries):
            if (OUT / 'runs' / entry['run_id']).exists():
                raise FileExistsError('partial or previous anchor; no automatic retry')
            beat.update(current_state=entry['state_id'], run_id=entry['run_id'], goal_category='anchor',
                        seed=entry['seed'], completed_processes=index)
            replay.run_one(entry, entry['run_id'], timeout_sec=900, timing_telemetry=True,
                timing_monitor_script=REPO / 'scripts/eat_nav_goal_subgoal_resource_monitor.py',
                timing_monitor_config=TIMING / 'resource-thresholds.json')
            summary = verify_anchor(entry)
            map_paths(entry, summary)
            results.append(read(OUT / f"anchor-{entry['state_id']}.json"))
            beat.update(completed_processes=index + 1, run_id=None)
    atomic(OUT / 'anchor-results-v2.json', {'status': 'pass', 'anchors': results,
                                              'schedule_sha256': sha(OUT / 'run-schedule-v2.json')})
    atomic(OUT / 'v7.done.json', {'status': 'pass', 'anchor_results_sha256': sha(OUT / 'anchor-results-v2.json'),
                                  'anchor_count': 4, 'candidate_outcome_count': 0})
    status({'stage': 'V7_ANCHORS_PASS', 'completed_processes': 4, 'total_processes': 48,
            'candidate_outcomes': 0, 'anchor_results_sha256': sha(OUT / 'anchor-results-v2.json')})
    print('V7_ANCHORS_PASS', len(results), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('anchors',), required=True)
    args = parser.parse_args()
    try:
        anchors()
    except BaseException as exc:
        status({'stage': 'v2_anchor_gate_fail', 'error': repr(exc), 'traceback': traceback.format_exc(),
                'candidate_outcomes': 0})
        raise
