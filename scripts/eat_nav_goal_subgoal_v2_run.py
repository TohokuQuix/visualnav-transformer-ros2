"""Fresh-process v2 anchors and conditionally approved live outcomes."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import subprocess
import threading
import time
import traceback

import numpy as np
import yaml

from visualnav_transformer.eat_nav.candidate_coverage_pilot import map_paths_from_anchor
from visualnav_transformer.eat_nav.command_preview_long_horizon import ANCHOR_KEYS, TRACE_KEYS, post_root
import eat_nav_goal_subgoal_run as replay
from eat_nav_goal_subgoal_dem_gate import verify_dem_history
from eat_nav_goal_subgoal_v2_select import REPO, DOC, OUT, CONFIG, sha, atomic

TIMING = Path('/tmp/eatnav-candidate-coverage/goal-ood-resource-timing-20261001')
SOURCE = REPO / 'scripts/eat_nav_goal_subgoal_sim.py'
REMOTE_SOURCE = str(OUT / 'sim-v2.py')
STORAGE_PROTOCOL = DOC / 'v2-storage-protocol.json'
REFERENCE = REPO / 'docs/quality-identifiability-gate/phase-c/reference-eat.yaml'
MANIFEST = Path('/tmp/eatnav-candidate-coverage/goal-ood/input-copy/manifest.json')
_FRAME_CACHE = None


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


def recorded_frames():
    global _FRAME_CACHE
    if _FRAME_CACHE is not None:
        return _FRAME_CACHE
    import gzip
    states = read(OUT / 'selected-states-v2.json')['states']
    wanted = {(tuple(s['episode_key']), step) for s in states
              for step in (s['global_step'] - 15, s['global_step'] - 10,
                           s['global_step'] - 5, s['global_step'])}
    found = {}
    manifest = read(MANIFEST)
    if sha(MANIFEST) != states[0]['manifest_sha256']:
        raise ValueError('raw manifest SHA drift')
    for chunk in manifest['chunks']:
        path = MANIFEST.parent / chunk['file']
        with gzip.open(path, 'rt') as stream:
            for line in stream:
                row = json.loads(line)
                key = ((row['session_id'], row['env_id'], row['episode_id']), row['step'])
                if key in wanted:
                    if key in found:
                        raise ValueError('duplicate recorded DEM frame identity')
                    found[key] = row
        if len(found) == len(wanted):
            break
    if set(found) != wanted:
        raise ValueError('recorded DEM frame missing')
    _FRAME_CACHE = found
    return found


def run_name(entry):
    return entry['run_id'] + '-attempt-1' if entry['order'] == 1 else entry['run_id']


def verify_saved_run(entry, actual_name=None):
    """Recheck frozen input/output bytes before reusing a completed process."""
    actual_name = actual_name or run_name(entry)
    folder = OUT / 'runs' / actual_name
    recovered_path = folder / 'recovered_complete_after_monitor_cleanup_race.json'
    recovered = recovered_path.exists()
    complete = read(recovered_path if recovered else folder / 'run-complete.json')
    started = read(folder / 'started.json')
    if started['entry'] != entry or started['run_name'] != actual_name or (
            recovered and (entry['order'] != 21 or complete['logical_run_id'] != actual_name)) or (
            not recovered and complete['identity'] != entry):
        raise ValueError('saved run identity differs')
    final = read(folder / 'final-status.json')
    if recovered:
        if (complete['classification'] != 'recovered_complete_after_monitor_cleanup_race'
                or complete['trace_steps'] != 120 or not complete['all_numeric_finite']
                or final['sim_exit'] != 0 or final['bridge_exit'] != 0 or final['residual_processes']
                or complete['started_sha256'] != sha(folder / 'started.json')
                or complete['final_status_sha256'] != sha(folder / 'final-status.json')
                or complete['resource_failure_sha256'] != sha(folder / 'resource-stop.failure.json')
                or complete['schedule_sha256'] != sha(OUT / 'run-schedule-v2.json')):
            raise ValueError('recovery evidence drift')
    elif (complete['protocol'] != 'fresh_bridge_per_run' or final['stage'] != 'COMPLETE'
            or ((folder / 'timing-monitor.done.json').exists() and
                read(folder / 'timing-monitor.done.json').get('status') not in ('complete', 'stopped'))):
        raise ValueError('saved run did not complete normally')
    if not recovered and entry['kind'] != 'anchor' and entry['order'] >= 22:
        done = read(folder / 'timing-monitor.done.json')
        if (done.get('status') != 'complete' or done.get('lifecycle') != 'COMPLETE_MARKER'
                or done.get('complete_marker_sha256') != sha(folder / 'run-complete.json')):
            raise ValueError('continuation cleanup verification missing or mismatched')
    for name, digest in complete['output_sha256'].items():
        if sha(folder / name) != digest:
            raise ValueError('saved run output SHA drift: ' + name)
    state_dir = OUT / 'inputs' / f"state-{entry['state_index']:03d}"
    paths = {'prefix': state_dir / 'prefix-actions.npz',
             'suffix': state_dir / 'suffix-actions-120.npz',
             'base_path': state_dir / f"{entry['candidate_id']}-base-path.json",
             'config': state_dir / 'config-120.yaml',
             'anchor': state_dir / 'logged-anchor.json'}
    if entry['kind'] != 'anchor':
        paths['map_path'] = state_dir / f"{entry['candidate_id']}-map-path.json"
    for name, digest in started['input_sha256'].items():
        if name not in paths or sha(paths[name]) != digest:
            raise ValueError('saved run input SHA drift: ' + name)
    if set(started['input_sha256']) != set(paths):
        raise ValueError('saved run input ledger incomplete')
    marker = read(folder / ('anchor-0.marker.json' if entry['kind'] == 'anchor' else 'live-0.marker.json'))
    for path, digest in marker['outputs_sha256'].items():
        name = Path(path).name
        if sha(folder / name) != digest:
            raise ValueError('saved simulator marker output SHA drift: ' + name)
    if marker['source']['runner_source'] != sha(SOURCE):
        raise ValueError('saved simulator source SHA drift')
    if marker['source']['reference_config'] != sha(REFERENCE) or marker['source']['manifest'] != sha(MANIFEST):
        raise ValueError('saved reference config or raw manifest SHA drift')
    if marker['source']['state'] != sha(OUT / 'inputs/selected-states.json'):
        raise ValueError('saved selected state input SHA drift')
    return {'run_complete_sha256': sha(recovered_path if recovered else folder / 'run-complete.json'),
            'started_sha256': sha(folder / 'started.json'),
            'marker_sha256': sha(folder / ('anchor-0.marker.json' if entry['kind'] == 'anchor' else 'live-0.marker.json'))}


def verify_anchor(entry):
    folder = OUT / 'runs' / run_name(entry)
    summary_name = 'anchor-0.json' if entry['kind'] == 'anchor' else 'live-0.json'
    summary = read(folder / summary_name)
    logged = read(OUT / 'inputs' / f"state-{entry['state_index']:03d}" / 'logged-anchor.json')
    state = summary['state']
    err = summary['anchor_error_vs_logged']
    if (state['state_id'] != entry['state_id'] or state['episode_key'] != entry['episode_key']
            or state['step'] != entry['anchor_step'] or state['local_step'] != entry['anchor_step']):
        raise ValueError('anchor identity/global/local step differs')
    if (err['position_m'] > .02 or err['orientation_rad'] > .03 or err['dem_rms_m'] > .03
            or err['mask_iou'] < .95
            or err['prefix_command_max_abs'] > 1e-5):
        raise ValueError('anchor pose/DEM/mask/command gate failed')
    frames = recorded_frames()
    steps = [entry['anchor_step'] - 15, entry['anchor_step'] - 10,
             entry['anchor_step'] - 5, entry['anchor_step']]
    key = tuple(entry['episode_key'])
    with np.load(OUT / 'selected-states-v2.npz') as selected:
        selected_dem = selected['dem'][entry['state_index']]
    numeric = verify_dem_history(entry=entry, summary=summary,
        recorded_frames=[frames[(key, step)] for step in steps], selected_dem=selected_dem,
        reference_config=REFERENCE,
        state_config=OUT / 'inputs' / f"state-{entry['state_index']:03d}" / 'config-120.yaml',
        source_sha256=sha(SOURCE), expected_source_sha256=read(OUT / 'v6.done.json')['source_sha256'])
    anchor = summary['anchor']
    if anchor['action_manager_action'] != logged['action'] or not finite(anchor):
        raise ValueError('logged action differs from ActionManager or nonfinite anchor')
    if len(summary['raw_joint_effort_field_order']) != 52:
        raise ValueError('raw effort joint order unavailable')
    result = {'status': 'pass', 'state_id': entry['state_id'], 'episode_key': entry['episode_key'],
              'terrain': entry['terrain'], 'global_prefix_steps': entry['prefix_steps'],
              'anchor_error': err, 'dem_history_numeric_gate': numeric,
              'action_manager_matches_logged_action': True,
              'critic_value_abs_vs_logged': abs(anchor['critic_value'] - logged['critic_value']),
              'actor_action_max_abs_vs_logged': max(abs(x - y) for x, y in zip(anchor['actor_action'], logged['action'])),
              'detailed_fields_saved': sorted(anchor),
              'historical_reference_NA': ['joint position/velocity', 'Actor observation/latent',
                                          'Critic observation', 'previous action', 'raw joint effort'],
              'run_complete_sha256': sha(folder / ('recovered_complete_after_monitor_cleanup_race.json'
                                                  if (folder / 'recovered_complete_after_monitor_cleanup_race.json').exists()
                                                  else 'run-complete.json')),
              'anchor_summary_sha256': sha(folder / summary_name),
              'simulator_elapsed_sec': read(folder / 'final-status.json')['elapsed_sec']}
    result['run_id'] = entry['run_id']
    if entry['kind'] != 'anchor':
        result['actual_run_name'] = run_name(entry)
    result['run_integrity'] = verify_saved_run(entry)
    result_name = (f"anchor-{entry['state_id']}-numeric-gate-v2.json" if entry['kind'] == 'anchor'
                   else f"run-{entry['order']:02d}-numeric-gate.json")
    atomic(OUT / result_name, result)
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
    schedule = read(OUT / 'run-schedule-v2.json')['runs']
    entries = [row for row in schedule if row['kind'] == 'anchor']
    if (OUT / 'v7-numeric-gate.done.json').exists():
        raise FileExistsError('v2 numeric-gate anchors already finalized')
    first = entries[0]
    if first['run_id'] != 'v2-train-000-anchor':
        raise ValueError('frozen first anchor changed')
    first_integrity = verify_saved_run(first)
    verify_anchor(first)
    atomic(OUT / 'v7-first-anchor-offline-numeric-gate.json', {
        'status': 'pass', 'state_id': first['state_id'], **first_integrity,
        'numeric_gate_sha256': sha(OUT / f"anchor-{first['state_id']}-numeric-gate-v2.json")})
    setup()
    results = []
    with Heartbeat('V7_ANCHORS_RUNNING', 4) as beat:
        for index, entry in enumerate(entries):
            beat.update(current_state=entry['state_id'], run_id=entry['run_id'], goal_category='anchor',
                        seed=entry['seed'], completed_processes=index)
            if index == 0:
                verify_saved_run(entry)
            else:
                if (OUT / 'runs' / entry['run_id']).exists():
                    verify_saved_run(entry)
                else:
                    replay.run_one(entry, entry['run_id'], timeout_sec=900, timing_telemetry=True,
                        timing_monitor_script=REPO / 'scripts/eat_nav_goal_subgoal_resource_monitor.py',
                        timing_monitor_config=TIMING / 'resource-thresholds.json')
                    verify_saved_run(entry)
            summary = verify_anchor(entry)
            map_paths(entry, summary)
            results.append(read(OUT / f"anchor-{entry['state_id']}-numeric-gate-v2.json"))
            beat.update(completed_processes=index + 1, run_id=None)
    atomic(OUT / 'anchor-results-v2-numeric-gate.json', {'status': 'pass', 'anchors': results,
                                              'schedule_sha256': sha(OUT / 'run-schedule-v2.json')})
    atomic(OUT / 'v7-numeric-gate.done.json', {'status': 'pass', 'anchor_results_sha256': sha(OUT / 'anchor-results-v2-numeric-gate.json'),
                                  'anchor_count': 4, 'candidate_outcome_count': 0})
    status({'stage': 'V7_ANCHORS_PASS', 'completed_processes': 4, 'total_processes': 48,
            'candidate_outcomes': 0, 'anchor_results_sha256': sha(OUT / 'anchor-results-v2-numeric-gate.json')})
    print('V7_ANCHORS_PASS', len(results), flush=True)


def offline_first_anchor():
    first = next(row for row in read(OUT / 'run-schedule-v2.json')['runs'] if row['kind'] == 'anchor')
    integrity = verify_saved_run(first)
    summary = verify_anchor(first)
    atomic(OUT / 'v7-first-anchor-offline-numeric-gate-v2.json', {
        'status': 'pass', 'state_id': first['state_id'], **integrity,
        'numeric_gate_sha256': sha(OUT / f"anchor-{first['state_id']}-numeric-gate-v2.json")})
    print('FIRST_ANCHOR_NUMERIC_GATE_PASS',
          [r['rms_m'] for r in read(OUT / f"anchor-{first['state_id']}-numeric-gate-v2.json")['dem_history_numeric_gate']['frames']],
          flush=True)


def load_trace(entry):
    with gzip.open(OUT / 'runs' / run_name(entry) / 'live-0.trace.json.gz', 'rt') as stream:
        return json.load(stream)


def duplicate_gate(entries):
    if len(entries) != 3 or any(entry['kind'] != 'duplicate' for entry in entries):
        raise ValueError('three fixed duplicate entries required')
    summaries = [read(OUT / 'runs' / run_name(e) / 'live-0.json') for e in entries]
    traces = [load_trace(e) for e in entries]
    if any(len(t) != 120 or any(row['terminal'] for row in t) for t in traces):
        raise ValueError('duplicate trace must have 120 nonterminal steps')
    if len({s['fixed_map_path_sha256'] for s in summaries}) != 1:
        raise ValueError('duplicate Path SHA differs')
    for key in ANCHOR_KEYS:
        if key == 'dem_history_sha256':
            continue
        if any(s['anchor'][key] != summaries[0]['anchor'][key] for s in summaries[1:]):
            raise ValueError('duplicate anchor state differs: ' + key)
    xy = []
    signal_rms = {key: [] for key in TRACE_KEYS if key != 'contact_force'}
    first = None
    exact = ('follower_command', 'command', 'actor_input', 'actor_latent',
             'actor_action', 'critic_observation', 'critic_value', 'reward',
             'contact_force', 'logical_sim_time_ns', 'tf_stamp_ns',
             'follower_command_stamp_ns')
    frame_diagnostics = []
    for other in traces[1:]:
        for k, (base, test) in enumerate(zip(traces[0], other)):
            a = np.asarray(post_root(base)[:2], dtype=float) - np.asarray(traces[0][0]['root_state'][:2], dtype=float)
            b = np.asarray(post_root(test)[:2], dtype=float) - np.asarray(other[0]['root_state'][:2], dtype=float)
            xy.extend((a - b).tolist())
            for key in signal_rms:
                if key in base and key in test:
                    x, y = np.asarray(base[key], dtype=float), np.asarray(test[key], dtype=float)
                    if x.shape == y.shape:
                        signal_rms[key].extend((x - y).ravel().tolist())
            if first is None:
                for signal in exact:
                    if base[signal] != test[signal]:
                        first = {'step': k, 'signal': signal}
                        break
            da = np.asarray(base['dem']['transformed'], dtype=np.float32)
            db = np.asarray(test['dem']['transformed'], dtype=np.float32)
            if da.shape != db.shape or da.shape != (2, 20, 20) or not np.isfinite(da).all() or not np.isfinite(db).all():
                raise ValueError('duplicate DEM shape/finite mismatch')
            va, vb = da[1] > .5, db[1] > .5
            common = va & vb
            if not common.any():
                raise ValueError('duplicate DEM lacks common valid cells')
            rms = float(np.sqrt(np.mean((da[0][common].astype(float) - db[0][common].astype(float)) ** 2)))
            iou = float(common.sum() / (va | vb).sum())
            if rms > .03 or iou < .95:
                raise ValueError(f'duplicate DEM numeric gate failed at step {k}: {rms}, {iou}')
            frame_diagnostics.append({'step': k, 'rms_m': rms, 'mask_iou': iou,
                                      'raw_sha_match': hashlib.sha256(da.tobytes()).hexdigest() == hashlib.sha256(db.tobytes()).hexdigest()})
    xy_rms = math.sqrt(float(np.mean(np.square(xy))))
    limit = yaml.safe_load(CONFIG.read_text())['duplicate_xy_rms_limit_m']
    if limit != .0105:
        raise ValueError('frozen duplicate XY RMS threshold changed')
    if xy_rms > limit or first is not None:
        raise ValueError(f'duplicate gate failed: XY RMS {xy_rms}, first signal {first}')
    result = {'status': 'pass', 'state_id': entries[0]['state_id'],
              'xy_rms_m': xy_rms, 'xy_rms_limit_m': limit,
              'first_exact_signal_difference': first,
              'signal_rms': {k: math.sqrt(float(np.mean(np.square(v)))) if v else None
                             for k, v in signal_rms.items()},
              'dem_numeric': {'max_rms_m': max(d['rms_m'] for d in frame_diagnostics),
                              'min_mask_iou': min(d['mask_iou'] for d in frame_diagnostics),
                              'raw_sha_match_count': sum(d['raw_sha_match'] for d in frame_diagnostics),
                              'comparison_count': len(frame_diagnostics)},
              'complete_marker_sha256': [sha(OUT / 'runs' / run_name(e) / 'run-complete.json') for e in entries]}
    atomic(OUT / f"duplicate-gate-{entries[0]['state_id']}.json", result)
    return result


def live():
    if read(OUT / 'v7-numeric-gate.done.json')['status'] != 'pass':
        raise ValueError('all four anchors must pass before live replay')
    schedule = read(OUT / 'run-schedule-v2.json')['runs']
    live_entries = [e for e in schedule if e['kind'] != 'anchor']
    if len(live_entries) != 44:
        raise ValueError('frozen live schedule count differs')
    storage_protocol = read(STORAGE_PROTOCOL)
    if (storage_protocol['storage_cap_bytes'] != 800_000_000
            or storage_protocol['bootstrap_total_bytes'] != read(OUT / 'runtime-forecast-v2.json')['storage_estimate_bytes']
            or storage_protocol['schedule_run_count'] != len(schedule)
            or storage_protocol['run_control_steps'] != 120):
        raise ValueError('v2 storage protocol differs from frozen limits')
    for anchor in (e for e in schedule if e['kind'] == 'anchor'):
        verify_saved_run(anchor)
        verify_anchor(anchor)
    failed = OUT / 'runs' / 'v2-train-000-nominal-00'
    first = live_entries[0]
    if first['order'] != 1 or first['run_id'] != failed.name or first['kind'] != 'duplicate':
        raise ValueError('fixed first live run changed')
    previous = read(failed / 'final-status.json')
    failure = read(failed / 'resource-stop.failure.json')
    if ((failed / 'run-complete.json').exists() or previous['stage'] != 'FAILED'
            or failure['reason'] != 'probe output reached or is projected above storage cap'
            or failure['heartbeat']['resource']['output_bytes'] >= 800_000_000
            or any((OUT / 'runs' / e['run_id']).exists() for e in live_entries[1:])):
        raise ValueError('attempt 0 does not match approved measurement-invalid case')
    atomic(OUT / 'v8-attempt-0-measurement-invalid.json', {
        'classification': 'measurement-invalid / monitor protocol error',
        'excluded_from_duplicate_and_outcome_analysis': True,
        'logical_run_id': first['run_id'], 'attempt': 0,
        'replacement_run_name': run_name(first),
        'frozen_entry_sha256': hashlib.sha256(json.dumps(first, sort_keys=True).encode()).hexdigest(),
        'started_sha256': sha(failed / 'started.json'),
        'final_status_sha256': sha(failed / 'final-status.json'),
        'resource_stop_sha256': sha(failed / 'resource-stop.failure.json'),
        'storage_protocol_sha256': sha(STORAGE_PROTOCOL),
        'schedule_sha256': sha(OUT / 'run-schedule-v2.json')})
    setup()
    started = OUT / 'v2-numeric-continuation-start.json'
    if not started.exists():
        atomic(started, {'start_unix': (OUT / 'v7-first-anchor-offline-numeric-gate.json').stat().st_mtime,
                         'schedule_sha256': sha(OUT / 'run-schedule-v2.json'), 'wall_limit_sec': 4 * 3600})
    clock = read(started)
    if clock['schedule_sha256'] != sha(OUT / 'run-schedule-v2.json') or clock['wall_limit_sec'] != 4 * 3600:
        raise ValueError('wall-clock or schedule marker differs')
    durations = []
    completed = 0
    with Heartbeat('V8_LIVE_RUNNING', 48) as beat:
        for state_index in range(4):
            group = [e for e in live_entries if e['state_index'] == state_index]
            if [e['kind'] for e in group] != ['duplicate'] * 3 + ['candidate'] * 8:
                raise ValueError('frozen state run order differs')
            anchor_entry = next(e for e in schedule if e['kind'] == 'anchor' and e['state_index'] == state_index)
            verify_saved_run(anchor_entry)
            verify_anchor(anchor_entry)
            for entry in group:
                elapsed = time.time() - clock['start_unix']
                remaining = 44 - completed
                median = float(np.median(durations)) if durations else 185.
                forecast = elapsed + remaining * median
                if elapsed >= 3 * 3600:
                    atomic(OUT / 'three-hour-forecast-numeric-gate.json',
                           {'elapsed_sec': elapsed, 'completed_runs': completed,
                            'median_run_sec': median, 'predicted_total_sec': forecast,
                            'limit_sec': 4 * 3600})
                if elapsed >= 4 * 3600 or forecast > 4 * 3600:
                    raise TimeoutError('four-hour wall forecast; no new run started')
                if bytes_in_root() > 800_000_000:
                    raise RuntimeError('0.8 GB storage cap reached')
                beat.update(run_id=run_name(entry), current_state=entry['state_id'],
                            goal_category=entry['goal_id'], seed=entry['seed'],
                            completed_processes=4 + completed, candidate_outcomes=sum(
                                (OUT / 'runs' / e['run_id'] / 'run-complete.json').exists()
                                for e in live_entries if e['kind'] == 'candidate'),
                            predicted_total_sec=forecast,
                            logical_run_id=entry['run_id'])
                folder = OUT / 'runs' / run_name(entry)
                if folder.exists():
                    verify_saved_run(entry)
                else:
                    begin = time.monotonic()
                    replay.run_one(entry, run_name(entry), timeout_sec=min(900, 4 * 3600 - elapsed),
                        timing_telemetry=True,
                        timing_monitor_script=REPO / 'scripts/eat_nav_goal_subgoal_resource_monitor.py',
                        timing_monitor_config=TIMING / 'resource-thresholds.json',
                        timing_monitor_extra_args=('--v2-schedule', OUT / 'run-schedule-v2.json',
                                                   '--v2-forecast', OUT / 'runtime-forecast-v2.json',
                                                   '--v2-storage-protocol', STORAGE_PROTOCOL))
                    durations.append(time.monotonic() - begin)
                    verify_saved_run(entry)
                verify_anchor(entry)
                completed += 1
                if entry == group[2]:
                    duplicate_gate(group[:3])
                beat.update(completed_processes=4 + completed, run_id=None)
    atomic(OUT / 'v8-numeric-gate.done.json', {'status': 'pass', 'live_count': completed,
        'candidate_count': 32, 'duplicate_count': 12,
        'schedule_sha256': sha(OUT / 'run-schedule-v2.json'),
        'wall_elapsed_sec': time.time() - clock['start_unix']})
    status({'stage': 'V8_LIVE_PASS', 'completed_processes': 48,
            'candidate_outcomes': 32, 'wall_elapsed_sec': time.time() - clock['start_unix']})
    print('V8_LIVE_PASS', completed, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('offline-first-anchor', 'anchors', 'live'), required=True)
    args = parser.parse_args()
    try:
        if args.stage == 'offline-first-anchor':
            offline_first_anchor()
        elif args.stage == 'anchors':
            anchors()
        else:
            live()
    except BaseException as exc:
        status({'stage': 'v2_numeric_gate_stopped', 'stage_requested': args.stage,
                'error': repr(exc), 'traceback': traceback.format_exc(),
                'candidate_outcomes': sum((OUT / 'runs' / e['run_id'] / 'run-complete.json').exists()
                  for e in read(OUT / 'run-schedule-v2.json')['runs'] if e['kind'] == 'candidate')})
        raise
