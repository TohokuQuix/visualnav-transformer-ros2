"""Resume-safe fresh-process exact-stamp pilot with per-state duplicate gates."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time

from visualnav_transformer.eat_nav.candidate_coverage_pilot import map_paths_from_anchor
from visualnav_transformer.eat_nav.command_preview_long_horizon import duplicate_120
import eat_nav_exact_stamp_run as exact
import eat_nav_sim_clock_run as old

REPO = Path('/tmp/eatnav-vnt-candidate-coverage')
DOC = REPO / 'docs/quality-identifiability-gate/candidate-coverage/evaluator-readiness-pilot'
ROOT = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-pilot')
HOST_INPUT = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-inputs')
CONTAINER = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness')
SIM = Path('/tmp/eatnav-candidate-coverage/evaluator-readiness-code/sim.py')
ENV = ('PYTHONPATH=/tmp/eatnav-candidate-coverage/evaluator-readiness-code/src:'
       '/tmp/eatnav-train:/tmp/eatnav-train/src:/tmp/eatnav-train/upstream-src:'
       '/tmp/eatnav-extended/isaac-source')


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
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(data, indent=2, sort_keys=True) + '\n'
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(value)
    tmp.replace(path)


def docker(*args):
    return subprocess.run(['docker', *map(str, args)], check=True, capture_output=True, text=True)


def folder(entry):
    return ROOT / 'runs' / entry['run_id']


def run_valid(entry):
    mark = folder(entry) / 'run-complete.json'
    if not mark.exists():
        return False
    data = read(mark)
    if data['identity'] != entry:
        raise ValueError('run identity marker mismatch')
    for name, digest in data['output_sha256'].items():
        if sha(folder(entry) / name) != digest:
            raise ValueError('run output SHA mismatch: ' + name)
    return True


def map_paths(entry, anchor):
    index = entry['state_index']
    host = HOST_INPUT / f'state-{index:03d}'
    table = read(host / 'candidate-table.json')
    paths = {}
    for row in table:
        candidate = row['candidate_id']
        path = host / f'{candidate}-base-path.json'
        if sha(path) != row['base_path_sha256']:
            raise ValueError('candidate base Path SHA mismatch')
        paths[candidate] = read(path)
    mapped = map_paths_from_anchor(anchor, paths)
    for name, points in mapped.items():
        target = host / f'{name}-map-path.json'
        if target.exists():
            import numpy as np
            old_points = np.asarray(read(target), dtype=float)
            new_points = np.asarray(points, dtype=float)
            if old_points.shape != new_points.shape or np.max(np.abs(old_points-new_points)) > 1e-6:
                raise ValueError('map Path changed across anchor replay')
        if not target.exists():
            atomic(target, points)
        docker('cp', target, f'isaac-lab-onix:{CONTAINER}/inputs/state-{index:03d}/{target.name}')
    atomic(ROOT / 'map-path-freeze' / f"{entry['state_id']}.json",
           {'anchor_sha256': sha(folder(entry) / 'anchor-0.json'),
            'map_path_sha256': {name: sha(host / f'{name}-map-path.json') for name in mapped}})


def execute(entry, deadline):
    target = folder(entry)
    target.mkdir(parents=True, exist_ok=True)
    if run_valid(entry):
        return read(target / 'run-complete.json')
    if any(target.iterdir()):
        prior_log = target / 'runner.log'
        prior = prior_log.read_text(errors='replace') if prior_log.exists() else ''
        preinit = (len(list(target.iterdir())) == 1 and
                   "can't open file '/tmp/eatnav-candidate-coverage/evaluator-readiness/evaluator-readiness-code/sim.py'" in prior and
                   'RECOVERY_STEP' not in prior and not (target / 'startup-attempt-0.log').exists())
        if preinit:
            prior_log.replace(target / 'startup-attempt-0.log')
            atomic(target / 'retry-history.json', {'reason': 'pre-simulation script path typo',
                   'previous_log_sha256': sha(target / 'startup-attempt-0.log'),
                   'step_zero_reached': False, 'retry_count': 1})
        else:
            raise RuntimeError('partial run requires inspection; no automatic measured-run retry: ' + str(target))
    index = entry['state_index']
    state = f'state-{index:03d}'
    candidate = entry['candidate_id']
    stage = 'anchor' if entry['kind'] == 'anchor' else 'live'
    remote = CONTAINER / 'runs' / entry['run_id']
    docker('exec', 'isaac-lab-onix', 'mkdir', '-p', remote)
    if stage == 'live':
        import socket
        with socket.create_connection(('127.0.0.1', 47819), timeout=3):
            pass
        freeze = read(ROOT / 'map-path-freeze' / f"{entry['state_id']}.json")
        expected = freeze['map_path_sha256'][candidate]
        if sha(HOST_INPUT / state / f'{candidate}-map-path.json') != expected:
            raise ValueError('candidate map Path SHA mismatch')
    command = ['docker', 'exec', '-e', ENV, 'isaac-lab-onix', '/isaac-sim/python.sh', str(SIM),
               '--headless', '--stage', stage, '--repeat-id', '0', '--logical-t0-ns', str(entry['logical_t0_ns']),
               '--control-steps', '120', '--manifest', '/tmp/eatnav-extended/output/manifest.json',
               '--prefix', str(CONTAINER / 'inputs' / state / 'prefix-actions.npz'),
               '--other-env-actions', str(CONTAINER / 'inputs' / state / 'suffix-actions-120.npz'),
               '--logged-anchor', str(CONTAINER / 'inputs' / state / 'logged-anchor.json'),
               '--state', str(CONTAINER / 'inputs/selected-states.json'),
               '--base-path', str(CONTAINER / 'inputs' / state / f'{candidate}-base-path.json'),
               '--fixed-map-path', str(CONTAINER / 'inputs' / state / f'{candidate}-map-path.json'),
               '--config', str(CONTAINER / 'inputs' / state / 'config-120.yaml'),
               '--reference-config', '/tmp/eatnav-candidate-coverage/baseline-worktree-8e8f4fd/docs/quality-identifiability-gate/phase-c/reference-eat.yaml',
               '--rl-checkpoint', '/workspace/isaaclab_onix/logs/rsl_rl/onix_rough/2026-07-24_18-20-16/model_10000.pt',
               '--output', str(remote)]
    memory = []
    stop = threading.Event()
    def watch():
        while not stop.wait(2):
            try:
                gpu = int(subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                                  text=True).splitlines()[0])
                rss = subprocess.check_output(['docker', 'stats', '--no-stream', '--format', '{{.MemUsage}}', 'isaac-lab-onix'],
                                              text=True).strip()
                memory.append((gpu, rss))
            except (subprocess.SubprocessError, ValueError, IndexError):
                pass
    thread = threading.Thread(target=watch, daemon=True)
    thread.start()
    started = time.monotonic()
    code = None
    try:
        with (target / 'runner.log').open('x') as log:
            log.write('BEGIN ' + json.dumps(entry, sort_keys=True) + '\n'); log.flush()
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=max(1, min(900, deadline - time.time())))
            code = process.returncode
            log.write(f'EXIT {code} elapsed_sec={time.monotonic()-started:.3f}\n')
    finally:
        stop.set(); thread.join(timeout=3)
    if code:
        raise RuntimeError(f'{entry["run_id"]}: simulator exit {code}; inspect {target / "runner.log"}')
    names = ['anchor-0.json', 'anchor-0.marker.json'] if stage == 'anchor' else [
        'live-0.json', 'live-0.trace.json.gz', 'live-0.clock.json.gz', 'live-0.marker.json']
    for name in names:
        docker('cp', f'isaac-lab-onix:{remote}/{name}', target / name)
    result = read(target / names[0])
    if result['state']['state_id'] != entry['state_id'] or result['requested_control_steps'] != 120:
        raise ValueError('run identity or duration mismatch')
    if stage == 'live':
        expected = read(ROOT / 'map-path-freeze' / f"{entry['state_id']}.json")['map_path_sha256'][candidate]
        if result['fixed_map_path_sha256'] != expected:
            raise ValueError('candidate Path SHA mismatch')
        exact.inspect_run(target, 120, entry['logical_t0_ns'])
        with gzip.open(target / 'live-0.trace.json.gz', 'rt') as stream:
            trace = json.load(stream)
        if len(trace) != 120 or any(row['terminal'] for row in trace):
            raise ValueError('unexpected reset or short rollout')
    peak = max((x[0] for x in memory), default=None)
    if peak is not None and peak > 12000:
        raise RuntimeError(f'unexpected GPU memory {peak} MiB')
    mark = {'identity': entry, 'output_sha256': {name: sha(target / name) for name in names},
            'elapsed_sec': time.monotonic() - started, 'gpu_peak_mib': peak,
            'docker_memory_samples': [x[1] for x in memory[-10:]]}
    atomic(target / 'run-complete.json', mark)
    return mark


def duplicate_gate(state_id, entries):
    paths = [folder(x) for x in entries]
    summaries = [read(path / 'live-0.json') for path in paths]
    traces = [old.trace(path / 'live-0.trace.json.gz') for path in paths]
    result = duplicate_120(summaries, traces, .0105)
    result['state_id'] = state_id
    result['run_sha256'] = [sha(path / 'run-complete.json') for path in paths]
    atomic(ROOT / 'duplicate-gates' / f'{state_id}.json', result)
    if not result['pass'] or not result['dem_sha_equal']:
        raise RuntimeError(f'{state_id}: duplicate gate failed: {result}')
    if any(value not in (0, None) for value in result['signal_rms'].values()):
        raise RuntimeError(f'{state_id}: nonzero duplicate signal variation')
    return result


def compress_closed_logs():
    log_dir = ROOT / 'follower-logs'
    files = list(log_dir.glob('reset-*-branch-*.log'))
    if not files:
        return []
    current = max(int(path.name.split('-')[1]) for path in files)
    compressed = []
    for path in files:
        if int(path.name.split('-')[1]) >= current:
            continue
        dest = path.with_suffix('.log.gz')
        if dest.exists():
            raise ValueError('closed-log gzip destination already exists')
        raw = path.read_bytes()
        raw_sha = hashlib.sha256(raw).hexdigest()
        temp = dest.with_suffix('.gz.tmp')
        with gzip.open(temp, 'wb', compresslevel=6) as stream:
            stream.write(raw)
        with gzip.open(temp, 'rb') as stream:
            if hashlib.sha256(stream.read()).hexdigest() != raw_sha:
                raise ValueError('closed-log gzip verification failed')
        temp.replace(dest)
        path.unlink()
        compressed.append({'path': str(dest), 'raw_sha256': raw_sha,
                           'compressed_sha256': sha(dest)})
    return compressed


def size_bytes(root):
    return sum(path.stat().st_size for path in root.rglob('*') if path.is_file())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--smoke-only', action='store_true')
    args = p.parse_args()
    config = read(DOC / 'pilot-config.json')
    schedule = read(DOC / 'schedule.json')['runs']
    selection = read(DOC / 'selection.json')
    frozen = read(DOC / 'input-freeze.json')
    if len(schedule) != 119 or sha(HOST_INPUT / 'input-freeze.json') != sha(DOC / 'input-freeze.json'):
        raise ValueError('schedule or input-freeze SHA mismatch')
    if not (HOST_INPUT / 'selected-states.json').exists():
        raise ValueError('selected states missing')
    ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    budget = ROOT / 'budget.json'
    if budget.exists():
        began = read(budget)['started_unix']
    else:
        began = started
        atomic(budget, {'started_unix': began, 'deadline_unix': began + config['wall_clock_cap_sec'],
                        'schedule_sha256': sha(DOC / 'schedule.json'),
                        'input_freeze_sha256': sha(DOC / 'input-freeze.json')})
    deadline = began + config['wall_clock_cap_sec']
    durations = []
    completed = 0
    try:
        for i, entry in enumerate(schedule):
            remaining = len(schedule) - i
            observed = max(durations[-4:], default=150.)
            forecast = remaining * observed
            if time.time() + forecast > deadline:
                raise TimeoutError(f'forecast exceeds seven-hour cap at run {i}: remaining={forecast:.1f}s')
            if size_bytes(ROOT) + remaining * 6_000_000 > config['storage_cap_bytes']:
                raise RuntimeError('forecast exceeds one-gigabyte storage cap')
            if entry['kind'] in ('nomad', 'bank'):
                gate = ROOT / 'duplicate-gates' / f"{entry['state_id']}.json"
                if not gate.exists() or not read(gate)['pass']:
                    raise RuntimeError('candidate before duplicate gate')
            atomic(ROOT / 'status.json', {'stage': 'RUNNING', 'completed_processes': completed,
                   'completed_rollouts': sum(run_valid(x) for x in schedule[:i] if x['kind'] != 'anchor'),
                   'total_rollouts': 112, 'next_run': entry['run_id'],
                   'elapsed_sec': time.time() - began, 'forecast_remaining_sec': forecast,
                   'size_bytes': size_bytes(ROOT), 'deadline_unix': deadline})
            result = execute(entry, deadline)
            durations.append(result['elapsed_sec'])
            completed += 1
            compressed = compress_closed_logs()
            if compressed:
                ledger = ROOT / 'compressed-log-sha.json'
                existing = read(ledger) if ledger.exists() else []
                atomic(ledger, existing + compressed)
            if entry['kind'] == 'anchor':
                map_paths(entry, read(folder(entry) / 'anchor-0.json'))
            if entry['kind'] == 'nominal' and entry['repeat'] == 2:
                duplicate_gate(entry['state_id'], schedule[i-2:i+1])
            atomic(ROOT / 'markers' / f"{entry['run_id']}.done.json",
                   {'run_id': entry['run_id'], 'run_complete_sha256': sha(folder(entry) / 'run-complete.json'),
                    'schedule_sha256': sha(DOC / 'schedule.json')})
            print('READINESS_RUN_COMPLETE', entry['run_id'], entry['terrain_type'], entry['kind'],
                  round(result['elapsed_sec'], 2), flush=True)
            if args.smoke_only and i == 1:
                atomic(ROOT / 'status.json', {'stage': 'SMOKE_COMPLETE', 'completed_processes': completed,
                       'elapsed_sec': time.time() - began, 'forecast_remaining_sec': (119-completed)*max(durations),
                       'size_bytes': size_bytes(ROOT)})
                return
        atomic(ROOT / 'status.json', {'stage': 'RAW_COMPLETE', 'completed_processes': 119,
               'completed_rollouts': 112, 'elapsed_sec': time.time() - began,
               'size_bytes': size_bytes(ROOT)})
    except BaseException as exc:
        atomic(ROOT / 'status.json', {'stage': 'STOPPED', 'reason': repr(exc),
               'completed_processes': completed, 'next_run': schedule[completed]['run_id'] if completed < 119 else None,
               'elapsed_sec': time.time() - began, 'size_bytes': size_bytes(ROOT)})
        raise


if __name__ == '__main__':
    main()
