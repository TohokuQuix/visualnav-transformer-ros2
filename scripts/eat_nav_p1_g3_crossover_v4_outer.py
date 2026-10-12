"""V4 outer supervisor: external-router evidence and terminal GPU classification.

This script only owns the observer and the staged inner host. The Zenoh
router remains external. The checked-in preparation manifest is deliberately
not runtime-authorized; a later approved attempt needs a fresh manifest/root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from eat_nav_p1_g1_lifecycle import verify_seal
from eat_nav_p1_g3_runtime_evidence_v4 import (
    check_communication, check_router_series, classify_gpu_bracket,
)


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path: Path, value: dict) -> None:
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def require_authorization(manifest: dict, process_id: str) -> None:
    if manifest.get('runtime_authorized') is not True:
        raise ValueError('runtime_not_authorized')
    if process_id not in manifest.get('authorized_process_ids', []):
        raise ValueError('batch_not_authorized')
    blocker = manifest.get('resume_blocker', {})
    if process_id == blocker.get('first_remaining_process') and \
            blocker.get('predecessor_status') == 'protocol_invalid':
        raise ValueError('predecessor_protocol_invalid')


def verify_outer_seal(root: Path) -> dict:
    root = Path(root)
    seal = json.loads((root/'outer-seal.json').read_text())
    ledger_path = root/'sha-ledger.json'
    if sha(ledger_path) != seal['sha_ledger_sha256']:
        raise ValueError('outer_ledger_sha_mismatch')
    ledger = json.loads(ledger_path.read_text())
    actual = {str(path.relative_to(root)) for path in root.rglob('*') if path.is_file()
              and path.name not in ('sha-ledger.json', 'outer-seal.json')}
    if actual != set(ledger):
        raise ValueError('outer_file_set_changed')
    if any(sha(root/name) != expected for name, expected in ledger.items()):
        raise ValueError('outer_file_sha_mismatch')
    return {'valid': True, 'file_count': len(ledger), 'stage': seal['stage']}


def verify_router_log_copies(rows: list[dict], observer_root: Path) -> bool:
    if not rows:
        raise ValueError('router_series_missing')
    for name in ('stdout', 'stderr'):
        copy = Path(observer_root)/f'router-{name}-copy.bin'
        if not copy.is_file() or sha(copy) != rows[-1].get(f'{name}_sha256'):
            raise ValueError('router_log_copy_sha_mismatch:' + name)
    return True


def assess_protocol(router_rows: list[dict], traces: list[list[dict]],
                    gpu_samples: list[dict], completion: dict,
                    *, run_start_ns: int, run_end_ns: int) -> dict:
    """Lifecycle and per-step transport are independent required evidence."""
    if (completion.get('sha_valid') is not True or
            completion.get('owner_identity_valid') is not True or
            completion.get('native_exit') != 0 or
            not isinstance(completion.get('written_wall_ns'), int)):
        raise ValueError('completion_unverified')
    router = check_router_series(router_rows, run_start_ns=run_start_ns,
                                 run_end_ns=run_end_ns)
    communication = check_communication(traces)
    if not gpu_samples:
        raise ValueError('gpu_series_missing')
    gpu = [classify_gpu_bracket(sample, completion) for sample in gpu_samples]
    if any(row['status'] == 'inconclusive_overlap' for row in gpu):
        raise ValueError('gpu_terminal_overlap_inconclusive')
    return {'pass': True, 'router': router, 'communication': communication,
            'gpu_sample_count': len(gpu),
            'gpu_terminal_classifications': [row for row in gpu if row['status'].startswith('terminal')]}


def _batch_evidence(batch: Path, manifest: dict) -> tuple[list[list[dict]], list[dict], dict]:
    seal = verify_seal(batch)
    if seal['stage'] != 'G3_crossover_batch_protocol_valid':
        raise ValueError('inner_batch_seal_not_protocol_valid')
    status = json.loads((batch/'g3-host-status.json').read_text())
    if status['failure'] is not None or status['native_exit'] != 0 or not status['validation']:
        raise ValueError('inner_host_validation_incomplete')
    cleanup = json.loads((batch/'cleanup-evidence.json').read_text())
    if cleanup['owned_residual'] or cleanup['bridge_residual'] or \
            not all(cleanup['writer_state'].values()):
        raise ValueError('inner_cleanup_incomplete')
    output = batch/'output'
    marker_path = output/'complete.json'
    marker = json.loads(marker_path.read_text())
    if marker['trace_sha256'] != [sha(output/f'trace-slot-{env}.jsonl') for env in range(4)]:
        raise ValueError('trace_sha_mismatch')
    traces = [[json.loads(line) for line in
               (output/f'trace-slot-{env}.jsonl').open()] for env in range(4)]
    cap = manifest['resource_cap']
    samples = []
    for line in (batch/'resource-series.jsonl').open():
        row = json.loads(line)
        gpu = row['gpu_attribution']
        samples.append({'sample_started_wall_ns': row['sample_started_wall_ns'],
                        'sample_finished_wall_ns': row['sample_finished_wall_ns'],
                        'failures': gpu['failures'],
                        'device_cap_pass': row['gpu_device_total_mib'] <= cap['gpu_device_total_mib']
                            and row['host_available_bytes'] >= cap['host_available_min_bytes']
                            and row['container_anon_plus_shmem_bytes'] <=
                            cap['container_anon_plus_shmem_max_bytes']
                            and row['persistent_root_bytes'] <=
                            cap['active_storage_max_bytes_per_process']
                            and row['elapsed_s'] <= cap['process_timeout_s'],
                        'attribution_gap_pass': gpu['bracket_gap_bytes'] <=
                            cap['gpu_attribution_gap_mib'] * 1048576})
    registered = json.loads((batch/'coord/registered-owner.json').read_text())
    completion = {'written_wall_ns': marker_path.stat().st_mtime_ns,
                  'sha_valid': True, 'native_exit': status['native_exit'],
                  'owner_identity_valid': status['owned_simulator'] == registered}
    return traces, samples, completion


def _stage_checked(manifest: dict, batch: Path, repo: Path,
                   attempt_manifest: Path) -> None:
    paths = {
        'g3_simulator_source_sha256': batch/'source.py',
        'g3_host_source_sha256': batch/'host-runner.py',
        'assignment_sha256': batch/'input/g3-assignment.json',
        'router_observer_source_sha256': repo/'scripts/eat_nav_p1_g3_router_observer_v4.py',
        'runtime_evidence_source_sha256': repo/'scripts/eat_nav_p1_g3_runtime_evidence_v4.py',
        'v4_outer_host_source_sha256': Path(__file__),
    }
    for key, path in paths.items():
        if sha(path) != manifest[key]:
            raise ValueError('v4_source_or_input_sha_mismatch:' + key)
    if sha(batch/'input/g3-contract.json') != sha(attempt_manifest):
        raise ValueError('v4_attempt_manifest_sha_mismatch')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-root', type=Path, required=True)
    parser.add_argument('--attempt-manifest', type=Path, required=True)
    parser.add_argument('--outer-root', type=Path, required=True)
    parser.add_argument('--router-log-dir', type=Path, required=True)
    parser.add_argument('--remote', required=True)
    parser.add_argument('--anchor-id', required=True)
    parser.add_argument('--batch-index', type=int, required=True)
    parser.add_argument('--execution-index', type=int, required=True)
    args = parser.parse_args()
    batch, outer = args.batch_root.resolve(), args.outer_root.resolve()
    manifest = json.loads(args.attempt_manifest.read_text())
    require_authorization(manifest, batch.name)
    repo = Path(__file__).resolve().parents[1]
    _stage_checked(manifest, batch, repo, args.attempt_manifest)
    if outer.exists():
        raise FileExistsError('fresh v4 outer root required')
    outer.mkdir(parents=True)
    write_new(outer/'launch-identity.json', {
        'attempt_manifest_sha256': sha(args.attempt_manifest),
        'batch_id': batch.name,
        'observer_source_sha256': manifest['router_observer_source_sha256'],
        'inner_host_source_sha256': manifest['g3_host_source_sha256'],
        'outer_host_source_sha256': manifest['v4_outer_host_source_sha256']})
    observer_source = outer/'router-observer-source.py'
    shutil.copy2(repo/'scripts/eat_nav_p1_g3_router_observer_v4.py', observer_source)
    if sha(observer_source) != manifest['router_observer_source_sha256']:
        raise ValueError('staged_router_observer_sha_mismatch')
    stop = outer/'observer-stop.json'
    observer_log = (outer/'observer-console.log').open('xb')
    observer = subprocess.Popen([sys.executable,
        str(observer_source),
        '--router-dir', str(args.router_log_dir), '--output-dir', str(outer/'router-observer'),
        '--stop-file', str(stop)], stdout=observer_log, stderr=subprocess.STDOUT)
    host = None
    host_exit = observer_exit = None
    started_ns = ended_ns = None
    run_error = None
    try:
        deadline = time.monotonic() + 10
        ready = outer/'router-observer/observer-ready.json'
        while not ready.is_file():
            if observer.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError('router_observer_not_ready')
            time.sleep(0.1)
        started_ns = time.time_ns()
        host_log = (outer/'host-console.log').open('xb')
        try:
            host = subprocess.Popen([sys.executable, str(batch/'host-runner.py'),
                '--root', str(batch), '--remote', args.remote,
                '--router-log-dir', str(args.router_log_dir),
                '--anchor-id', args.anchor_id, '--batch-index', str(args.batch_index),
                '--execution-index', str(args.execution_index)],
                stdout=host_log, stderr=subprocess.STDOUT)
            try:
                host_exit = host.wait(timeout=660)
            except subprocess.TimeoutExpired:
                host.send_signal(signal.SIGINT)
                try:
                    host_exit = host.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    host.terminate()
                    host_exit = host.wait(timeout=10)
                raise TimeoutError('v4_inner_host_timeout')
        finally:
            host_log.close()
        ended_ns = time.time_ns()
    except BaseException as exc:
        run_error = type(exc).__name__ + ':' + str(exc)
        ended_ns = time.time_ns()
    finally:
        write_new(stop, {'stop_requested_wall_ns': time.time_ns()})
        try:
            observer_exit = observer.wait(timeout=10)
        except subprocess.TimeoutExpired:
            observer.terminate()
            observer_exit = observer.wait(timeout=5)
        observer_log.close()
    write_new(outer/'runtime-window.json', {
        'run_start_wall_ns': started_ns, 'run_end_wall_ns': ended_ns,
        'inner_host_exit': host_exit, 'observer_exit': observer_exit,
        'prevalidation_error': run_error})
    try:
        if run_error is not None:
            raise ValueError(run_error)
        if host is None or host_exit != 0:
            raise ValueError('inner_host_failed')
        if observer_exit != 0 or not (outer/'router-observer/observer-complete.json').is_file():
            raise ValueError('router_observer_failed')
        rows = [json.loads(line) for line in
                (outer/'router-observer/router-series.jsonl').open()]
        verify_router_log_copies(rows, outer/'router-observer')
        traces, samples, completion = _batch_evidence(batch, manifest)
        decision = assess_protocol(rows, traces, samples, completion,
                                   run_start_ns=started_ns, run_end_ns=ended_ns)
        decision['decision'] = 'protocol_valid'
    except Exception as exc:
        decision = {'decision': 'protocol_invalid_or_inconclusive',
                    'failure': type(exc).__name__ + ':' + str(exc)}
    write_new(outer/'v4-decision.json', decision)
    write_new(outer/'sha-ledger.json', {str(path.relative_to(outer)): sha(path)
        for path in sorted(outer.rglob('*')) if path.is_file() and
        path.name != 'sha-ledger.json'})
    write_new(outer/'outer-seal.json', {
        'stage': decision['decision'],
        'sha_ledger_sha256': sha(outer/'sha-ledger.json'),
        'inner_host_seal_sha256': sha(batch/'host-seal.json')
            if (batch/'host-seal.json').is_file() else None})
    verify_outer_seal(outer)
    if decision['decision'] != 'protocol_valid':
        raise RuntimeError(decision['failure'])
    print('V4_BATCH_PASS', batch)


if __name__ == '__main__':
    main()
