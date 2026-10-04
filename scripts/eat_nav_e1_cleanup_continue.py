"""One-shot E1 continuation after the noncandidate lifecycle smoke passes."""
from __future__ import annotations

import json
from pathlib import Path
import time
import traceback

import eat_nav_e1_recovery_continue as recovery
import eat_nav_e1_run_one as replay
import eat_nav_e1_schedule as schedule
from eat_nav_e1_router_recovery import atomic, sha


ROOT = Path('/tmp/eatnav-e1-cleanup-fixed-20261004')
PRIOR = Path('/tmp/eatnav-e1-zenoh-recovery-20261004')


def preflight():
    smoke=json.loads((ROOT/'smoke/lifecycle-smoke-result.json').read_text())
    if smoke['status']!='pass' or smoke['fixture']!='readiness-002' or smoke['E1_candidate']:
        raise ValueError('noncandidate lifecycle smoke gate did not pass')
    recovery.configure_output_root(ROOT)
    replay.REMOTE_CODE=str(ROOT/'eat_nav_e1_sim.py')
    recovery.check_router(ROOT)
    readiness=json.loads((ROOT/'router-readiness.json').read_text())
    if readiness['gate']['status']!='pass':raise ValueError('external router readiness failed')
    ledger=json.loads((ROOT/'prior-artifact-sha-ledger.json').read_text())
    for record in ledger['records']:
        if sha(record['path'])!=record['sha256']:
            raise ValueError('prior artifact SHA changed '+record['path'])
    audit=schedule.freeze_sources()
    old=json.loads((PRIOR/'source-freeze.json').read_text())
    for field in ('E1_continuation_manifest_sha256','E1_gates_sha256',
                  'E1_safety_stops_sha256','RL_checkpoint_sha256',
                  'reference_config_sha256','container_frozen_sha256',
                  'map_path_sha256','formal_anchor_result_sha256'):
        if audit[field]!=old[field]:raise ValueError('frozen E1 input changed '+field)
    changed={'run_one','schedule','monitor'}
    for field,digest in old['source_binary_sha256'].items():
        if field not in changed and audit['source_binary_sha256'][field]!=digest:
            raise ValueError('non-monitor E1 source changed '+field)
    if (ROOT/'runs').exists() and any((ROOT/'runs').iterdir()):
        raise FileExistsError('new output root already has E1 attempts')
    return audit


def main():
    audit=preflight()
    if (ROOT/'cleanup-fixed-continuation-started.marker.json').exists() or (ROOT/'continuation-started.marker.json').exists():
        raise FileExistsError('continuation already started')
    recovery.install_transport_guard(ROOT)
    start=time.time_ns()
    atomic(ROOT/'cleanup-fixed-continuation-started.marker.json',{
        'start_wall_ns':start,'pending_valid_processes':18,
        'smoke_result_sha256':sha(ROOT/'smoke/lifecycle-smoke-result.json'),
        'attempt_0_1_excluded':True})
    atomic(ROOT/'continuation-started.marker.json',{
        'start_wall_ns':start,'pending_processes':18,
        'cleanup_fixed_marker_sha256':sha(ROOT/'cleanup-fixed-continuation-started.marker.json')})
    try:
        schedule.run()
        counts=schedule.process_counts(ROOT)
        if counts['protocol_valid_process_count']!=18:
            raise ValueError('18 protocol-valid processes required')
        atomic(ROOT/'cleanup-fixed-final-decision.json',{'status':'cleanup_protocol_recovered_e1_complete',
            **counts,'continuation_elapsed_sec':(time.time_ns()-start)/1e9,
            'source_freeze_sha256':sha(ROOT/'source-freeze.json')})
    except BaseException as exc:
        counts=schedule.process_counts(ROOT)
        status=('attempt_2_measurement_invalid_stop' if counts['protocol_valid_process_count']==0
                else 'cleanup_protocol_recovered_e1_duplicate_gate_fail' if 'duplicate gate' in repr(exc)
                else 'cleanup_protocol_recovered_e1_runtime_stop' if isinstance(exc,(TimeoutError,RuntimeError))
                else 'cleanup_protocol_recovered_e1_incomplete')
        atomic(ROOT/'cleanup-fixed-final-decision.json',{'status':status,'error':repr(exc),
            'traceback':traceback.format_exc(),**counts,
            'continuation_elapsed_sec':(time.time_ns()-start)/1e9})
        raise


if __name__=='__main__':main()
