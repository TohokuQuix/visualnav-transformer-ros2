"""Resume the frozen E1 schedule in a separate root after external router readiness.

The router is inspected, never started or stopped here. The failed attempt-0 root
remains untouched; the same frozen simulator source is used in the new root.
"""
from __future__ import annotations

import json
from pathlib import Path
import socket
import traceback

import eat_nav_e1_run_one as replay
import eat_nav_e1_schedule as schedule
from eat_nav_e1_router_recovery import atomic, recovery_clock_start, sha

OLD = Path('/tmp/eatnav-e1-feasibility-20261003')
RECOVERY = Path('/tmp/eatnav-e1-zenoh-recovery-20261004')


def configure_output_root(root):
    root = Path(root)
    schedule.ROOT = root
    schedule.REMOTE = str(root)
    replay.DOC = root
    replay.ROOT = root
    replay.INPUT = root/'inputs'
    replay.REMOTE_ROOT = str(root/'runs')
    replay.REMOTE_INPUT_ROOT = root
    # E1's frozen simulator source remains the original SHA-identified copy.
    # The existing per-PID monitor resolves this exact source path.
    replay.REMOTE_CODE = str(OLD/'eat_nav_e1_sim.py')


def preflight(root):
    root = Path(root)
    readiness = json.loads((root/'router-readiness.json').read_text())
    if readiness['gate']['status'] != 'pass':
        raise ValueError('router readiness gate did not pass')
    old_marker = OLD/'runs/g0e-train-000-nominal_duplicate-00/failure-0.marker.json'
    preserved = json.loads((root/'attempt-0-preservation.json').read_text())
    if preserved['failure_marker_sha256'] != sha(old_marker):
        raise ValueError('attempt-0 failure marker changed')
    router = json.loads((root/'router-launch-provenance.json').read_text())
    proc = Path('/proc')/str(router['pid'])
    if not proc.is_dir() or proc.joinpath('exe').resolve() != Path('/opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd'):
        raise RuntimeError('dedicated external router identity lost')
    with socket.create_connection(('127.0.0.1', 7447), timeout=2):
        pass
    log = (root/'router.log').read_text()
    if 'Started Zenoh router' not in log or any(x in log.lower() for x in ('error', 'failed', 'panic')):
        raise RuntimeError('external router log has no clean startup')
    audit = schedule.freeze_sources()
    prior = json.loads((OLD/'source-freeze.json').read_text())
    for key in ('E1_continuation_manifest_sha256', 'E1_gates_sha256',
                'E1_safety_stops_sha256', 'source_binary_sha256'):
        if audit[key] != prior[key]:
            raise ValueError('frozen E1 input/source changed: ' + key)
    if sha(OLD/'eat_nav_e1_sim.py') != audit['source_binary_sha256']['simulator']:
        raise ValueError('frozen Docker simulator source changed')
    return audit


def main():
    configure_output_root(RECOVERY)
    try:
        audit = preflight(RECOVERY)
        recovery_clock_start(RECOVERY,
            OLD/'runs/g0e-train-000-nominal_duplicate-00', pending=18,
            readiness_status='pass')
        schedule.run()
        atomic(RECOVERY/'recovery-final-status.json', {'status': 'E1_COMPLETE',
               'new_valid_processes': 18, 'router_owned_by_runner': False,
               'source_freeze_sha256': sha(RECOVERY/'source-freeze.json')})
    except BaseException as exc:
        atomic(RECOVERY/'recovery-stop.json', {'status': 'E1_STOPPED',
               'error': repr(exc), 'traceback': traceback.format_exc(),
               'router_owned_by_runner': False,
               'completed_valid_processes': len(list((RECOVERY/'runs').glob('*/run-complete.json')))})
        raise


if __name__ == '__main__':
    main()
