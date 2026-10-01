"""Separate two-hour continuation of the frozen v2 schedule after audit recovery."""
from __future__ import annotations

import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import eat_nav_goal_subgoal_v2_run as v

LIMIT_SEC = 7200
CAP_BYTES = 800_000_000
START = v.OUT / 'v2-continuation-start.json'
REPORT = v.OUT / 'v2-continuation-preflight.json'


def complete_path(entry):
    folder = v.OUT / 'runs' / v.run_name(entry)
    regular = folder / 'run-complete.json'
    recovery = folder / 'recovered_complete_after_monitor_cleanup_race.json'
    if regular.exists() and recovery.exists():
        raise ValueError('ambiguous completion marker: ' + entry['run_id'])
    return regular if regular.exists() else recovery if recovery.exists() else None


def remote_run_sizes():
    raw = subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'sh', '-c',
                                   'du -sb ' + str(v.OUT / 'runs') + '/*'], text=True, stderr=subprocess.STDOUT)
    return {Path(line.split('\t', 1)[1]).name: int(line.split('\t', 1)[0])
            for line in raw.splitlines()}


def sizes():
    host = v.bytes_in_root()
    remote = int(subprocess.check_output(['docker', 'exec', 'isaac-lab-onix', 'du', '-sb', str(v.OUT)],
                                         text=True).split()[0])
    return host, remote


def measured(schedule):
    durations = []
    physical = []
    remotes = remote_run_sizes()
    for entry in schedule:
        marker = complete_path(entry)
        if marker is None:
            continue
        folder = marker.parent
        if entry['kind'] != 'anchor':
            durations.append(float(v.read(folder / 'final-status.json')['elapsed_sec']))
            physical.append(sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
                            + remotes[v.run_name(entry)])
    if not durations or not physical:
        raise ValueError('no completed live measurements')
    return max(durations), max(physical), len(durations)


def preflight(schedule):
    if v.sha(v.OUT / 'run-schedule-v2.json') != v.read(v.OUT / 'v4.done.json')['schedule_sha256']:
        raise ValueError('frozen schedule SHA drift')
    if v.read(v.OUT / 'v7-numeric-gate.done.json')['status'] != 'pass':
        raise ValueError('anchor numeric gate incomplete')
    recovery = schedule[21]
    if recovery['run_id'] != 'v2-train-001-near_left_lateral-seed-1002':
        raise ValueError('recovery identity drift')
    if complete_path(recovery).name != 'recovered_complete_after_monitor_cleanup_race.json':
        raise ValueError('recovery marker missing')
    for entry in schedule:
        if complete_path(entry):
            v.verify_saved_run(entry)
    remaining = [entry for entry in schedule if complete_path(entry) is None]
    initial_orders = [22, 23, 25, 26, 27, 28, 29, 30, 31, 32,
                      33, 34, 35, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47]
    if [entry['order'] for entry in remaining] != (initial_orders[-len(remaining):] if remaining else []):
        raise ValueError('remaining frozen run order changed')
    longest, largest, measured_count = measured(schedule)
    host, remote = sizes()
    prediction = int(math.ceil((len(remaining) * longest + 90) * 1.2))
    storage_prediction = host + remote + len(remaining) * largest
    original = v.read(v.OUT / 'v2-numeric-continuation-start.json')
    result = {'classification': 'v2 continuation preflight', 'schedule_sha256': v.sha(v.OUT / 'run-schedule-v2.json'),
              'recovery_marker_sha256': v.sha(complete_path(recovery)),
              'completed_live_measurements': measured_count, 'remaining_orders': [e['order'] for e in remaining],
              'longest_completed_live_sec': longest, 'largest_completed_physical_run_bytes': largest,
              'predicted_remaining_with_20_percent_margin_sec': prediction,
              'additional_limit_sec': LIMIT_SEC, 'actual_host_bytes': host, 'actual_docker_bytes': remote,
              'predicted_final_bytes': storage_prediction, 'storage_cap_bytes': CAP_BYTES,
              'original_four_hour_gate_start_unix': original['start_unix'],
              'original_four_hour_gate_elapsed_sec_at_preflight': time.time() - original['start_unix'],
              'original_four_hour_gate_exceeded': time.time() - original['start_unix'] > 14400}
    if REPORT.exists():
        saved = v.read(REPORT)
        if saved['schedule_sha256'] != result['schedule_sha256'] or saved['recovery_marker_sha256'] != result['recovery_marker_sha256']:
            raise ValueError('continuation preflight provenance drift')
    else:
        v.atomic(REPORT, result)
    if prediction > LIMIT_SEC or storage_prediction > CAP_BYTES or host + remote > CAP_BYTES:
        raise RuntimeError('continuation timing or storage preflight gate failed')
    return remaining


def main():
    schedule = v.read(v.OUT / 'run-schedule-v2.json')['runs']
    if len(schedule) != 48 or [e['order'] for e in schedule] != list(range(48)):
        raise ValueError('frozen schedule malformed')
    remaining = preflight(schedule)
    if not START.exists():
        v.atomic(START, {'start_unix': time.time(), 'limit_sec': LIMIT_SEC,
                         'schedule_sha256': v.sha(v.OUT / 'run-schedule-v2.json'),
                         'preflight_sha256': v.sha(REPORT)})
    start = v.read(START)
    if start['schedule_sha256'] != v.sha(v.OUT / 'run-schedule-v2.json') or start['limit_sec'] != LIMIT_SEC:
        raise ValueError('continuation start marker drift')
    v.setup()
    for entry in schedule:
        marker = complete_path(entry)
        if marker is not None and entry['kind'] == 'duplicate':
            v.verify_saved_run(entry)
    for index in range(4):
        group = [e for e in schedule if e['state_index'] == index and e['kind'] == 'duplicate']
        if len(group) != 3:
            raise ValueError('duplicate group malformed')
        if all(complete_path(e) for e in group):
            v.duplicate_gate(group)
    for entry in remaining:
        if complete_path(entry):
            v.verify_saved_run(entry)
            continue
        elapsed = time.time() - start['start_unix']
        pending = [e for e in schedule if complete_path(e) is None]
        longest, largest, count = measured(schedule)
        host, remote = sizes()
        predicted_end = elapsed + (len(pending) * longest + 30) * 1.2
        predicted_bytes = host + remote + len(pending) * largest
        if elapsed >= LIMIT_SEC or predicted_end > LIMIT_SEC:
            raise TimeoutError('additional two-hour gate or 20%-margin forecast exceeded')
        if host + remote > CAP_BYTES or predicted_bytes > CAP_BYTES:
            raise RuntimeError('0.8 GB storage gate exceeded')
        v.status({'stage': 'V2_CONTINUATION_RUNNING', 'run_id': entry['run_id'],
                  'order': entry['order'], 'remaining': len(pending), 'elapsed_sec': elapsed,
                  'predicted_end_sec': predicted_end, 'predicted_final_bytes': predicted_bytes})
        v.replay.run_one(entry, v.run_name(entry), timeout_sec=min(900, LIMIT_SEC - elapsed),
            timing_telemetry=True,
            timing_monitor_script=v.REPO / 'scripts/eat_nav_goal_subgoal_resource_monitor.py',
            timing_monitor_config=v.TIMING / 'resource-thresholds.json',
            timing_monitor_extra_args=('--v2-schedule', v.OUT / 'run-schedule-v2.json',
                                       '--v2-forecast', v.OUT / 'runtime-forecast-v2.json',
                                       '--v2-storage-protocol', v.STORAGE_PROTOCOL),
            monitor_cleanup_protocol=True)
        v.verify_saved_run(entry)
        v.verify_anchor(entry)
        if entry['kind'] == 'duplicate' and entry == [e for e in schedule
                if e['state_index'] == entry['state_index'] and e['kind'] == 'duplicate'][-1]:
            v.duplicate_gate([e for e in schedule if e['state_index'] == entry['state_index']
                              and e['kind'] == 'duplicate'])
    if any(complete_path(e) is None for e in schedule):
        raise ValueError('schedule incomplete after continuation')
    v.atomic(v.OUT / 'v2-continuation.done.json', {'status': 'pass', 'processes': 48,
        'recovered_processes': 1, 'candidate_outcomes': 32,
        'additional_elapsed_sec': time.time() - start['start_unix'],
        'original_four_hour_gate_exceeded': True,
        'schedule_sha256': v.sha(v.OUT / 'run-schedule-v2.json')})
    v.status({'stage': 'V2_CONTINUATION_PASS', 'completed_processes': 48,
              'candidate_outcomes': 32, 'additional_elapsed_sec': time.time() - start['start_unix']})


if __name__ == '__main__':
    try:
        main()
    except BaseException as exc:
        v.status({'stage': 'V2_CONTINUATION_STOPPED', 'error': repr(exc),
                  'traceback': traceback.format_exc(),
                  'additional_elapsed_sec': time.time() - v.read(START)['start_unix'] if START.exists() else None})
        raise
