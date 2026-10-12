"""Pure v4 evidence gates; no router or simulator lifecycle side effects."""
from __future__ import annotations

from visualnav_transformer.eat_nav.p1_longhorizon_protocol import validate_step_order


ROUTER_SAMPLE_INTERVAL_NS = 500_000_000
ROUTER_MAX_SAMPLE_GAP_NS = 1_500_000_000
EXPECTED_EXIT_FLAGS = frozenset({
    'owned_start_tick_mismatch', 'simulator_disappeared_in_bracket',
    'owned_process_disappeared_in_bracket',
})


def check_router_series(rows: list[dict], *, run_start_ns: int,
                        run_end_ns: int) -> dict:
    """Reject identity, heartbeat, or listener gaps; sampling is not continuity proof."""
    if not rows or run_start_ns >= run_end_ns:
        raise ValueError('observer_series_missing')
    rows = sorted(rows, key=lambda row: row['sample_started_wall_ns'])
    if (rows[0]['sample_finished_wall_ns'] > run_start_ns or
            rows[-1]['sample_started_wall_ns'] < run_end_ns):
        raise ValueError('observer_run_window_uncovered')
    first = rows[0]
    required = ('pid', 'start_tick', 'launcher_pid', 'launcher_start_tick',
                'zid', 'endpoints', 'router_parent_pid', 'router_session_id',
                'router_cgroup', 'router_cmdline', 'launcher_session_id',
                'launcher_cgroup', 'launcher_cmdline')
    if any(first.get(key) in (None, '', []) for key in required):
        raise ValueError('router_identity_missing')
    for i, row in enumerate(rows):
        if row.get('pid') is None or row.get('start_tick') is None:
            raise ValueError('router_missing')
        if row.get('launcher_pid') is None or row.get('launcher_start_tick') is None:
            raise ValueError('launcher_missing')
        if (row['pid'], row['start_tick']) != (first['pid'], first['start_tick']):
            raise ValueError('router_pid_reused')
        if (row['launcher_pid'], row['launcher_start_tick']) != (
                first['launcher_pid'], first['launcher_start_tick']):
            raise ValueError('launcher_pid_reused')
        metadata = ('router_parent_pid', 'router_session_id', 'router_cgroup',
                    'router_cmdline', 'launcher_session_id', 'launcher_cgroup',
                    'launcher_cmdline')
        if row['router_parent_pid'] != row['launcher_pid'] or any(
                row.get(key) != first[key] for key in metadata):
            raise ValueError('router_process_metadata_changed')
        if row.get('zid') != first['zid']:
            raise ValueError('zid_changed')
        if row.get('endpoints') != first['endpoints']:
            raise ValueError('endpoints_changed')
        if set(row.get('reachable_endpoints', [])) != set(first['endpoints']):
            raise ValueError('endpoint_unreachable')
        if row.get('listener_owned') is not True:
            raise ValueError('listener_missing')
        if not row.get('stdout_available') or not row.get('stderr_available'):
            raise ValueError('log_unavailable')
        if row.get('misses'):
            raise ValueError('observer_sample_missing')
        if i:
            prev = rows[i - 1]
            if (row['observer_heartbeat_seq'] != prev['observer_heartbeat_seq'] + 1 or
                    row['sample_started_wall_ns'] - prev['sample_started_wall_ns'] >
                    ROUTER_MAX_SAMPLE_GAP_NS or
                    row['sample_started_wall_ns'] <= prev['sample_started_wall_ns']):
                raise ValueError('observer_heartbeat_gap')
    return {'pass': True, 'sample_count': len(rows),
            'sampling_guarantee': 'sampled_lifecycle_only_not_continuous_connectivity'}


def classify_gpu_bracket(sample: dict, completion: dict | None) -> dict:
    """Only post-completion process-exit flags may be waived in protocol v4."""
    start = sample.get('sample_started_wall_ns')
    end = sample.get('sample_finished_wall_ns')
    if not isinstance(start, int) or not isinstance(end, int) or start > end:
        raise ValueError('gpu_bracket_time_invalid')
    if not sample.get('device_cap_pass') or not sample.get('attribution_gap_pass'):
        raise ValueError('gpu_resource_or_gap_failure')
    failures = set(sample.get('failures', []))
    if not completion or not completion.get('sha_valid') or \
            not completion.get('owner_identity_valid') or completion.get('native_exit') != 0 or \
            not isinstance(completion.get('written_wall_ns'), int):
        if failures:
            raise ValueError('completion_unverified')
        return {'status': 'active_valid'}
    written = completion['written_wall_ns']
    if end < written:
        if failures:
            raise ValueError('active_gpu_attribution_failure')
        return {'status': 'active_valid'}
    if start < written <= end:
        return {'status': 'inconclusive_overlap' if failures else 'overlap_valid'}
    unexpected = failures - EXPECTED_EXIT_FLAGS
    if unexpected:
        raise ValueError('terminal_gpu_failure:' + ','.join(sorted(unexpected)))
    return {'status': 'terminal_expected_exit' if failures else 'terminal_valid',
            'waived_exit_flags': sorted(failures)}


def check_communication(traces: list[list[dict]]) -> dict:
    """Independent per-step transport/control evidence; cannot prove router uptime."""
    if not traces or any(not rows for rows in traces):
        raise ValueError('communication_trace_missing')
    expected_stamps = None
    for env, rows in enumerate(traces):
        stamps = []
        for step, row in enumerate(rows):
            if row.get('control_step') != step:
                raise ValueError('control_step_gap')
            timing = row.get('timing_protocol', {})
            validate_step_order(timing)
            stamp = timing['stamp_ns']
            if stamps and stamp != stamps[-1] + 50_000_000:
                raise ValueError('control_step_stamp_gap')
            stamps.append(stamp)
            accepted = row.get('bridge_acceptance', {})
            if accepted.get('accepted') is not True or accepted.get('reject_reason'):
                raise ValueError('command_not_accepted')
            if accepted.get('stamp_ns') != stamp or accepted.get('tf_stamp_ns') != stamp:
                raise ValueError('bridge_stamp_mismatch')
            follower = row.get('follower_diagnostic', {})
            if follower.get('tf_success') is not True:
                raise ValueError('tf_failure')
            if follower.get('tf_request') != 'exact_stamp' or any(
                    follower.get(key) != stamp for key in ('tf_stamp_ns', 'command_stamp_ns')):
                raise ValueError('follower_stamp_mismatch')
            if timing.get('reset_called') or timing.get('terminated') or timing.get('truncated') or \
                    row.get('terminal') is not None:
                raise ValueError('episode_interrupted')
        if expected_stamps is None:
            expected_stamps = stamps
        elif stamps != expected_stamps:
            raise ValueError('env_stamp_sequence_mismatch')
    return {'pass': True, 'env_count': len(traces), 'steps_per_env': len(expected_stamps),
            'meaning': 'per_step_control_delivery_not_router_lifecycle'}
