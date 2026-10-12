"""Synthetic gates for router, communication, and GPU terminal evidence."""
import copy

import pytest

from eat_nav_p1_g3_runtime_evidence_v4 import (
    check_router_series, check_communication, classify_gpu_bracket,
)


def router_sample(time_ns=0):
    return {
        'sample_started_wall_ns': time_ns,
        'sample_finished_wall_ns': time_ns + 10_000_000,
        'observer_heartbeat_seq': time_ns // 500_000_000,
        'pid': 6026, 'start_tick': 306994,
        'launcher_pid': 6019, 'launcher_start_tick': 306981,
        'router_parent_pid': 6019, 'router_session_id': 500,
        'router_cgroup': 'router.scope', 'router_cmdline': 'rmw_zenohd',
        'launcher_parent_pid': 1, 'launcher_session_id': 500,
        'launcher_cgroup': 'router.scope', 'launcher_cmdline': 'ros2 run rmw_zenoh_cpp rmw_zenohd',
        'zid': 'abc123', 'endpoints': ['127.0.0.1:7447'],
        'reachable_endpoints': ['127.0.0.1:7447'],
        'listener_owned': True, 'stdout_available': True,
        'stderr_available': True, 'misses': [],
    }


def test_router_continuous_series_passes():
    rows = [router_sample(i * 500_000_000) for i in range(5)]
    assert check_router_series(rows, run_start_ns=600_000_000,
                               run_end_ns=1_600_000_000)['pass']


@pytest.mark.parametrize('mutation,reason', [
    (lambda r: r.update(pid=None), 'router_missing'),
    (lambda r: r.update(start_tick=333), 'router_pid_reused'),
    (lambda r: r.update(launcher_pid=None), 'launcher_missing'),
    (lambda r: r.update(zid='changed'), 'zid_changed'),
    (lambda r: r.update(reachable_endpoints=[]), 'endpoint_unreachable'),
    (lambda r: r.update(listener_owned=False), 'listener_missing'),
    (lambda r: r.update(stdout_available=False), 'log_unavailable'),
    (lambda r: r.update(router_cgroup='other.scope'), 'router_process_metadata_changed'),
])
def test_router_identity_faults_fail_closed(mutation, reason):
    rows = [router_sample(i * 500_000_000) for i in range(5)]
    mutation(rows[2])
    with pytest.raises(ValueError, match=reason):
        check_router_series(rows, run_start_ns=600_000_000,
                            run_end_ns=1_600_000_000)


def test_router_heartbeat_gap_is_not_assumed_healthy():
    rows = [router_sample(i * 500_000_000) for i in (0, 1, 4, 5)]
    with pytest.raises(ValueError, match='observer_heartbeat_gap'):
        check_router_series(rows, run_start_ns=600_000_000,
                            run_end_ns=2_100_000_000)


def test_deleted_path_is_valid_only_with_fd_copy():
    rows = [router_sample(i * 500_000_000) for i in range(5)]
    rows[2]['stdout_path_deleted'] = True
    rows[2]['stdout_available'] = True
    assert check_router_series(rows, run_start_ns=600_000_000,
                               run_end_ns=1_600_000_000)['pass']
    rows[2]['stdout_available'] = False
    with pytest.raises(ValueError, match='log_unavailable'):
        check_router_series(rows, run_start_ns=600_000_000,
                            run_end_ns=1_600_000_000)


def gpu_sample(start=200, end=300, failures=()):
    return {'sample_started_wall_ns': start, 'sample_finished_wall_ns': end,
            'failures': list(failures), 'device_cap_pass': True,
            'attribution_gap_pass': True}


def completion():
    return {'written_wall_ns': 150, 'sha_valid': True,
            'native_exit': 0, 'owner_identity_valid': True}


def test_gpu_post_completion_pid_disappearance_is_terminal_only():
    result = classify_gpu_bracket(gpu_sample(failures=(
        'owned_start_tick_mismatch', 'simulator_disappeared_in_bracket')),
        completion())
    assert result['status'] == 'terminal_expected_exit'


def test_gpu_live_failure_stays_failed():
    with pytest.raises(ValueError, match='active_gpu_attribution_failure'):
        classify_gpu_bracket(gpu_sample(50, 100,
            ('simulator_disappeared_in_bracket',)), completion())


def test_gpu_boundary_overlap_is_inconclusive():
    assert classify_gpu_bracket(gpu_sample(100, 200,
        ('simulator_disappeared_in_bracket',)), completion())['status'] == 'inconclusive_overlap'


@pytest.mark.parametrize('change', [
    {'native_exit': None}, {'sha_valid': False}, {'owner_identity_valid': False},
])
def test_gpu_terminal_requires_valid_completion(change):
    marker = {**completion(), **change}
    with pytest.raises(ValueError, match='completion_unverified'):
        classify_gpu_bracket(gpu_sample(failures=('owned_start_tick_mismatch',)), marker)


def comm_row(step=0):
    stamp = 12_000_000_000 + step * 50_000_000
    return {'control_step': step, 'timing_protocol': {
        'events': ['clock', 'tf', 'command', 'bridge_acceptance', 'physics'],
        'stamp_ns': stamp, 'clock_ns': stamp, 'tf_ns': stamp,
        'command_ns': stamp, 'bridge_accept_ns': stamp,
        'physics_input_ns': stamp, 'command_accepted_steady_ns': 10 + step,
        'physics_begin_steady_ns': 11 + step, 'reset_called': False,
        'terminated': False, 'truncated': False},
        'bridge_acceptance': {'accepted': True, 'stamp_ns': stamp,
                              'tf_stamp_ns': stamp, 'reject_reason': None},
        'follower_diagnostic': {'tf_success': True, 'tf_stamp_ns': stamp,
                                'command_stamp_ns': stamp, 'tf_request': 'exact_stamp'},
        'terminal': None}


def test_per_step_communication_passes_and_detects_missing_or_bad_step():
    rows = [comm_row(i) for i in range(3)]
    assert check_communication([rows, copy.deepcopy(rows)])['pass']
    bad = copy.deepcopy(rows); bad[1]['bridge_acceptance']['accepted'] = False
    with pytest.raises(ValueError, match='command_not_accepted'):
        check_communication([bad])
    with pytest.raises(ValueError, match='control_step_gap'):
        check_communication([[rows[0], rows[2]]])
    bad = copy.deepcopy(rows); bad[2]['follower_diagnostic']['tf_success'] = False
    with pytest.raises(ValueError, match='tf_failure'):
        check_communication([bad])
