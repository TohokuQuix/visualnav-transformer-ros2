"""V4 outer evidence must gate old batch evidence without router ownership."""
import copy
import pytest

from eat_nav_p1_g3_crossover_v4_outer import (
    assess_protocol, require_authorization, verify_outer_seal,
    verify_router_log_copies,
)


def _row(step):
    stamp = 12_000_000_000 + 50_000_000 * step
    return {'control_step': step, 'timing_protocol': {
        'events': ['clock', 'tf', 'command', 'bridge_acceptance', 'physics'],
        'stamp_ns': stamp, 'clock_ns': stamp, 'tf_ns': stamp,
        'command_ns': stamp, 'bridge_accept_ns': stamp,
        'physics_input_ns': stamp, 'command_accepted_steady_ns': step,
        'physics_begin_steady_ns': step + 1, 'reset_called': False,
        'terminated': False, 'truncated': False},
        'bridge_acceptance': {'accepted': True, 'stamp_ns': stamp,
                              'tf_stamp_ns': stamp, 'reject_reason': None},
        'follower_diagnostic': {'tf_success': True, 'tf_stamp_ns': stamp,
                                'command_stamp_ns': stamp, 'tf_request': 'exact_stamp'},
        'terminal': None}


def _router_rows():
    return [{'sample_started_wall_ns': i * 500_000_000,
             'sample_finished_wall_ns': i * 500_000_000 + 1,
             'observer_heartbeat_seq': i,
             'pid': 11, 'start_tick': 100, 'launcher_pid': 10,
             'launcher_start_tick': 99, 'zid': 'abc',
             'router_parent_pid': 10, 'router_session_id': 500,
             'router_cgroup': 'router.scope', 'router_cmdline': 'rmw_zenohd',
             'launcher_session_id': 500, 'launcher_cgroup': 'router.scope',
             'launcher_cmdline': 'ros2 run rmw_zenoh_cpp rmw_zenohd',
             'endpoints': ['127.0.0.1:7447'],
             'reachable_endpoints': ['127.0.0.1:7447'],
             'listener_owned': True, 'stdout_available': True,
             'stderr_available': True, 'misses': []} for i in range(5)]


def test_outer_requires_fresh_explicit_runtime_authorization():
    with pytest.raises(ValueError, match='runtime_not_authorized'):
        require_authorization({'runtime_authorized': False}, 'batch-a')
    with pytest.raises(ValueError, match='batch_not_authorized'):
        require_authorization({'runtime_authorized': True,
                               'authorized_process_ids': ['batch-b']}, 'batch-a')
    with pytest.raises(ValueError, match='predecessor_protocol_invalid'):
        require_authorization({'runtime_authorized': True,
                               'authorized_process_ids': ['batch-a'],
                               'resume_blocker': {'first_remaining_process': 'batch-a',
                                                  'predecessor_status': 'protocol_invalid'}},
                              'batch-a')


def test_integrated_lifecycle_and_communication_both_required():
    traces = [[_row(i) for i in range(3)] for _ in range(4)]
    samples = [{'sample_started_wall_ns': 0, 'sample_finished_wall_ns': 100,
                'failures': [], 'device_cap_pass': True,
                'attribution_gap_pass': True},
               {'sample_started_wall_ns': 200, 'sample_finished_wall_ns': 300,
                'failures': ['owned_start_tick_mismatch'],
                'device_cap_pass': True, 'attribution_gap_pass': True}]
    result = assess_protocol(_router_rows(), traces, samples,
                             {'written_wall_ns': 150, 'sha_valid': True,
                              'owner_identity_valid': True, 'native_exit': 0},
                             run_start_ns=600_000_000, run_end_ns=1_600_000_000)
    assert result['pass'] and result['router']['pass'] and result['communication']['pass']
    bad = copy.deepcopy(traces); bad[0][1]['bridge_acceptance']['accepted'] = False
    with pytest.raises(ValueError, match='command_not_accepted'):
        assess_protocol(_router_rows(), bad, samples,
                        {'written_wall_ns': 150, 'sha_valid': True,
                         'owner_identity_valid': True, 'native_exit': 0},
                        run_start_ns=600_000_000, run_end_ns=1_600_000_000)
    with pytest.raises(ValueError, match='completion_unverified'):
        clean_samples = copy.deepcopy(samples)
        clean_samples[-1]['failures'] = []
        assess_protocol(_router_rows(), traces, clean_samples,
                        {'written_wall_ns': 150, 'sha_valid': True,
                         'owner_identity_valid': False, 'native_exit': 0},
                        run_start_ns=600_000_000, run_end_ns=1_600_000_000)


def test_outer_has_no_router_management_call():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'scripts/eat_nav_p1_g3_crossover_v4_outer.py').read_text()
    assert 'router-observer' in source
    assert 'observer-ready.json' in source
    assert 'check_router_series' in source
    assert 'check_communication' in source
    assert 'router.pid' not in source or 'os.kill' not in source


def test_outer_seal_detects_post_seal_write(tmp_path):
    import hashlib
    import json
    item = tmp_path / 'evidence.txt'
    item.write_text('sealed')
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    ledger = tmp_path / 'sha-ledger.json'
    ledger.write_text(json.dumps({'evidence.txt': digest(item)}))
    seal = tmp_path / 'outer-seal.json'
    seal.write_text(json.dumps({'sha_ledger_sha256': digest(ledger),
                                'stage': 'protocol_valid'}))
    assert verify_outer_seal(tmp_path)['valid']
    item.write_text('changed')
    with pytest.raises(ValueError, match='outer_file_sha_mismatch'):
        verify_outer_seal(tmp_path)


def test_router_log_copy_matches_final_observer_sha(tmp_path):
    import hashlib
    for name, content in [('stdout', b'router'), ('stderr', b'')]:
        (tmp_path/f'router-{name}-copy.bin').write_bytes(content)
    rows = [{'stdout_sha256': hashlib.sha256(b'router').hexdigest(),
             'stderr_sha256': hashlib.sha256(b'').hexdigest()}]
    assert verify_router_log_copies(rows, tmp_path)
    (tmp_path/'router-stdout-copy.bin').write_bytes(b'changed')
    with pytest.raises(ValueError, match='router_log_copy_sha_mismatch'):
        verify_router_log_copies(rows, tmp_path)
