import hashlib
import json
from pathlib import Path

import pytest

from eat_nav_e1_lifecycle import (LifecycleMachine, read_complete_json,
    marker_matches_run, protocol_valid_count)


def sample(now, **changes):
    value = dict(now_ns=now, simulator_tree=True, host_alive=True,
        host_start_ticks=70, step_count=10, completion=None, resource_stop=None,
        heartbeat_stale=False, protocol_error=False, router_alive=True,
        bridge_alive=True, followers_alive=True, nonfinite=False, final=None)
    value.update(changes)
    return value


def completed(now, **changes):
    value = dict(simulator_tree=False, step_count=120,
        completion={'valid': True, 'trace_closed': True, 'trace_sha_match': True,
                    'step_count': 120, 'normal_completion': True})
    value.update(changes)
    return sample(now, **value)


def final(**changes):
    value = dict(docker_exit=0, bridge_exit=0, followers_exit_ok=True,
        run_complete_exists=True, output_sha_match=True, residual_count=0,
        zombie_count=0, cleanup_marker=True)
    value.update(changes)
    return value


def running():
    m = LifecycleMachine(expected_steps=120, grace_ns=30_000_000_000)
    assert m.observe(sample(0)).state == 'RUNNING'
    return m


def test_normal_finalization_waits_for_docker_exec_and_marker():
    m = running()
    assert m.observe(completed(1_000_000_000)).state == 'FINALIZING'
    assert m.observe(completed(3_000_000_000)).state == 'FINALIZING'
    assert m.observe(completed(4_000_000_000, final=final())).state == 'COMPLETE'


def test_router_must_remain_alive_during_finalization():
    m = running();m.observe(completed(1_000_000_000))
    assert m.observe(completed(2_000_000_000, router_alive=False)).reason == 'router_disconnected'


@pytest.mark.parametrize('bad,reason', [
    ({'step_count': 119, 'completion': None}, 'before_expected_steps'),
    ({'completion': None}, 'completion_marker_missing'),
    ({'completion': {'valid': True, 'trace_closed': True,
                     'trace_sha_match': False, 'step_count': 120,
                     'normal_completion': True}}, 'trace_sha'),
])
def test_pid_disappearance_requires_complete_valid_trace(bad, reason):
    m = running()
    assert m.observe(completed(1_000_000_000, **bad)).reason == reason


@pytest.mark.parametrize('bad,reason', [
    ({'docker_exit': 1}, 'docker_exit'),
    ({'output_sha_match': False}, 'output_sha'),
    ({'residual_count': 1}, 'residual'),
    ({'zombie_count': 1}, 'zombie'),
])
def test_finalization_rejects_failed_cleanup(bad, reason):
    m = running();m.observe(completed(1_000_000_000))
    assert m.observe(completed(2_000_000_000, final=final(**bad))).reason == reason


def test_missing_run_complete_fails_after_fixed_grace():
    m = running();m.observe(completed(1_000_000_000))
    assert m.observe(completed(30_000_000_000)).state == 'FINALIZING'
    assert m.observe(completed(31_000_000_001)).reason == 'finalization_grace'


@pytest.mark.parametrize('bad,reason', [
    ({'resource_stop': 'GPU_cap'}, 'GPU_cap'),
    ({'heartbeat_stale': True}, 'heartbeat_stall'),
    ({'router_alive': False}, 'router_disconnected'),
    ({'protocol_error': True}, 'protocol_error'),
])
def test_runtime_stops_do_not_enter_finalization(bad, reason):
    m = running()
    assert m.observe(sample(1_000_000_000, **bad)).reason == reason


def test_partial_json_is_not_a_completion_marker(tmp_path):
    path = tmp_path/'live-0.marker.json'
    path.write_text('{"outputs_sha256":')
    assert read_complete_json(path) is None


def test_stale_marker_cannot_be_used_for_another_run():
    marker = {'outputs_sha256': {'/tmp/old/runs/a/live-0.trace.json.gz': 'a'*64}}
    assert not marker_matches_run(marker, '/tmp/new/runs/b', 'b'*64)


def test_pid_reuse_is_detected_by_process_start_ticks():
    m = running()
    assert m.observe(sample(1_000_000_000, host_start_ticks=71)).reason == 'host_pid_reused'


def test_old_attempt_markers_do_not_count_as_protocol_valid(tmp_path):
    runs = tmp_path/'runs';(runs/'a').mkdir(parents=True)
    (runs/'a'/'run-complete.json').write_text('{}')
    (tmp_path/'attempt-0.marker.json').write_text('{}')
    (tmp_path/'attempt-1.marker.json').write_text('{}')
    assert protocol_valid_count(runs) == 0
    (runs/'a'/'protocol-valid.json').write_text('{"status":"valid","run_complete_sha256":"abc"}')
    assert protocol_valid_count(runs) == 1


def test_saved_attempt_one_passes_artifact_verifier_but_not_protocol_valid_count():
    from eat_nav_e1_monitor import finalization_status, verify_complete
    run = Path('/tmp/eatnav-e1-zenoh-recovery-20261004/runs/g0e-train-000-nominal_duplicate-00')
    assert finalization_status(run)['output_sha_match'] is True
    assert verify_complete(run) == hashlib.sha256((run/'run-complete.json').read_bytes()).hexdigest()
    assert protocol_valid_count(run.parent) == 0


def test_schedule_refuses_saved_attempt_one_without_protocol_marker(monkeypatch):
    import eat_nav_e1_schedule as schedule
    root = Path('/tmp/eatnav-e1-zenoh-recovery-20261004')
    row = json.loads((root/'continuation-manifest.json').read_text())['pending_rows'][0]
    monkeypatch.setattr(schedule, 'ROOT', root)
    with pytest.raises(ValueError, match='cleanup gate incomplete'):
        schedule.completed_rows([row])
