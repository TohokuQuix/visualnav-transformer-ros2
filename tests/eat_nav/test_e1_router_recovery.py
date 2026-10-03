import pytest

from eat_nav_e1_router_recovery import verify_readiness, recovery_clock_start


def sample(clock_ns, subscribers=None):
    return {
        'clock_now_ns': clock_ns,
        'clock_publishers': 1,
        'use_sim_time': True,
        'status_counts': [3] * 8,
        'command_sequence': [0] * 8,
        'callback_count': [0] * 8,
        'rejected': [0] * 8,
        'protocol_errors': [],
        'follower_alive': [True] * 8,
        'path_subscribers': subscribers or [1] * 8,
        'tf_ready': [True] * 8,
    }


def test_readiness_requires_stable_eight_subscribers_and_advancing_clock():
    result = verify_readiness([sample(100), sample(150), sample(200)])
    assert result['status'] == 'pass'
    assert result['total_path_subscribers'] == 8
    with pytest.raises(ValueError, match='subscriber'):
        verify_readiness([sample(100), sample(150, [0] + [1]*7), sample(200)])
    with pytest.raises(ValueError, match='clock'):
        verify_readiness([sample(100), sample(100), sample(100)])


def test_readiness_rejects_pre_path_command_and_unready_tf():
    rows = [sample(100), sample(150), sample(200)]
    rows[1]['command_sequence'][3] = 1
    with pytest.raises(ValueError, match='command'):
        verify_readiness(rows)
    rows = [sample(100), sample(150), sample(200)]
    rows[2]['tf_ready'][4] = False
    with pytest.raises(ValueError, match='TF'):
        verify_readiness(rows)


def test_recovery_clock_rejects_attempt_zero_or_existing_marker(tmp_path):
    old = tmp_path/'attempt0'
    old.mkdir()
    (old/'failure-0.marker.json').write_text('{}')
    new = tmp_path/'recovery'
    new.mkdir()
    marker = recovery_clock_start(new, old, pending=18, readiness_status='pass')
    assert marker['pending_valid_processes'] == 18
    assert (new/'recovery-continuation-started.marker.json').exists()
    assert (new/'continuation-started.marker.json').exists()
    with pytest.raises(FileExistsError):
        recovery_clock_start(new, old, pending=18, readiness_status='pass')


def test_recovery_runner_uses_new_output_root_without_rewriting_old(monkeypatch, tmp_path):
    import eat_nav_e1_schedule as schedule
    import eat_nav_e1_run_one as replay
    from eat_nav_e1_recovery_continue import configure_output_root

    old = schedule.ROOT
    new = tmp_path/'new'
    try:
        configure_output_root(new)
        assert schedule.ROOT == new
        assert replay.ROOT == replay.DOC == new
        assert replay.REMOTE_ROOT == str(new/'runs')
        assert replay.REMOTE_CODE == str(old/'eat_nav_e1_sim.py')
    finally:
        configure_output_root(old)
