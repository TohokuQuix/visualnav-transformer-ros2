"""Index and bootstrap contracts for the offline 40-transition audit."""
import pytest

from eat_nav_g40_teacher_review import return_at_horizon, validate_trace_index


def rows(count=41):
    return [
        {
            'critic_value': float(k * 10), 'reward': float(k + 1),
            'root_state': [float(k)], 'post_root_state': [float(k + 1)],
            'logical_sim_time_ns': 1000000000 + 50000000 * k,
            'terminal': None,
        }
        for k in range(count)
    ]


def test_40_rewards_and_pre_step_value_40_are_used():
    trace = rows()
    validate_trace_index(trace, 40)
    result = return_at_horizon(trace, 40, 0.5)
    assert result['reward_discounted'] == pytest.approx(sum((0.5 ** k) * (k + 1) for k in range(40)))
    assert result['bootstrap_value'] == 400.0
    assert result['bootstrap_index'] == 40
    assert result['reward_undiscounted'] == sum(range(1, 41))
    assert result['advantage'] == pytest.approx(result['g'] - 0.0)


def test_first_boundary_mismatch_fails_with_step():
    trace = rows()
    trace[1]['root_state'][0] += 0.1
    with pytest.raises(ValueError, match='boundary step 0'):
        validate_trace_index(trace, 40)


def test_physical_terminal_omits_bootstrap():
    trace = rows()
    trace[3]['terminal'] = {'terminated': True, 'truncated': False,
                            'termination_reason': ['tilt_over']}
    result = return_at_horizon(trace, 40, 0.5)
    assert result['observed_transitions'] == 4
    assert result['bootstrap_value'] is None
    assert result['g'] == pytest.approx(1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 4)


def test_timeout_requires_pre_reset_terminal_value_to_bootstrap():
    trace = rows()
    trace[3]['terminal'] = {'terminated': False, 'truncated': True,
                            'termination_reason': ['time_out']}
    result = return_at_horizon(trace, 40, 0.5)
    assert result['g'] is None
    assert result['status'] == 'timeout_missing_terminal_pre_reset_value'
    result = return_at_horizon(trace, 40, 0.5, terminal_pre_reset_value=20.)
    assert result['g'] == pytest.approx(1 + 0.5 * 2 + 0.25 * 3 + 0.125 * 4 + 0.5 ** 4 * 20)


def test_measurement_invalid_is_not_failure_reward():
    trace = rows()
    trace[3]['terminal'] = {'kind': 'measurement_invalid'}
    result = return_at_horizon(trace, 40, 0.5)
    assert result['g'] is None
    assert result['status'] == 'excluded_measurement_invalid'
