"""ROS-free demo command preprocessing follows the bridge's turn escape rule."""

import pytest

from visualnav_transformer.eat_nav.demo_turn_escape import TurnStuckEscape


STEP_NS = 50_000_000


def test_low_yaw_response_triggers_then_alternates_forward_and_back():
    escape = TurnStuckEscape()
    for step in range(40):
        command, state = escape.apply((0.0, 0.0, 0.4), 0.1, step * STEP_NS)
        assert command == (0.0, 0.0, 0.4)
        assert not state['active']
    command, state = escape.apply((0.0, 0.0, 0.4), 0.1, 40 * STEP_NS)
    assert command == pytest.approx((0.2, 0.0, 0.4))
    assert state['active'] and state['direction'] == 1.0
    command, state = escape.apply((0.0, 0.0, 0.4), 0.1, 60 * STEP_NS)
    assert command == pytest.approx((-0.2, 0.0, 0.4))
    assert state['direction'] == -1.0
    command, state = escape.apply((0.0, 0.0, 0.4), 0.1, 80 * STEP_NS)
    assert command == pytest.approx((0.2, 0.0, 0.4))
    assert state['direction'] == 1.0


def test_yaw_rate_error_scales_offset_and_preserves_follower_command():
    escape = TurnStuckEscape()
    for step in range(41):
        command, state = escape.apply((0.08, 0.0, -0.3), -0.05, step * STEP_NS)
    assert command == pytest.approx((0.28, 0.0, -0.3))
    assert state['offset_mps'] == pytest.approx(0.2)
    assert state['follower_command'] == (0.08, 0.0, -0.3)


def test_response_recovers_and_zero_command_clears_state():
    escape = TurnStuckEscape()
    for step in range(41):
        escape.apply((0.0, 0.0, 0.4), 0.0, step * STEP_NS)
    command, state = escape.apply((0.0, 0.0, 0.4), 0.3, 41 * STEP_NS)
    assert command == (0.0, 0.0, 0.4)
    assert not state['active']
    command, state = escape.apply((0.0, 0.0, 0.0), 0.0, 42 * STEP_NS)
    assert command == (0.0, 0.0, 0.0)
    assert not state['active']
    command, state = escape.apply((0.0, 0.0, 0.4), 0.0, 43 * STEP_NS)
    assert command == (0.0, 0.0, 0.4)
    assert not state['active']


def test_no_imu_or_invalid_time_fails_closed_and_reset_clears_history():
    escape = TurnStuckEscape()
    with pytest.raises(ValueError, match='yaw'):
        escape.apply((0.0, 0.0, 0.4), float('nan'), 0)
    for step in range(41):
        escape.apply((0.0, 0.0, 0.4), 0.0, step * STEP_NS)
    escape.reset()
    command, state = escape.apply((0.0, 0.0, 0.4), 0.0, 41 * STEP_NS)
    assert command == (0.0, 0.0, 0.4)
    assert not state['active']
    with pytest.raises(ValueError, match='stamp'):
        escape.apply((0.0, 0.0, 0.4), 0.0, 41 * STEP_NS)
