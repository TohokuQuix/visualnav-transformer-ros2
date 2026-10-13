"""Demo reset semantics, separate from the checkpoint's training environment."""

from types import SimpleNamespace

import pytest

from visualnav_transformer.eat_nav.demo_contract import sensor_stamp_valid
from visualnav_transformer.eat_nav.demo_reset_policy import (
    DemoSensorClock,
    assert_finite_sim_state,
    configure_demo_reset_policy,
)


def test_only_tilt_can_reset_and_fold_mode_is_zero_after_each_reset():
    tilt = object()
    cfg = SimpleNamespace(
        terminations=SimpleNamespace(time_out=object(), non_finite_state=object(), tilt_over=tilt),
        events=SimpleNamespace(reset_flipper_fold_mode=SimpleNamespace(
            params={'modes': (0., 1., -1.), 'probabilities': (1., 1., 1.)})),
    )
    configure_demo_reset_policy(cfg)
    assert cfg.terminations.tilt_over is tilt
    assert cfg.terminations.time_out is None
    assert cfg.terminations.non_finite_state is None
    assert cfg.events.reset_flipper_fold_mode.params == {
        'modes': (0.,), 'probabilities': (1.,)}


def test_unexpected_termination_term_fails_closed():
    cfg = SimpleNamespace(
        terminations=SimpleNamespace(time_out=object(), non_finite_state=object(),
                                     tilt_over=object(), base_contact=object()),
        events=SimpleNamespace(reset_flipper_fold_mode=SimpleNamespace(params={})),
    )
    with pytest.raises(ValueError, match='unexpected'):
        configure_demo_reset_policy(cfg)


def test_sensor_clock_survives_timeout_boundary_and_fall_reset():
    clock = DemoSensorClock(step_ns=50_000_000)
    assert clock.global_sensor_stamp(39_750_000_000) == 39_750_000_000
    # No timeout reset at the old 800-step boundary: sensor time continues.
    assert clock.global_sensor_stamp(40_000_000_000) == 40_000_000_000
    clock.after_fall(completed_step=924)
    assert clock.global_sensor_stamp(0) == 46_250_000_000
    assert sensor_stamp_valid(46_250_000_000, clock.global_sensor_stamp(0))
    assert sensor_stamp_valid(46_450_000_000, clock.global_sensor_stamp(0))
    assert not sensor_stamp_valid(46_600_000_000, clock.global_sensor_stamp(0))
    assert clock.global_sensor_stamp(250_000_000) == 46_500_000_000
    clock.after_fall(completed_step=1000)
    assert clock.global_sensor_stamp(0) == 50_050_000_000


def test_sensor_clock_rejects_invalid_or_nonmonotonic_episode_origin():
    clock = DemoSensorClock(step_ns=50_000_000)
    with pytest.raises(ValueError, match='sensor'):
        clock.global_sensor_stamp(-1)
    clock.after_fall(completed_step=10)
    with pytest.raises(ValueError, match='reset'):
        clock.after_fall(completed_step=9)


def test_nonfinite_robot_state_is_fatal_instead_of_reset():
    import torch

    robot = SimpleNamespace(data=SimpleNamespace(
        root_state_w=torch.zeros((1, 13)), joint_pos=torch.zeros((1, 4)),
        joint_vel=torch.zeros((1, 4))))
    assert_finite_sim_state(robot)
    robot.data.joint_vel[0, 2] = float('nan')
    with pytest.raises(ValueError, match='joint_vel'):
        assert_finite_sim_state(robot)
