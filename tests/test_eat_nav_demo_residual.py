import math
import pytest

from visualnav_transformer.eat_nav.demo_residual import summarize_cancel, capture_physical_state, compare_step_fields


def test_cancel_summary_separates_path_length_net_displacement_and_time_bins():
    rows = [
        {'event': 'cancel', 'step': 2, 'stamp_ns': 100_000_000},
        {'event': 'step', 'step': 0, 'stamp_ns': 0, 'pose_pre': [0, 0, 0],
         'pose_post': [0, 0, 0], 'command': [1, 0, 0],
         'actor_input_command': [1, 0, 0], 'actor_action': [0]*6, 'done': False},
        {'event': 'step', 'step': 1, 'stamp_ns': 50_000_000, 'pose_pre': [0, 0, 0],
         'pose_post': [1, 0, 0], 'command': [0, 0, 0],
         'actor_input_command': [0, 0, 0], 'actor_action': [0]*6, 'done': False},
        {'event': 'step', 'step': 2, 'stamp_ns': 100_000_000, 'pose_pre': [1, 0, 0],
         'pose_post': [2, 0, .1], 'command': [0, 0, 0],
         'actor_input_command': [0, 0, 0], 'actor_action': [1]*6, 'done': False},
        {'event': 'step', 'step': 3, 'stamp_ns': 150_000_000, 'pose_pre': [2, 0, .1],
         'pose_post': [1, 0, .2], 'command': [0, 0, 0],
         'actor_input_command': [0, 0, 0], 'actor_action': [1]*6, 'done': False},
    ]
    result = summarize_cancel(rows)
    assert result['cancel_step'] == 2
    assert result['cumulative_xy_m'] == 2
    assert result['net_xy_m'] == 0
    assert result['max_xy_displacement_m'] == 1
    assert math.isclose(result['yaw_change_rad'], .2)
    assert result['segments_m']['0.0-0.5'] == 2
    assert result['post_cancel_command_nonzero_steps'] == 0
    assert result['post_cancel_actor_input_nonzero_steps'] == 0
    assert result['last_root_velocity_source'] == 'finite_difference_pose'


def test_cancel_summary_rejects_missing_or_discontinuous_step():
    rows = [
        {'event': 'cancel', 'step': 2, 'stamp_ns': 100_000_000},
        {'event': 'step', 'step': 2, 'stamp_ns': 100_000_000, 'pose_pre': [0, 0, 0],
         'pose_post': [1, 0, 0], 'command': [0, 0, 0],
         'actor_input_command': [0, 0, 0], 'actor_action': [0]*6, 'done': False},
        {'event': 'step', 'step': 4, 'stamp_ns': 200_000_000, 'pose_pre': [1, 0, 0],
         'pose_post': [2, 0, 0], 'command': [0, 0, 0],
         'actor_input_command': [0, 0, 0], 'actor_action': [0]*6, 'done': False},
    ]
    import pytest
    with pytest.raises(ValueError, match='noncontiguous'):
        summarize_cancel(rows)


def test_capture_physical_state_labels_actual_isaac_tensors():
    import torch
    from types import SimpleNamespace
    data = SimpleNamespace(
        root_lin_vel_w=torch.tensor([[.1, .2, .3]]),
        root_ang_vel_w=torch.tensor([[0., 0., .4]]),
        joint_pos=torch.tensor([[1., 2.]]),
        joint_vel=torch.tensor([[3., 4.]]),
        applied_torque=torch.tensor([[5., 6.]]),
    )
    robot = SimpleNamespace(data=data, joint_names=['left_track', 'right_track'])
    sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[0., 0., 7.]]])))
    got = capture_physical_state(robot, sensor)
    assert got['root_lin_vel_w'] == pytest.approx([.1, .2, .3])
    assert got['joint_vel'] == [3., 4.]
    assert got['applied_torque'] == [5., 6.]
    assert got['contact_net_forces_w'] == [[0., 0., 7.]]
    assert got['joint_names'] == ['left_track', 'right_track']
    assert got['torque_meaning'] == 'Isaac ArticulationData.applied_torque; simulated applied joint torque, not measured effort'


def test_cancel_summary_prefers_post_physics_root_velocity_when_logged():
    rows = [
        {'event': 'cancel', 'step': 0, 'stamp_ns': 0},
        {'event': 'step', 'step': 0, 'stamp_ns': 0, 'pose_pre': [0., 0., 0.],
         'pose_post': [.01, 0., 0.], 'command': [0., 0., 0.],
         'actor_input_command': [0., 0., 0.], 'actor_action': [0.]*6,
         'physical_post': {'root_lin_vel_w': [.4, .3, 0.]}, 'done': False},
    ]
    result = summarize_cancel(rows)
    assert result['last_xy_speed_mps'] == pytest.approx(.5)
    assert result['last_root_velocity_source'] == 'robot.data.root_lin_vel_w_post'


def test_compare_step_fields_reports_first_signal_and_rejects_stamp_mismatch():
    a = [{'step': 0, 'stamp_ns': 0, 'command': [0, 0, 0], 'actor_action': [1]},
         {'step': 1, 'stamp_ns': 50_000_000, 'command': [0, 0, 0], 'actor_action': [1]}]
    b = [dict(row) for row in a]
    b[1]['actor_action'] = [2]
    assert compare_step_fields(a, b, ('command', 'actor_action')) == {
        'command': {'equal': True, 'first_mismatch_step': None},
        'actor_action': {'equal': False, 'first_mismatch_step': 1}}
    b[1]['stamp_ns'] = 100_000_000
    with pytest.raises(ValueError, match='stamp'):
        compare_step_fields(a, b, ('command',))
