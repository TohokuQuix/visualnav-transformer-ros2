import math

from eat_nav_p1_g3_motion_residual_review import DT, project_s, row_metrics, window_metrics


def state(x, y, angle):
    return [x, y, 0, math.cos(angle / 2), 0, 0, math.sin(angle / 2), 0, 0, 0, 0, 0, 0]


def test_command_compares_with_same_physics_interval_in_body_frame():
    row = {
        'control_step': 0, 'logical_sim_time_ns': 0,
        'root_state': state(0, 0, math.pi / 2),
        'post_root_state': state(0, 0.01, math.pi / 2 + 0.02),
        'follower_command': [0.3, 0, 0.5],
        'follower_diagnostic': {'heading_error': 0.1, 'goal_reached': False},
        'contact_net_forces_w': [[3, 4, 0]], 'applied_torque': [2, -4],
        'reward': 1.25, 'critic_value': 5.0,
    }
    result = row_metrics(row, [0, 0, 0], [[0, 0], [0, 1]], [0, 1])
    assert math.isclose(result['v_actual_interval_mps'], 0.2, abs_tol=1e-12)
    assert math.isclose(result['v_residual_signed_mps'], 0.1, abs_tol=1e-12)
    assert math.isclose(result['yaw_actual_interval_radps'], 0.4, abs_tol=1e-12)
    assert math.isclose(result['yaw_residual_signed_radps'], 0.1, abs_tol=1e-12)
    assert math.isclose(result['path_progress_delta_m'], 0.01, abs_tol=1e-12)
    assert result['contact_pre_step_mean_body_force_N'] == 5
    assert result['torque_pre_step_mean_abs_Nm'] == 3
    assert window_metrics([result])['reward_post_step_sum'] == 1.25


def test_projection_clamps_ends_and_handles_segments():
    path = [[0, 0], [1, 0], [1, 1]]
    assert project_s([-1, 0], path) == 0
    assert project_s([1, 0.4], path) == 1.4
    assert project_s([1, 2], path) == 2


def test_residual_signed_absolute_and_rmse_are_separate():
    base = {'v_residual_signed_mps': 1., 'yaw_residual_signed_radps': -2.}
    a = dict(base)
    b = dict(base, v_residual_signed_mps=-1., yaw_residual_signed_radps=2.)
    for row in (a, b):
        row.update({name: 0. for name in (
            'heading_abs_rad', 'nominal_forward_cap_mps', 'v_command_mps',
            'v_actual_interval_mps', 'yaw_command_radps', 'yaw_actual_interval_radps',
            'contact_pre_step_mean_body_force_N', 'torque_pre_step_mean_abs_Nm',
            'commanded_forward_delta_m', 'actual_forward_delta_m',
            'commanded_yaw_delta_rad', 'actual_yaw_delta_rad',
            'path_progress_delta_m', 'common_goal_progress_delta_m', 'reward_post_step')})
        row['zero_forward_cap'] = False
    result = window_metrics([a, b])
    assert result['v_residual_signed_mean'] == 0
    assert result['v_residual_abs_mean'] == result['v_residual_rmse'] == 1
    assert result['yaw_residual_signed_mean'] == 0
    assert result['yaw_residual_abs_mean'] == result['yaw_residual_rmse'] == 2
