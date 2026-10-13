import math

import pytest

from eat_nav_p1_teacher_labels import label_window


def state(x):
    return [x, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]


def rows(n=40, arrival=25):
    return [{'step': i, 'stamp_ns': i * 50_000_000,
             'root_pre': state(i * 0.01), 'root_post': state((i+1)*0.01),
             'command': [0.3 if i < arrival else 0., 0, 0],
             'reward': 1.0, 'goal_reached_pre': i >= arrival,
             'torque_Nm': [2.0], 'body_force_world_N': [[3.0,4.0,0]],
             'terminated': False, 'censored': False} for i in range(n)]


def test_fixed_40_step_label_has_equal_exposure_and_early_arrival_hold():
    result = label_window(rows(), common_goal_xy=(1., 0.),
                          allowables={'schema':'eatnav_pain_allowables_v1',
                                      'joint_torque_ref_Nm':{}, 'body_force_ref_N':{}},
                          joint_names=['j0'], body_names=['b0'])
    assert result['status'] == 'complete_2s'
    assert result['arrival_step'] == 25
    assert result['active_command_steps'] == 25
    assert result['common_goal_progress_m'] == pytest.approx(0.4)
    assert result['reward_gamma_1_sum'] == 40
    assert result['pain']['status'] == 'missing_allowables'


def test_short_termination_is_censored_not_imputed():
    truncated = rows(12)
    truncated[-1]['terminated'] = True
    result = label_window(truncated, common_goal_xy=(1.,0.),
                          allowables={'schema':'eatnav_pain_allowables_v1',
                                      'joint_torque_ref_Nm':{}, 'body_force_ref_N':{}},
                          joint_names=['j0'], body_names=['b0'])
    assert result['status'] == 'censored_before_2s'
    assert result['reward_gamma_1_sum'] is None
    assert result['observed_steps'] == 12


def test_gamma_is_applied_to_logged_reward_without_new_dt_factor():
    result = label_window(rows(), common_goal_xy=(1.,0.), gamma=0.5,
                          allowables={'schema':'eatnav_pain_allowables_v1',
                                      'joint_torque_ref_Nm':{}, 'body_force_ref_N':{}},
                          joint_names=['j0'], body_names=['b0'])
    assert result['reward_discounted_sum'] == pytest.approx(2.0)


def test_common_goal_uses_tile_local_pose():
    translated = rows()
    for row in translated:
        row['root_pre'][0] += 5.0
        row['root_post'][0] += 5.0
    result = label_window(translated, common_goal_xy=(1., 0.),
                          env_origin_xy=(5., 0.),
                          allowables={'schema':'eatnav_pain_allowables_v1',
                                      'joint_torque_ref_Nm':{}, 'body_force_ref_N':{}},
                          joint_names=['j0'], body_names=['b0'])
    assert result['common_goal_progress_m'] == pytest.approx(0.4)
