"""Offline 40-control-step labels from one aligned P1 teacher trace."""

from __future__ import annotations

import math

from visualnav_transformer.eat_nav.p1_teacher_capture import compute_pain


SCHEMA = 'eatnav_p1_teacher_labels_v1'


def _yaw(state):
    w, x, y, z = state[3:7]
    return math.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))


def _wrap(value):
    return math.atan2(math.sin(value), math.cos(value))


def label_window(rows, *, common_goal_xy, allowables, joint_names, body_names,
                 env_origin_xy=(0.0, 0.0), gamma=1.0):
    if not 0 < gamma <= 1 or not rows or len(common_goal_xy) != 2 or len(env_origin_xy) != 2:
        raise ValueError('teacher_window_contract')
    observed = rows[:40]
    for i, row in enumerate(observed):
        if row['step'] != i or row['stamp_ns'] != i * 50_000_000:
            raise ValueError('teacher_timestamp_alignment')
    arrival = next((row['step'] for row in observed if row['goal_reached_pre']), None)
    complete = len(observed) == 40 and not any(row['terminated'] or row['censored']
                                                for row in observed)
    result = {'schema': SCHEMA,
              'status': 'complete_2s' if complete else 'censored_before_2s',
              'observed_steps': len(observed), 'arrival_step': arrival,
              'active_command_steps': sum(any(abs(float(x)) > 1e-9 for x in row['command'])
                                          for row in observed),
              'reward_gamma_1_sum': None,
              'reward_discounted_sum': None,
              'common_goal_progress_m': None,
              'motion_error_forward_abs_mean_mps': None,
              'motion_error_yaw_abs_mean_radps': None,
              'pain': {'status': 'not_evaluated_for_censored_window'}}
    if not complete:
        return result
    pre = observed[0]['root_pre']
    post = observed[-1]['root_post']
    pre_local = (pre[0] - env_origin_xy[0], pre[1] - env_origin_xy[1])
    post_local = (post[0] - env_origin_xy[0], post[1] - env_origin_xy[1])
    result['common_goal_progress_m'] = (math.dist(pre_local, common_goal_xy) -
                                        math.dist(post_local, common_goal_xy))
    result['reward_gamma_1_sum'] = sum(float(row['reward']) for row in observed)
    result['reward_discounted_sum'] = sum(gamma**i * float(row['reward'])
                                          for i, row in enumerate(observed))
    forward_error = []
    yaw_error = []
    for row in observed:
        a,b = row['root_pre'],row['root_post']
        angle = _yaw(a)
        velocity = ((b[0]-a[0])*math.cos(angle)+(b[1]-a[1])*math.sin(angle))/0.05
        omega = _wrap(_yaw(b)-angle)/0.05
        forward_error.append(abs(float(row['command'][0])-velocity))
        yaw_error.append(abs(float(row['command'][2])-omega))
    result['motion_error_forward_abs_mean_mps'] = sum(forward_error)/40
    result['motion_error_yaw_abs_mean_radps'] = sum(yaw_error)/40
    result['pain'] = compute_pain(
        {'torque_Nm':[row['torque_Nm'] for row in observed],
         'body_force_world_N':[row['body_force_world_N'] for row in observed]},
        allowables,joint_names=joint_names,body_names=body_names,dt=0.05)
    return result
