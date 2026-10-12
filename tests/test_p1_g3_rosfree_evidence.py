import copy

import pytest

from visualnav_transformer.eat_nav.p1_g3_rosfree_evidence import validate_control_trace


def row(step, *, goal=False):
    stamp = 2_000_000_000 + 50_000_000 * step
    command = [0., 0., 0.] if goal else [.3, 0., 0.]
    timing = {
        'step': step, 'stamp_ns': stamp, 'pose_stamp_ns': stamp,
        'core_call_stamp_ns': stamp, 'command_injection_stamp_ns': stamp,
        'actor_input_stamp_ns': stamp, 'physics_input_ns': stamp,
        'next_pose_stamp_ns': stamp + 50_000_000,
        'events': ['pose', 'core', 'command_injection', 'actor', 'physics', 'next_pose'],
        'pose_capture_steady_ns': step * 100 + 1, 'core_call_steady_ns': step * 100 + 2,
        'command_injection_steady_ns': step * 100 + 3, 'actor_begin_steady_ns': step * 100 + 4,
        'physics_begin_steady_ns': step * 100 + 5, 'next_pose_capture_steady_ns': step * 100 + 6,
        'episode_length_before': 20 + step, 'episode_length_after': 21 + step,
        'terminated': False, 'truncated': False, 'reset_called': False,
        'next_pose': [0, 0, 0],
    }
    return {'control_step': step, 'logical_sim_time_ns': stamp,
            'follower_command': command, 'applied_command': command,
            'label_exposure': True,
            'follower_diagnostic': {'goal_reached': goal, 'linear_recovery': False,
                                    'angular_recovery': False},
            'core_evidence': {'env_id': 0, 'pose_frame': 'tile_local', 'pose': [0, 0, 0],
                              'stamp_ns': stamp, 'suppressed_after_exposure': False,
                              'actor_command_slice': [0, 3], 'critic_command_slice': [0, 3],
                              'command_observation': command,
                              'command_observation_previous_raw': command,
                              'command_observation_latency_alpha': 0.0},
            'actor_input': command, 'critic_observation': command,
            'terminal': None, 'timing_protocol': timing}


def test_trace_accepts_exact_causal_order_and_goal_hold():
    rows = [row(0), *[row(step, goal=True) for step in range(1, 42)]]
    assert validate_control_trace(rows, 0)['rows'] == 42


@pytest.mark.parametrize('edit', [
    lambda r: r[0]['timing_protocol'].update({'actor_begin_steady_ns': 0}),
    lambda r: r[0]['core_evidence'].update({'stamp_ns': 99}),
    lambda r: r[1].update({'applied_command': [.2, 0., 0.]}),
    lambda r: r[1]['timing_protocol'].update({'reset_called': True}),
    lambda r: r[2]['core_evidence'].update({'pose_frame': 'map'}),
    lambda r: r[2].update({'follower_command': [.3, 0., 0.]}),
])
def test_trace_rejects_protocol_drift(edit):
    rows = [row(0), *[row(step, goal=True) for step in range(1, 42)]]
    edit(rows)
    with pytest.raises(ValueError):
        validate_control_trace(rows, 0)
