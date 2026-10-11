import math

import pytest

from visualnav_transformer.eat_nav.p1_paired_command import intervene


def record(heading=0.3, *, goal=False, recovery=False, reverse=False):
    yaw = 0.0
    desired = heading + (math.pi if reverse else 0.0)
    turning = abs(heading) >= 0.6
    vx = 0.0 if turning else (-1 if reverse else 1) * 0.3 * (1 - abs(heading) / math.pi)
    return {'phase': 'command_pre_publish', 'command_stamp_ns': 2_000_000_000,
            'tf_stamp_ns': 2_000_000_000, 'tf_success': True, 'path_points': 8,
            'goal_reached': goal, 'mode': 'default', 'linear_recovery': recovery,
            'angular_recovery': False, 'turn_in_place': turning,
            'heading_error': heading, 'tf_x': 0.0, 'tf_y': 0.0, 'tf_yaw': yaw,
            'carrot_x': math.cos(desired), 'carrot_y': math.sin(desired),
            'vx': 0.0 if goal else vx, 'wz': heading * 0.5}


def apply(row, condition):
    return intervene([row['vx'], 0.0, row['wz']], row, condition,
                     stamp_ns=2_000_000_000)


def test_soft_scale_only_changes_forward_component():
    row = record()
    assert apply(row, 'B')[0][0] == row['vx']
    for condition in ('S', 'T'):
        command, reason = apply(row, condition)
        assert command == [0.3, 0.0, row['wz']]
        assert reason == 'soft_scale_removed'


def test_turn_stop_preserved_in_s_and_removed_only_in_t():
    row = record(0.8)
    assert apply(row, 'S')[0][0] == 0.0
    command, reason = apply(row, 'T')
    assert command == [0.3, 0.0, row['wz']]
    assert reason == 'ordinary_turn_stop_removed'


@pytest.mark.parametrize('kwargs,reason', [
    ({'goal': True}, 'goal_stop_preserved'),
    ({'recovery': True}, 'recovery_preserved'),
])
def test_unrelated_branches_preserved(kwargs, reason):
    row = record(**kwargs)
    command, observed = apply(row, 'T')
    assert command[0] == row['vx']
    assert observed == reason


def test_reverse_direction_kept_while_soft_scale_removed():
    row = record(reverse=True)
    command, reason = apply(row, 'S')
    assert command == [-0.3, 0.0, row['wz']]
    assert reason == 'soft_scale_removed'


def test_mismatched_stamp_and_formula_fail_closed():
    row = record()
    row['command_stamp_ns'] += 50_000_000
    with pytest.raises(ValueError, match='stamp'):
        apply(row, 'T')
    row = record()
    row['vx'] = 0.1
    with pytest.raises(ValueError, match='formula'):
        apply(row, 'T')
