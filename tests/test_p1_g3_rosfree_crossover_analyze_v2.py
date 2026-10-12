import pytest

from eat_nav_p1_g3_rosfree_crossover_analyze_v2 import add_display_mode


@pytest.mark.parametrize('flags, expected', [
    ({'goal_reached': True, 'linear_recovery': False, 'angular_recovery': False}, 'goal'),
    ({'goal_reached': False, 'linear_recovery': True, 'angular_recovery': False}, 'linear_recovery'),
    ({'goal_reached': False, 'linear_recovery': False, 'angular_recovery': True}, 'angular_recovery'),
    ({'goal_reached': False, 'linear_recovery': False, 'angular_recovery': False}, 'tracking'),
])
def test_mode_is_derived_without_mutating_raw(flags, expected):
    raw = [{'follower_diagnostic': dict(flags), 'control_step': 0}]
    derived = add_display_mode(raw)
    assert derived[0]['follower_diagnostic']['mode'] == expected
    assert 'mode' not in raw[0]['follower_diagnostic']


def test_legacy_mode_is_rejected():
    with pytest.raises(ValueError, match='unexpected legacy'):
        add_display_mode([{'follower_diagnostic': {'mode': 'tracking'}}])
