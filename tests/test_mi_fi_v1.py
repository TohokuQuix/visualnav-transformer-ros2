from types import SimpleNamespace

import pytest


def _cfg():
    return SimpleNamespace(events=SimpleNamespace(
        reset_flipper_fold_mode=SimpleNamespace(params={'modes': (0.0,), 'probabilities': (1.0,)}),
        reset_robot_joints=SimpleNamespace(params={
            'flipper_joint_names': ['rr', 'rl', 'fr', 'fl'],
            'pose_probabilities': (0.0, 1.0, 0.0),
            'folded_angle': 3.14159265359,
            'front_rest_angle': 0.0,
        }),
    ))


def test_condition_selects_existing_reset_pose_without_changing_mode():
    from visualnav_transformer.eat_nav.mi_fi_v1 import configure_initialization
    a, b = _cfg(), _cfg()
    assert configure_initialization(a, 'front_folded') == (0.0, 0.0, 3.14159265359, 3.14159265359)
    assert configure_initialization(b, 'rear_folded') == (3.14159265359, 3.14159265359, 0.0, 0.0)
    assert a.events.reset_robot_joints.params['pose_probabilities'] == (0.0, 1.0, 0.0)
    assert b.events.reset_robot_joints.params['pose_probabilities'] == (0.0, 0.0, 1.0)
    assert b.events.reset_flipper_fold_mode.params == {'modes': (0.0,), 'probabilities': (1.0,)}


def test_condition_rejects_unfrozen_reset_contract():
    from visualnav_transformer.eat_nav.mi_fi_v1 import configure_initialization
    cfg = _cfg()
    cfg.events.reset_robot_joints.params['flipper_joint_names'][0] = 'fr'
    with pytest.raises(ValueError):
        configure_initialization(cfg, 'rear_folded')


def test_schedule_has_two_candidate_free_repeats_and_six_candidates_per_condition():
    from visualnav_transformer.eat_nav.mi_fi_v1 import pilot_plan
    rows = pilot_plan(2026101305)
    assert len(rows) == 16
    assert [r['condition'] for r in rows] == ['front_folded'] * 8 + ['rear_folded'] * 8
    assert [r['stage'] for r in rows[:8]] == ['prepath'] * 2 + ['candidate'] * 6
    assert [r['stage'] for r in rows[8:]] == ['prepath'] * 2 + ['candidate'] * 6


def test_gate_p_never_compares_across_initialization_conditions():
    from visualnav_transformer.eat_nav.mi_fi_v1 import compare_gate_p
    import numpy as np
    base = {'identity': {'flipper_initialization_mode': 'front_folded'},
            'arrays': {'root': np.zeros((4, 13))}}
    other = {'identity': {'flipper_initialization_mode': 'rear_folded'},
             'arrays': {'root': np.zeros((4, 13))}}
    assert not compare_gate_p(base, other)['pass']
