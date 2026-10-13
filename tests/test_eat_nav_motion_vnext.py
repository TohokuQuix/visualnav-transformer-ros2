import math

import numpy as np
import pytest

from visualnav_transformer.eat_nav.motion_vnext import motion_label_from_trace, goal_progress_scores
from visualnav_transformer.eat_nav.arc_bank import arc_waypoints, audit_arc, STAGE1_ANGLES, STAGE2_ANGLES


def _trace(yaw=math.pi / 2, end=(0., 0.6), terminal=None):
    quat = [math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)]
    rows = []
    for step in range(40):
        rows.append({'control_step': step, 'logical_sim_time_ns': 5_000_000_000 + step * 50_000_000,
                     'root_state': [0., 0., 0., *quat],
                     'post_root_state': [end[0] * (step + 1) / 40,
                                         end[1] * (step + 1) / 40, 0., *quat],
                     'terminal': terminal if step == 12 else None})
    return rows


def test_motion_label_uses_start_yaw_and_exactly_40_steps():
    result = motion_label_from_trace(_trace())
    assert result['available'] is True
    np.testing.assert_allclose(result['delta_body_xy_m'], [.6, 0.], atol=1e-7)
    assert result['last_control_step'] == 39


def test_terminated_and_short_windows_are_censored_without_zero_label():
    for rows in (_trace(terminal={'termination_reason': 'tilt_over'}), _trace()[:39]):
        result = motion_label_from_trace(rows)
        assert result['available'] is False
        assert result['delta_body_xy_m'] is None
    assert motion_label_from_trace(_trace(terminal={'termination_reason': 'tilt_over'}))[
        'termination_reason'] == 'tilt_over'


def test_motion_label_rejects_time_or_step_discontinuity():
    rows = _trace()
    rows[2]['control_step'] = 3
    with pytest.raises(ValueError, match='step'):
        motion_label_from_trace(rows)


def test_goal_only_changes_score_not_predictions():
    displacement = np.array([[.5, 0.], [0., .5]], dtype=np.float32)
    np.testing.assert_allclose(goal_progress_scores(displacement, [1., 0.]),
                               [0.5, 1. - math.sqrt(1.25)], atol=1e-6)
    np.testing.assert_allclose(goal_progress_scores(displacement, [0., 1.]),
                               [1. - math.sqrt(1.25), 0.5], atol=1e-6)
    assert np.array_equal(displacement, np.array([[.5, 0.], [0., .5]], dtype=np.float32))


def test_arc_bank_geometry_symmetry_and_endpoint_bearing():
    assert STAGE1_ANGLES == (0, -30, 30, -60, 60, -90, 90)
    assert STAGE2_ANGLES == (-120, 120, -150, 150, -180, 180)
    direct = arc_waypoints(0)
    left = arc_waypoints(90)
    right = arc_waypoints(-90)
    assert direct.shape == left.shape == right.shape == (8, 2)
    np.testing.assert_array_equal(direct[0], [0., 0.])
    np.testing.assert_allclose(left[:, 0], right[:, 0], atol=1e-12)
    np.testing.assert_allclose(left[:, 1], -right[:, 1], atol=1e-12)
    audit = audit_arc(left, 90)
    assert audit['geometry_valid']
    assert audit['endpoint_bearing_deg'] == pytest.approx(45., abs=1.)
    assert audit['tangent_change_deg'] == pytest.approx(90., abs=1.)
    assert audit['path_length_m'] == pytest.approx(.75, rel=.01)
    assert not np.array_equal(arc_waypoints(180), arc_waypoints(-180))


def test_arc_geometry_rejects_repeated_points():
    with pytest.raises(ValueError, match='waypoint'):
        audit_arc(np.zeros((8, 2)), 0)


def test_arc_audit_detects_a_crossing_custom_polyline():
    path = np.array([[0., 0.], [1., 1.], [0., 1.], [1., 0.],
                     [2., 0.], [3., 0.], [4., 0.], [5., 0.]])
    result = audit_arc(path, 0)
    assert result['self_intersection']
    assert not result['geometry_valid']


def test_motion_predictor_shapes_mask_and_goal_absence():
    import torch
    from visualnav_transformer.eat_nav.motion_model import MotionPredictor

    model = MotionPredictor()
    dem = torch.zeros(2, 4, 2, 20, 20)
    actor = torch.zeros(2, 22)
    paths = torch.zeros(2, 3, 8, 2)
    mask = torch.tensor([[1, 1, 0], [0, 1, 1]], dtype=torch.bool)
    output = model(dem, actor, paths, mask)
    assert output.shape == (2, 3, 2)
    assert torch.isfinite(output[mask]).all()
    assert torch.isnan(output[~mask]).all()
    with pytest.raises((TypeError, ValueError)):
        model(dem, actor, paths, mask, torch.zeros(2, 2))


def test_motion_normalization_and_group_split_train_only():
    from visualnav_transformer.eat_nav.motion_dataset import (
        fit_motion_normalization, validate_state_group_split,
    )
    labels = np.asarray([[1., 2.], [3., 4.], [100., 200.]])
    norm = fit_motion_normalization(labels[:2])
    np.testing.assert_allclose(norm['mean'], [2., 3.])
    np.testing.assert_allclose(norm['std'], [1., 1.])
    samples = [{'id': 'a', 'group': 'one', 'terrain_seed': 1},
               {'id': 'b', 'group': 'one', 'terrain_seed': 1},
               {'id': 'c', 'group': 'two', 'terrain_seed': 2},
               {'id': 'd', 'group': 'three', 'terrain_seed': 3}]
    validate_state_group_split(samples, {'train': ['a', 'b'], 'validation': ['c'], 'test': ['d']})
    with pytest.raises(ValueError, match='terrain seed'):
        validate_state_group_split(samples, {'train': ['a', 'b'], 'validation': ['c'], 'test': ['d']},
                                   extra_terrain_seeds={'two': 1})


def test_body_arc_to_tile_local_preserves_initial_pose_and_orientation():
    from visualnav_transformer.eat_nav.motion_collector import body_path_to_tile_local
    path = arc_waypoints(90)
    actual = body_path_to_tile_local(path, (10., 20., math.pi / 2), (8., 18.))
    np.testing.assert_allclose(actual[0], [2., 2.], atol=1e-12)
    np.testing.assert_allclose(actual[-1], [2. - path[-1, 1], 2. + path[-1, 0]], atol=1e-12)


def test_prepath_gate_rejects_candidate_dependent_state():
    from visualnav_transformer.eat_nav.motion_collector import compare_prepath
    reference = {'root': np.zeros((4, 13)), 'joint': np.ones((4, 6)),
                 'actor_pre_noise': np.zeros((4, 22)), 'dem': np.zeros((4, 4, 2, 20, 20)),
                 'material_sha256': 'same', 'terrain_sha256': 'same', 'seed': 42}
    assert compare_prepath(reference, reference)['pass']
    changed = dict(reference)
    changed['actor_pre_noise'] = reference['actor_pre_noise'].copy()
    changed['actor_pre_noise'][0, 0] = 0.01
    assert not compare_prepath(reference, changed)['pass']
