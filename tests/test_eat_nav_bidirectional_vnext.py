"""Contracts for the new forward/reverse experimental bank."""

import math

import numpy as np
import pytest


def test_reverse_direct_geometry_and_actual_follower(tmp_path):
    from visualnav_transformer.eat_nav.bidirectional_bank import candidate
    from visualnav_transformer.eat_nav.p1_follower_core import PureFollower, compile_core

    reverse = candidate(-1, 0)
    assert reverse.waypoints.shape == (8, 2)
    np.testing.assert_allclose(reverse.waypoints[0], [0, 0])
    np.testing.assert_allclose(reverse.waypoints[-1], [-0.75, 0], atol=1e-12)
    np.testing.assert_allclose(np.diff(reverse.waypoints[:, 0]), -0.75 / 7)
    assert np.isfinite(reverse.waypoints).all()
    follower = PureFollower(reverse.waypoints.tolist(), library=compile_core(tmp_path))
    try:
        command, diagnostic = follower.step_diagnostic(0, 0, 0, 1_000_000_000)
        assert diagnostic['travel_direction'] == -1
        assert command[0] < 0
        assert abs(command[2]) < 1e-8
    finally:
        follower.close()


@pytest.mark.parametrize('direction', [-1, 1])
@pytest.mark.parametrize('angle', [-90, -60, -30, 0, 30, 60, 90])
def test_unified_bank_geometry(direction, angle):
    from visualnav_transformer.eat_nav.bidirectional_bank import candidate, audit_candidate

    item = candidate(direction, angle)
    audit = audit_candidate(item)
    assert audit['geometry_valid']
    assert item.direction == direction
    assert item.arc_angle_deg == angle
    assert item.waypoints.shape == (8, 2)
    assert item.waypoints[-1, 0] * direction > 0
    assert audit['path_length_m'] == pytest.approx(0.75, rel=0.01)
    assert audit['body_yaw_change_deg'] == pytest.approx(angle)


def test_motion_predictor_requires_direction_and_uses_it():
    import torch
    from visualnav_transformer.eat_nav.bidirectional_motion_model import BidirectionalMotionPredictor

    model = BidirectionalMotionPredictor()
    dem = torch.zeros(1, 4, 2, 20, 20)
    state = torch.zeros(1, 22)
    paths = torch.zeros(1, 2, 8, 2)
    direction = torch.tensor([[[1.], [-1.]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    result = model(dem, state, paths, direction, mask)
    assert result.shape == (1, 2, 2)
    assert torch.isfinite(result).all()
    assert not torch.equal(result[0, 0], result[0, 1])
    with pytest.raises(ValueError):
        model(dem, state, paths, torch.zeros_like(direction), mask)


def test_yaw_label_preserves_reverse_negative_x():
    from visualnav_transformer.eat_nav.bidirectional_labels import motion_label_with_yaw

    rows = []
    for step in range(40):
        yaw = 0.1 * (step + 1) / 40
        rows.append({'control_step': step,
                     'logical_sim_time_ns': 1_000_000_000 + step * 50_000_000,
                     'root_state': [0., 0., 0., 1., 0., 0., 0.],
                     'post_root_state': [-0.3 * (step + 1) / 40, 0., 0.,
                                         math.cos(yaw / 2), 0., 0., math.sin(yaw / 2)],
                     'terminal': None})
    label = motion_label_with_yaw(rows)
    assert label['available']
    np.testing.assert_allclose(label['delta_body_xy_m'], [-0.3, 0], atol=1e-8)
    assert label['delta_body_yaw_rad'] == pytest.approx(0.1)


def test_container_stage_uses_full_attempt_identity(tmp_path):
    from scripts.eat_nav_bidirectional_collect_host import container_stage

    a = container_stage(tmp_path / 'r0' / 'forward_direct')
    b = container_stage(tmp_path / 'r1' / 'forward_direct')
    assert a != b
    assert a.startswith('/tmp/eatnav-bidirectional-')


def test_simulator_output_identity_validation_rejects_stale_stage(tmp_path):
    import json
    from scripts.eat_nav_bidirectional_collect_host import validate_output_identity

    output = tmp_path / 'sim-output'
    output.mkdir()
    (output / 'input-manifest.json').write_text(json.dumps({
        'seed': 1, 'direction': 1, 'arc_angle_deg': 0,
        'source_sha256': 'expected', 'checkpoint_sha256': 'checkpoint'}))
    expected = {'seed': 2, 'direction': 1, 'angle_deg': 0,
                'source_sha256': 'expected', 'rl_checkpoint_sha256': 'checkpoint'}
    with pytest.raises(ValueError, match='seed'):
        validate_output_identity(output, expected)
    expected['seed'] = 1
    validate_output_identity(output, expected)


def test_bidirectional_dataset_contract_keeps_signed_reverse_and_group_split():
    from visualnav_transformer.eat_nav.bidirectional_dataset_schema import validate_group

    group = {'state_group_id': 'terrain-a:seed-1:env-0', 'terrain_seed': 1,
             'dem': np.zeros((4, 2, 20, 20), dtype=np.float32),
             'actor_state': np.zeros(22, dtype=np.float32),
             'paths': np.zeros((2, 8, 2), dtype=np.float32),
             'direction': np.array([[1], [-1]], dtype=np.int8),
             'motion_xy': np.array([[.4, 0], [-.2, 0]], dtype=np.float32),
             'delta_yaw': np.array([0.1, -0.1], dtype=np.float32),
             'valid': np.array([True, True])}
    validate_group(group)
    bad = dict(group, direction=np.array([[1], [0]], dtype=np.int8))
    with pytest.raises(ValueError, match='direction'):
        validate_group(bad)
    bad = dict(group, motion_xy=np.array([[.4, 0], [.2, 0]], dtype=np.float32))
    validate_group(bad)  # Actual displacement is observed data; sign is never rewritten.
