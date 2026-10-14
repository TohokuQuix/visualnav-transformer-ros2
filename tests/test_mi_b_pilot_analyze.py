import numpy as np

from scripts.eat_nav_mi_b_pilot_analyze import contrast_vectors, posture_interaction


def test_contrast_vectors_identifies_wrong_arc_effect():
    actual = np.zeros((2, 6, 2), np.float32)
    actual[:, 1, 0] = .1
    prediction = np.zeros_like(actual)
    prediction[:, 1, 0] = -.1
    row = contrast_vectors(prediction, actual)['forward_+30']
    assert row['measured_mean_xy_m'] == [pytest_approx(.1), pytest_approx(0)]
    assert row['predicted_mean_xy_m'][0] == pytest_approx(-.1)
    assert row['same_direction_fraction'] == 0


def pytest_approx(value):
    import pytest
    return pytest.approx(value)


def test_posture_interaction_pairs_same_seed_and_env():
    target = np.zeros((4, 6, 2), np.float32)
    target[2:, 1, 0] = .1
    prediction = target.copy()
    metadata = [
        {'terrain_seed': 7, 'env_tile_id': 0, 'posture': 'front_folded'},
        {'terrain_seed': 7, 'env_tile_id': 1, 'posture': 'front_folded'},
        {'terrain_seed': 7, 'env_tile_id': 0, 'posture': 'rear_folded'},
        {'terrain_seed': 7, 'env_tile_id': 1, 'posture': 'rear_folded'},
    ]
    result = posture_interaction(prediction, target, metadata)
    assert result['paired_conditions'] == 2
    assert result['interaction_rmse_m'] == 0
