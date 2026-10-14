import numpy as np

from visualnav_transformer.eat_nav.motion_14_metrics import evaluate_predictions


def test_paired_metrics_perfect_and_seed_clustered():
    target = np.random.default_rng(4).normal(size=(4, 14, 2)).astype(np.float32)
    rows = [{'terrain_seed': i // 2, 'posture': 'front_folded' if i % 2 == 0 else 'rear_folded',
             'terrain_type': 'random_rough', 'terrain_level': 2, 'env_tile_id': 0}
            for i in range(4)]
    perfect = evaluate_predictions(target, target, rows, bootstrap_replicates=30)
    assert perfect['absolute_xy_rmse_m'] == 0
    assert perfect['contrast_rmse_m'] == 0
    assert perfect['independent_seed_count'] == 2
    shifted = evaluate_predictions(target + .1, target, rows, bootstrap_replicates=30)
    assert shifted['absolute_xy_rmse_m'] > 0
    assert shifted['contrast_rmse_m'] < 1e-6
