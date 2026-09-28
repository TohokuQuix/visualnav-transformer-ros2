import numpy as np

from visualnav_transformer.eat_nav.quality import EmpiricalCDF, fit_train_statistics, p10


def test_p10_uses_raw_values_and_keeps_low_tail():
    assert p10([0, 10, 20, 30, 40]) == 4.0
    assert p10([-20, 30, 35, 40]) < 0


def test_train_only_ecdf_maps_heldout_without_refit():
    cdf = EmpiricalCDF.fit([0, 10, 20, 30])
    assert cdf.transform(-10) == 0
    assert cdf.transform(10) == 0.5
    assert cdf.transform(100) == 1
    assert len(cdf.values) == 4


def test_action_stats_use_only_valid_train_deltas_and_change_with_length():
    a = {'action': np.array([[1., 2.], [3., 4.], [999., 999.]]), 'action_mask': np.array([1, 1, 0]), 'critic_raw': np.array([0., 10.])}
    b = {'action': np.array([[5., 6.], [999., 999.], [999., 999.]]), 'action_mask': np.array([1, 0, 0]), 'critic_raw': np.array([20., 30.])}
    stats, cdf = fit_train_statistics([a, b])
    np.testing.assert_allclose(stats.mean, [3., 4.])
    assert stats.count == 3
    assert cdf.values == (1., 21.)
    shortened, _ = fit_train_statistics([{'action': x['action'][:1], 'action_mask': x['action_mask'][:1], 'critic_raw': x['critic_raw']} for x in (a,b)])
    np.testing.assert_allclose(shortened.mean, [3., 4.])
    assert shortened.count == 2
