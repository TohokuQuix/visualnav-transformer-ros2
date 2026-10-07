import numpy as np

from eat_nav_corridor_prepare import classify_group, profile_from_corridor


def test_profile_shape_and_invalid_fraction():
    corridor = np.zeros((4, 2, 16, 7), dtype=np.float32)
    corridor[:, 1] = 1
    corridor[:, 1, 5, 0] = 0
    profile = profile_from_corridor(corridor, .8)
    assert profile.shape[0] == 16
    assert profile.shape[1] >= 8
    assert profile[5, -1] > profile[4, -1]


def test_support_classification_is_geometry_only():
    a = np.zeros((4, 2, 16, 7), dtype=np.float32)
    b = a.copy()
    a[:, 1] = b[:, 1] = 1
    b[:, 0, 12:, :] = .12
    thresholds = {'min_candidate_valid_ratio': .5, 'flat_height_std_max_m': .03,
                  'flat_longitudinal_step_max_m': .03, 'differential_pair_rms_min_m': .03,
                  'differential_step_difference_min_m': .03,
                  'nonflat_station_height_delta_min_m': .03,
                  'transition_first_nonflat_range_m': [.15, .55],
                  'cluster_pair_rms_max_m': .03}
    assert classify_group({'a': a, 'b': b}, .8, thresholds)['class'] == 'differential_corridor'
