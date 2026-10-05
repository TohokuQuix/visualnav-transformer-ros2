"""Review calculations must preserve source integrity and statistical units."""
import numpy as np
import pytest

from eat_nav_evaluator_design_audit import same
from eat_nav_evaluator_design_analyze import pairwise_agreement
from eat_nav_hard_state_support import sample_profile


def test_outcome_comparison_rejects_first_numeric_change():
    same({'delta_v': 0.25, 'reason': []}, {'delta_v': 0.25, 'reason': []}, 'run')
    with pytest.raises(ValueError, match=r'run.delta_v'):
        same({'delta_v': 0.25}, {'delta_v': 0.2500001}, 'run')


def test_pairwise_tie_is_excluded_without_hidden_epsilon():
    result = pairwise_agreement([0., 0., 1.], [0., 1., 1.], 0., 0.)
    assert result['comparable_pairs'] == 1
    assert result['agreement'] == 1.
    assert pairwise_agreement([0., 0.], [1., 2.])['agreement'] is None


def test_base_yaw_dem_profile_uses_mask_and_fixed_length():
    dem = np.zeros((2, 20, 20), dtype=np.float32)
    dem[1] = 1
    dem[0, 10, 10] = 0.1
    heights, mask = sample_profile(dem, [[0., 0.], [.1, 0.]])
    assert heights.shape == (64,)
    assert mask.shape == (64,)
    assert heights[0] == pytest.approx(.1)
    dem[1, 10, 10] = 0
    _, mask2 = sample_profile(dem, [[0., 0.], [.1, 0.]])
    assert not mask2[0]
