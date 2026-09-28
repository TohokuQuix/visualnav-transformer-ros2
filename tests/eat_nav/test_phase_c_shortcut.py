import numpy as np
from visualnav_transformer.eat_nav.phase_c_shortcut import trajectory_input, endpoint_fixed_shuffle


def test_gte_carries_only_endpoint_and_gts_cannot_reconstruct_straight_endpoint():
    a = np.array([[[1., 0.]] * 8, [[2., 0.]] * 8], dtype=np.float32)
    gte = trajectory_input(a, 'GTE')
    gts = trajectory_input(a, 'GTS')
    assert np.all(gte[:, :-1] == 0)
    assert np.allclose(gte[:, -1, 0], [8, 16])
    assert np.allclose(gts, 0)
    assert np.allclose(trajectory_input(a, 'G'), 0)
    assert np.allclose(trajectory_input(a, 'GT'), a)


def test_endpoint_fixed_shuffle_changes_middle_but_preserves_each_endpoint():
    a = np.zeros((3, 8, 2), dtype=np.float32)
    a[0, 0, 0] = 2; a[0, -1, 0] = -1
    a[1, 2, 1] = 2; a[1, -1, 1] = -1
    a[2, 4, 0] = 3; a[2, -1, 0] = -2
    shuffled = endpoint_fixed_shuffle(a, np.array(['rough'] * 3), seed=1)
    assert np.allclose(shuffled.sum(axis=1), a.sum(axis=1))
    assert not np.allclose(shuffled[:, :-1], a[:, :-1])
