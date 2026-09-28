import numpy as np
import pytest

from visualnav_transformer.eat_nav.phase_c_reference import CompleteOnlyView, fit_complete_train_statistics


class FakeDataset:
    def __init__(self, windows):
        self.windows = windows


def window(step, mask, value):
    return {
        'episode_key': ('session', 0, 1), 'step': step,
        'action_mask': np.asarray(mask, dtype=bool),
        'action': np.full((8, 2), value, dtype=np.float32),
        'dem': np.zeros((4, 2, 20, 20), dtype=np.float32),
        'goal': np.array([value, 0], dtype=np.float32),
        'critic_raw': np.array([value], dtype=np.float32),
        'critic_mask': np.array([True]),
        'quality_raw': value,
    }


def test_complete_view_excludes_partial_and_zero_and_preserves_ids():
    data = FakeDataset([window(1, [True]*8, 1), window(2, [True]*7+[False], 50),
                        window(3, [False]*8, 100), window(4, [True]*8, 3)])
    view = CompleteOnlyView(data, 8)
    assert len(view) == 2
    assert view.identities == [(('session', 0, 1), 1), (('session', 0, 1), 4)]
    assert view[0]['action_mask'].all()


def test_complete_train_statistics_ignore_partial_and_validation():
    train = CompleteOnlyView(FakeDataset([window(1, [True]*8, 1),
                                         window(2, [True]*7+[False], 50),
                                         window(3, [True]*8, 3)]), 8)
    validation = CompleteOnlyView(FakeDataset([window(4, [True]*8, 100)]), 8)
    stats, ecdf = fit_complete_train_statistics(train)
    assert stats.count == 16
    assert np.allclose(stats.mean, [2, 2])
    assert np.allclose(stats.std, [1, 1])
    assert len(ecdf.values) == 2
    assert validation[0]['action'][0, 0] == 100


def test_complete_view_rejects_duplicate_identity():
    with pytest.raises(ValueError, match='duplicate'):
        CompleteOnlyView(FakeDataset([window(1, [True]*8, 1), window(1, [True]*8, 2)]), 8)
