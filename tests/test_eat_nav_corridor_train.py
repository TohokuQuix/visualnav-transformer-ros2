import numpy as np
import torch

from eat_nav_corridor_train import (fit_train_scalers, group_pair_loss,
                                    folds_for_shard, _pair_plan, _fast_pair_loss)


def test_scalers_ignore_test_group_and_preserve_binary_masks():
    common = np.array([[1., 0.], [3., 1.], [1000., 1.]], dtype=np.float32)
    targets = np.array([[1., 2.], [3., 4.], [1000., 1000.]], dtype=np.float32)
    scaler = fit_train_scalers(common, targets, ['a', 'a', 'test'], [0, 1], [False, True])
    np.testing.assert_allclose(scaler['common_mean'], [2., 0.])
    np.testing.assert_allclose(scaler['common_scale'][1], 1.)
    np.testing.assert_allclose(scaler['target_mean'], [2., 3.])


def test_pair_loss_uses_strict_labels_and_group_macro():
    predictions = torch.tensor([[1., 0., 0.], [0., 0., 1.]], requires_grad=True)
    rows = [{'group': 'g', 'candidate': 'a'}, {'group': 'g', 'candidate': 'b'}]
    pairs = {'g': [{'candidate_i': 'a', 'candidate_j': 'b',
                    'reward_label': 'i_better', 'goal_label': 'j_better'}]}
    loss = group_pair_loss(predictions, rows, [0, 1], pairs, (.25, .25))
    assert 0 < loss.item() < 1
    loss.backward()
    assert predictions.grad is not None


def test_shards_partition_folds_without_changing_order():
    folds = [{'fold_id': str(i)} for i in range(7)]
    portions = [folds_for_shard(folds, shard, 3) for shard in range(3)]
    assert [[int(item['fold_id']) for item in part] for part in portions] == [[0, 3, 6], [1, 4], [2, 5]]


def test_vectorized_pair_plan_matches_reference_loss():
    predictions = torch.tensor([[1., 0., 0.], [0., 0., 1.]], requires_grad=True)
    rows = [{'group': 'g', 'candidate': 'a'}, {'group': 'g', 'candidate': 'b'}]
    pairs = {'g': [{'candidate_i': 'a', 'candidate_j': 'b',
                    'reward_label': 'i_better', 'goal_label': 'j_better'}]}
    reference = group_pair_loss(predictions, rows, [0, 1], pairs, (.25, .25))
    actual = _fast_pair_loss(predictions, _pair_plan(rows, [0, 1], pairs))
    torch.testing.assert_close(actual, reference)
