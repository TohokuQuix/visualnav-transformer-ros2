import numpy as np
import pytest
import torch

from visualnav_transformer.eat_nav.mi_b_pilot import (
    PilotPredictor, contrast_indices, validate_grouped_arrays,
    paired_metrics, normalized_pair_loss, split_indices,
    fit_normalization, transform_arrays,
)


def fixture_arrays():
    n, k = 3, 6
    return dict(
        dem=np.zeros((n, 4, 2, 20, 20), np.float32),
        state=np.zeros((n, 22), np.float32),
        paths=np.zeros((n, k, 8, 2), np.float32),
        direction=np.tile(np.array([1, 1, 1, -1, -1, -1], np.float32), (n, 1))[:, :, None],
        mask=np.ones((n, k), bool),
        target=np.zeros((n, k, 2), np.float32),
    )


def test_group_contract_rejects_invalid_shape_or_direction():
    arrays = fixture_arrays()
    validate_grouped_arrays(arrays)
    arrays['direction'][0, 0, 0] = 0
    with pytest.raises(ValueError, match='direction'):
        validate_grouped_arrays(arrays)


def test_no_path_cannot_separate_same_direction_candidates():
    arrays = fixture_arrays()
    arrays['paths'][:, 1, :, 1] = .2
    model = PilotPredictor('no_path', feature_dim=16)
    tensors = [torch.from_numpy(arrays[k]) for k in ('dem', 'state', 'paths', 'direction', 'mask')]
    out = model(*tensors)
    assert out.shape == (3, 6, 2)
    assert torch.equal(out[:, 0], out[:, 1])
    assert torch.equal(out[:, 3], out[:, 4])


def test_direction_and_path_can_change_prediction():
    arrays = fixture_arrays()
    arrays['paths'][:, 1, :, 1] = .2
    model = PilotPredictor('full', feature_dim=16)
    out = model(*[torch.from_numpy(arrays[k]) for k in ('dem', 'state', 'paths', 'direction', 'mask')])
    assert not torch.equal(out[:, 0], out[:, 1])
    assert not torch.equal(out[:, 0], out[:, 3])


def test_mask_and_paired_loss_use_four_arc_contrasts():
    pred = torch.zeros((2, 6, 2))
    target = torch.zeros_like(pred)
    target[:, 1, 0] = 1
    target[:, 4, 1] = 2
    mask = torch.ones((2, 6), dtype=torch.bool)
    assert contrast_indices() == ((1, 0), (2, 0), (4, 3), (5, 3))
    loss = normalized_pair_loss(pred, target, mask)
    assert torch.isclose(loss, torch.tensor((1 + 4) / 6 + (1 + 4) / 4, dtype=torch.float32))
    mask[:, 4] = False
    assert torch.isfinite(normalized_pair_loss(pred, target, mask))


def test_paired_metrics_count_groups_not_tiles_as_seeds():
    arrays = fixture_arrays()
    arrays['target'][:, 1, 0] = .1
    pred = np.zeros_like(arrays['target'])
    metrics = paired_metrics(pred, arrays['target'], arrays['mask'],
                             seed_ids=np.array([7, 7, 8]))
    assert metrics['group_count'] == 3
    assert metrics['seed_count'] == 2
    assert metrics['contrast_rmse_m'] > 0


def test_split_rejects_seed_leakage_and_covers_groups():
    metadata = [{'terrain_seed': 7}, {'terrain_seed': 7},
                {'terrain_seed': 8}, {'terrain_seed': 9}]
    assert [v.tolist() for v in split_indices(metadata, {
        'train': [7], 'validation': [8], 'test': [9]}).values()] == [[0, 1], [2], [3]]
    with pytest.raises(ValueError, match='seed'):
        split_indices(metadata, {'train': [7], 'validation': [7, 8], 'test': [9]})


def test_normalization_uses_only_train_and_preserves_dem_mask():
    arrays = fixture_arrays()
    arrays['dem'][:, :, 1] = 1
    arrays['dem'][0, :, 0] = .1
    arrays['dem'][1, :, 0] = .2
    arrays['dem'][2, :, 0] = 99
    arrays['state'][0] = 1
    arrays['state'][1] = 2
    arrays['state'][2] = 999
    arrays['target'][0] = 1
    arrays['target'][1] = 2
    arrays['target'][2] = 999
    norm = fit_normalization(arrays, np.array([0, 1]))
    assert norm['dem_height_mean'] == pytest.approx(.15)
    assert norm['target_mean'] == pytest.approx([1.5, 1.5])
    changed = transform_arrays(arrays, norm)
    assert np.array_equal(changed['dem'][:, :, 1], arrays['dem'][:, :, 1])
    assert np.array_equal(changed['paths'], arrays['paths'])

