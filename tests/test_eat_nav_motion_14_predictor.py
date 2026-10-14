import numpy as np
import torch
import json
from pathlib import Path

from visualnav_transformer.eat_nav.motion_14_predictor import (
    CANDIDATE_IDS, Motion14Predictor, contrast_indices, normalized_pair_loss,
    validate_grouped_arrays,
)
from visualnav_transformer.eat_nav.mi_b_pilot import PilotPredictor


def fake_arrays(n=2, k=14):
    direction = np.tile(np.array([1] * 7 + [-1] * 7, np.float32)[None, :, None],
                        (n, 1, 1))
    return {'dem': np.zeros((n, 4, 2, 20, 20), np.float32),
            'state': np.zeros((n, 22), np.float32),
            'paths': np.zeros((n, k, 8, 2), np.float32),
            'direction': direction, 'mask': np.ones((n, k), bool),
            'target': np.zeros((n, k, 2), np.float32)}


def test_fourteen_slot_shape_and_six_pairs_per_direction():
    arrays = fake_arrays()
    validate_grouped_arrays(arrays)
    assert len(contrast_indices()) == 12
    values = tuple(torch.from_numpy(arrays[key]) for key in
                   ('dem', 'state', 'paths', 'direction', 'mask'))
    model = Motion14Predictor('full')
    output = model(*values)
    assert output.shape == (2, 14, 2)
    assert torch.isfinite(output).all()
    assert normalized_pair_loss(output, output, values[-1]).item() == 0


def test_architecture_matches_mi_b_and_training_order_is_frozen():
    old = PilotPredictor('full').state_dict()
    new = Motion14Predictor('full').state_dict()
    assert set(old) == set(new)
    assert all(old[name].shape == new[name].shape for name in old)
    contract = json.loads((Path(__file__).resolve().parents[1]/'docs/eat-nav/'
                           'motion-predictor-vnext/motion_14_training_contract.json').read_text())
    assert contract['candidate_count'] == 14
    assert contract['candidate_order'] == list(CANDIDATE_IDS)


def test_no_path_cannot_separate_arcs_within_direction():
    arrays = fake_arrays()
    arrays['paths'] = np.random.default_rng(1).normal(size=(2, 14, 8, 2)).astype(np.float32)
    values = tuple(torch.from_numpy(arrays[key]) for key in
                   ('dem', 'state', 'paths', 'direction', 'mask'))
    model = Motion14Predictor('no_path')
    with torch.no_grad():
        output = model(*values)
    assert torch.allclose(output[:, :7], output[:, :1].expand(-1, 7, -1), atol=1e-7, rtol=0)
    assert torch.allclose(output[:, 7:], output[:, 7:8].expand(-1, 7, -1), atol=1e-7, rtol=0)
