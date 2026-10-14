"""Stage C adopted model parity for the ROS2 worker boundary."""

from pathlib import Path

import numpy as np
import torch

from visualnav_transformer.eat_nav.motion_14_predictor import Motion14Predictor
from visualnav_transformer.eat_nav.ros2_phase2.persistent_worker import ModelWorkerClient
from visualnav_transformer.eat_nav.ros2_phase2.predictor_client import (
    PATHS, prepare_model_input, validate_adoption_artifacts,
)


ROOT = Path('/home/kojima/lib/visualnav-transformer-ros2/.eatnav-motion-vnext-evidence/'
            'motion-14-training-20261014-04')
CHECKPOINT = ROOT/'training/full-seed123/best.pt'
WORKER_PYTHON = Path('/home/kojima/anaconda3/envs/tinyvla/bin/python')


def test_adopted_checkpoint_worker_matches_direct_inference_on_sealed_group():
    results, normalization = validate_adoption_artifacts(ROOT/'results.json', CHECKPOINT)
    assert results['selected_full_seed'] == 123
    with np.load(ROOT/'grouped.npz', allow_pickle=False) as saved:
        dem = saved['dem'][0].copy()
        state = saved['state'][0].copy()
        assert np.array_equal(saved['paths'][0], PATHS)
    values = prepare_model_input(dem, state, PATHS,
                                 np.asarray([1]*7+[-1]*7, np.float32), normalization)
    payload = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    direct = Motion14Predictor('full').eval()
    direct.load_state_dict(payload['model_state'], strict=True)
    with torch.no_grad():
        scaled = direct(*(torch.from_numpy(value) for value in values))[0].numpy()
    expected = (scaled.astype(np.float64)*np.asarray(normalization['target_std']) +
                np.asarray(normalization['target_mean'])).astype(np.float32)
    with ModelWorkerClient(WORKER_PYTHON, 'learned14_adopted', CHECKPOINT,
                           results=ROOT/'results.json',
                           normalization=ROOT/'normalization.json') as worker:
        actual, digest = worker.predict_adopted(dem, state)
    assert digest == results['variants']['full']['123']['checkpoint_sha256']
    assert actual.shape == (14, 2)
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
