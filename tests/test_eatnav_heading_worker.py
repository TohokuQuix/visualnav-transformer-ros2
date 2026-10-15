"""The packaged yaw head preserves adopted XY output and stays display-only."""

from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.ros2_phase2.persistent_worker import ModelWorkerClient


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / 'checkpoints/eat_nav/motion14'
HEADING = ROOT / 'checkpoints/eat_nav/motion14_heading/h2a-seed123/best.pt'
PYTHON = Path('/home/kojima/anaconda3/envs/tinyvla/bin/python')


def test_packaged_yaw_head_preserves_xy_predictor_output():
    dem = np.zeros((4, 2, 20, 20), np.float32)
    dem[:, 1] = 1
    state = np.zeros(22, np.float32)
    args = (PYTHON, 'learned14_adopted', TRAIN / 'training/full-seed123/best.pt')
    kwargs = {'results': TRAIN / 'results.json',
              'normalization': TRAIN / 'normalization.json'}
    with ModelWorkerClient(*args, **kwargs) as xy_worker, \
            ModelWorkerClient(*args, heading_checkpoint=HEADING, **kwargs) as yaw_worker:
        xy, _ = xy_worker.predict_adopted(dem, state)
        headed_xy, yaw, _ = yaw_worker.predict_heading(dem, state)
    np.testing.assert_allclose(headed_xy, xy, atol=1e-6, rtol=0)
    assert yaw.shape == (14,)
    assert np.isfinite(yaw).all()
