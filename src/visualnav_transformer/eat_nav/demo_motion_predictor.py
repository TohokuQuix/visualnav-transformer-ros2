"""Pinned MI-B Pilot motion inference for the three-Path GUI demonstration.

The pilot learned six candidate slots. The GUI keeps its three bank Paths in
the forward slots and masks three unused slots. This is an experimental input distribution,
not a validated MI-B navigation selector.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

from .mi_b_pilot import PilotPredictor


DEMO_CHECKPOINT_SHA256 = 'ff6cd2e03ee2c4832e35e2d4a935815bbd4ecb0717a8a34d445fc48e6d77bb2d'
TRAINING_MODEL_SHA256 = 'fd34cfbfc235be84f355d94edc3fcb358667d4b1ba2c9d8a62ae0465a61ec357'
TRAINING_CONTRACT_SHA256 = '4e8be933ada6d446f83b513d4ce59991158095277662f46db7bf0ae9b89551e8'


def _finite_shape(value, shape, name):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f'{name} shape/nonfinite')
    return array


def prepare_demo_motion_inputs(dem, state, paths, normalization):
    """Normalize measured inputs and mask unused MI-B candidate slots."""
    dem = _finite_shape(dem, (4, 2, 20, 20), 'DEM')
    state = _finite_shape(state, (22,), 'Actor state')
    paths = _finite_shape(paths, (3, 8, 2), 'candidate Paths')
    if not np.isin(dem[:, 1], (0., 1.)).all():
        raise ValueError('DEM valid mask')
    height_mean = float(normalization['dem_height_mean'])
    height_std = float(normalization['dem_height_std'])
    state_mean = _finite_shape(normalization['state_mean'], (22,), 'state mean')
    state_std = _finite_shape(normalization['state_std'], (22,), 'state std')
    if not np.isfinite([height_mean, height_std]).all() or height_std <= 0 or (state_std <= 0).any():
        raise ValueError('normalization scale')
    scaled_dem = dem.copy()
    valid = scaled_dem[:, 1].astype(bool)
    height = scaled_dem[:, 0]
    height[valid] = (height[valid] - height_mean) / height_std
    height[~valid] = 0.
    scaled_state = (state - state_mean) / state_std
    six_paths = np.zeros((1, 6, 8, 2), dtype=np.float32)
    six_paths[0, :3] = paths
    direction = np.asarray([1, 1, 1, -1, -1, -1], np.float32).reshape(1, 6, 1)
    mask = np.asarray([[True, True, True, False, False, False]], dtype=bool)
    return scaled_dem[None], scaled_state[None], six_paths, direction, mask


def local_goal_progress(predicted_body_xy, goal_body_xy):
    """Score the predicted 2 s body displacement against the current subgoal."""
    predicted = np.asarray(predicted_body_xy, dtype=np.float32)
    goal = _finite_shape(goal_body_xy, (2,), 'local goal')
    if predicted.shape != (3, 2) or not np.isfinite(predicted).all() or np.linalg.norm(goal) <= 0:
        raise ValueError('predicted motion/local goal contract')
    return (np.linalg.norm(goal) - np.linalg.norm(goal[None] - predicted, axis=1)).astype(np.float32)


class DemoMotionPredictor:
    def __init__(self, model, normalization, checkpoint_sha256):
        self.model = model.eval()
        self.normalization = normalization
        self.checkpoint_sha256 = checkpoint_sha256

    @torch.no_grad()
    def predict(self, dem, state, paths, goal_body_xy):
        values = prepare_demo_motion_inputs(dem, state, paths, self.normalization)
        tensors = tuple(torch.from_numpy(value) for value in values)
        scaled = self.model(*tensors)[0, :3].cpu().numpy()
        mean = _finite_shape(self.normalization['target_mean'], (2,), 'target mean')
        std = _finite_shape(self.normalization['target_std'], (2,), 'target std')
        if (std <= 0).any():
            raise ValueError('target normalization scale')
        motion = scaled * std + mean
        if not np.isfinite(motion).all():
            raise ValueError('predicted motion nonfinite')
        return motion, local_goal_progress(motion, goal_body_xy)


def load_demo_motion_predictor(path):
    """Load the validation-selected Full seed-456 checkpoint by SHA."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DEMO_CHECKPOINT_SHA256:
        raise ValueError('MI-B demo checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    source = payload.get('source', {})
    if payload.get('variant') != 'full' or payload.get('training_seed') != 456 or \
            source.get('model_sha256') != TRAINING_MODEL_SHA256 or \
            source.get('contract_sha256') != TRAINING_CONTRACT_SHA256 or \
            hashlib.sha256(Path(__file__).with_name('mi_b_pilot.py').read_bytes()).hexdigest() != TRAINING_MODEL_SHA256:
        raise ValueError('MI-B demo checkpoint provenance mismatch')
    norm = payload.get('normalization')
    if not isinstance(norm, dict) or norm.get('fitted_from') != 'train_only':
        raise ValueError('MI-B normalization provenance mismatch')
    model = PilotPredictor('full')
    model.load_state_dict(payload['model_state'], strict=True)
    return DemoMotionPredictor(model, norm, digest)
