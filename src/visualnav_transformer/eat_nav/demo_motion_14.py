"""Adopted fourteen-Path Motion Predictor adapter for the Isaac GUI demo."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np
import torch

from .bidirectional_bank import SCHEMA as BANK_SCHEMA, bank
from .motion_14_predictor import CANDIDATE_IDS, Motion14Predictor


DEMO_CHECKPOINT_SHA256 = '83a565f0c1b662c278da20ce851297d890f974f39a86df73b971fd0a2c0d22fa'
TRAINING_MODEL_SHA256 = 'a4737ed81cc4cc1b87c2ca19c83fdb7c7e919ee53ba8e9c5bc6fd0ad97db2678'
TRAINING_CONTRACT_SHA256 = '7e6df9a49b5042f7d4f30012a3cf44d8b23e9c1474f5da9d010e7541d7c90f75'


def _finite_shape(value, shape, name):
    result = np.asarray(value, dtype=np.float32)
    if result.shape != shape or not np.isfinite(result).all():
        raise ValueError(f'{name} shape/nonfinite')
    return result


def candidates_from_frozen_bank(robot_pose_map):
    """Return the training bank in its exact body-frame order and map copy."""
    pose = np.asarray(robot_pose_map, dtype=np.float64)
    if pose.shape != (3,) or not np.isfinite(pose).all():
        raise ValueError('robot pose shape/nonfinite')
    items = bank()
    from .motion_14_stage_a import candidate_role
    if len(items) != 14 or tuple(candidate_role(item.direction, item.arc_angle_deg)
                                 for item in items) != CANDIDATE_IDS:
        raise ValueError('candidate bank/model order mismatch')
    c, s = math.cos(pose[2]), math.sin(pose[2])
    rotation = np.asarray([[c, -s], [s, c]], dtype=np.float64)
    body = {role: item.waypoints for role, item in zip(CANDIDATE_IDS, items)}
    mapped = {role: item.waypoints @ rotation.T + pose[:2]
              for role, item in zip(CANDIDATE_IDS, items)}
    return body, mapped, {'bank': BANK_SCHEMA, 'candidate_count': len(items),
                          'candidate_order': list(CANDIDATE_IDS),
                          'direction': {role: item.direction for role, item in zip(CANDIDATE_IDS, items)},
                          'arc_angle_deg': {role: item.arc_angle_deg for role, item in zip(CANDIDATE_IDS, items)}}


def prepare_motion_14_inputs(dem, state, body_paths, normalization):
    dem = _finite_shape(dem, (4, 2, 20, 20), 'DEM').copy()
    state = _finite_shape(state, (22,), 'Actor state')
    if tuple(body_paths) != CANDIDATE_IDS:
        raise ValueError('candidate order differs from training')
    items = bank()
    paths = np.stack([_finite_shape(body_paths[role], (8, 2), role)
                      for role in CANDIDATE_IDS]).astype(np.float32)
    if not all(np.array_equal(paths[i], item.waypoints.astype(np.float32))
               for i, item in enumerate(items)):
        raise ValueError('candidate Path differs from frozen training bank')
    if not np.isin(dem[:, 1], (0., 1.)).all():
        raise ValueError('DEM valid mask')
    if normalization.get('fitted_from') != 'train_only':
        raise ValueError('normalization provenance')
    height_mean, height_std = (float(normalization[key]) for key in
                               ('dem_height_mean', 'dem_height_std'))
    state_mean = _finite_shape(normalization['state_mean'], (22,), 'state mean')
    state_std = _finite_shape(normalization['state_std'], (22,), 'state std')
    if not np.isfinite((height_mean, height_std)).all() or height_std <= 0 or (state_std <= 0).any():
        raise ValueError('normalization scale')
    valid = dem[:, 1].astype(bool)
    height = dem[:, 0]
    height[valid] = (height[valid] - height_mean) / height_std
    height[~valid] = 0.
    scaled_state = (state - state_mean) / state_std
    direction = np.asarray([item.direction for item in items], dtype=np.float32).reshape(1, 14, 1)
    return dem[None], scaled_state[None], paths[None], direction, np.ones((1, 14), dtype=bool)


def global_goal_progress(predicted_body_xy, robot_pose_map, goal_map):
    predicted = _finite_shape(predicted_body_xy, (14, 2), 'predicted body XY')
    pose = _finite_shape(robot_pose_map, (3,), 'robot map pose')
    goal = _finite_shape(goal_map, (2,), 'global goal map XY')
    delta = goal - pose[:2]
    c, s = math.cos(float(pose[2])), math.sin(float(pose[2]))
    goal_body = np.asarray([c * delta[0] + s * delta[1],
                            -s * delta[0] + c * delta[1]], dtype=np.float32)
    if np.linalg.norm(goal_body) <= 0:
        raise ValueError('global goal coincides with robot')
    return (np.linalg.norm(goal_body) - np.linalg.norm(goal_body[None] - predicted, axis=1)).astype(np.float32)


class DemoMotion14Predictor:
    def __init__(self, model, normalization, checkpoint_sha256):
        self.model = model.eval()
        self.normalization = normalization
        self.checkpoint_sha256 = checkpoint_sha256

    @torch.no_grad()
    def predict(self, dem, state, body_paths, robot_pose_map, goal_map):
        inputs = prepare_motion_14_inputs(dem, state, body_paths, self.normalization)
        tensor_inputs = tuple(torch.from_numpy(value) for value in inputs)
        scaled = self.model(*tensor_inputs)[0].cpu().numpy()
        target_mean = _finite_shape(self.normalization['target_mean'], (2,), 'target mean')
        target_std = _finite_shape(self.normalization['target_std'], (2,), 'target std')
        if (target_std <= 0).any():
            raise ValueError('target normalization scale')
        motion = scaled * target_std + target_mean
        if not np.isfinite(motion).all():
            raise ValueError('predicted motion nonfinite')
        return motion, global_goal_progress(motion, robot_pose_map, goal_map)


def load_motion_14_demo_predictor(path):
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != DEMO_CHECKPOINT_SHA256:
        raise ValueError('adopted 14-Path checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    source = payload.get('source', {})
    if payload.get('variant') != 'full' or payload.get('training_seed') != 123 or \
            payload.get('candidate_count') != 14 or \
            tuple(payload.get('candidate_order', ())) != CANDIDATE_IDS or \
            source.get('model_sha256') != TRAINING_MODEL_SHA256 or \
            source.get('contract_sha256') != TRAINING_CONTRACT_SHA256 or \
            hashlib.sha256(Path(__file__).with_name('motion_14_predictor.py').read_bytes()).hexdigest() != TRAINING_MODEL_SHA256:
        raise ValueError('adopted checkpoint provenance mismatch')
    norm = payload.get('normalization')
    if not isinstance(norm, dict) or norm.get('fitted_from') != 'train_only':
        raise ValueError('14-Path normalization provenance mismatch')
    model = Motion14Predictor('full')
    model.load_state_dict(payload['model_state'], strict=True)
    return DemoMotion14Predictor(model, norm, digest)
