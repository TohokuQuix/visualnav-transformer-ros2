"""Stable score API; no ROS or simulator dependency."""

from __future__ import annotations

import numpy as np
import torch

from .preprocessing import prepare_inputs
from .preprocessing import SCHEMA as INPUT_SCHEMA
from .model import BaselineEvaluator
from .labels import SCHEMA as LABEL_SCHEMA, LABELS


class NoValidCandidateError(ValueError):
    pass


class Evaluator:
    def __init__(self, model, normalization, device='cpu', label_transform=None):
        self.model = model.to(device).eval()
        self.normalization = normalization
        self.device = torch.device(device)
        self.label_transform = label_transform or {'mean': 0., 'std': 1.}

    @torch.no_grad()
    def predict(self, dem, robot_state, candidate_paths, candidate_mask=None):
        inputs = prepare_inputs(dem, robot_state, candidate_paths, self.normalization)
        batch, candidates = inputs['paths'].shape[:2]
        if candidate_mask is None:
            mask = np.ones((batch, candidates), dtype=bool)
        else:
            mask = np.asarray(candidate_mask)
            if mask.shape != (batch, candidates) or mask.dtype != np.bool_:
                raise ValueError('candidate mask shape/dtype')
        if not mask.any(axis=1).all():
            raise NoValidCandidateError('no-valid-candidate')
        tensors = {key: torch.as_tensor(value, device=self.device) for key, value in inputs.items()}
        scores = self.model(tensors['dem'], tensors['state'], tensors['paths'])
        scores = scores * float(self.label_transform['std']) + float(self.label_transform['mean'])
        scores = scores.masked_fill(~torch.as_tensor(mask, device=self.device), float('-inf'))
        return scores.cpu().numpy()


def load_checkpoint(path, *, device='cpu', expected_label_type=None,
                    expected_dataset_sha256=None):
    """Load a trusted local pilot checkpoint with explicit schema checks."""
    payload = torch.load(path, map_location=device)
    if payload.get('schema') != 'eatnav_baseline_checkpoint_v1' or \
            payload.get('input_schema') != INPUT_SCHEMA or \
            payload.get('label_schema') != LABEL_SCHEMA or \
            payload.get('label_type') not in LABELS or \
            payload.get('normalization', {}).get('schema') != INPUT_SCHEMA:
        raise ValueError('checkpoint schema mismatch')
    if expected_label_type is not None and payload['label_type'] != expected_label_type:
        raise ValueError('checkpoint label mismatch')
    if expected_dataset_sha256 is not None and \
            payload.get('dataset_sha256') != expected_dataset_sha256:
        raise ValueError('checkpoint dataset provenance mismatch')
    model = BaselineEvaluator(**payload['model_config'])
    model.load_state_dict(payload['model_state'])
    if model.config['dem_channels'] != 8 or model.config['state_dim'] != 22 or \
            model.config['path_points'] != 8 or \
            payload['normalization'].get('path_points') != 8:
        raise ValueError('checkpoint input shape mismatch')
    stats=payload['normalization']
    for key,count in (('state_mean',22),('state_std',22),('path_mean',2),('path_std',2)):
        values=stats.get(key)
        if not isinstance(values,list) or len(values)!=count or not np.isfinite(values).all():
            raise ValueError('checkpoint normalization shape/nonfinite')
    if (np.asarray(stats['state_std'])<=0).any() or (np.asarray(stats['path_std'])<=0).any() or \
            not np.isfinite([stats.get('dem_height_mean'),stats.get('dem_height_std')]).all() or \
            stats['dem_height_std']<=0:
        raise ValueError('checkpoint normalization scale')
    transform = payload.get('label_transform')
    if not isinstance(transform, dict) or not all(isinstance(transform.get(key),(int,float))
            for key in ('mean','std')) or not np.isfinite([transform['mean'],transform['std']]).all() or \
            transform['std'] <= 0:
        raise ValueError('checkpoint label normalization mismatch')
    return Evaluator(model, payload['normalization'], device=device,
                     label_transform=transform), payload
