"""Goal-independent predictor of two-second body-frame root displacement."""

from __future__ import annotations

import torch
from torch import nn

from .evaluator.model import BaselineEvaluator


SCHEMA = 'eatnav_motion_predictor_v1'


class MotionPredictor(BaselineEvaluator):
    """Reuse the pilot CNN/state/Path encoders with a two-coordinate head."""

    def __init__(self, *, feature_dim: int = 64):
        super().__init__(dem_channels=8, state_dim=22, path_points=8,
                         feature_dim=feature_dim)
        self.fusion[-1] = nn.Linear(32, 2)
        self.config = {'schema': SCHEMA, 'dem_history': 4, 'dem_channels_per_frame': 2,
                       'dem_size': [20, 20], 'state_dim': 22, 'path_points': 8,
                       'feature_dim': feature_dim, 'output_dim': 2,
                       'label': 'body_frame_delta_xy_2s'}

    def forward(self, dem: torch.Tensor, state: torch.Tensor,
                paths: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        if dem.ndim != 5 or tuple(dem.shape[1:]) != (4, 2, 20, 20) or \
                state.ndim != 2 or state.shape != (dem.shape[0], 22) or \
                paths.ndim != 4 or paths.shape[0] != dem.shape[0] or \
                tuple(paths.shape[2:]) != (8, 2) or \
                candidate_mask.shape != paths.shape[:2] or \
                candidate_mask.dtype != torch.bool or \
                not bool(candidate_mask.any(dim=1).all()):
            raise ValueError('motion model input/mask contract')
        if not torch.isfinite(dem).all() or not torch.isfinite(state).all() or \
                not torch.isfinite(paths).all():
            raise ValueError('nonfinite motion model input')
        batch, candidates = paths.shape[:2]
        flat_dem = dem.reshape(batch, 8, 20, 20)
        shared = torch.cat((self.dem_encoder(flat_dem), self.state_encoder(state)), dim=-1)
        encoded_paths = self.path_encoder(paths.reshape(batch * candidates, -1))
        encoded_paths = encoded_paths.reshape(batch, candidates, -1)
        fused = torch.cat((shared[:, None].expand(-1, candidates, -1), encoded_paths), dim=-1)
        prediction = self.fusion(fused)
        return prediction.masked_fill(~candidate_mask[..., None], float('nan'))


def masked_motion_mse(prediction: torch.Tensor, target: torch.Tensor,
                      valid: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3 or \
            prediction.shape[-1] != 2 or valid.shape != prediction.shape[:2] or \
            valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError('motion loss shape/validity')
    selected_prediction = prediction[valid]
    selected_target = target[valid]
    if not torch.isfinite(selected_prediction).all() or not torch.isfinite(selected_target).all():
        raise ValueError('nonfinite selected motion target/prediction')
    return torch.square(selected_prediction - selected_target).sum(dim=-1).mean()
