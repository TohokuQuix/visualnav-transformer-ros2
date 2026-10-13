"""Separate direction-aware model version; old forward-only model is intact."""

from __future__ import annotations

import torch
from torch import nn

from .evaluator.model import BaselineEvaluator


SCHEMA = 'eatnav_bidirectional_motion_predictor_v1'


class BidirectionalMotionPredictor(BaselineEvaluator):
    def __init__(self, *, feature_dim: int = 64):
        super().__init__(dem_channels=8, state_dim=22, path_points=8,
                         feature_dim=feature_dim)
        self.fusion[0] = nn.Linear(feature_dim + 32 + 32 + 1, 64)
        self.fusion[-1] = nn.Linear(32, 2)
        self.config = {'schema': SCHEMA, 'dem_history': 4, 'dem_size': [20, 20],
                       'dem_channels_per_frame': 2, 'state_dim': 22,
                       'path_points': 8, 'direction_shape': '[B,K,1]',
                       'direction_values': [-1, 1], 'feature_dim': feature_dim,
                       'output_dim': 2, 'label': 'body_frame_delta_xy_2s'}

    def forward(self, dem: torch.Tensor, state: torch.Tensor, paths: torch.Tensor,
                direction: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
        if dem.ndim != 5 or tuple(dem.shape[1:]) != (4, 2, 20, 20) or \
                state.shape != (dem.shape[0], 22) or paths.ndim != 4 or \
                paths.shape[0] != dem.shape[0] or tuple(paths.shape[2:]) != (8, 2) or \
                direction.shape != (*paths.shape[:2], 1) or \
                candidate_mask.shape != paths.shape[:2] or \
                candidate_mask.dtype != torch.bool or \
                not bool(candidate_mask.any(dim=1).all()):
            raise ValueError('bidirectional motion input shape/mask')
        if not torch.isfinite(dem).all() or not torch.isfinite(state).all() or \
                not torch.isfinite(paths).all() or not torch.isfinite(direction).all() or \
                not bool(((direction == 1) | (direction == -1)).all()):
            raise ValueError('bidirectional motion input value')
        batch, count = paths.shape[:2]
        shared = torch.cat((self.dem_encoder(dem.reshape(batch, 8, 20, 20)),
                            self.state_encoder(state)), dim=-1)
        encoded_path = self.path_encoder(paths.reshape(batch * count, -1)).reshape(batch, count, -1)
        fused = torch.cat((shared[:, None].expand(-1, count, -1), encoded_path,
                           direction.to(dtype=shared.dtype)), dim=-1)
        prediction = self.fusion(fused)
        return prediction.masked_fill(~candidate_mask[..., None], float('nan'))
