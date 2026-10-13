"""Small CNN plus Path/state MLP, shared across candidate count K."""

from __future__ import annotations

import torch
from torch import nn


class BaselineEvaluator(nn.Module):
    def __init__(self, dem_channels=8, state_dim=22, path_points=8, feature_dim=64):
        super().__init__()
        self.config = {'dem_channels': dem_channels, 'state_dim': state_dim,
                       'path_points': path_points, 'feature_dim': feature_dim}
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(dem_channels, 16, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(32, feature_dim), nn.ReLU())
        self.path_encoder = nn.Sequential(nn.Linear(path_points*2, 64), nn.ReLU(),
                                          nn.Linear(64, 32), nn.ReLU())
        self.state_encoder = nn.Sequential(nn.Linear(state_dim, 32), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(feature_dim+32+32, 64), nn.ReLU(),
                                    nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, dem, state, paths):
        if dem.ndim != 4 or state.ndim != 2 or paths.ndim != 4 or \
                dem.shape[0] != state.shape[0] or dem.shape[0] != paths.shape[0] or \
                dem.shape[1] != self.config['dem_channels'] or \
                state.shape[1] != self.config['state_dim'] or \
                paths.shape[2:] != (self.config['path_points'], 2):
            raise ValueError('model input shape mismatch')
        batch, candidates = paths.shape[:2]
        shared = torch.cat((self.dem_encoder(dem), self.state_encoder(state)), dim=-1)
        path = self.path_encoder(paths.reshape(batch*candidates, -1)).reshape(batch, candidates, -1)
        fused = torch.cat((shared[:, None].expand(-1, candidates, -1), path), dim=-1)
        return self.fusion(fused).squeeze(-1)
