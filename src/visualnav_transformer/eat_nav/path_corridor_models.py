"""Frozen A1–A4 small evaluator architectures for the offline comparison."""
from __future__ import annotations

import torch
from torch import nn


class _Conv2D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1), nn.GroupNorm(4, 16), nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=(2, 1), padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.Conv2d(32, 32, 3, stride=(2, 2), padding=1), nn.GroupNorm(8, 32), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, 64), nn.SiLU())

    def forward(self, value):
        return self.features(value)


class PathCorridorEvaluator(nn.Module):
    def __init__(self, variant: str, common_dim: int, profile_dim: int):
        super().__init__()
        if variant not in ('A1', 'A2', 'A3', 'A4'):
            raise ValueError(f'unknown architecture {variant}')
        self.variant = variant
        self.common = nn.Sequential(nn.Linear(common_dim, 64), nn.SiLU(),
                                    nn.Linear(64, 64), nn.SiLU())
        if variant == 'A1':
            self.dem = nn.Sequential(nn.Flatten(), nn.Linear(8 * 20 * 20, 16), nn.SiLU(),
                                     nn.Linear(16, 64), nn.SiLU())
        elif variant == 'A2':
            self.dem = nn.Sequential(nn.Conv1d(profile_dim, 16, 3, padding=1),
                                     nn.GroupNorm(4, 16), nn.SiLU(),
                                     nn.Conv1d(16, 32, 3, stride=2, padding=1),
                                     nn.GroupNorm(8, 32), nn.SiLU(),
                                     nn.AdaptiveAvgPool1d(1), nn.Flatten(),
                                     nn.Linear(32, 64), nn.SiLU())
        else:
            self.dem = _Conv2D()
        self.full_branch = _Conv2D() if variant == 'A4' else None
        fusion_dim = 192 if variant == 'A4' else 128
        self.fusion = nn.Sequential(nn.Linear(fusion_dim, 384), nn.SiLU(),
                                    nn.Linear(384, 384), nn.SiLU(), nn.Linear(384, 6))

    def forward(self, common, full_dem, corridor, profile):
        common_embedding = self.common(common)
        if self.variant == 'A1':
            dem_embedding = self.dem(full_dem)
        elif self.variant == 'A2':
            dem_embedding = self.dem(profile)
        else:
            dem_embedding = self.dem(corridor)
        inputs = [common_embedding, dem_embedding]
        if self.full_branch is not None:
            inputs.append(self.full_branch(full_dem))
        return self.fusion(torch.cat(inputs, dim=1))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
