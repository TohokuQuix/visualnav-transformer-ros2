"""Goal-free MI-B architecture extended to the frozen 14-candidate bank."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .bidirectional_bank import bank
from .motion_14_stage_a import candidate_role


CANDIDATE_IDS = tuple(candidate_role(path.direction, path.arc_angle_deg)
                      for path in bank())
CANDIDATE_COUNT = len(CANDIDATE_IDS)


def contrast_indices() -> tuple[tuple[int, int], ...]:
    return tuple((arc, direct) for direct in (0, 7)
                 for arc in range(direct + 1, direct + 7))


def validate_grouped_arrays(arrays: dict) -> None:
    n = len(arrays['dem'])
    shapes = {'dem': (n, 4, 2, 20, 20), 'state': (n, 22),
              'paths': (n, CANDIDATE_COUNT, 8, 2),
              'direction': (n, CANDIDATE_COUNT, 1),
              'mask': (n, CANDIDATE_COUNT),
              'target': (n, CANDIDATE_COUNT, 2)}
    for name, shape in shapes.items():
        value = np.asarray(arrays[name])
        if value.shape != shape or (name != 'mask' and not np.isfinite(value).all()):
            raise ValueError(f'{name} shape/finite contract')
    if np.asarray(arrays['mask']).dtype != np.bool_ or not arrays['mask'].all():
        raise ValueError('fourteen complete candidate masks required')
    expected = np.asarray([1] * 7 + [-1] * 7)
    if not np.array_equal(arrays['direction'][:, :, 0], np.tile(expected, (n, 1))):
        raise ValueError('candidate direction/order contract')
    if not np.isin(arrays['dem'][:, :, 1], [0, 1]).all():
        raise ValueError('DEM validity mask contract')


class Motion14Predictor(nn.Module):
    """The MI-B Full/ablation CNN+MLP with candidate axis read from input."""

    def __init__(self, variant: str, feature_dim: int = 32):
        super().__init__()
        if variant not in ('full', 'no_dem', 'no_body', 'no_path'):
            raise ValueError('unknown model variant')
        self.variant = variant
        self.dem_encoder = nn.Sequential(
            nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(12, 24, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(24, 24, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Linear(24, feature_dim), nn.ReLU())
        self.state_encoder = nn.Sequential(nn.Linear(22, 32), nn.ReLU(),
                                           nn.Linear(32, 32), nn.ReLU())
        self.path_encoder = nn.Sequential(nn.Linear(17, 48), nn.ReLU(),
                                          nn.Linear(48, 32), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(feature_dim + 64, 64), nn.ReLU(),
                                  nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, dem: torch.Tensor, state: torch.Tensor,
                paths: torch.Tensor, direction: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        n, k = paths.shape[:2]
        if k != CANDIDATE_COUNT or tuple(dem.shape) != (n, 4, 2, 20, 20) or \
                state.shape != (n, 22) or paths.shape != (n, k, 8, 2) or \
                direction.shape != (n, k, 1) or mask.shape != (n, k) or \
                mask.dtype != torch.bool:
            raise ValueError('14-candidate model input contract')
        if self.variant == 'no_dem':
            dem = torch.zeros_like(dem)
        if self.variant == 'no_body':
            state = torch.zeros_like(state)
        if self.variant == 'no_path':
            paths = torch.zeros_like(paths)
        dem_feature = self.dem_encoder(dem.reshape(n, 8, 20, 20))
        state_feature = self.state_encoder(state)
        path_feature = self.path_encoder(torch.cat((paths.reshape(n, k, 16), direction), -1))
        shared = torch.cat((dem_feature, state_feature), -1)[:, None].expand(-1, k, -1)
        return self.head(torch.cat((shared, path_feature), -1)).masked_fill(~mask[..., None], 0)


def normalized_pair_loss(pred: torch.Tensor, target: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
    if pred.shape != target.shape or pred.ndim != 3 or \
            pred.shape[1:] != (CANDIDATE_COUNT, 2) or \
            mask.shape != pred.shape[:2] or mask.dtype != torch.bool or not mask.any():
        raise ValueError('fourteen-pair loss shape')
    absolute = (pred[mask] - target[mask]).square().sum(-1).mean()
    contrasts = []
    for arc, direct in contrast_indices():
        valid = mask[:, arc] & mask[:, direct]
        if valid.any():
            residual = (pred[:, arc] - pred[:, direct]) - (target[:, arc] - target[:, direct])
            contrasts.append(residual[valid].square().sum(-1))
    if not contrasts:
        raise ValueError('no paired Arc/Direct samples')
    return absolute + torch.cat(contrasts).mean()
