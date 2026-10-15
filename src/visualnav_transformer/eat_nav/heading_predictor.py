"""Frozen XY predictor with a separately trained two-second yaw head."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch
from torch import nn

from .motion_14_predictor import CANDIDATE_IDS, Motion14Predictor


HEADING_CHECKPOINT_SHA256 = '4984b3fa636e7378f3653b996bc6f6f94420946f3fd26877a56761851bc06641'


class HeadingMotion14Predictor(nn.Module):
    def __init__(self, base: Motion14Predictor):
        super().__init__()
        if base.variant != 'full':
            raise ValueError('heading head requires adopted Full XY model')
        self.base = base.eval()
        for parameter in self.base.parameters():
            parameter.requires_grad_(False)
        self.yaw_head = nn.Sequential(nn.Linear(32, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, dem, state, paths, direction, mask):
        xy = self.base(dem, state, paths, direction, mask)
        n, k = paths.shape[:2]
        with torch.no_grad():
            dem_feature = self.base.dem_encoder(dem.reshape(n, 8, 20, 20))
            state_feature = self.base.state_encoder(state)
            path_feature = self.base.path_encoder(
                torch.cat((paths.reshape(n, k, 16), direction), dim=-1))
            shared = torch.cat((dem_feature, state_feature), dim=-1)[:, None].expand(-1, k, -1)
            hidden = self.base.head[:4](torch.cat((shared, path_feature), dim=-1))
        yaw = self.yaw_head(hidden).masked_fill(~mask[..., None], 0)
        return torch.cat((xy, yaw), dim=-1)


def load_heading_model(base, normalization, base_checkpoint_sha256, checkpoint_path):
    """Load the frozen yaw head while keeping the adopted XY base unchanged."""
    path = Path(checkpoint_path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != HEADING_CHECKPOINT_SHA256:
        raise ValueError('heading checkpoint SHA mismatch')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload.get('schema') != 'eatnav_heading_h2a_frozen_xy_v1' or \
            payload.get('base_checkpoint_sha256') != base_checkpoint_sha256 or \
            tuple(payload.get('candidate_order', ())) != CANDIDATE_IDS or \
            payload.get('normalization') != normalization:
        raise ValueError('heading checkpoint provenance mismatch')
    model = HeadingMotion14Predictor(base)
    model.yaw_head.load_state_dict(payload['yaw_head_state'], strict=True)
    return model.eval()
