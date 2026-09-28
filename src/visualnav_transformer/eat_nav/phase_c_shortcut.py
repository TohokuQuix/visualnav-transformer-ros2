"""Endpoint and shape controls for the frozen Phase B trajectory shortcut audit."""
from __future__ import annotations

import numpy as np


def trajectory_input(action, variant):
    action = np.asarray(action, dtype=np.float32)
    if action.ndim != 3 or action.shape[2] != 2:
        raise ValueError('action must have shape [N,T,2]')
    if variant == 'GT':
        return action.copy()
    if variant == 'G':
        return np.zeros_like(action)
    endpoint = action.sum(axis=1)
    if variant == 'GTE':
        result = np.zeros_like(action)
        result[:, -1] = endpoint
        return result
    if variant == 'GTS':
        # Residual cumulative path after removing the origin-to-endpoint line.
        # Its increments sum exactly zero, so the endpoint cannot be reconstructed.
        return action - endpoint[:, None, :] / action.shape[1]
    raise ValueError(f'unknown shortcut variant: {variant}')


def endpoint_fixed_shuffle(action, terrain, seed=613):
    action = np.asarray(action, dtype=np.float32)
    terrain = np.asarray(terrain)
    if len(action) != len(terrain):
        raise ValueError('terrain/action length mismatch')
    residual = trajectory_input(action, 'GTS')
    rng = np.random.default_rng(seed)
    donor = np.arange(len(action))
    for kind in np.unique(terrain):
        ids = np.flatnonzero(terrain == kind)
        if len(ids) < 2:
            raise ValueError(f'terrain {kind} has fewer than two shuffle donors')
        order = rng.permutation(ids)
        donor[order] = np.roll(order, 1)
    endpoint = action.sum(axis=1)
    return residual[donor] + endpoint[:, None, :] / action.shape[1]
