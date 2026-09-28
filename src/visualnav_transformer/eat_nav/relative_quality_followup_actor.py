"""Strict recorded actor observation availability check."""
from __future__ import annotations
import numpy as np


def actor_observation_status(row,expected_dim):
    if 'actor_observation' not in row:return 'missing'
    value=np.asarray(row['actor_observation'],dtype=float)
    if value.shape!=(expected_dim,) or not np.isfinite(value).all():return 'invalid'
    return 'available'
