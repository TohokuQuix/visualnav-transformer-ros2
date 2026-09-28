"""Fixed-terrain, window-preserving permutation for post-hoc Quality Probes."""
from __future__ import annotations
import numpy as np


def terrain_permutation(terrain,rng):
    terrain=np.asarray(terrain)
    result=np.arange(len(terrain))
    for name in np.unique(terrain):
        indices=np.flatnonzero(terrain==name)
        result[indices]=rng.permutation(indices)
    return result
