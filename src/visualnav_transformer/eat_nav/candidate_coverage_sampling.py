"""Fixed-seed Quality-disabled reference NoMaD candidate sampling."""
from __future__ import annotations
import numpy as np


def sample_paths(model, dem, goal, seeds):
    """Sample one Path per seed with identical DEM and goal, and no Quality."""
    if len(seeds) != len(set(seeds)) or not seeds:
        raise ValueError('diffusion seeds must be distinct and nonempty')
    paths = []
    for seed in seeds:
        path = model.sample(dem, goal, None, seed=int(seed))[0]
        paths.append(path.detach().cpu().numpy())
    result = np.stack(paths)
    if result.ndim != 3 or result.shape[2] != 2:
        raise ValueError('sampled trajectory shape must be [K,T,2]')
    return result


def prepare_state_candidates(model, dem, goal, seeds, *, slight_scale, strong_scale):
    """Use seed order unchanged and the existing fixed bank, without outcomes."""
    from .candidate_coverage_offline import build_candidate_records
    nomad_paths = sample_paths(model, dem, goal, seeds)
    relative_goal = goal[0].detach().cpu().numpy()
    return build_candidate_records(nomad_paths, seeds, relative_goal,
                                   slight_scale, strong_scale)
