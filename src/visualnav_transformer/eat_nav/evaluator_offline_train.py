"""Small offline-only supervised smoke helpers; no simulator or ROS imports."""

from __future__ import annotations

import numpy as np


def fit_scaler(values, groups):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or len(values) != len(groups) or not np.isfinite(values).all():
        raise ValueError("invalid scaler input")
    unique = sorted(set(groups))
    group_means = np.stack([values[np.asarray(groups) == group].mean(axis=0) for group in unique])
    mean = group_means.mean(axis=0)
    group_second = np.stack([(values[np.asarray(groups) == group] ** 2).mean(axis=0)
                             for group in unique]).mean(axis=0)
    scale = np.sqrt(np.maximum(group_second - mean**2, 0.0))
    scale[scale < 1e-6] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def loso_splits(groups):
    if not groups:
        raise ValueError("no groups")
    return [(held, [i for i, group in enumerate(groups) if group != held],
             [i for i, group in enumerate(groups) if group == held])
            for held in sorted(set(groups))]


def pareto_set(scores):
    """Names of nondominated candidates for larger-is-better reward and goal."""
    names = list(scores)
    values = {name: np.asarray(scores[name], dtype=float) for name in names}
    if any(v.shape != (2,) or not np.isfinite(v).all() for v in values.values()):
        raise ValueError("invalid reward/goal vector")
    return {name for name in names if not any(
        other != name and np.all(values[other] >= values[name]) and
        np.any(values[other] > values[name]) for other in names)}
