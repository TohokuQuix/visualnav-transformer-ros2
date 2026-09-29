"""Outcome-free candidate identity and path geometry for NoMaD coverage."""

from __future__ import annotations

import math

import numpy as np


_BANK_NAMES = ("nominal", "slight_left", "slight_right", "strong_left", "strong_right")


def candidate_sets(nomad_paths, diffusion_seeds, bank_paths):
    """Keep every seeded candidate, including invalid ones, in frozen order."""
    nomad = np.asarray(nomad_paths, dtype=float)
    bank = np.asarray(bank_paths, dtype=float)
    if nomad.ndim != 3 or nomad.shape[2] != 2 or nomad.shape[1] < 2 or len(nomad) != 32:
        raise ValueError("NoMaD candidates must have shape [32,T,2], T>=2")
    if len(diffusion_seeds) != 32 or len(set(diffusion_seeds)) != 32:
        raise ValueError("32 unique diffusion seeds required")
    if bank.shape != (5, nomad.shape[1], 2):
        raise ValueError("fixed bank horizon must match NoMaD")
    def record(points, *, candidate_id, seed=None, bank_name=None):
        finite = bool(np.isfinite(points).all())
        return {"candidate_id": candidate_id, "seed": seed, "bank_name": bank_name,
                "finite": finite,
                "points": [[float(v) if math.isfinite(v) else None for v in point]
                           for point in points]}
    nomad_rows = [record(path, candidate_id=f"nomad-seed-{seed}", seed=int(seed))
                  for seed, path in zip(diffusion_seeds, nomad)]
    bank_rows = [record(path, candidate_id=f"bank-{name}", bank_name=name)
                 for name, path in zip(_BANK_NAMES, bank)]
    return {"nomad_k8": nomad_rows[:8], "nomad_k32": nomad_rows,
            "fixed_bank": bank_rows,
            "hybrid_k8": nomad_rows[:8] + bank_rows,
            "hybrid_k32": nomad_rows + bank_rows}


def pairwise_path_rms(first, second):
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 2:
        raise ValueError("paths must have matching [T,2] shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite path")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def _cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _self_intersection(points):
    # Adjacent segments share a waypoint and are not intersections.
    for i in range(len(points) - 1):
        for j in range(i + 2, len(points) - 1):
            a, b = points[i:i + 2]
            c, d = points[j:j + 2]
            if (_cross(a, b, c) * _cross(a, b, d) < 0 and
                    _cross(c, d, a) * _cross(c, d, b) < 0):
                return True
    return False


def path_metrics(path):
    """Measure base-frame trajectory geometry without outcomes or Critic data."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or len(path) < 2:
        raise ValueError("path must have shape [T,2] with T>=2")
    if not np.isfinite(path).all():
        raise ValueError("nonfinite path")
    points = np.vstack((np.zeros((1, 2)), path))
    delta = np.diff(points, axis=0)
    angles = np.unwrap(np.arctan2(delta[:, 1], delta[:, 0]))
    return {
        "endpoint_xy_m": path[-1].tolist(),
        "endpoint_heading_rad": float(angles[-1]),
        "path_length_m": float(np.linalg.norm(delta, axis=1).sum()),
        "curvature_rad": float(np.abs(np.diff(angles)).sum()),
        "lateral_displacement_m": float(path[-1, 1]),
        "forward_m": float(np.maximum(delta[:, 0], 0).sum()),
        "reverse_m": float(np.maximum(-delta[:, 0], 0).sum()),
        "self_intersection": _self_intersection(points),
    }


def path_distribution(paths, *, cluster_rms_m):
    """Quantify pairwise geometry and independent path modes."""
    paths = np.asarray(paths, dtype=float)
    if paths.ndim != 3 or paths.shape[2] != 2 or len(paths) < 2:
        raise ValueError('paths must have shape [K,T,2], K>=2')
    if not np.isfinite(paths).all():
        raise ValueError('nonfinite path')
    if cluster_rms_m <= 0:
        raise ValueError('cluster threshold must be positive')
    flat = paths.reshape(len(paths), -1)
    pairwise = np.sqrt(np.mean((flat[:, None] - flat[None, :]) ** 2, axis=-1))
    upper = pairwise[np.triu_indices(len(paths), 1)]
    centered = flat - flat.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    variance = singular ** 2
    effective_rank = float(variance.sum() ** 2 / np.square(variance).sum()) if variance.sum() else 0.
    parent = list(range(len(paths)))
    def find(index):
        while parent[index] != index:
            index = parent[index]
        return index
    for i in range(len(paths)):
        for j in range(i + 1, len(paths)):
            if pairwise[i, j] < cluster_rms_m:
                parent[find(j)] = find(i)
    return {
        'pairwise_rms_median_m': float(np.median(upper)),
        'pairwise_rms_p95_m': float(np.quantile(upper, .95)),
        'effective_rank': effective_rank,
        'endpoint_covariance_m2': np.cov(paths[:, -1, :].T).tolist(),
        'path_cluster_count': len({find(i) for i in range(len(paths))}),
    }


def build_candidate_records(nomad_paths, diffusion_seeds, goal, slight_scale, strong_scale):
    """Join frozen NoMaD samples with the unchanged endpoint-aligned bank."""
    from .feasible_set import maneuver_bank
    paths = np.asarray(nomad_paths, dtype=float)
    if paths.ndim != 3 or len(paths) != 32 or not np.isfinite(paths[0]).all():
        raise ValueError('nominal NoMaD candidate is invalid')
    bank = maneuver_bank(paths[0], goal, slight_scale, strong_scale)
    ordered = np.stack([bank[name] for name in _BANK_NAMES])
    return candidate_sets(paths, diffusion_seeds, ordered)
