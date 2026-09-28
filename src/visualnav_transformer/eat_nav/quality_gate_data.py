"""Integrity and paired-condition primitives for the Quality diagnostic."""

from __future__ import annotations

from collections import Counter
import numpy as np


def assert_split_integrity(splits: dict[str, list[dict]]) -> bool:
    seed_split = {}
    episode_split = {}
    for split, windows in splits.items():
        for window in windows:
            seed = int(window["terrain_seed"])
            episode = tuple(window["episode_key"])
            if seed in seed_split and seed_split[seed] != split:
                raise ValueError(f"terrain seed leakage: {seed}")
            if episode in episode_split and episode_split[episode] != split:
                raise ValueError(f"episode leakage: {episode}")
            seed_split[seed] = split
            episode_split[episode] = split
    return True


def build_canonical_pair(low: dict, high: dict) -> tuple[dict, dict]:
    if low["episode_key"] == high["episode_key"]:
        raise ValueError("paired samples must come from different episodes")
    if not low["quality"] < high["quality"]:
        raise ValueError("low and high Quality are out of order")
    common_dem = np.array(low["dem"], copy=True)
    common_goal = np.array(low["goal"], copy=True)
    left = dict(low, dem=common_dem.copy(), goal=common_goal.copy())
    right = dict(high, dem=common_dem.copy(), goal=common_goal.copy())
    if not np.array_equal(left["dem"], right["dem"]) or not np.array_equal(left["goal"], right["goal"]):
        raise AssertionError("canonical inputs must be exactly identical")
    return left, right


def select_overfit_pairs(windows: list[dict], candidates: list[tuple], *,
                         min_rms: float, min_direction: float, count: int,
                         max_episode_uses: int) -> list[tuple]:
    used_windows = set()
    episode_uses = Counter()
    result = []
    for row in sorted(candidates, key=lambda r: (-r[2], r[0], r[1])):
        lo, hi, rms, direction = row[:4]
        if rms < min_rms or direction < min_direction or lo in used_windows or hi in used_windows:
            continue
        a, b = windows[lo], windows[hi]
        if a["episode_key"] == b["episode_key"]:
            continue
        if episode_uses[a["episode_key"]] >= max_episode_uses or episode_uses[b["episode_key"]] >= max_episode_uses:
            continue
        result.append(row)
        used_windows.update((lo, hi))
        episode_uses[a["episode_key"]] += 1
        episode_uses[b["episode_key"]] += 1
        if len(result) >= count:
            break
    return result


def audit_windows(windows: list[dict]) -> dict:
    """Summarize windows without treating terminal-window counts as episode counts."""
    from scipy.stats import spearmanr

    if not windows:
        raise ValueError("empty split")
    q = np.asarray([w["quality"] for w in windows], dtype=np.float64)
    if not np.isfinite(q).all() or np.min(q) < 0 or np.max(q) > 1:
        raise ValueError("nonfinite or out-of-range Quality")
    bins = np.minimum((q * 4).astype(int), 3)
    goals = np.stack([w["goal"] for w in windows])
    actions = np.stack([w["action"] for w in windows])
    masks = np.stack([w["action_mask"] for w in windows]).astype(bool)
    dems = np.stack([w["dem"] for w in windows])
    if not np.isfinite(goals).all() or not np.isfinite(actions).all() or not np.isfinite(dems).all():
        raise ValueError("nonfinite model input or trajectory")
    xy = np.cumsum(actions, axis=1)
    last = np.maximum(masks.sum(axis=1) - 1, 0)
    endpoints = xy[np.arange(len(windows)), last]
    progress = endpoints[:, 0]
    path = np.linalg.norm(actions, axis=-1).sum(axis=1)
    angles = np.arctan2(actions[:, :, 1], actions[:, :, 0])
    turns = np.abs(np.arctan2(np.sin(np.diff(angles, axis=1)), np.cos(np.diff(angles, axis=1))))
    total_turning = (turns * (masks[:, 1:] & masks[:, :-1])).sum(axis=1)
    distances = np.linalg.norm(goals, axis=1)
    valid_dem = dems[:, :, 1] > 0.5
    dem_height = dems[:, :, 0]
    mask_fraction = valid_dem.mean(axis=(1, 2, 3))
    height_mean = (dem_height * valid_dem).sum(axis=(1, 2, 3)) / np.maximum(valid_dem.sum(axis=(1, 2, 3)), 1)
    height_std = np.sqrt((np.square(dem_height - height_mean[:, None, None, None]) * valid_dem).sum(axis=(1, 2, 3)) / np.maximum(valid_dem.sum(axis=(1, 2, 3)), 1))
    def corr(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return None
        value = float(spearmanr(a, b).statistic)
        return value if np.isfinite(value) else None
    by_terrain = {}
    for terrain in sorted({w["terrain_type"] for w in windows}):
        ix = [i for i, w in enumerate(windows) if w["terrain_type"] == terrain]
        by_terrain[terrain] = {"count": len(ix), "q_median": float(np.median(q[ix])),
                               "bins": {str(b): int(np.sum(bins[ix] == b)) for b in range(4)}}
    reasons = ["+".join(w.get("termination_reason", ())) or "none" for w in windows]
    by_reason = {}
    for reason in sorted(set(reasons)):
        ix = [i for i, r in enumerate(reasons) if r == reason]
        by_reason[reason] = {"count": len(ix), "q_median": float(np.median(q[ix])),
                             "bins": {str(b): int(np.sum(bins[ix] == b)) for b in range(4)}}
    stats = {"forward_progress": progress, "endpoint_x": endpoints[:, 0],
             "endpoint_y": endpoints[:, 1], "path_length": path,
             "total_turning": total_turning,
             **{f"xy_delta_{point}_{axis}": actions[:, point, axis]
                for point in range(actions.shape[1]) for axis in range(2)}}
    return {"window_count": len(windows), "episode_count": len({tuple(w["episode_key"]) for w in windows}),
            "q_bin_counts": {str(b): int(np.sum(bins == b)) for b in range(4) if np.any(bins == b)},
            "q_quantiles": np.quantile(q, [0, .1, .25, .5, .75, .9, 1]).tolist(),
            "terrain_type_counts": dict(Counter(w["terrain_type"] for w in windows)),
            "q_by_terrain": by_terrain,
            "termination_window_counts": dict(Counter(reasons)), "q_by_termination": by_reason,
            "valid_action_fraction": float(masks.mean()),
            "q_spearman": {"goal_distance": corr(q, distances), "goal_x": corr(q, goals[:, 0]),
                           "goal_y": corr(q, goals[:, 1]), "dem_mask_fraction": corr(q, mask_fraction),
                           "dem_height_mean": corr(q, height_mean), "dem_height_std": corr(q, height_std),
                           **{name: corr(q, value) for name, value in stats.items()}},
            "goal_distance_spearman": {name: corr(distances, value) for name, value in stats.items()},
            "q_vs_goal_distance_by_bin": {str(b): float(np.median(distances[bins == b]))
                                          for b in range(4) if np.any(bins == b)}}


def raw_dem_status(row: dict) -> str:
    """DEM is intentionally absent between 4 Hz samples on the 20 Hz raw stream."""
    if 'dem' not in row or row['dem'] is None:
        return 'not_sampled'
    dem = row['dem']
    return 'invalid' if dem.get('transformed') is None or dem.get('error') else 'valid'
