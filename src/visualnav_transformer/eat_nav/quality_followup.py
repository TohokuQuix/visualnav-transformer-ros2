"""Frozen follow-up diagnostic audit, ladder selection, and restart primitives."""
from __future__ import annotations

from collections import Counter
import numpy as np


from .quality_followup_io import atomic_json, make_stage_marker, sha256_file, verify_stage_marker


def is_complete_window(window: dict, point_count: int) -> bool:
    mask = np.asarray(window['action_mask'])
    return (mask.shape == (point_count,) and mask.dtype == np.bool_ and bool(mask.all())
            and np.asarray(window['action']).shape == (point_count, 2)
            and np.isfinite(window['action']).all() and np.isfinite(window['goal']).all()
            and np.isfinite(window['dem']).all())


def window_identity(window: dict) -> tuple:
    return (tuple(window['episode_key']), int(window['step']))


def validate_complete_splits(splits: dict[str, list[dict]], point_count: int, expected_heldout: int) -> dict:
    from .quality_gate_data import assert_split_integrity
    assert_split_integrity(splits)
    by_split = {}
    seen = set()
    for split, windows in splits.items():
        complete = [w for w in windows if is_complete_window(w, point_count)]
        ids = [window_identity(w) for w in complete]
        if len(set(ids)) != len(ids) or any(i in seen for i in ids):
            raise ValueError('duplicate or cross-split window identity')
        seen.update(ids)
        by_split[split] = {'total':len(windows),'complete':len(complete),'partial_or_zero':len(windows)-len(complete),
                           'terrain_seeds':len({w['terrain_seed'] for w in complete}),
                           'episodes':len({tuple(w['episode_key']) for w in complete})}
    if by_split['test']['complete'] != expected_heldout:
        raise ValueError(f"held-out complete count mismatch: {by_split['test']['complete']} != {expected_heldout}")
    return by_split


def pair_id(pair: dict) -> str:
    return f"{pair['low_index']}:{pair['high_index']}"


def select_ladder_pairs(rows: list[dict], cap: int = 3) -> dict[str, list[dict]]:
    if len(rows) != 64:
        raise ValueError('frozen ladder requires exactly 64 pairs')
    used_windows = set()
    counts = Counter()
    for row in rows:
        lo, hi = int(row['low_index']), int(row['high_index'])
        if lo in used_windows or hi in used_windows or lo == hi:
            raise ValueError('window reused in frozen pairs')
        used_windows.update((lo, hi))
        for name in ('low_episode','high_episode'):
            counts[tuple(row[name])] += 1
        if tuple(row['low_episode']) == tuple(row['high_episode']):
            raise ValueError('same-episode pair')
    if max(counts.values()) > cap:
        raise ValueError('episode cap exceeded in frozen pairs')
    median = float(np.median([row['trajectory_rms_m'] for row in rows]))
    ordered = sorted(rows, key=lambda r: (abs(r['trajectory_rms_m']-median), pair_id(r)))
    b1 = ordered[0]
    b8 = [b1]
    for terrain in sorted({row['terrain_type'] for row in rows} - {b1['terrain_type']}):
        b8.append(next(row for row in ordered if row['terrain_type'] == terrain))
    for row in ordered:
        if len(b8) == 8: break
        if row not in b8: b8.append(row)
    if len(b8) != 8:
        raise ValueError('insufficient frozen pairs for B8')
    return {'B1':[b1], 'B8':b8, 'B64':list(rows)}


def stop_overfit(losses: list[float], ceiling: float, window: int = 50, relative_delta: float = 0.01) -> tuple[bool, str | None]:
    if not losses or not np.isfinite(losses).all():
        raise ValueError('nonfinite or absent fixed-bank loss')
    if losses[-1] <= ceiling:
        return True, 'loss_ceiling'
    if len(losses) >= window:
        past = float(losses[-window])
        improvement = (past - float(losses[-1])) / max(abs(past), 1e-12)
        if improvement < relative_delta:
            return True, 'plateau'
    return False, None


