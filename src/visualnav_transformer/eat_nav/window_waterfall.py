"""Compare legacy complete-window and EAT terminal-aware window definitions."""

from __future__ import annotations

from collections import Counter
import math

from .dataset import _build_window, _nearest


def count_window_waterfall(episodes, config, split_map=None):
    counts = Counter()
    current_complete_ids = set()
    legacy_ids = set()
    by_split = {name: Counter() for name in ('train', 'val', 'test')}
    history_ns = round(config.dem_history_interval_sec * 1e9)
    future_ns = round(config.trajectory_interval_sec * 1e9)
    tolerance_ns = round(min(config.trajectory_interval_sec, config.dem_history_interval_sec, .05) * .4 * 1e9)
    for key, rows in episodes.items():
        split = split_map[rows[0]['terrain_seed']] if split_map else None
        local = by_split[split] if split else Counter()
        counts['episodes'] += 1
        counts['raw_records'] += len(rows)
        times = [row['timestamp_ns'] for row in rows]
        for i, row in enumerate(rows):
            if row['dem'] is None:
                continue
            counts['dem_samples'] += 1
            local['dem_samples'] += 1
            history_indices = [_nearest(times, row['timestamp_ns'] - offset * history_ns, tolerance_ns)
                               for offset in range(config.context_size, -1, -1)]
            if any(j is None or rows[j]['dem'] is None for j in history_indices):
                continue
            if any(rows[j]['dem'].shape != (2, config.dem_height_cells, config.dem_width_cells)
                   for j in history_indices):
                continue
            counts['history_complete'] += 1
            local['history_complete'] += 1
            future_indices = [_nearest(times, row['timestamp_ns'] + offset * future_ns, tolerance_ns)
                              for offset in range(1, config.len_traj_pred + 1)]
            if all(j is not None and j > i for j in future_indices):
                counts['raw_future_complete'] += 1
                local['raw_future_complete'] += 1
            window = _build_window(key, rows, i, config)
            if window is None:
                continue
            counts['valid_training_windows'] += 1
            local['valid_training_windows'] += 1
            valid = int(window['action_mask'].sum())
            if valid == config.len_traj_pred:
                counts['training_future_complete'] += 1
                local['training_future_complete'] += 1
                current_complete_ids.add((key, row['step']))
            elif valid == 0:
                counts['training_future_zero_valid'] += 1
                local['training_future_zero_valid'] += 1
            else:
                counts['training_future_terminal_padded'] += 1
                local['training_future_terminal_padded'] += 1
            # Every source lookup is into this episode's row array. Independently
            # reconstruct the last valid point to check the goal source.
            last_valid = next((j for j in reversed(future_indices) if j is not None and j > i), None)
            if last_valid is not None and valid:
                dx, dy = rows[last_valid]['pose'] - row['pose']
                c, s = math.cos(row['yaw']), math.sin(row['yaw'])
                gx, gy = c * dx + s * dy, -s * dx + c * dy
                scale = max(1., math.hypot(gx, gy) / config.goal_max_radius_m)
                if max(abs(window['goal'][0] - gx / scale), abs(window['goal'][1] - gy / scale)) > 1e-5:
                    counts['goal_source_mismatch'] += 1
        # The older duplication audit used regular 5-step anchors, strict
        # ``i + 40 < len(rows)`` and four exact 5-step DEM offsets.
        for i in range(15, len(rows) - 40, 5):
            if all(rows[i - offset]['dem'] is not None for offset in (15, 10, 5, 0)):
                counts['legacy_strict_complete_windows'] += 1
                local['legacy_strict_complete_windows'] += 1
                legacy_ids.add((key, rows[i]['step']))
    counts['current_complete_and_legacy'] = len(current_complete_ids & legacy_ids)
    counts['current_complete_only'] = len(current_complete_ids - legacy_ids)
    for key, step in current_complete_ids - legacy_ids:
        phase = episodes[key][0]['step'] % 5
        counts[f'current_only_episode_start_phase_{phase}'] += 1
    counts['legacy_only'] = len(legacy_ids - current_complete_ids)
    counts['cross_episode_sources'] = 0  # All lookups above use one episode-keyed row array.
    return {'counts': dict(counts), 'by_split': {s: dict(c) for s,c in by_split.items()},
            'cross_episode_sources': 0, 'valid_training_windows': counts['valid_training_windows']}
