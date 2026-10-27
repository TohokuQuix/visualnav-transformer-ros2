"""Post-process passive long-horizon traces into event and censor labels."""

from __future__ import annotations

import math


SCHEMA = 'eatnav_passive_long_horizon_events_v1'
EVENT_NAMES = ('near_fall', 'tip_over', 'linear_stall', 'turn_stall')


def effective_tilt_deg(root_state) -> float:
    """Angle between the chassis +Z axis and world +Z, in degrees."""
    w, x, y, z = map(float, root_state[3:7])
    del w
    cosine = max(-1.0, min(1.0, 1.0 - 2.0 * (x * x + y * y)))
    return math.degrees(math.acos(cosine))


def _state_for_row(row):
    terminal = row.get('terminal')
    if terminal and terminal.get('pre_reset_state'):
        return terminal['pre_reset_state']['root_state']
    return row.get('post_root_state', row['root_state'])


def _termination_reasons(row):
    terminal = row.get('terminal') or {}
    reasons = terminal.get('termination_reasons')
    if reasons is not None:
        return set(reasons)
    reason = terminal.get('termination_reason')
    return {reason} if reason else set()


def _event_rows(rows):
    events = {name: [False] * len(rows) for name in EVENT_NAMES}
    tilts = [effective_tilt_deg(_state_for_row(row)) for row in rows]

    high_tilt_start = None
    for index, tilt in enumerate(tilts):
        above_near_fall = tilt >= 60.0 - 1e-6
        high_tilt_start = index if above_near_fall and high_tilt_start is None else \
            high_tilt_start if above_near_fall else None
        if high_tilt_start is not None and index - high_tilt_start + 1 >= 5:
            events['near_fall'][high_tilt_start] = True

        if tilt >= 85.0 or 'tilt_over' in _termination_reasons(rows[index]):
            events['tip_over'][index] = True

    cycle_indices = {}
    for index, row in enumerate(rows):
        cycle_indices.setdefault(int(row.get('candidate_cycle', 0)), []).append(index)
    for indices in cycle_indices.values():
        if len(indices) < 40:
            continue
        first = rows[indices[0]]['follower_diagnostic']
        last_index = indices[-1]
        last = rows[last_index]['follower_diagnostic']
        commanded_linear = float(last.get('commanded_distance', 0.0))
        actual_linear = float(last.get('actual_distance', 0.0))
        first_remaining = float(first.get('remaining_path_distance', 0.0))
        last_remaining = float(last.get('remaining_path_distance', 0.0))
        progress = first_remaining - last_remaining
        if commanded_linear >= 0.20 and actual_linear / commanded_linear <= 0.25 and progress <= 0.05:
            events['linear_stall'][last_index] = True

        turn_rows = [rows[index] for index in indices]
        commanded_rates = [abs(float(row['follower_command'][2])) for row in turn_rows]
        measured_rates = [abs(float(row['body_angular_velocity_radps'][2]))
                          for row in turn_rows
                          if row.get('body_angular_velocity_radps') is not None]
        if len(measured_rates) == len(indices) and all(rate >= 0.01 for rate in commanded_rates) and \
                all(measured <= commanded * 0.50
                    for measured, commanded in zip(measured_rates, commanded_rates)):
            events['turn_stall'][last_index] = True

    return events, tilts


def label_trace(rows, *, control_hz: int = 20, horizons_sec=(1, 2, 5)) -> dict:
    """Create future-event labels; unknown future past trial end stays censored."""
    if not rows or control_hz <= 0 or any(horizon <= 0 for horizon in horizons_sec):
        raise ValueError('non-empty trace and positive rates/horizons required')
    for index, row in enumerate(rows):
        if row.get('control_step') != index:
            raise ValueError('trace control steps must be contiguous from zero')
        if index and row['logical_sim_time_ns'] - rows[index - 1]['logical_sim_time_ns'] != \
                round(1e9 / control_hz):
            raise ValueError('trace timestamps must match control rate')

    event_rows, tilts = _event_rows(rows)
    labels = []
    for index, row in enumerate(rows):
        horizon_labels = {}
        for horizon in horizons_sec:
            steps = int(horizon * control_hz)
            future_end = min(len(rows), index + 1 + steps)
            future = range(index + 1, future_end)
            observed = future_end - (index + 1) >= steps
            event_values = {}
            for event_name in EVENT_NAMES:
                occurred = any(event_rows[event_name][future_index] for future_index in future)
                event_values[event_name] = True if occurred else False if observed else None
            any_event = any(value is True for value in event_values.values())
            event_values['event_free'] = False if any_event else True if observed else None
            horizon_labels[f'{horizon}s'] = {
                'observed': observed or any_event,
                'censored': not observed and not any_event,
                'events': event_values,
            }
        labels.append({
            'control_step': index,
            'effective_tilt_deg': tilts[index],
            'events_at_step': {name: event_rows[name][index] for name in EVENT_NAMES},
            'horizons': horizon_labels,
        })

    return {
        'schema': SCHEMA,
        'control_hz': control_hz,
        'horizons_sec': list(horizons_sec),
        'window_count': len(labels),
        'events': {name: sum(values) for name, values in event_rows.items()},
        'labels': labels,
    }


def label_traces(traces, *, control_hz: int = 20, horizons_sec=(1, 2, 5)) -> list[dict]:
    return [label_trace(rows, control_hz=control_hz, horizons_sec=horizons_sec)
            for rows in traces]