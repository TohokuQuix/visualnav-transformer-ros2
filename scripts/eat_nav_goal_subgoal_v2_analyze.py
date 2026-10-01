"""Evaluate only a complete, SHA-verified 44-run v2 pilot.

The four selected states are the statistical units.  Goal/seed pairs stay
within each state; no held-out state or outcome-dependent reselection is used.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from visualnav_transformer.eat_nav.command_preview_long_horizon import endpoint_metrics, post_root
from eat_nav_goal_subgoal_v2_run import OUT, read, sha, run_name, verify_saved_run
from eat_nav_goal_subgoal_v2_select import atomic

GOAL_TOLERANCE_M = .2
MODE_SEPARATION_M = .05
DELTA_V_EPSILON = .01


def trace(entry):
    with gzip.open(OUT / 'runs' / run_name(entry) / 'live-0.trace.json.gz', 'rt') as stream:
        return json.load(stream)


def yaw_from_root(root):
    w, x, y, z = map(float, root[3:7])
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def goal_map(summary, goal_xy):
    xy = np.asarray(summary['map_path'][0], dtype=float)
    yaw = yaw_from_root(summary['anchor']['root_state'])
    c, s = math.cos(yaw), math.sin(yaw)
    gx, gy = map(float, goal_xy)
    return xy + np.array([c * gx - s * gy, s * gx + c * gy])


def polyline_distance(points, polyline):
    p = np.asarray(points, dtype=float)
    path = np.asarray(polyline, dtype=float)
    distances = []
    for a, b in zip(path[:-1], path[1:]):
        direction = b - a
        denom = float(direction @ direction)
        fraction = np.clip((p - a) @ direction / denom, 0, 1) if denom else np.zeros(len(p))
        closest = a + fraction[:, None] * direction
        distances.append(np.linalg.norm(p - closest, axis=1))
    return np.min(np.stack(distances), axis=0)


def path_distance(a, b):
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != y.shape:
        raise ValueError('Path shapes differ')
    return float(np.sqrt(np.mean(np.sum((x - y) ** 2, axis=-1))))


def series_distance(a, b, key):
    x = np.asarray([row[key] for row in a], dtype=float)
    y = np.asarray([row[key] for row in b], dtype=float)
    n = min(len(x), len(y), 40)
    if n != 40 or x.shape[1:] != y.shape[1:]:
        return None
    return float(np.sqrt(np.mean((x[:n] - y[:n]) ** 2)))


def realized_distance(a, b):
    if min(len(a), len(b)) < 40:
        return None
    def xy(rows):
        origin = np.asarray(rows[0]['root_state'][:2], dtype=float)
        return np.asarray([post_root(row)[:2] for row in rows[:40]], dtype=float) - origin
    return path_distance(xy(a), xy(b))


def greedy_modes(rows, separation):
    representatives = []
    for row in rows:
        if not representatives or all(row['distance_to'][other['run_id']] > separation for other in representatives):
            representatives.append(row)
    return len(representatives)


def run_metrics(entry, summary, rows):
    if not rows or not all(np.isfinite(row['root_state']).all() for row in rows):
        raise ValueError('empty or nonfinite trace')
    anchor_xy = np.asarray(summary['map_path'][0], dtype=float)
    target = goal_map(summary, entry['goal_xy'])
    root = np.asarray([post_root(row)[:2] for row in rows], dtype=float)
    planned = np.asarray(summary['map_path'], dtype=float)
    distance = np.linalg.norm(root - target, axis=1)
    first_reach = np.flatnonzero(distance <= GOAL_TOLERANCE_M)
    primary = min(len(rows), 40)
    secondary = min(len(rows), 120)
    path_error = polyline_distance(root, planned)
    primary_reach = bool(np.any(distance[:primary] <= GOAL_TOLERANCE_M))
    reach_time = float((first_reach[0] + 1) / 20) if len(first_reach) else None
    post_reach_deviation = (float(np.max(distance[first_reach[0]:])) if len(first_reach) else None)
    initial_distance = float(np.linalg.norm(anchor_xy - target))
    progress = float((initial_distance - distance[primary - 1]) / initial_distance) if initial_distance else None
    common = {'state_id': entry['state_id'], 'terrain': entry['terrain'], 'run_id': entry['run_id'],
              'goal_class': entry['goal_class'], 'goal_id': entry['goal_id'], 'seed': entry['seed'],
              'train_support_class': entry['goal_class'], 'control_steps': len(rows),
              'terminated_early': bool(rows[-1]['terminal']),
              'catastrophic_termination': bool(rows[-1]['terminal'] and rows[-1]['terminal'].get('terminated')),
              'goal_xy_map': target.tolist(), 'goal_initial_distance_m': initial_distance,
              'planned_endpoint_error_m': float(np.linalg.norm(planned[-1] - target)),
              'path_sha256': entry['base_path_sha256'],
              'input_sha256': sha(OUT / 'runs' / run_name(entry) / 'run-complete.json')}
    horizons = {}
    for name, count in [('primary_0_2s', primary), ('secondary_0_6s', secondary)]:
        d = distance[:count]
        e = path_error[:count]
        horizons[name] = {'observed_steps': count, 'complete': count == (40 if name.startswith('primary') else 120),
                          'endpoint_error_m': float(d[-1]),
                          'reach_0p2m': bool(np.any(d <= GOAL_TOLERANCE_M)),
                          'cross_track_p50_m': float(np.quantile(e, .5)),
                          'cross_track_p95_m': float(np.quantile(e, .95)),
                          'path_endpoint_distance_m': float(np.linalg.norm(root[count - 1] - planned[-1])),
                          'progress_ratio': float((initial_distance - d[-1]) / initial_distance) if initial_distance else None}
    common.update({'reach_time_sec': reach_time, 'post_reach_max_deviation_m': post_reach_deviation,
                   'primary_reach': primary_reach, 'primary_progress_ratio': progress,
                   'horizons': horizons})
    return common


def main():
    schedule = read(OUT / 'run-schedule-v2.json')['runs']
    marker = read(OUT / 'v8-numeric-gate.done.json')
    if marker['status'] != 'pass' or marker['schedule_sha256'] != sha(OUT / 'run-schedule-v2.json'):
        raise ValueError('44-run completion marker missing or schedule changed')
    entries = [row for row in schedule if row['kind'] != 'anchor']
    if len(entries) != 44:
        raise ValueError('frozen live count differs')
    for entry in entries:
        verify_saved_run(entry)
        result = read(OUT / f"run-{entry['order']:02d}-numeric-gate.json")
        if result['dem_history_numeric_gate']['status'] != 'pass':
            raise ValueError('per-run numeric DEM gate missing')
    traces = {e['run_id']: trace(e) for e in entries}
    summaries = {e['run_id']: read(OUT / 'runs' / run_name(e) / 'live-0.json') for e in entries}
    by_state = defaultdict(list)
    for entry in entries:
        by_state[entry['state_id']].append(entry)
    execution = []
    relative = []
    diversity = []
    schema = []
    for state_id, group in by_state.items():
        duplicates = [e for e in group if e['kind'] == 'duplicate']
        candidates = [e for e in group if e['kind'] == 'candidate']
        gate = read(OUT / f'duplicate-gate-{state_id}.json')
        if gate['status'] != 'pass' or len(duplicates) != 3 or len(candidates) != 8:
            raise ValueError('state duplicate gate or candidate count missing')
        for entry in candidates:
            execution.append(run_metrics(entry, summaries[entry['run_id']], traces[entry['run_id']]))
        by_run = {e['run_id']: e for e in candidates}
        distance_rows = []
        for entry in candidates:
            row = {'run_id': entry['run_id'], 'goal_class': entry['goal_class'],
                   'distance_to': {}}
            for other in candidates:
                row['distance_to'][other['run_id']] = realized_distance(
                    traces[entry['run_id']], traces[other['run_id']])
            distance_rows.append(row)
        core = [r for r in distance_rows if r['goal_class'] not in ('far_ood', 'far-OOD')]
        core_ids = [r['run_id'] for r in core]
        if all(all(r['distance_to'][other] is not None for other in core_ids) for r in core):
            mode_count = greedy_modes(core, max(MODE_SEPARATION_M, gate['xy_rms_m']))
        else:
            mode_count = None
        p_distances = {}
        c_distances = {}
        r_distances = {}
        for index, left in enumerate(candidates):
            for right in candidates[index + 1:]:
                key = left['run_id'] + '|' + right['run_id']
                p_distances[key] = path_distance(summaries[left['run_id']]['map_path'],
                                                 summaries[right['run_id']]['map_path'])
                c_distances[key] = series_distance(traces[left['run_id']], traces[right['run_id']], 'command')
                r_distances[key] = realized_distance(traces[left['run_id']], traces[right['run_id']])
        diversity.append({'state_id': state_id, 'terrain': group[0]['terrain'],
                          'duplicate_xy_rms_m': gate['xy_rms_m'],
                          'planned_path_distance_m': p_distances,
                          'command_distance': c_distances,
                          'realized_distance_m': r_distances,
                          'core_mode_count': mode_count,
                          'core_mode_separation_m': max(MODE_SEPARATION_M, gate['xy_rms_m'])})
        outcomes = []
        for entry in group:
            rows = traces[entry['run_id']]
            summary = summaries[entry['run_id']]
            if len(rows) >= 40 and (len(rows) > 40 or summary['suffix_end_critic_value'] is not None):
                anchor = np.asarray(summary['map_path'][0], dtype=float)
                target = goal_map(summary, entry['goal_xy'])
                metrics = endpoint_metrics(rows, 40, rows[40]['critic_value'] if len(rows) > 40 else
                                           summary['suffix_end_critic_value'], anchor, target,
                                           summary['map_path'])
            else:
                metrics = None
            if len(rows) == 120 and summary['suffix_end_critic_value'] is not None:
                anchor = np.asarray(summary['map_path'][0], dtype=float)
                target = goal_map(summary, entry['goal_xy'])
                secondary = endpoint_metrics(rows, 120, summary['suffix_end_critic_value'],
                                             anchor, target, summary['map_path'])
            else:
                secondary = None
            outcomes.append({'run_id': entry['run_id'], 'kind': entry['kind'],
                             'goal_class': entry['goal_class'], 'seed': entry['seed'],
                             'primary_40_step': metrics,
                             'secondary_120_step': secondary,
                             'termination': rows[-1]['terminal'] if rows else None,
                             'raw_joint_effort_peak_abs': float(np.max(np.abs(
                                 np.asarray([r['raw_joint_effort'] for r in rows], dtype=float))))})
        candidate_outcomes = [o for o in outcomes if o['kind'] == 'candidate' and o['primary_40_step']]
        duplicate_outcomes = [o for o in outcomes if o['kind'] == 'duplicate' and o['primary_40_step']]
        values = [o['primary_40_step']['delta_v'] for o in candidate_outcomes]
        nominal = [o['primary_40_step']['delta_v'] for o in duplicate_outcomes]
        delta_range = max(values) - min(values) if len(values) == 8 else None
        nominal_range = max(nominal) - min(nominal) if len(nominal) == 3 else None
        required = max(nominal_range, DELTA_V_EPSILON) if nominal_range is not None else None
        other_difference = (len({round(o['primary_40_step'][key], 9) for o in candidate_outcomes}) > 1
                            for key in ('advantage_h', 'drawdown'))
        relative.append({'state_id': state_id, 'terrain': group[0]['terrain'],
                         'outcomes': outcomes, 'candidate_delta_v_range': delta_range,
                         'nominal_delta_v_range': nominal_range,
                         'required_delta_v_range': required,
                         'advantage_or_drawdown_differs': any(other_difference),
                         'pass': bool(delta_range is not None and required is not None and
                                      delta_range > required and any(
                                          len({round(o['primary_40_step'][key], 9) for o in candidate_outcomes}) > 1
                                          for key in ('advantage_h', 'drawdown')))})
        required_fields = ('dem_history', 'mission_logged_goal', 'candidate_local_goal',
                           'planned_candidate_path', 'follower_command', 'actor_input',
                           'actor_latent', 'previous_action', 'root_state', 'joint_position',
                           'raw_joint_effort')
        schema.append({'state_id': state_id, 'terrain': group[0]['terrain'],
                       'candidate_schema_complete': all(all(
                           key in traces[e['run_id']][0] and traces[e['run_id']][0][key] is not None
                           for key in required_fields) for e in candidates),
                       'raw_joint_effort_widths': sorted({len(traces[e['run_id']][0]['raw_joint_effort'])
                                                          for e in candidates}),
                       'real_robot_equivalent_effort': 'NA / not verified'})
    core = [row for row in execution if row['goal_class'] in ('id', 'near_ood', 'ID', 'near-OOD')]
    by_state_reach = {state: sum(row['primary_reach'] for row in core if row['state_id'] == state) /
                      sum(row['state_id'] == state for row in core)
                      for state in by_state} if len(core) == 16 else {}
    reach_rate = sum(row['primary_reach'] for row in core) / len(core) if core else None
    gs1 = bool(reach_rate is not None and reach_rate >= .75 and
               sum(rate >= .75 for rate in by_state_reach.values()) >= 3 and
               not any(row['catastrophic_termination'] for row in execution))
    gs2 = sum((row['core_mode_count'] or 0) >= 2 for row in diversity) >= 3
    gs3 = sum(row['pass'] for row in relative) >= 3
    gs4 = all(row['candidate_schema_complete'] and row['raw_joint_effort_widths'] == [52] for row in schema)
    atomic(OUT / 'goal-execution-results-v2.json', {'runs': execution, 'core_primary_reach_rate': reach_rate,
        'state_core_primary_reach_rate': by_state_reach, 'GS1_pass': gs1})
    atomic(OUT / 'pcr-diversity-results-v2.json', {'states': diversity, 'GS2_pass': gs2})
    atomic(OUT / 'relative-outcome-results-v2.json', {'states': relative, 'GS3_pass': gs3})
    atomic(OUT / 'evaluator-input-readiness-v2.json', {'states': schema, 'GS4_simulator_schema_pass': gs4})
    decision = ('proceed_support_aware_subgoal_collection' if all((gs1, gs2, gs3, gs4)) else
                'local_subgoal_execution_not_supported' if not gs1 else
                'goal_execution_ok_but_outcome_diversity_insufficient')
    overall = ('v2_pilot_complete_gs_pass' if all((gs1, gs2, gs3, gs4)) else
               'v2_pilot_complete_gs_partial' if any((gs1, gs2, gs3, gs4)) else
               'v2_pilot_complete_gs_fail')
    atomic(OUT / 'final-decision-v2-schedule-storage.json', {'status': overall, 'decision': decision,
        'GS1': gs1, 'GS2': gs2, 'GS3': gs3, 'GS4_simulator': gs4,
        'heldout_used': False, 'real_robot_equivalent_effort': 'NA / not verified',
        'candidate_count': 32, 'duplicate_count': 12,
        'schedule_sha256': sha(OUT / 'run-schedule-v2.json'),
        'inputs_sha256': {name: sha(OUT / name) for name in (
            'goal-execution-results-v2.json', 'pcr-diversity-results-v2.json',
            'relative-outcome-results-v2.json', 'evaluator-input-readiness-v2.json')}})
    print('V10_V11_ANALYSIS_COMPLETE', overall, flush=True)


if __name__ == '__main__':
    main()
