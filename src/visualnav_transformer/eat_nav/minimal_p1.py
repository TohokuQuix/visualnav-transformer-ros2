"""Outcome-blind four-anchor P1 gate, manifest and resource preflight."""

from __future__ import annotations

import itertools
import math
from pathlib import Path

SLOTS = (('first', 'R-differential'), ('second', 'R-differential'),
         ('first', 'R-transition-common'), ('second', 'R-transition-common'))
BEARINGS = (-20, -10, 0, 10, 20)


def _valid(row: dict, role: str) -> bool:
    if (row['primary_role'] != role or not row['physical_eligible'] or
            not row['hard_valid'] or not row['follower_compatible'] or
            len(row['dem_history_sha256']) != 4 or len(row['body_state_actor']) != 22 or
            not all(math.isfinite(float(v)) for v in row['body_state_actor']) or
            len(row['paths']) != 5 or not all(p['known'] for p in row['paths'])):
        return False
    if role == 'R-differential':
        return row['pair_count'] >= 1 and row['max_pair_rms_m'] >= .01
    return any(p['class_proxy'] == 'transition' and p['onset_m'] is not None
               and 0 <= p['onset_m'] <= .6 and
               p['class_diagnostic']['maximum_step_m'] >= .05 for p in row['paths'])


def _role_margin(row: dict, role: str) -> float:
    if role == 'R-differential':
        return float(row['max_pair_rms_m'])
    return max(float(p['class_diagnostic']['maximum_step_m']) - .05
               for p in row['paths'] if p['class_proxy'] == 'transition'
               and p['onset_m'] is not None and p['onset_m'] <= .6)


def _first_onset(row: dict) -> float:
    values = [float(p['onset_m']) for p in row['paths']
              if p['class_proxy'] == 'transition' and p['onset_m'] is not None
              and p['onset_m'] <= .6]
    return min(values) if values else math.inf


def select_four(rows: list[dict]) -> dict:
    choices = [[row for row in rows if row['seed_label'] == seed and _valid(row, role)]
               for seed, role in SLOTS]
    alternatives = [[{'anchor_id': x['anchor_id'], 'role_margin': _role_margin(x, role),
                      'first_onset_m': _first_onset(x) if role == 'R-transition-common' else None,
                      'source_family': x['source_family'], 'selection_rank': x['selection_rank']}
                     for x in slot] for (_, role), slot in zip(SLOTS, choices)]
    if any(not slot for slot in choices):
        return {'pass': False, 'decision': 'minimal_p1_anchor_support_invalid',
                'selected': [], 'alternatives_by_slot': alternatives,
                'missing_slots': [dict(seed_label=seed, primary_role=role)
                                  for (seed, role), slot in zip(SLOTS, choices) if not slot]}
    def score(group):
        margins = tuple(-_role_margin(row, role) for row, (_, role) in zip(group, SLOTS))
        onsets = tuple(_first_onset(row) for row, (_, role) in zip(group, SLOTS)
                       if role == 'R-transition-common')
        identities = {(row['seed_label'], row['source_family']) for row in group}
        return (margins, onsets, -len(identities),
                tuple(row['selection_rank'] for row in group),
                tuple(row['anchor_id'] for row in group))
    combinations = [group for group in itertools.product(*choices)
                    if len({row['anchor_id'] for row in group}) == 4]
    if not combinations:
        return {'pass': False, 'decision': 'minimal_p1_anchor_support_invalid',
                'selected': [], 'alternatives_by_slot': alternatives,
                'missing_slots': []}
    selected = min(combinations, key=score)
    return {'pass': True, 'decision': None,
            'selected': [row['anchor_id'] for row in selected],
            'alternatives_by_slot': alternatives, 'missing_slots': [],
            'selected_source_identity_count': len({(row['seed_label'], row['source_family'])
                                                   for row in selected})}


def build_manifest(anchors: list[dict], curves: list[dict], *, output_root: str,
                   source_sha: dict, seed: int) -> dict:
    if len(anchors) != 4 or len({a['anchor_id'] for a in anchors}) != 4:
        raise ValueError('exactly four distinct anchors required')
    if [int(c['bearing_deg']) for c in curves] != list(BEARINGS):
        raise ValueError('five frozen primitive bearings and order required')
    rows = []
    for index, anchor in enumerate(anchors):
        for curve in curves:
            bearing = int(curve['bearing_deg'])
            for repeat in range(3 if bearing == 0 else 1):
                run_id = f"p1-{index:02d}-{anchor['anchor_id']}-b{bearing:+03d}-r{repeat:02d}"
                if run_id != Path(run_id).name:
                    raise ValueError('unsafe run identity')
                rows.append({'order': len(rows), 'physical_run_id': run_id,
                             'goal_label_id': run_id+':goal:+00',
                             'anchor_id': anchor['anchor_id'],
                             'terrain_seed': anchor['terrain_seed'],
                             'source_family': anchor['source_family'],
                             'terrain_role': anchor['primary_role'],
                             'primitive_bearing_deg': bearing, 'repeat_id': repeat,
                             'candidate_path_sha256': curve['path_sha256'],
                             'local_primitive_endpoint': {'frame_id': 'base',
                                                          'base_xy_m': list(curve['endpoint'])},
                             'navigation_global_goal': {'frame_id': 'base',
                                                        'base_xy_m': [.75, 0.]},
                             'random_seed': int(seed), 'candidate_retry_count': 0,
                             'output_directory': str(Path(output_root)/'runs'/run_id),
                             'source_sha256': source_sha})
    if len(rows) != 28 or len({r['physical_run_id'] for r in rows}) != 28:
        raise ValueError('28 unique physical runs required')
    return {'schema': 'eatnav_minimal_p1_manifest_v1', 'rows': rows,
            'fresh_process_count': 28, 'direct_repeat_process_count': 12,
            'non_direct_process_count': 16, 'candidate_retry_count': 0}


def forecast_resource(measured: dict, *, wall_cap_hours: float, storage_cap_gb: float,
                      container_anon_shmem_cap_gib: float = 6.,
                      host_available_ram_gib: float | None = None) -> dict:
    p1 = measured['P1_28_processes']
    gpu_peak = float(measured['measured_gpu_device_peak_mib'])
    gpu_with_margin = 1.2 * gpu_peak
    gpu_cap = float(measured['gpu_device_total_hard_cap_mib'])
    anon = measured.get('measured_process_tree_anon_plus_shmem_peak_gib')
    failures = []
    if p1['p95_plus_20_percent_hours'] > wall_cap_hours: failures.append('wall_time_margin')
    if p1['storage_plus_20_percent_gb'] > storage_cap_gb: failures.append('storage_margin')
    if gpu_with_margin > gpu_cap: failures.append('gpu_device_margin')
    if anon is not None and 1.2 * float(anon) > container_anon_shmem_cap_gib:
        failures.append('container_anon_shmem_margin')
    if host_available_ram_gib is not None and anon is not None and 1.2 * float(anon) > host_available_ram_gib:
        failures.append('host_available_ram_margin')
    return {'pass': not failures, 'failures': failures, 'margin_factor': 1.2,
            'gpu_device_peak_measured_mib': gpu_peak,
            'gpu_device_peak_with_margin_mib': gpu_with_margin,
            'gpu_device_cap_mib': gpu_cap,
            'wall_hours_with_margin': p1['p95_plus_20_percent_hours'],
            'wall_cap_hours': wall_cap_hours,
            'storage_gb_with_margin': p1['storage_plus_20_percent_gb'],
            'storage_cap_gb': storage_cap_gb,
            'container_anon_shmem_gib_with_margin': 1.2 * float(anon) if anon is not None else None,
            'container_anon_shmem_cap_gib': container_anon_shmem_cap_gib,
            'host_available_ram_gib': host_available_ram_gib}
