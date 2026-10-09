"""Read-only 112-scene versus P1 start-state comparison."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def _array(value):
    return np.asarray(value, dtype=np.float64)


def _max_abs(a, b):
    a, b = _array(a), _array(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return math.inf
    return float(np.max(np.abs(a-b)))


def _orientation(a, b):
    a, b = _array(a), _array(b)
    return float(2 * math.acos(min(1.0, abs(float(np.dot(a, b))) /
                                   max(1e-12, float(np.linalg.norm(a)*np.linalg.norm(b))))))


def compare_prepath(records: list[dict], plan: dict, backup: Path, tolerance: dict) -> dict:
    """Compare all 28 target envs; a failed gate never authorizes Path publication."""
    by_id = {record['physical_run_id']: record for record in records}
    if len(by_id) != 28 or len(records) != 28:
        raise ValueError('pre-Path record count/duplicate mismatch')
    result = []
    for item in plan['targets']:
        run_id = item['physical_run_id']
        observed = by_id[run_id]
        run = Path(backup) / 'runs' / run_id / 'run'
        pose = item['source_pose_id']
        reference_state = json.loads((run / 'trace' / f'{pose}.jsonl').read_text().splitlines()[-1])['state']
        reference_actor = json.loads((run / 'actor' / pose / 'settled_step_200.json').read_text())
        gold = reference_state['root_state_w']
        live = observed['root_state_w']
        source_origin = item['source_tile_origin_xy']
        target_origin = observed['target_tile_origin_xy']
        root_local_error = _max_abs([live[0]-target_origin[0], live[1]-target_origin[1], live[2]],
                                    [gold[0]-source_origin[0], gold[1]-source_origin[1], gold[2]])
        checks = {
            'terrain_seed_exact': observed['terrain_metadata']['seed'] == item['terrain_seed'],
            'terrain_type_exact': observed['terrain_metadata']['terrain_type'] == item['terrain_type'],
            'settling_no_termination': not observed['terminated_during_settling'],
            'root_local_position': root_local_error <= tolerance['root_position_m'],
            'root_orientation': _orientation(live[3:7], gold[3:7]) <= tolerance['root_orientation_rad'],
            'root_velocity': _max_abs(live[7:13], gold[7:13]) <= tolerance['root_velocity_mps_or_radps'],
            'joint_position': _max_abs(observed['joint_position'], reference_state['joint_position']) <= tolerance['joint_position_rad'],
            'joint_velocity': _max_abs(observed['joint_velocity'], reference_state['joint_velocity']) <= tolerance['joint_velocity_radps'],
            'actor_input': _max_abs(observed['actor_input'], reference_actor['values']) <= tolerance['actor_post_delay_max_abs'],
        }
        dem_metrics = []
        for j, live_frame in enumerate(observed['dem_frames']):
            gold_frame = json.loads((run / 'dem' / pose / f'frame-{j:02d}.json').read_text())
            live_dem = _array(live_frame['transformed'])
            gold_dem = _array(gold_frame['transformed'])
            mask_match = bool(np.array_equal(live_dem[1] > .5, gold_dem[1] > .5)) if live_dem.shape == gold_dem.shape else False
            height_diff = live_dem[0]-gold_dem[0] if live_dem.shape == gold_dem.shape else np.asarray([math.inf])
            rms = float(np.sqrt(np.mean(height_diff**2)))
            peak = float(np.max(np.abs(height_diff)))
            passed = (live_frame['finite'] and live_dem.shape == (2, 20, 20) and
                      mask_match and rms <= tolerance['dem_height_rms_m'] and
                      peak <= tolerance['dem_height_max_abs_m'])
            checks[f'dem_frame_{j}'] = bool(passed)
            dem_metrics.append({'frame': j, 'mask_exact': mask_match, 'height_rms_m': rms,
                                'height_max_abs_m': peak})
        failures = [key for key, ok in checks.items() if not ok]
        result.append({'physical_run_id': run_id, 'target_env_index': item['target_env_index'],
                       'checks': checks, 'first_failing_signal': failures[0] if failures else None,
                       'root_local_max_abs_m': root_local_error,
                       'actor_input_max_abs': _max_abs(observed['actor_input'], reference_actor['values']),
                       'dem_metrics': dem_metrics})
    return {'schema': 'p1_batch_prepath_audit_v1', 'target_count': 28,
            'pass': all(row['first_failing_signal'] is None for row in result),
            'Path_publications': 0, 'rows': result}
